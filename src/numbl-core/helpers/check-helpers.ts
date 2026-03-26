/**
 * Helpers for writing builtin check functions.
 * These make check implementations shorter and easier to read.
 */

import {
  FloatXArray,
  type FloatXArrayType,
  isRuntimeChar,
  isRuntimeNumber,
  isRuntimeString,
} from "../runtime/types.js";
import { colMajorIndex, RTV, RuntimeValue } from "../runtime/index.js";
import { RuntimeError } from "../runtime/index.js";
import { getIBuiltin, inferJitType } from "../interpreter/builtins/types.js";

/** Ensure data is Float64Array (needed by LAPACK bridges). */
export function toF64(data: FloatXArrayType): Float64Array {
  return data instanceof Float64Array ? data : new Float64Array(data);
}

/**
 * Extract and normalize a string argument at runtime.
 * Strips surrounding quotes and lowercases the result.
 * Works with both raw string args and RuntimeChar/RuntimeString values.
 */
export function parseStringArgLower(arg: unknown): string {
  if (typeof arg === "string") {
    return arg.replace(/^['"]|['"]$/g, "").toLowerCase();
  }
  if (arg && typeof arg === "object" && "value" in arg) {
    return String((arg as { value: unknown }).value)
      .replace(/^['"]|['"]$/g, "")
      .toLowerCase();
  }
  return String(arg).toLowerCase();
}

/**
 * Parse runtime economy-mode argument (0 or 'econ') used by qr, svd, etc.
 * Returns true for economy mode, false otherwise.
 */
export function parseEconArgRuntime(arg: RuntimeValue | undefined): boolean {
  if (arg === undefined) return false;
  if (isRuntimeNumber(arg) && arg === 0) return true;
  if (isRuntimeString(arg) && arg.toLowerCase() === "econ") return true;
  if (isRuntimeChar(arg) && arg.value.toLowerCase() === "econ") return true;
  return false;
}

/**
 * Build a (possibly complex) eigenvector matrix from LAPACK's packed real format.
 *
 * For complex conjugate eigenvalue pairs, LAPACK stores eigenvectors as:
 *   Column j:   real part
 *   Column j+1: imaginary part
 * Eigenvector for eigenvalue j   is V(:,j) + i*V(:,j+1)
 * Eigenvector for eigenvalue j+1 is V(:,j) - i*V(:,j+1)
 */
export function buildEigenvectorMatrix(
  packedV: Float64Array,
  wi: Float64Array,
  n: number,
  hasComplex: boolean
) {
  if (!hasComplex) {
    return RTV.tensor(new FloatXArray(packedV), [n, n]);
  }

  const realPart = new FloatXArray(n * n);
  const imagPart = new FloatXArray(n * n);

  let j = 0;
  while (j < n) {
    if (Math.abs(wi[j]) === 0) {
      for (let i = 0; i < n; i++) {
        realPart[colMajorIndex(i, j, n)] = packedV[colMajorIndex(i, j, n)];
      }
      j++;
    } else {
      for (let i = 0; i < n; i++) {
        const re = packedV[colMajorIndex(i, j, n)];
        const im = packedV[colMajorIndex(i, j + 1, n)];
        realPart[colMajorIndex(i, j, n)] = re;
        imagPart[colMajorIndex(i, j, n)] = im;
        realPart[colMajorIndex(i, j + 1, n)] = re;
        imagPart[colMajorIndex(i, j + 1, n)] = -im;
      }
      j += 2;
    }
  }

  return RTV.tensor(realPart, [n, n], imagPart);
}

/**
 * Build a tensor, attaching the imaginary part only when it contains non-zero values.
 * Replaces the repeated `hasComplex ? RTV.tensor(re, shape, im) : RTV.tensor(re, shape)` pattern.
 */
export function maybeComplexTensor(
  re: FloatXArrayType | Float64Array,
  shape: number[],
  im: FloatXArrayType | Float64Array | undefined
): ReturnType<typeof RTV.tensor> {
  const imag = im && im.some(v => v !== 0) ? new FloatXArray(im) : undefined;
  return RTV.tensor(new FloatXArray(re), shape, imag);
}

/**
 * Build a diagonal matrix from a vector of values (column-major).
 * For square matrices pass just n; for rectangular pass [rows, cols].
 * Optionally includes an imaginary diagonal.
 */
export function buildDiagMatrix(
  realVals: Float64Array | FloatXArrayType,
  imagVals: Float64Array | FloatXArrayType | undefined,
  size: number | [number, number]
): ReturnType<typeof RTV.tensor> {
  const [rows, cols] = typeof size === "number" ? [size, size] : size;
  const k = Math.min(rows, cols, realVals.length);
  const dReal = new FloatXArray(rows * cols);
  for (let i = 0; i < k; i++) dReal[colMajorIndex(i, i, rows)] = realVals[i];
  if (imagVals && imagVals.some(v => v !== 0)) {
    const dImag = new FloatXArray(rows * cols);
    for (let i = 0; i < k; i++) dImag[colMajorIndex(i, i, rows)] = imagVals[i];
    return RTV.tensor(dReal, [rows, cols], dImag);
  }
  return RTV.tensor(dReal, [rows, cols]);
}

/**
 * In-place Gauss-Jordan elimination with partial pivoting on a column-major
 * augmented matrix [A | B] of size `rows × totalCols`.
 */
export function gaussJordanEliminate(
  aug: FloatXArrayType,
  rows: number,
  totalCols: number
): void {
  for (let col = 0; col < rows; col++) {
    let maxVal = Math.abs(aug[col * rows + col]);
    let maxRow = col;
    for (let row = col + 1; row < rows; row++) {
      const val = Math.abs(aug[col * rows + row]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      for (let j = 0; j < totalCols; j++) {
        const tmp = aug[j * rows + col];
        aug[j * rows + col] = aug[j * rows + maxRow];
        aug[j * rows + maxRow] = tmp;
      }
    }
    const pivot = aug[col * rows + col];
    if (Math.abs(pivot) < 1e-14) continue;
    for (let j = 0; j < totalCols; j++) aug[j * rows + col] /= pivot;
    for (let row = 0; row < rows; row++) {
      if (row === col) continue;
      const factor = aug[col * rows + row];
      for (let j = 0; j < totalCols; j++) {
        aug[j * rows + row] -= factor * aug[j * rows + col];
      }
    }
  }
}

/**
 * Call a registered builtin from within another builtin's apply().
 * Convenience wrapper that includes the caller name in error messages.
 */
export function applyBuiltin(
  caller: string,
  name: string,
  args: RuntimeValue[],
  nargout: number
): RuntimeValue {
  const ib = getIBuiltin(name);
  if (!ib) throw new RuntimeError(`${caller}: builtin '${name}' not found`);
  const argTypes = args.map(inferJitType);
  const resolution = ib.resolve(argTypes, nargout);
  if (!resolution)
    throw new RuntimeError(`${caller}: builtin '${name}' returned no result`);
  const result = resolution.apply(args, nargout);
  if (Array.isArray(result)) return result[0];
  return result;
}
