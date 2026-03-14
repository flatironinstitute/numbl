/**
 * Matrix inversion builtin function
 */

import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { register } from "../registry.js";
import { out, toF64, unknownMatrix, isMatrixLike } from "./check-helpers.js";

// ── Matrix inversion helpers ─────────────────────────────────────────────────

/**
 * Invert an n×n real matrix via LAPACK (dgetrf + dgetri).
 * Uses the native addon when available, otherwise falls back to ts-lapack.
 * Input/output are column-major Float64Arrays.
 */
function invLapack(data: FloatXArrayType, n: number): Float64Array {
  const bridge = getEffectiveBridge("inv");
  return bridge.inv(toF64(data), n);
}

/**
 * Invert an n×n complex matrix via LAPACK (zgetrf + zgetri).
 * Input/output are column-major Float64Arrays for real and imaginary parts.
 * Returns null if the bridge is unavailable.
 */
function invLapackComplex(
  dataRe: FloatXArrayType,
  dataIm: FloatXArrayType,
  n: number
): { re: Float64Array; im: Float64Array } | null {
  const bridge = getLapackBridge();
  if (!bridge || !bridge.invComplex) return null;
  return bridge.invComplex(toF64(dataRe), toF64(dataIm), n);
}

/**
 * Invert an n×n complex matrix using Gauss-Jordan elimination (pure JS).
 * Input: real and imaginary parts in column-major layout.
 * Output: inverted real and imaginary parts in column-major layout.
 */
function invComplexJS(
  dataRe: FloatXArrayType,
  dataIm: FloatXArrayType,
  n: number
): { re: FloatXArrayType; im: FloatXArrayType } {
  // Build augmented matrix [A | I] in row-major layout
  // Each element is complex: (re, im)
  // Row r occupies columns [0..n-1] (from A) and [n..2n-1] (from I)
  const augRe = new Float64Array(n * 2 * n);
  const augIm = new Float64Array(n * 2 * n);

  // Fill A (column-major → row-major) and I
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      augRe[row * 2 * n + col] = dataRe[row + col * n];
      augIm[row * 2 * n + col] = dataIm[row + col * n];
    }
    augRe[row * 2 * n + n + row] = 1; // identity block
    augIm[row * 2 * n + n + row] = 0;
  }

  // Forward (and backward) elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot row (maximum magnitude)
    let maxRow = col;
    let maxMag = augRe[col * 2 * n + col] ** 2 + augIm[col * 2 * n + col] ** 2;
    for (let row = col + 1; row < n; row++) {
      const mag = augRe[row * 2 * n + col] ** 2 + augIm[row * 2 * n + col] ** 2;
      if (mag > maxMag) {
        maxMag = mag;
        maxRow = row;
      }
    }
    if (maxMag < 1e-300)
      throw new RuntimeError("inv: matrix is singular or nearly singular");

    // Swap rows col ↔ maxRow
    if (maxRow !== col) {
      for (let k = 0; k < 2 * n; k++) {
        let tmp = augRe[col * 2 * n + k];
        augRe[col * 2 * n + k] = augRe[maxRow * 2 * n + k];
        augRe[maxRow * 2 * n + k] = tmp;
        tmp = augIm[col * 2 * n + k];
        augIm[col * 2 * n + k] = augIm[maxRow * 2 * n + k];
        augIm[maxRow * 2 * n + k] = tmp;
      }
    }

    // Scale pivot row so the pivot element becomes 1
    // For complex division: 1/(a+bi) = (a-bi)/(a²+b²)
    const pivotRe = augRe[col * 2 * n + col];
    const pivotIm = augIm[col * 2 * n + col];
    const pivotMagSq = pivotRe * pivotRe + pivotIm * pivotIm;
    const invPivotRe = pivotRe / pivotMagSq;
    const invPivotIm = -pivotIm / pivotMagSq;

    for (let k = col; k < 2 * n; k++) {
      const re = augRe[col * 2 * n + k];
      const im = augIm[col * 2 * n + k];
      augRe[col * 2 * n + k] = re * invPivotRe - im * invPivotIm;
      augIm[col * 2 * n + k] = re * invPivotIm + im * invPivotRe;
    }

    // Eliminate this column in every other row
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factorRe = augRe[row * 2 * n + col];
      const factorIm = augIm[row * 2 * n + col];
      if (factorRe === 0 && factorIm === 0) continue;
      for (let k = col; k < 2 * n; k++) {
        const multRe =
          factorRe * augRe[col * 2 * n + k] - factorIm * augIm[col * 2 * n + k];
        const multIm =
          factorRe * augIm[col * 2 * n + k] + factorIm * augRe[col * 2 * n + k];
        augRe[row * 2 * n + k] -= multRe;
        augIm[row * 2 * n + k] -= multIm;
      }
    }
  }

  // Extract the right half (the inverse) and convert back to column-major
  const resultRe = new FloatXArray(n * n);
  const resultIm = new FloatXArray(n * n);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      resultRe[row + col * n] = augRe[row * 2 * n + n + col];
      resultIm[row + col * n] = augIm[row * 2 * n + n + col];
    }
  }
  return { re: resultRe, im: resultIm };
}

export function registerInv(): void {
  /**
   * Matrix inversion: B = inv(A)
   * Uses the native LAPACK addon when available (Node.js), otherwise falls
   * back to the pure-TypeScript ts-lapack implementation.
   */
  register("inv", [
    {
      check: (argTypes, nargout) => {
        if (argTypes.length !== 1 || nargout !== 1) return null;
        const A = argTypes[0];
        if (!isMatrixLike(A)) return null;
        return out(A.kind === "Unknown" ? unknownMatrix() : A);
      },
      apply: args => {
        if (args.length < 1) throw new RuntimeError("inv requires 1 argument");
        const A = args[0];

        // Complex scalar shortcut
        if (isRuntimeComplexNumber(A)) {
          const { re, im } = A;
          const magSq = re * re + im * im;
          if (magSq === 0) throw new RuntimeError("inv: argument is singular");
          // 1/(a+bi) = (a-bi)/(a²+b²)
          return RTV.complex(re / magSq, -im / magSq);
        }

        // Real scalar shortcut
        if (isRuntimeNumber(A)) {
          if (A === 0) throw new RuntimeError("inv: argument is singular");
          return RTV.num(1 / A);
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("inv: argument must be numeric");

        const [m, n] = tensorSize2D(A);
        if (m !== n) throw new RuntimeError("inv: matrix must be square");

        // Complex matrix inversion: try LAPACK first
        if (A.imag !== undefined) {
          const lapackResult = invLapackComplex(A.data, A.imag, n);
          if (lapackResult) {
            return RTV.tensor(
              new FloatXArray(lapackResult.re),
              [n, n],
              new FloatXArray(lapackResult.im)
            );
          }
          // JS fallback for complex matrices
          const result = invComplexJS(A.data, A.imag, n);
          return RTV.tensor(result.re, [n, n], result.im);
        }

        // Real matrix inversion via LAPACK (native or ts-lapack fallback)
        return RTV.tensor(new FloatXArray(invLapack(A.data, n)), [n, n]);
      },
    },
  ]);
}
