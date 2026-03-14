/**
 * det and trace builtin functions
 */

import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import {
  FloatXArray,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { register } from "../registry.js";
import { out, toF64, isMatrixLike } from "./check-helpers.js";
import { IType } from "../../lowering/itemTypes.js";

/**
 * Compute determinant via LU decomposition with partial pivoting.
 * Input is column-major. Returns the determinant as a number.
 */
function detJS(data: Float32Array | Float64Array, n: number): number {
  // Copy to row-major working array
  const a = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      a[r * n + c] = data[r + c * n]; // column-major → row-major
    }
  }

  let det = 1;
  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxRow = col;
    let maxVal = Math.abs(a[col * n + col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(a[row * n + col]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }

    if (maxVal === 0) return 0; // Singular

    if (maxRow !== col) {
      // Swap rows
      for (let k = 0; k < n; k++) {
        const tmp = a[col * n + k];
        a[col * n + k] = a[maxRow * n + k];
        a[maxRow * n + k] = tmp;
      }
      det *= -1; // Row swap flips sign
    }

    const pivot = a[col * n + col];
    det *= pivot;

    for (let row = col + 1; row < n; row++) {
      const factor = a[row * n + col] / pivot;
      for (let k = col; k < n; k++) {
        a[row * n + k] -= factor * a[col * n + k];
      }
    }
  }

  return det;
}

/**
 * Compute determinant of a complex matrix via LU decomposition with partial pivoting.
 * Inputs are column-major real and imaginary parts. Returns [detRe, detIm].
 */
function detComplexJS(
  dataRe: Float32Array | Float64Array,
  dataIm: Float32Array | Float64Array,
  n: number
): [number, number] {
  // Copy to row-major working arrays
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      re[r * n + c] = dataRe[r + c * n];
      im[r * n + c] = dataIm[r + c * n];
    }
  }

  let detRe = 1;
  let detIm = 0;
  for (let col = 0; col < n; col++) {
    // Partial pivot by magnitude
    let maxRow = col;
    let maxVal =
      re[col * n + col] * re[col * n + col] +
      im[col * n + col] * im[col * n + col];
    for (let row = col + 1; row < n; row++) {
      const v =
        re[row * n + col] * re[row * n + col] +
        im[row * n + col] * im[row * n + col];
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }

    if (maxVal === 0) return [0, 0];

    if (maxRow !== col) {
      for (let k = 0; k < n; k++) {
        let tmp = re[col * n + k];
        re[col * n + k] = re[maxRow * n + k];
        re[maxRow * n + k] = tmp;
        tmp = im[col * n + k];
        im[col * n + k] = im[maxRow * n + k];
        im[maxRow * n + k] = tmp;
      }
      detRe = -detRe;
      detIm = -detIm;
    }

    const pivRe = re[col * n + col];
    const pivIm = im[col * n + col];
    // det *= pivot (complex multiply)
    const newDetRe = detRe * pivRe - detIm * pivIm;
    const newDetIm = detRe * pivIm + detIm * pivRe;
    detRe = newDetRe;
    detIm = newDetIm;

    // Eliminate below pivot: factor = row / pivot (complex division)
    const pivMag2 = pivRe * pivRe + pivIm * pivIm;
    for (let row = col + 1; row < n; row++) {
      const rRe = re[row * n + col];
      const rIm = im[row * n + col];
      const fRe = (rRe * pivRe + rIm * pivIm) / pivMag2;
      const fIm = (rIm * pivRe - rRe * pivIm) / pivMag2;
      for (let k = col; k < n; k++) {
        re[row * n + k] -= fRe * re[col * n + k] - fIm * im[col * n + k];
        im[row * n + k] -= fRe * im[col * n + k] + fIm * re[col * n + k];
      }
    }
  }

  return [detRe, detIm];
}

export function registerDet(): void {
  register("det", [
    {
      check: (argTypes, nargout) => {
        if (argTypes.length !== 1 || nargout !== 1) return null;
        if (!isMatrixLike(argTypes[0])) return null;
        return out(IType.num());
      },
      apply: args => {
        if (args.length !== 1)
          throw new RuntimeError("det requires 1 argument");
        const A = args[0];
        if (isRuntimeNumber(A)) return A;
        if (isRuntimeLogical(A)) return RTV.num(A ? 1 : 0);
        if (!isRuntimeTensor(A))
          throw new RuntimeError("det: argument must be a matrix");
        const [m, n] = tensorSize2D(A);
        if (m !== n) throw new RuntimeError("det: matrix must be square");
        if (A.imag) {
          const [detRe, detIm] = detComplexJS(toF64(A.data), toF64(A.imag), n);
          if (Math.abs(detIm) < 1e-15) return RTV.num(detRe);
          return RTV.complex(detRe, detIm);
        }
        return RTV.num(detJS(toF64(A.data), n));
      },
    },
  ]);

  register("trace", [
    {
      check: (argTypes, nargout) => {
        if (argTypes.length !== 1 || nargout !== 1) return null;
        if (!isMatrixLike(argTypes[0])) return null;
        return out(IType.num());
      },
      apply: args => {
        if (args.length !== 1)
          throw new RuntimeError("trace requires 1 argument");
        const A = args[0];
        if (isRuntimeNumber(A)) return A;
        if (!isRuntimeTensor(A))
          throw new RuntimeError("trace: argument must be a matrix");
        const [rows, cols] = tensorSize2D(A);
        const n = Math.min(rows, cols);
        let sum = 0;
        for (let i = 0; i < n; i++) {
          sum += A.data[i + i * rows]; // column-major: element (i,i)
        }
        return RTV.num(sum);
      },
    },
  ]);

  register("cross", [
    {
      check: (argTypes, nargout) => {
        if (argTypes.length !== 2 || nargout !== 1) return null;
        return { outputTypes: [IType.Unknown] };
      },
      apply: args => {
        if (args.length !== 2)
          throw new RuntimeError("cross requires 2 arguments");
        const a = args[0],
          b = args[1];
        if (!isRuntimeTensor(a) || !isRuntimeTensor(b))
          throw new RuntimeError("cross: arguments must be vectors");

        const aRows = a.shape[0];
        const aCols = a.shape.length >= 2 ? a.shape[1] : 1;
        const bRows = b.shape[0];
        const bCols = b.shape.length >= 2 ? b.shape[1] : 1;

        // Determine if inputs are column vectors (Nx1), row vectors (1xN), or matrices (3xM)
        const isColVector = aRows === 3 && aCols === 1;
        const isRowVector = aRows === 1 && aCols === 3;
        const isMatrix = aRows === 3 && aCols > 1;

        if (isColVector || isRowVector) {
          // Vector case
          if (a.data.length !== 3 || b.data.length !== 3)
            throw new RuntimeError("cross: vectors must have 3 elements");
          const ax = a.data[0],
            ay = a.data[1],
            az = a.data[2];
          const bx = b.data[0],
            by = b.data[1],
            bz = b.data[2];
          const result = new FloatXArray(3);
          result[0] = ay * bz - az * by;
          result[1] = az * bx - ax * bz;
          result[2] = ax * by - ay * bx;
          return RTV.tensor(result, isColVector ? [3, 1] : [1, 3]);
        } else if (isMatrix) {
          // Matrix case: column-wise cross products
          if (bRows !== 3 || bCols !== aCols)
            throw new RuntimeError("cross: matrix dimensions must agree");
          const cols = aCols;
          const result = new FloatXArray(3 * cols);
          for (let c = 0; c < cols; c++) {
            const off = c * 3; // column-major, 3 rows
            const ax = a.data[off],
              ay = a.data[off + 1],
              az = a.data[off + 2];
            const bx = b.data[off],
              by = b.data[off + 1],
              bz = b.data[off + 2];
            result[off] = ay * bz - az * by;
            result[off + 1] = az * bx - ax * bz;
            result[off + 2] = ax * by - ay * bx;
          }
          return RTV.tensor(result, [3, cols]);
        } else {
          throw new RuntimeError(
            "cross: inputs must be 3-element vectors or 3xN matrices"
          );
        }
      },
    },
  ]);
}
