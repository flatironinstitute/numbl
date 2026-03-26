/**
 * det and trace builtin functions
 */

import {
  RTV,
  RuntimeError,
  tensorSize2D,
  toNumber,
} from "../../runtime/index.js";
import {
  FloatXArray,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { register } from "../registry.js";
import { out, toF64, isMatrixLike } from "../check-helpers.js";
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
        if ((argTypes.length !== 2 && argTypes.length !== 3) || nargout !== 1)
          return null;
        return { outputTypes: [IType.Unknown] };
      },
      apply: args => {
        if (args.length < 2 || args.length > 3)
          throw new RuntimeError("cross requires 2 or 3 arguments");
        const a = args[0],
          b = args[1];
        if (!isRuntimeTensor(a) || !isRuntimeTensor(b))
          throw new RuntimeError("cross: arguments must be vectors or arrays");

        const shape = a.shape;
        // Validate shapes match
        if (
          shape.length !== b.shape.length ||
          shape.some((s, i) => s !== b.shape[i])
        )
          throw new RuntimeError("cross: A and B must have the same size");

        // Determine the dimension to operate along
        let dim: number;
        if (args.length === 3) {
          dim = toNumber(args[2]);
          if (!Number.isInteger(dim) || dim < 1)
            throw new RuntimeError("cross: dim must be a positive integer");
        } else {
          // Find first dimension of size 3
          dim = shape.indexOf(3) + 1; // 1-based
          if (dim === 0)
            throw new RuntimeError(
              "cross: A and B must have at least one dimension of length 3"
            );
        }

        const dimIdx = dim - 1; // 0-based
        if (dimIdx >= shape.length || shape[dimIdx] !== 3)
          throw new RuntimeError(
            `cross: size(A,${dim}) and size(B,${dim}) must be 3`
          );

        const totalLen = a.data.length;
        const result = new FloatXArray(totalLen);

        // Compute strides for column-major layout
        // stride[d] = product of shape[0..d-1]
        const strides = new Array(shape.length);
        strides[0] = 1;
        for (let d = 1; d < shape.length; d++)
          strides[d] = strides[d - 1] * shape[d - 1];

        const dimStride = strides[dimIdx];

        // Iterate over all positions excluding the cross dimension.
        // outerStride = stride of dimension dimIdx+1 (or totalLen if last dim)
        const outerStride =
          dimIdx + 1 < shape.length ? strides[dimIdx + 1] : totalLen;
        const innerSize = dimStride; // elements before the cross dim
        const numOuter = totalLen / outerStride; // number of outer blocks

        // For each cross product, compute the base index (offset of element 0
        // along the cross dimension). We iterate block-by-block.
        for (let outer = 0; outer < numOuter; outer++) {
          const blockBase = outer * outerStride;
          for (let inner = 0; inner < innerSize; inner++) {
            const base = blockBase + inner;
            const i0 = base;
            const i1 = base + dimStride;
            const i2 = base + 2 * dimStride;
            const ax = a.data[i0],
              ay = a.data[i1],
              az = a.data[i2];
            const bx = b.data[i0],
              by = b.data[i1],
              bz = b.data[i2];
            result[i0] = ay * bz - az * by;
            result[i1] = az * bx - ax * bz;
            result[i2] = ax * by - ay * bx;
          }
        }

        return RTV.tensor(result, [...shape]);
      },
    },
  ]);
}
