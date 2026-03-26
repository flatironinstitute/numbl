/**
 * vecnorm – vector-wise norm along a dimension
 */

import { RTV, toNumber, RuntimeError } from "../../runtime/index.js";
import {
  FloatXArray,
  isRuntimeNumber,
  isRuntimeComplexNumber,
  isRuntimeTensor,
  type RuntimeTensor,
} from "../../runtime/types.js";
import { register, builtinSingle } from "../registry.js";
import { forEachSlice, copyTensor } from "../reduction-helpers.js";

function vecnormImpl(
  args: import("../../runtime/types.js").RuntimeValue[]
): import("../../runtime/types.js").RuntimeValue {
  if (args.length < 1)
    throw new RuntimeError("vecnorm requires at least 1 argument");

  const v = args[0];

  // Parse p (default 2)
  let p = 2;
  if (args.length >= 2) p = toNumber(args[1]);

  // Scalar input
  if (isRuntimeNumber(v)) return RTV.num(Math.abs(v));
  if (isRuntimeComplexNumber(v)) return RTV.num(Math.hypot(v.re, v.im));

  if (!isRuntimeTensor(v))
    throw new RuntimeError("vecnorm: argument must be numeric");

  // Parse dim (default: first non-singleton)
  let dim: number;
  if (args.length >= 3) {
    dim = Math.round(toNumber(args[2]));
  } else {
    // Find first dimension with size > 1
    const idx = v.shape.findIndex(d => d > 1);
    dim = idx === -1 ? 1 : idx + 1;
  }

  return vecnormAlongDim(v, p, dim);
}

function vecnormAlongDim(
  v: RuntimeTensor,
  p: number,
  dim: number
): import("../../runtime/types.js").RuntimeValue {
  const dimIdx = dim - 1;

  // dim exceeds rank: return abs(A)
  if (dimIdx >= v.shape.length) {
    const result = new FloatXArray(v.data.length);
    const imag = v.imag;
    for (let i = 0; i < v.data.length; i++) {
      result[i] = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
    }
    return RTV.tensor(result, [...v.shape]);
  }

  const info = forEachSlice(v.shape, dim, () => {});
  if (!info) return copyTensor(v);

  const result = new FloatXArray(info.totalElems);
  const imag = v.imag;

  forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
    if (p === Infinity) {
      let m = 0;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        if (a > m) m = a;
      }
      result[outIdx] = m;
    } else if (p === -Infinity) {
      let m = Infinity;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        if (a < m) m = a;
      }
      result[outIdx] = m;
    } else {
      let s = 0;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        s += Math.pow(a, p);
      }
      result[outIdx] = Math.pow(s, 1 / p);
    }
  });

  return RTV.tensor(result, info.resultShape);
}

export function registerVecnorm(): void {
  register(
    "vecnorm",
    builtinSingle(args => vecnormImpl(args))
  );
}
