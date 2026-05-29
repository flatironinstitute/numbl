// JS sibling of `tensor_reshape_nd.h`. Reshape a real tensor to an
// N-D shape, supporting one `-1` auto-infer slot. Same error
// behaviour as the C side: throws (instead of `abort()`) on bad
// inputs.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

export function mtoc2_reshape_nd(input, ndim, dims) {
  let inTotal = 1;
  for (const d of input.shape) inTotal *= d;
  let inferIdx = -1;
  let explicitProd = 1;
  for (let i = 0; i < ndim; i++) {
    if (dims[i] === -1) {
      if (inferIdx !== -1) {
        throw new Error("reshape: at most one '[]' auto-infer slot allowed");
      }
      inferIdx = i;
    } else if (dims[i] < 0) {
      throw new Error(
        `reshape: dim ${i + 1} must be a non-negative integer (got ${dims[i]})`
      );
    } else {
      explicitProd *= dims[i];
    }
  }
  const resolved = new Array(ndim);
  for (let i = 0; i < ndim; i++) resolved[i] = dims[i];
  let outTotal;
  if (inferIdx !== -1) {
    if (explicitProd === 0 && inTotal !== 0) {
      throw new Error(
        `reshape: input has ${inTotal} elements but explicit dims around '[]' multiply to 0`
      );
    }
    if (explicitProd > 0 && inTotal % explicitProd !== 0) {
      throw new Error(
        `reshape: input has ${inTotal} elements, not divisible by ${explicitProd}`
      );
    }
    resolved[inferIdx] = explicitProd === 0 ? 0 : inTotal / explicitProd;
    outTotal = inTotal;
  } else {
    outTotal = explicitProd;
    if (inTotal !== outTotal) {
      throw new Error(
        `reshape: number of elements must not change (in=${inTotal}, out=${outTotal})`
      );
    }
  }
  const out = mtoc2_tensor_alloc_nd(ndim, resolved);
  if (outTotal > 0) out.data.set(input.data.subarray(0, outTotal));
  return out;
}
