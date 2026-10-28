// JS sibling of `tensor_reshape_nd_complex.h`. Reshape a complex
// tensor to an N-D shape. Same auto-infer / element-count rules as
// the real reshape; both lanes are reinterpreted (no rearrangement).

import { jit_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";
import {
  jit_raise_bail,
  jit_raise_error,
  jit_raise_error_deferred,
} from "../system/raise.js";

export function jit_reshape_nd_complex(input, ndim, dims) {
  let inTotal = 1;
  for (const d of input.shape) inTotal *= d;
  let inferIdx = -1;
  let explicitProd = 1;
  for (let i = 0; i < ndim; i++) {
    if (dims[i] === -1) {
      if (inferIdx !== -1) {
        jit_raise_bail("reshape: at most one '[]' auto-infer slot allowed");
      }
      inferIdx = i;
    } else if (dims[i] < 0) {
      jit_raise_error_deferred();
    } else {
      explicitProd *= dims[i];
    }
  }
  const resolved = new Array(ndim);
  for (let i = 0; i < ndim; i++) resolved[i] = dims[i];
  let outTotal;
  if (inferIdx !== -1) {
    if (explicitProd === 0 && inTotal !== 0) {
      jit_raise_error_deferred();
    }
    if (explicitProd > 0 && inTotal % explicitProd !== 0) {
      jit_raise_error("", "reshape: number of elements must not change");
    }
    resolved[inferIdx] = explicitProd === 0 ? 0 : inTotal / explicitProd;
    outTotal = inTotal;
  } else {
    outTotal = explicitProd;
    if (inTotal !== outTotal) {
      jit_raise_error("", "reshape: number of elements must not change");
    }
  }
  const out = jit_tensor_alloc_nd_complex(ndim, resolved);
  if (outTotal > 0) {
    out.data.set(input.data.subarray(0, outTotal));
    if (input.imag !== undefined) {
      out.imag.set(input.imag.subarray(0, outTotal));
    }
  }
  return out;
}
