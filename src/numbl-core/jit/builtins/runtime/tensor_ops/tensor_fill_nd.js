// JS sibling of `tensor_fill_nd.h`. Like zeros/ones but takes the
// fill value as a leading argument — used by the `nan` / `Inf` shape-
// constructor branches and by `repmat(scalar, ...)`. Complex variant
// takes `(re, im)` and fills both lanes.

import { jit_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { jit_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";

export function jit_tensor_fill_nd(value, ndim, dims) {
  const t = jit_tensor_alloc_nd(ndim, dims);
  t.data.fill(value);
  return t;
}

export function jit_tensor_fill_nd_complex(re, im, ndim, dims) {
  const t = jit_tensor_alloc_nd_complex(ndim, dims);
  t.data.fill(re);
  t.imag.fill(im);
  return t;
}
