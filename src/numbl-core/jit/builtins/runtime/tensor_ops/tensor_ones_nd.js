// JS sibling of `tensor_ones_nd.h`. Fill the freshly-allocated
// tensor with `1.0`.

import { jit_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

export function jit_tensor_ones_nd(ndim, dims) {
  const t = jit_tensor_alloc_nd(ndim, dims);
  t.data.fill(1);
  return t;
}
