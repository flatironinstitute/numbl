// JS sibling of `tensor_zeros_nd.h`. Float64Array initialises to
// zeros automatically, so `_zeros_nd` is just an alias for the
// allocator.

import { jit_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

export function jit_tensor_zeros_nd(ndim, dims) {
  return jit_tensor_alloc_nd(ndim, dims);
}
