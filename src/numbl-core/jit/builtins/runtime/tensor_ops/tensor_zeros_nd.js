// JS sibling of `tensor_zeros_nd.h`. Float64Array initialises to
// zeros automatically, so `_zeros_nd` is just an alias for the
// allocator.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

export function mtoc2_tensor_zeros_nd(ndim, dims) {
  return mtoc2_tensor_alloc_nd(ndim, dims);
}
