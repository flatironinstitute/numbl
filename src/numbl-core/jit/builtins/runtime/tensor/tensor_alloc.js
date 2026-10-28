// JS sibling of `tensor_alloc.h`. 2-D allocator (R × C). The C side
// hardcodes a 2-D fast path next to the more general `_alloc_nd`;
// JS keeps both for structural parity.

import { jit_tensor_make } from "./tensor.js";

export function jit_tensor_alloc(rows, cols) {
  return jit_tensor_make([rows, cols], new Float64Array(rows * cols));
}
