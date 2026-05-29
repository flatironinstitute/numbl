// JS sibling of `tensor_alloc.h`. 2-D allocator (R × C). The C side
// hardcodes a 2-D fast path next to the more general `_alloc_nd`;
// JS keeps both for structural parity.

import { mtoc2_tensor_make } from "./tensor.js";

export function mtoc2_tensor_alloc(rows, cols) {
  return mtoc2_tensor_make([rows, cols], new Float64Array(rows * cols));
}
