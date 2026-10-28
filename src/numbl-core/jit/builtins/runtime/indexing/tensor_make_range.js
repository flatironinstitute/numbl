// JS sibling of `tensor_make_range.h`. Build a 1×N row tensor for a
// `start:step:end` range used as a value (outside an index slot and
// outside a for-loop iterable).

import { jit_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { jit_loop_count } from "./loop_count.js";
import { jit_range_value } from "./range_value.js";

export function jit_tensor_make_range(start, step, end) {
  const n = jit_loop_count(start, end, step);
  const t = jit_tensor_alloc_nd(2, [1, n]);
  for (let k = 0; k < n; k++) {
    t.data[k] = jit_range_value(start, step, end, n, k);
  }
  return t;
}
