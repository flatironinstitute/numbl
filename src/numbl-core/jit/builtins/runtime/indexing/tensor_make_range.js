// JS sibling of `tensor_make_range.h`. Build a 1×N row tensor for a
// `start:step:end` range used as a value (outside an index slot and
// outside a for-loop iterable).

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { mtoc2_loop_count } from "./loop_count.js";
import { mtoc2_range_value } from "./range_value.js";

export function mtoc2_tensor_make_range(start, step, end) {
  const n = mtoc2_loop_count(start, end, step);
  const t = mtoc2_tensor_alloc_nd(2, [1, n]);
  for (let k = 0; k < n; k++) {
    t.data[k] = mtoc2_range_value(start, step, end, n, k);
  }
  return t;
}
