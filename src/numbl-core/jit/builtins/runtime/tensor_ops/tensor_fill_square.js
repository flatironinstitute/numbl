// JS sibling of `tensor_fill_square.h`. Single-eval helper for
// `nan(n)` / `Inf(n)` style square-fill constructors.

import { mtoc2_tensor_fill_nd } from "./tensor_fill_nd.js";

export function mtoc2_tensor_fill_square(value, n) {
  return mtoc2_tensor_fill_nd(value, 2, [n, n]);
}
