// JS sibling of `tensor_fill_square.h`. Single-eval helper for
// `nan(n)` / `Inf(n)` style square-fill constructors.

import { jit_tensor_fill_nd } from "./tensor_fill_nd.js";

export function jit_tensor_fill_square(value, n) {
  return jit_tensor_fill_nd(value, 2, [n, n]);
}
