// JS sibling of `tensor_ones_square.h`. See `tensor_zeros_square.js`
// for the rationale.

import { mtoc2_tensor_ones_nd } from "./tensor_ones_nd.js";

export function mtoc2_tensor_ones_square(n) {
  return mtoc2_tensor_ones_nd(2, [n, n]);
}
