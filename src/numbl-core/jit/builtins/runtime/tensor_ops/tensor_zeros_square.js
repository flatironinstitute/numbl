// JS sibling of `tensor_zeros_square.h`. Single-eval helper for
// `zeros(n)` with a runtime `n` — keeps the source-level expression
// from being evaluated twice (the C side passes `n` as a parameter
// to avoid macro-style double-evaluation; the JS analogue keeps the
// parallel call shape so the emitter stays structurally aligned).

import { mtoc2_tensor_zeros_nd } from "./tensor_zeros_nd.js";

export function mtoc2_tensor_zeros_square(n) {
  return mtoc2_tensor_zeros_nd(2, [n, n]);
}
