/**
 * `triu(A [, k])` — upper triangular part of a matrix around the k-th
 * diagonal. Keeps entries where `j - i >= k`, zeros the rest.
 *
 *   triu(A)     ≡ triu(A, 0): zero everything strictly below the main
 *                 diagonal.
 *   triu(A, k)  with `k > 0` shifts the kept band upward (kept block
 *                 starts at the k-th super-diagonal); `k < 0` shifts
 *                 downward (kept block extends into the |k| sub-
 *                 diagonals).
 *
 * `k` must be a statically-known integer literal. Real and complex
 * inputs both supported; rank > 2 is deferred with `UnsupportedConstruct`.
 * Mirrors numbl's `triu` in `interpreter/builtins/array-extras.ts`.
 */

import { defineTriangular, jsTriu, jsTriuComplex } from "./_triangular.js";

export const triu = defineTriangular({
  name: "triu",
  cHelper: "mtoc2_tensor_triu",
  cHelperComplex: "mtoc2_tensor_triu_complex",
  keep: (i, j, k) => j - i >= k,
  jsHelper: jsTriu,
  jsHelperComplex: jsTriuComplex,
});
