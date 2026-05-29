/**
 * `tril(A [, k])` — lower triangular part of a matrix around the k-th
 * diagonal. Keeps entries where `i - j >= -k` (equivalently
 * `j - i <= k`), zeros the rest.
 *
 *   tril(A)     ≡ tril(A, 0): zero everything strictly above the main
 *                 diagonal.
 *   tril(A, k)  with `k > 0` widens the kept block upward by `k`
 *                 super-diagonals; `k < 0` narrows it (kept band
 *                 starts `|k|` below the main diagonal).
 *
 * `k` must be a statically-known integer literal. Real and complex
 * inputs both supported; rank > 2 is deferred with `UnsupportedConstruct`.
 * Mirrors numbl's `tril` in `interpreter/builtins/array-extras.ts`.
 */

import { defineTriangular, jsTril, jsTrilComplex } from "./_triangular.js";

export const tril = defineTriangular({
  name: "tril",
  cHelper: "mtoc2_tensor_tril",
  cHelperComplex: "mtoc2_tensor_tril_complex",
  keep: (i, j, k) => i - j >= -k,
  jsHelper: jsTril,
  jsHelperComplex: jsTrilComplex,
});
