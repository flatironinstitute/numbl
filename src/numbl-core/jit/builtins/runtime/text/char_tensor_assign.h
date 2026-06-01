/* mtoc2 runtime helper: consume-and-replace assignment for char tensors.
 *
 * Frees `*lhs`'s current backing (only if owned — `mtoc2_char_tensor_free`
 * checks the flag) and moves `rhs` into place. Codegen guarantees every
 * RHS is either a freshly-allocated owned handle (from a future concat
 * helper, or `mtoc2_char_tensor_copy`) or a literal handle (owned=0 from
 * `mtoc2_char_tensor_from_literal`); both shapes are valid.
 */

static void mtoc2_char_tensor_assign(mtoc2_char_tensor_t *lhs,
                                       mtoc2_char_tensor_t rhs) {
  mtoc2_char_tensor_free(lhs);
  *lhs = rhs;
}
