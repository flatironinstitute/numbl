/* mtoc2 runtime helper: consume-and-replace assignment for strings.
 *
 * Frees `*lhs`'s current backing (only if owned — `mtoc2_string_free`
 * checks the flag) and moves `rhs` into place. Codegen guarantees
 * every RHS handed here is either a freshly-allocated owned handle
 * (e.g. `mtoc2_string_copy` result) or a literal handle (owned=0 from
 * `mtoc2_string_from_literal`); both shapes are valid moves into
 * `*lhs`.
 */

static void mtoc2_string_assign(mtoc2_string_t *lhs, mtoc2_string_t rhs) {
  mtoc2_string_free(lhs);
  *lhs = rhs;
}
