/* JIT runtime helper: consume-and-replace assignment for strings.
 *
 * Frees `*lhs`'s current backing (only if owned — `jit_string_free`
 * checks the flag) and moves `rhs` into place. Codegen guarantees
 * every RHS handed here is either a freshly-allocated owned handle
 * (e.g. `jit_string_copy` result) or a literal handle (owned=0 from
 * `jit_string_from_literal`); both shapes are valid moves into
 * `*lhs`.
 */

static void jit_string_assign(jit_string_t *lhs, jit_string_t rhs) {
  jit_string_free(lhs);
  *lhs = rhs;
}
