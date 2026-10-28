/* JIT runtime helper: consume-and-replace assignment for char tensors.
 *
 * Frees `*lhs`'s current backing (only if owned — `jit_char_tensor_free`
 * checks the flag) and moves `rhs` into place. Codegen guarantees every
 * RHS is either a freshly-allocated owned handle (from a future concat
 * helper, or `jit_char_tensor_copy`) or a literal handle (owned=0 from
 * `jit_char_tensor_from_literal`); both shapes are valid.
 */

static void jit_char_tensor_assign(jit_char_tensor_t *lhs,
                                       jit_char_tensor_t rhs) {
  jit_char_tensor_free(lhs);
  *lhs = rhs;
}
