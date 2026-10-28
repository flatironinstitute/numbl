/* @snippet name=jit_char_tensor_empty deps=jit_char_tensor_t */
/* JIT runtime helper: zero-initialized empty char tensor handle.
 *
 * Used at every predeclaration site so a char tensor variable starts
 * in a known state. `jit_char_tensor_assign` overwrites it on first
 * assignment. The empty handle has `owned = 0` so a stray free of an
 * uninitialized variable is a no-op.
 */

static jit_char_tensor_t jit_char_tensor_empty(void) {
  jit_char_tensor_t t;
  t.data = (char *)0;
  t.rows = 0;
  t.cols = 0;
  t.owned = 0;
  return t;
}
