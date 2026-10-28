/* @snippet name=jit_tensor_empty deps=jit_tensor_t */
/* mtoc runtime helper: empty/zero-initialized tensor.
 *
 * Returns the canonical "no buffers, zero shape" placeholder used to
 * predeclare every tensor variable. Subsequent assignments overwrite
 * the struct via `jit_tensor_assign`, which frees these (NULL)
 * buffers as a no-op and installs the new value.
 */

static jit_tensor_t jit_tensor_empty(void) {
  jit_tensor_t out = { NULL, NULL, 0, {0} };
  return out;
}
