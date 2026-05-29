/* mtoc runtime helper: empty/zero-initialized tensor.
 *
 * Returns the canonical "no buffers, zero shape" placeholder used to
 * predeclare every tensor variable. Subsequent assignments overwrite
 * the struct via `mtoc2_tensor_assign`, which frees these (NULL)
 * buffers as a no-op and installs the new value.
 */

static mtoc2_tensor_t mtoc2_tensor_empty(void) {
  mtoc2_tensor_t out = { NULL, NULL, 0, {0} };
  return out;
}
