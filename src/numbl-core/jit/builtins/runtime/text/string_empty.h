/* mtoc2 runtime helper: zero-initialized empty string handle.
 *
 * Used at every predeclaration site so a string variable starts in a
 * known state; `mtoc2_string_assign` will overwrite it on first
 * assignment. The empty handle is treated as the empty string by the
 * text-view helpers (`mtoc2_disp_text` skips the write when `data` is
 * NULL or `len` is 0). `owned` starts as 0 so a stray free of an
 * uninitialized variable is a no-op.
 */

static mtoc2_string_t mtoc2_string_empty(void) {
  mtoc2_string_t s;
  s.data = (const char *)0;
  s.len = 0;
  s.owned = 0;
  return s;
}
