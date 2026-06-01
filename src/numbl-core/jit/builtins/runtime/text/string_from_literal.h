/* mtoc2 runtime helper: build a string handle pointing at a C string
 * literal in `.rodata`.
 *
 * Returns `{src, len, 0}` — owned=0 marks the buffer as not heap-
 * allocated, so `mtoc2_string_free` will leave it untouched. Callers
 * are expected to pass a pointer to a string with static storage
 * duration (codegen always emits a literal here, so this is safe).
 *
 * `len` is the byte length excluding the trailing NUL the C literal
 * carries; codegen passes `sizeof(literal) - 1` (the source length).
 */

static mtoc2_string_t mtoc2_string_from_literal(const char *src, long len) {
  mtoc2_string_t s;
  s.data = src;
  s.len = len;
  s.owned = 0;
  return s;
}
