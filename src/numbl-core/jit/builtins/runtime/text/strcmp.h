/* JIT runtime helpers: byte-equality / ASCII-case-fold-equality on
 * a pair of text views. Used by the `strcmp` and `strcmpi` builtins.
 *
 * Both return `double` (`1.0` if equal, `0.0` otherwise) so they fit
 * directly into the JIT's scalar-logical convention. Length mismatch
 * is not an error — the inputs simply compare unequal, matching
 * numbl's lax behavior.
 *
 * `jit_strcmpi` folds ASCII letters only (A..Z → a..z); locale and
 * Unicode case rules are out of scope — same as numbl, which uses
 * JS `String.prototype.toLowerCase` on inputs that are ASCII in
 * practice.
 */

static double jit_strcmp(jit_text_view_t a, jit_text_view_t b) {
  if (a.len != b.len) return 0.0;
  for (long i = 0; i < a.len; i++) {
    if (a.data[i] != b.data[i]) return 0.0;
  }
  return 1.0;
}

static double jit_strcmpi(jit_text_view_t a, jit_text_view_t b) {
  if (a.len != b.len) return 0.0;
  for (long i = 0; i < a.len; i++) {
    char ca = a.data[i];
    char cb = b.data[i];
    if (ca >= 'A' && ca <= 'Z') ca = (char)(ca + 32);
    if (cb >= 'A' && cb <= 'Z') cb = (char)(cb + 32);
    if (ca != cb) return 0.0;
  }
  return 1.0;
}
