/* mtoc2 string runtime: the C representation for every numbl scalar
 * string (double-quoted) value mtoc2 emits.
 *
 * numbl distinguishes `string` (scalar handle, `length("hi") == 1`)
 * from `char` (1×N row of bytes, `length('hi') == 2`). mtoc2 carries
 * both as distinct lattice kinds; this header is the string side.
 *
 * Storage:
 *   - `data`  — pointer to the byte sequence. NULL in the empty state
 *               (zero-initialized variable); never NULL once populated.
 *   - `len`   — length in BYTES (not code points). UTF-8 by convention;
 *               the runtime never inspects code points so it works for
 *               any byte string.
 *   - `owned` — 1 iff `data` was allocated by an mtoc2 helper and must
 *               be `free()`-d on disposal. 0 for handles pointing at
 *               a C string literal in `.rodata` (the common case from
 *               `mtoc2_string_from_literal`); freeing those is
 *               undefined behavior, so the free helper checks the flag.
 */

typedef struct {
  const char *data;
  long len;
  int owned;
} mtoc2_string_t;
