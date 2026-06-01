/* mtoc2 runtime helper: deep-copy a string handle into a fresh
 * heap buffer.
 *
 * Returns `{malloc'd, len, 1}` — the caller owns the new buffer and
 * is responsible for releasing it (typically via `mtoc2_string_assign`,
 * which takes ownership, or `mtoc2_string_free`). The byte content is
 * memcpy'd verbatim; encoding is not inspected. Used wherever the
 * codegen wants a stand-alone owned copy of a string — notably the
 * `c = a;` assignment path emits
 * `mtoc2_string_assign(&c, mtoc2_string_copy(a));`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mtoc2_string_t mtoc2_string_copy(mtoc2_string_t s) {
  mtoc2_string_t out;
  if (s.len <= 0 || s.data == (const char *)0) {
    /* Empty input → empty owned output. Avoid a zero-byte malloc
     * (implementation-defined). */
    out.data = (const char *)0;
    out.len = 0;
    out.owned = 1;
    return out;
  }
  char *buf = (char *)malloc((size_t)s.len);
  if (!buf) {
    fprintf(stderr, "mtoc2: out of memory in mtoc2_string_copy (%ld bytes)\n",
            s.len);
    abort();
  }
  memcpy(buf, s.data, (size_t)s.len);
  out.data = buf;
  out.len = s.len;
  out.owned = 1;
  return out;
}
