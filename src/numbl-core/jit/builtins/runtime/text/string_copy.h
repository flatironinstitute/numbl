/* JIT runtime helper: deep-copy a string handle into a fresh
 * heap buffer.
 *
 * Returns `{malloc'd, len, 1}` — the caller owns the new buffer and
 * is responsible for releasing it (typically via `jit_string_assign`,
 * which takes ownership, or `jit_string_free`). The byte content is
 * memcpy'd verbatim; encoding is not inspected. Used wherever the
 * codegen wants a stand-alone owned copy of a string — notably the
 * `c = a;` assignment path emits
 * `jit_string_assign(&c, jit_string_copy(a));`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static jit_string_t jit_string_copy(jit_string_t s) {
  jit_string_t out;
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
    fprintf(stderr, "C-JIT: out of memory in jit_string_copy (%ld bytes)\n",
            s.len);
    abort();
  }
  memcpy(buf, s.data, (size_t)s.len);
  out.data = buf;
  out.len = s.len;
  out.owned = 1;
  return out;
}
