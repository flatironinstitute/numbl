/* mtoc2 runtime helper: sprintf — return formatted text as an owned
 * string or char-array, matching numbl's `sprintf` shape rule:
 *   - char-typed format ('...')  → returns `mtoc2_char_tensor_t`
 *   - string-typed format ("...") → returns `mtoc2_string_t`
 *
 * The format kind is decided statically (by the format arg's lattice
 * type); codegen picks `mtoc2_sprintf_char` or `mtoc2_sprintf_str`
 * accordingly. Both share a growable-buffer writer over the shared
 * format engine.
 *
 * The returned handle owns its byte buffer; the caller passes it to
 * `mtoc2_string_assign` / `mtoc2_char_tensor_assign` (or frees it
 * directly via the matching `_free`). Empty results are returned as
 * owned handles with NULL data — no zero-byte malloc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  char *data;
  long len;
  long cap;
} mtoc2__sprintf_buf_t;

static void mtoc2__sprintf_writer(void *ctx, const char *bytes, long len) {
  mtoc2__sprintf_buf_t *b = (mtoc2__sprintf_buf_t *)ctx;
  if (len <= 0) return;
  if (b->len + len > b->cap) {
    long new_cap = b->cap == 0 ? 64 : b->cap * 2;
    while (new_cap < b->len + len) new_cap *= 2;
    char *nb = (char *)realloc(b->data, (size_t)new_cap);
    if (!nb) {
      fprintf(stderr, "mtoc2: out of memory in mtoc2_sprintf\n");
      abort();
    }
    b->data = nb;
    b->cap = new_cap;
  }
  memcpy(b->data + b->len, bytes, (size_t)len);
  b->len += len;
}

static mtoc2_string_t mtoc2_sprintf_str(mtoc2_text_view_t fmt, int nargs,
                                        const mtoc2_fprintf_arg_t *args) {
  mtoc2__sprintf_buf_t b;
  b.data = (char *)0;
  b.len = 0;
  b.cap = 0;
  mtoc2__format_walk(mtoc2__sprintf_writer, &b, fmt, nargs, args);
  mtoc2_string_t s;
  if (b.len == 0) {
    free(b.data);
    s.data = (const char *)0;
    s.len = 0;
    s.owned = 1;
    return s;
  }
  /* Trim to exact size — sprintf results may be long-lived. */
  char *trimmed = (char *)realloc(b.data, (size_t)b.len);
  s.data = trimmed ? trimmed : b.data;
  s.len = b.len;
  s.owned = 1;
  return s;
}

static mtoc2_char_tensor_t mtoc2_sprintf_char(mtoc2_text_view_t fmt, int nargs,
                                              const mtoc2_fprintf_arg_t *args) {
  mtoc2__sprintf_buf_t b;
  b.data = (char *)0;
  b.len = 0;
  b.cap = 0;
  mtoc2__format_walk(mtoc2__sprintf_writer, &b, fmt, nargs, args);
  mtoc2_char_tensor_t c;
  if (b.len == 0) {
    free(b.data);
    c.data = (char *)0;
    c.rows = 0;
    c.cols = 0;
    c.owned = 1;
    return c;
  }
  char *trimmed = (char *)realloc(b.data, (size_t)b.len);
  c.data = trimmed ? trimmed : b.data;
  c.rows = 1;
  c.cols = b.len;
  c.owned = 1;
  return c;
}
