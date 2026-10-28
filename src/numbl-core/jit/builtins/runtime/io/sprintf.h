/* @snippet name=jit_sprintf_str deps=jit_format_engine,jit_string_t,jit_char_tensor_t aliases=jit_sprintf_char */
/* JIT runtime helper: sprintf — return formatted text as an owned
 * string or char-array, matching numbl's `sprintf` shape rule:
 *   - char-typed format ('...')  → returns `jit_char_tensor_t`
 *   - string-typed format ("...") → returns `jit_string_t`
 *
 * The format kind is decided statically (by the format arg's lattice
 * type); codegen picks `jit_sprintf_char` or `jit_sprintf_str`
 * accordingly. Both share a growable-buffer writer over the shared
 * format engine.
 *
 * The returned handle owns its byte buffer; the caller passes it to
 * `jit_string_assign` / `jit_char_tensor_assign` (or frees it
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
} jit__sprintf_buf_t;

static void jit__sprintf_writer(void *ctx, const char *bytes, long len) {
  jit__sprintf_buf_t *b = (jit__sprintf_buf_t *)ctx;
  if (len <= 0) return;
  if (b->len + len > b->cap) {
    long new_cap = b->cap == 0 ? 64 : b->cap * 2;
    while (new_cap < b->len + len) new_cap *= 2;
    char *nb = (char *)realloc(b->data, (size_t)new_cap);
    if (!nb) {
      jit_raise_bailf("out of memory in jit_sprintf");
    }
    b->data = nb;
    b->cap = new_cap;
  }
  memcpy(b->data + b->len, bytes, (size_t)len);
  b->len += len;
}

static jit_string_t jit_sprintf_str(jit_text_view_t fmt, int nargs,
                                        const jit_fprintf_arg_t *args) {
  jit__sprintf_buf_t b;
  b.data = (char *)0;
  b.len = 0;
  b.cap = 0;
  jit__format_walk(jit__sprintf_writer, &b, fmt, nargs, args);
  jit_string_t s;
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

static jit_char_tensor_t jit_sprintf_char(jit_text_view_t fmt, int nargs,
                                              const jit_fprintf_arg_t *args) {
  jit__sprintf_buf_t b;
  b.data = (char *)0;
  b.len = 0;
  b.cap = 0;
  jit__format_walk(jit__sprintf_writer, &b, fmt, nargs, args);
  jit_char_tensor_t c;
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
