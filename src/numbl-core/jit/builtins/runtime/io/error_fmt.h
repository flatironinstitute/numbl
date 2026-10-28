/* @snippet name=jit_error_fmt deps=jit_format_engine */
/* JIT runtime helper: `error(...)` raises a MATLAB error through the
 * raise channel (`raise.h`), which the host turns into numbl's
 * RuntimeError with the same message and identifier.
 *
 * Mirrors the interpreter's `error` (interpreter/builtins/utility.ts):
 *   error(msg)                  message taken literally
 *   error(id, msg)              message formatted, identifier id
 *   error(fmt, a1, ...)         message formatted
 *   error(id, fmt, a1, ...)     message formatted, identifier id
 * The id-vs-message decision is made at lowering (the first argument must
 * be a literal), so the codegen picks the helper and passes the id ("" for
 * none).
 *
 * Formatting uses the shared format engine, so the spec set / escape
 * rules / arg cycling match `fprintf`.
 */

typedef struct {
  char *data;
  size_t len;
  size_t cap;
} jit__error_buf_t;

static void jit__error_writer(void *ctx, const char *bytes, long len) {
  jit__error_buf_t *b = (jit__error_buf_t *)ctx;
  if (len <= 0) return;
  if (b->len + (size_t)len + 1 > b->cap) {
    size_t cap = b->cap * 2 + (size_t)len + 64;
    char *nb = (char *)realloc(b->data, cap);
    if (!nb) jit_raise_bail("out of memory formatting an error message");
    b->data = nb;
    b->cap = cap;
  }
  memcpy(b->data + b->len, bytes, (size_t)len);
  b->len += (size_t)len;
}

static void jit_error_fmt(const char *id, jit_text_view_t fmt, int nargs,
                          const jit_fprintf_arg_t *args) {
  /* The buffer is a tracked allocation: the raise guard frees it. */
  jit__error_buf_t b = {NULL, 0, 0};
  jit__format_walk(jit__error_writer, &b, fmt, nargs, args);
  jit_raise_n(JIT_RAISE_ERROR, id, b.data ? b.data : "", b.len);
}

static void jit_error_text(const char *id, jit_text_view_t msg) {
  jit_raise_n(JIT_RAISE_ERROR, id, msg.data ? msg.data : "",
              msg.len > 0 ? (size_t)msg.len : 0);
}
