/* @snippet name=jit_warning_fmt deps=jit_format_engine,jit_host_output aliases=jit_warning_text */
/* JIT runtime helper: `warning(...)` hands the identifier ("" for none) and
 * message to the host, which checks the warning state and writes
 * `Warning: <msg>\n`. Mirrors the interpreter's `warning`
 * (interpreter/builtins/output.ts): `jit_warning_fmt` formats the message,
 * `jit_warning_text` takes it literally (a lone `warning(msg)`).
 *
 * The pair travels as one record on the host output hook (`host_output.h`),
 * like plot records (`plot_dispatch.h`):
 *
 *   \x1e jit:warning \t <id> \t <msg> \n
 *
 * with `id` and `msg` escaped as JSON string bodies, so neither contains a
 * raw tab or newline. The host (`cOutputSink` in
 * executors/jit/hostHelpers.ts) decodes it.
 */

typedef struct {
  char *data;
  size_t len;
  size_t cap;
} jit__warning_buf_t;

static void jit__warning_put(jit__warning_buf_t *b, const char *bytes,
                             size_t len) {
  if (b->len + len > b->cap) {
    size_t cap = b->cap * 2 + len + 64;
    char *nb = (char *)realloc(b->data, cap);
    if (!nb) jit_raise_bail("out of memory formatting a warning message");
    b->data = nb;
    b->cap = cap;
  }
  memcpy(b->data + b->len, bytes, len);
  b->len += len;
}

/* Append `bytes` escaped as the body of a JSON string. */
static void jit__warning_put_escaped(jit__warning_buf_t *b, const char *bytes,
                                     long len) {
  for (long i = 0; i < len; i++) {
    unsigned char c = (unsigned char)bytes[i];
    if (c == '"' || c == '\\') {
      char esc[2] = {'\\', (char)c};
      jit__warning_put(b, esc, 2);
    } else if (c == '\n') {
      jit__warning_put(b, "\\n", 2);
    } else if (c == '\t') {
      jit__warning_put(b, "\\t", 2);
    } else if (c < 0x20) {
      char esc[8];
      int n = snprintf(esc, sizeof(esc), "\\u%04x", c);
      jit__warning_put(b, esc, (size_t)n);
    } else {
      char ch = (char)c;
      jit__warning_put(b, &ch, 1);
    }
  }
}

static void jit__warning_writer(void *ctx, const char *bytes, long len) {
  jit__warning_put_escaped((jit__warning_buf_t *)ctx, bytes, len);
}

static void jit__warning_begin(jit__warning_buf_t *b, const char *id) {
  jit__warning_put(b, "\x1ejit:warning\t", 13);
  jit__warning_put_escaped(b, id, (long)strlen(id));
  jit__warning_put(b, "\t", 1);
}

static void jit__warning_end(jit__warning_buf_t *b) {
  jit__warning_put(b, "\n", 1);
  jit_stdout(b->data, (long)b->len);
  free(b->data);
}

static void jit_warning_fmt(const char *id, jit_text_view_t fmt, int nargs,
                            const jit_fprintf_arg_t *args) {
  jit__warning_buf_t b = {NULL, 0, 0};
  jit__warning_begin(&b, id);
  jit__format_walk(jit__warning_writer, &b, fmt, nargs, args);
  jit__warning_end(&b);
}

static void jit_warning_text(const char *id, jit_text_view_t msg) {
  jit__warning_buf_t b = {NULL, 0, 0};
  jit__warning_begin(&b, id);
  jit__warning_put_escaped(&b, msg.data ? msg.data : "",
                           msg.len > 0 ? msg.len : 0);
  jit__warning_end(&b);
}
