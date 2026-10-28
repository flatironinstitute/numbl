/* @snippet name=jit_plot_dispatch deps=jit_format_engine,jit_host_output */
/* JIT runtime: plot dispatch — single helper for every plotting
 * builtin (`plot`, `surf`, `imagesc`, `bar`, `figure`, `hold`, …).
 *
 * Numbl plots by accumulating plot instructions in the runtime. Compiled
 * C writes each plotting call as one line of JSON through the host
 * output hook (`host_output.h`), prefixed by the ASCII RS (`\x1e`)
 * sentinel plus the tag `jit:plot\t`; the host (`cOutputSink` in
 * executors/jit/hostHelpers.ts) turns each record into a plot
 * instruction. RS never occurs in ordinary output, so records are found
 * by prefix without escaping user output.
 *
 * The wire shape per call is:
 *
 *   \x1e jit:plot \t {"call":"<name>","args":[<arg>, ...]} \n
 *
 * Argument encoding (one element per source-level arg):
 *   double   ->  bare numeric (NaN / ±Infinity rendered as `null`,
 *                matching JS `JSON.stringify` behavior so the
 *                viewer round-trips through `JSON.parse`)
 *   text     ->  {"kind":"text","data":"<utf-8 with JSON escapes>"}
 *                (char and string sources both flatten to this —
 *                MATLAB plot semantics don't distinguish them)
 *   tensor   ->  {"kind":"tensor","dims":[d1,...,dN],
 *                 "data":[v1, ...]}  (column-major flatten;
 *                 non-finite elements as `null`)
 *
 * Argument transport reuses `jit_fprintf_arg_t` from
 * `format_engine.h` (slot kinds DOUBLE / TEXT / TENSOR). Type-level
 * rejection of unsupported arg kinds (complex / struct / class /
 * handle / Void / Unknown) happens at lowering — the dispatcher
 * itself never receives them.
 *
 * Records go through the host output hook (`host_output.h`); the host
 * turns them into plot instructions (`cOutputSink` in
 * executors/jit/hostHelpers.ts).
 */

static void jit__plot_emit(const char *s, long n) { jit_stdout(s, n); }

static void jit__plot_emit_char(int c) {
  char ch = (char)c;
  jit_stdout(&ch, 1);
}

static void jit__plot_emit_cstr(const char *s) {
  jit__plot_emit(s, (long)strlen(s));
}

static void jit__plot_emit_double(double x) {
  if (!isfinite(x)) {
    jit__plot_emit_cstr("null");
    return;
  }
  char buf[32];
  long n;
  if (fabs(x) < 1e16 && x == (double)(long long)x) {
    n = (long)snprintf(buf, sizeof(buf), "%lld", (long long)x);
  } else {
    n = (long)snprintf(buf, sizeof(buf), "%.17g", x);
  }
  jit__plot_emit(buf, n);
}

static void jit__plot_emit_json_string(const char *data, long len) {
  jit__plot_emit_char('"');
  for (long i = 0; i < len; i++) {
    unsigned char c = (unsigned char)data[i];
    if (c == '"' || c == '\\') {
      char esc[2] = { '\\', (char)c };
      jit__plot_emit(esc, 2);
    } else if (c == '\n') {
      jit__plot_emit("\\n", 2);
    } else if (c == '\r') {
      jit__plot_emit("\\r", 2);
    } else if (c == '\t') {
      jit__plot_emit("\\t", 2);
    } else if (c == '\b') {
      jit__plot_emit("\\b", 2);
    } else if (c == '\f') {
      jit__plot_emit("\\f", 2);
    } else if (c < 0x20) {
      char esc[8];
      long n = (long)snprintf(esc, sizeof(esc), "\\u%04x", c);
      jit__plot_emit(esc, n);
    } else {
      /* Bytes ≥ 0x20 (including high-bit UTF-8 continuation) pass
       * through verbatim; JSON allows raw UTF-8 in string content. */
      jit__plot_emit_char((int)c);
    }
  }
  jit__plot_emit_char('"');
}

static void jit__plot_emit_arg(const jit_fprintf_arg_t *a) {
  switch (a->kind) {
    case JIT_FA_DOUBLE:
      jit__plot_emit_double(a->u.d);
      return;
    case JIT_FA_TEXT:
    case JIT_FA_CHAR: {
      jit_text_view_t tv = a->u.t;
      long n = tv.len > 0 ? tv.len : 0;
      const char *data = tv.data ? tv.data : "";
      jit__plot_emit_cstr("{\"kind\":\"text\",\"data\":");
      jit__plot_emit_json_string(data, n);
      jit__plot_emit_char('}');
      return;
    }
    case JIT_FA_TENSOR: {
      const jit_tensor_t *t = a->u.tensor;
      jit__plot_emit_cstr("{\"kind\":\"tensor\",\"dims\":[");
      long total = 1;
      for (int k = 0; k < t->ndim; k++) {
        if (k > 0) jit__plot_emit_char(',');
        char buf[24];
        long n = (long)snprintf(buf, sizeof(buf), "%ld", (long)t->dims[k]);
        jit__plot_emit(buf, n);
        total *= t->dims[k];
      }
      jit__plot_emit_cstr("],\"data\":[");
      for (long i = 0; i < total; i++) {
        if (i > 0) jit__plot_emit(",", 1);
        jit__plot_emit_double(t->real[i]);
      }
      jit__plot_emit_cstr("]}");
      return;
    }
    default:
      jit__plot_emit_cstr("null");
      return;
  }
}

static void jit_plot_dispatch(const char *name, int nargs,
                                const jit_fprintf_arg_t *args) {
  jit__plot_emit_cstr("\x1ejit:plot\t{\"call\":");
  jit__plot_emit_json_string(name, (long)strlen(name));
  jit__plot_emit_cstr(",\"args\":[");
  for (int i = 0; i < nargs; i++) {
    if (i > 0) jit__plot_emit(",", 1);
    jit__plot_emit_arg(&args[i]);
  }
  jit__plot_emit_cstr("]}\n");
}
