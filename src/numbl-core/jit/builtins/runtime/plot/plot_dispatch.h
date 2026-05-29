/* mtoc2 runtime: plot dispatch — single helper for every plotting
 * builtin (`plot`, `surf`, `imagesc`, `bar`, `figure`, `hold`, …).
 *
 * Numbl handles plotting by accumulating an in-memory "instruction
 * stream" that a separate viewer renders; the CLI never writes plot
 * output to stdout. mtoc2 mirrors that data-shape on the wire: each
 * plotting call emits one line of JSON on stdout, prefixed by the
 * ASCII RS (`\x1e`) sentinel plus the literal tag `mtoc2:plot\t`.
 * A launcher / viewer process tees stdout, splits on the prefix,
 * and feeds the records into numbl's own plot module — no C
 * plotting code needed.
 *
 * Why RS as the sentinel: RS (0x1E, "record separator") is a C0
 * control character that practical numeric / text output never
 * contains, so the splitter can identify plot lines purely by
 * prefix without escaping user output. The cross-runner drops
 * these lines globally before the byte-for-byte stdout compare
 * (numbl produces no such lines, so the drop is a no-op there).
 *
 * The wire shape per call is:
 *
 *   \x1e mtoc2:plot \t {"call":"<name>","args":[<arg>, ...]} \n
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
 * Argument transport reuses `mtoc2_fprintf_arg_t` from
 * `format_engine.h` (slot kinds DOUBLE / TEXT / TENSOR). Type-level
 * rejection of unsupported arg kinds (complex / struct / class /
 * handle / Void / Unknown) happens at lowering — the dispatcher
 * itself never receives them.
 *
 * `fflush(stdout)` after every record makes plots stream live when
 * stdout is piped to a viewer; cost is one syscall per call and
 * negligible compared to user-level disp output.
 */

static void mtoc2__plot_emit(const char *s, long n) {
  if (n > 0) fwrite(s, 1, (size_t)n, stdout);
}

static void mtoc2__plot_emit_cstr(const char *s) {
  mtoc2__plot_emit(s, (long)strlen(s));
}

static void mtoc2__plot_emit_double(double x) {
  if (!isfinite(x)) {
    mtoc2__plot_emit_cstr("null");
    return;
  }
  char buf[32];
  long n;
  if (fabs(x) < 1e16 && x == (double)(long long)x) {
    n = (long)snprintf(buf, sizeof(buf), "%lld", (long long)x);
  } else {
    n = (long)snprintf(buf, sizeof(buf), "%.17g", x);
  }
  mtoc2__plot_emit(buf, n);
}

static void mtoc2__plot_emit_json_string(const char *data, long len) {
  fputc('"', stdout);
  for (long i = 0; i < len; i++) {
    unsigned char c = (unsigned char)data[i];
    if (c == '"' || c == '\\') {
      char esc[2] = { '\\', (char)c };
      mtoc2__plot_emit(esc, 2);
    } else if (c == '\n') {
      mtoc2__plot_emit("\\n", 2);
    } else if (c == '\r') {
      mtoc2__plot_emit("\\r", 2);
    } else if (c == '\t') {
      mtoc2__plot_emit("\\t", 2);
    } else if (c == '\b') {
      mtoc2__plot_emit("\\b", 2);
    } else if (c == '\f') {
      mtoc2__plot_emit("\\f", 2);
    } else if (c < 0x20) {
      char esc[8];
      long n = (long)snprintf(esc, sizeof(esc), "\\u%04x", c);
      mtoc2__plot_emit(esc, n);
    } else {
      /* Bytes ≥ 0x20 (including high-bit UTF-8 continuation) pass
       * through verbatim; JSON allows raw UTF-8 in string content. */
      fputc((int)c, stdout);
    }
  }
  fputc('"', stdout);
}

static void mtoc2__plot_emit_arg(const mtoc2_fprintf_arg_t *a) {
  switch (a->kind) {
    case MTOC2_FA_DOUBLE:
      mtoc2__plot_emit_double(a->u.d);
      return;
    case MTOC2_FA_TEXT: {
      mtoc2_text_view_t tv = a->u.t;
      long n = tv.len > 0 ? tv.len : 0;
      const char *data = tv.data ? tv.data : "";
      mtoc2__plot_emit_cstr("{\"kind\":\"text\",\"data\":");
      mtoc2__plot_emit_json_string(data, n);
      fputc('}', stdout);
      return;
    }
    case MTOC2_FA_TENSOR: {
      const mtoc2_tensor_t *t = a->u.tensor;
      mtoc2__plot_emit_cstr("{\"kind\":\"tensor\",\"dims\":[");
      long total = 1;
      for (int k = 0; k < t->ndim; k++) {
        if (k > 0) fputc(',', stdout);
        char buf[24];
        long n = (long)snprintf(buf, sizeof(buf), "%ld", (long)t->dims[k]);
        mtoc2__plot_emit(buf, n);
        total *= t->dims[k];
      }
      mtoc2__plot_emit_cstr("],\"data\":[");
      for (long i = 0; i < total; i++) {
        if (i > 0) fputc(',', stdout);
        mtoc2__plot_emit_double(t->real[i]);
      }
      mtoc2__plot_emit_cstr("]}");
      return;
    }
    default:
      mtoc2__plot_emit_cstr("null");
      return;
  }
}

static void mtoc2_plot_dispatch(const char *name, int nargs,
                                const mtoc2_fprintf_arg_t *args) {
  mtoc2__plot_emit_cstr("\x1emtoc2:plot\t{\"call\":");
  mtoc2__plot_emit_json_string(name, (long)strlen(name));
  mtoc2__plot_emit_cstr(",\"args\":[");
  for (int i = 0; i < nargs; i++) {
    if (i > 0) fputc(',', stdout);
    mtoc2__plot_emit_arg(&args[i]);
  }
  mtoc2__plot_emit_cstr("]}\n");
  fflush(stdout);
}
