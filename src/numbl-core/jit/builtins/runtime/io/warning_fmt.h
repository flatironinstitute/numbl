/* @snippet name=jit_warning_fmt deps=jit_format_engine aliases=jit_warning_fmt_id */
/* JIT runtime helper: warning — write a `Warning: <formatted text>\n`
 * line to stdout. Matches numbl's `warning(...)` shape
 * (src/numbl-core/runtime/specialBuiltins.ts — `rt.output("Warning: " + ... + "\n")`).
 *
 * Two forms:
 *  - `jit_warning_fmt(fmt, nargs, args)` — bare `warning(fmt, ...)`.
 *  - `jit_warning_fmt_id(id, fmt, nargs, args)` — id-bearing
 *    `warning('id:topic', fmt, ...)`. The C-JIT ignores the id
 *    (it has no `lastwarn`-style state to update); the JS sibling
 *    matches numbl by including the id only in side-channel state if
 *    we ever add it. For now both forms produce the same prefix.
 *
 * Format engine is shared with sprintf / fprintf — see format_engine.h.
 */

#include <stdio.h>

static void jit__warning_writer(void *ctx, const char *bytes, long len) {
  (void)ctx;
  if (len > 0) fwrite(bytes, 1, (size_t)len, stdout);
}

static void jit_warning_fmt(jit_text_view_t fmt, int nargs,
                              const jit_fprintf_arg_t *args) {
  fputs("Warning: ", stdout);
  jit__format_walk(jit__warning_writer, NULL, fmt, nargs, args);
  fputc('\n', stdout);
}

static void jit_warning_fmt_id(jit_text_view_t id, jit_text_view_t fmt,
                                 int nargs, const jit_fprintf_arg_t *args) {
  (void)id; /* no lastwarn state to set */
  jit_warning_fmt(fmt, nargs, args);
}
