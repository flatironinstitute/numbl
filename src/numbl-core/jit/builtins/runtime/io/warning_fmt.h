/* mtoc2 runtime helper: warning — write a `Warning: <formatted text>\n`
 * line to stdout. Matches numbl's `warning(...)` shape
 * (numbl/src/numbl-core/runtime/specialBuiltins.ts:190 — `rt.output("Warning: " + ... + "\n")`).
 *
 * Two forms:
 *  - `mtoc2_warning_fmt(fmt, nargs, args)` — bare `warning(fmt, ...)`.
 *  - `mtoc2_warning_fmt_id(id, fmt, nargs, args)` — id-bearing
 *    `warning('id:topic', fmt, ...)`. The c-aot path ignores the id
 *    (it has no `lastwarn`-style state to update); the JS sibling
 *    matches numbl by including the id only in side-channel state if
 *    we ever add it. For now both forms produce the same prefix.
 *
 * Format engine is shared with sprintf / fprintf — see format_engine.h.
 */

#include <stdio.h>

static void mtoc2__warning_writer(void *ctx, const char *bytes, long len) {
  (void)ctx;
  if (len > 0) fwrite(bytes, 1, (size_t)len, stdout);
}

static void mtoc2_warning_fmt(mtoc2_text_view_t fmt, int nargs,
                              const mtoc2_fprintf_arg_t *args) {
  fputs("Warning: ", stdout);
  mtoc2__format_walk(mtoc2__warning_writer, NULL, fmt, nargs, args);
  fputc('\n', stdout);
}

static void mtoc2_warning_fmt_id(mtoc2_text_view_t id, mtoc2_text_view_t fmt,
                                 int nargs, const mtoc2_fprintf_arg_t *args) {
  (void)id; /* c-aot has no lastwarn state to set */
  mtoc2_warning_fmt(fmt, nargs, args);
}
