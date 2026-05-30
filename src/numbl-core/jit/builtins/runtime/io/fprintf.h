/* mtoc2 runtime helper: fprintf — write formatted text to stdout.
 *
 * Matches numbl's `fprintf` semantics
 * (numbl/src/numbl-core/runtime/specialBuiltins.ts):
 *   - `fprintf(fmt, args...)`         → write to stdout
 *   - `fprintf(1, fmt, args...)`      → write to stdout
 *   - `fprintf(2, fmt, args...)`      → numbl routes fid=2 to its
 *     single `output` stream too; mtoc2 matches by sending fid=2 to
 *     stdout (cross-runner parity requires byte-for-byte stdout match).
 *
 * The fid is resolved at lowering (literal 1 or 2 only); the codegen
 * always emits `mtoc2_fprintf(stdout, …)`. Other fids are deferred at
 * lowering with an UnsupportedConstruct span.
 *
 * Format engine is shared with sprintf — see `format_engine.h` for the
 * spec set, escape handling, and arg-cycling rules.
 */

#include <stdio.h>

static void mtoc2__fprintf_writer(void *ctx, const char *bytes, long len) {
  (void)ctx; /* fid 1 and 2 both route through the host output hook */
  mtoc2_stdout(bytes, len);
}

static void mtoc2_fprintf(FILE *f, mtoc2_text_view_t fmt,
                          int nargs, const mtoc2_fprintf_arg_t *args) {
  mtoc2__format_walk(mtoc2__fprintf_writer, (void *)f, fmt, nargs, args);
}
