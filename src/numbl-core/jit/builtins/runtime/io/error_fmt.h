/* mtoc2 runtime helper: error(fmt, args...) — write formatted text to
 * stderr and abort.
 *
 * numbl's `error(...)` throws a RuntimeError; the CLI prints the message
 * and exits non-zero. mtoc2-emitted programs match by writing the
 * formatted message to stderr (+ newline) and calling `exit(1)`.
 *
 * Uses the shared format engine, so the spec set / escape rules / arg
 * cycling match `fprintf`. The optional MATLAB id form
 * (`error('Comp:Mn', fmt, ...)`) is resolved at lowering — the codegen
 * skips the id slot and passes only fmt + remaining args here, so the
 * runtime helper is identical to a single-format error.
 */

#include <stdio.h>
#include <stdlib.h>

static void mtoc2__error_writer(void *ctx, const char *bytes, long len) {
  (void)ctx;
  if (len > 0) fwrite(bytes, 1, (size_t)len, stderr);
}

static void mtoc2_error_fmt(mtoc2_text_view_t fmt, int nargs,
                            const mtoc2_fprintf_arg_t *args) {
  mtoc2__format_walk(mtoc2__error_writer, (void *)0, fmt, nargs, args);
  fputc('\n', stderr);
  exit(1);
}
