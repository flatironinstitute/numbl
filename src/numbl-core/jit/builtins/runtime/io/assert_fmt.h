/* mtoc2 runtime helper: `assert(cond, fmt, args...)` — printf-style
 * assertion failure.
 *
 * Tests `cond` for MATLAB truthiness (non-zero and not NaN); on
 * failure, writes "mtoc2: assertion failed: " followed by the
 * formatted body to stderr and aborts. Uses the shared format engine
 * (`mtoc2__format_walk`), so the spec set / escape rules / arg
 * cycling match `fprintf` / `error`.
 *
 * Routed to by the lowerer whenever `assert(cond, ...)` has 2+ args
 * AND the message is either an opaque text variable or carries
 * additional format args. The simpler 2-arg literal-message form
 * stays on `mtoc2_assert_scalar` (no engine activation needed).
 */

#include <stdio.h>
#include <stdlib.h>

static void mtoc2__assert_fmt_writer(void *ctx, const char *bytes, long len) {
  (void)ctx;
  if (len > 0) fwrite(bytes, 1, (size_t)len, stderr);
}

static void mtoc2_assert_scalar_fmt(double cond, mtoc2_text_view_t fmt,
                                    int nargs,
                                    const mtoc2_fprintf_arg_t *args) {
  if (cond != 0.0 && cond == cond) return;
  fputs("mtoc2: assertion failed: ", stderr);
  mtoc2__format_walk(mtoc2__assert_fmt_writer, (void *)0, fmt, nargs, args);
  fputc('\n', stderr);
  abort();
}
