/* mtoc2 runtime helper: scalar `assert(cond, msg)`. Tests `cond` for
 * MATLAB truthiness (non-zero and not NaN); fires `abort()` after
 * printing the message to stderr if the test fails.
 *
 * Numbl's `assert` (`interpreter/builtins/utility.ts:151`) accepts
 * tensor conds too — every element must be non-zero and not NaN. We
 * only support scalar conds in v1 (chunkie_simple's `assert(n >= 0,
 * ...)` is scalar-comparison-result anyway). Tensor-cond support is
 * a small followup.
 *
 * `msg` may be NULL or empty; the printer falls back to a generic
 * "assertion failed" line in that case.
 */
#include <stdio.h>
#include <stdlib.h>

static void mtoc2_assert_scalar(double cond, const char *msg) {
  /* `cond != cond` filters NaN — numbl's `isNaN(v)` check. */
  if (cond != 0.0 && cond == cond) return;
  fputs("mtoc2: assertion failed", stderr);
  if (msg != NULL && msg[0] != '\0') {
    fputs(": ", stderr);
    fputs(msg, stderr);
  }
  fputc('\n', stderr);
  abort();
}
