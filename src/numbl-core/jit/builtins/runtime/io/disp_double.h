/* mtoc runtime helper: disp(x) for a real-scalar double.
 *
 * Mirrors numbl's `formatNumber` (numbl/src/numbl-core/runtime/display.ts)
 * so cross-runner test output matches. The actual formatting lives in
 * `format_double.h`; this snippet only wraps it with a print + newline.
 */

#include <stdio.h>

static void mtoc2_disp_double(double x) {
  char buf[64];
  mtoc2_format_double(buf, sizeof(buf), x);
  mtoc2_stdout_s(buf);
  mtoc2_stdout("\n", 1);
}
