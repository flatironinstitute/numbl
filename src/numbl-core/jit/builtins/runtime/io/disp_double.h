/* mtoc runtime helper: disp(x) for a real-scalar double.
 *
 * Mirrors numbl's `formatNumber` (src/numbl-core/runtime/display.ts)
 * so output matches the interpreter. The actual formatting lives in
 * `format_double.h`; this snippet only wraps it with a print + newline.
 */

#include <stdio.h>

static void jit_disp_double(double x) {
  char buf[64];
  jit_format_double(buf, sizeof(buf), x);
  jit_stdout_s(buf);
  jit_stdout("\n", 1);
}
