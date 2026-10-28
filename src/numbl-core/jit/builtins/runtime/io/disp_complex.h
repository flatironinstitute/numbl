/* @snippet name=jit_disp_complex deps=jit_format_complex,jit_host_output */
/* JIT runtime helper: disp(z) for a scalar `double _Complex`.
 *
 * Mirrors numbl's `formatComplex` (via jit_format_complex) so
 * output matches the interpreter byte-for-byte. The actual
 * formatting lives in `format_complex.h`; this snippet only wraps
 * it with a print + newline.
 */

#include <complex.h>
#include <stdio.h>

static void jit_disp_complex(double _Complex z) {
  char buf[160];
  jit_format_complex(buf, sizeof(buf), z);
  jit_stdout_s(buf);
  jit_stdout("\n", 1);
}
