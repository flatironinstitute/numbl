/* mtoc2 runtime helper: disp(z) for a scalar `double _Complex`.
 *
 * Mirrors numbl's `formatComplex` (via mtoc2_format_complex) so
 * cross-runner test output matches byte-for-byte. The actual
 * formatting lives in `format_complex.h`; this snippet only wraps
 * it with a print + newline.
 */

#include <complex.h>
#include <stdio.h>

static void mtoc2_disp_complex(double _Complex z) {
  char buf[160];
  mtoc2_format_complex(buf, sizeof(buf), z);
  printf("%s\n", buf);
}
