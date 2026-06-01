/* mtoc2 runtime helper: format a `double _Complex` the way numbl's
 * `formatComplex` does, into a caller-provided buffer.
 *
 * Layout rules (mirrors numbl/runtime/display.ts::formatComplex):
 *   - im == 0       → "<re>"            (formatNumber on the real part)
 *   - re == 0       → "<im>i"
 *   - im < 0        → "<re> - <|im|>i"  (note the spaces)
 *   - otherwise     → "<re> + <im>i"
 *
 * Each component is rendered through `mtoc2_format_double` so the
 * scalar formatting stays in lockstep with numbl's `formatNumber`.
 *
 * Returns the number of characters written (excluding trailing NUL).
 */

#include <complex.h>
#include <stdio.h>
#include <string.h>

static int mtoc2_format_complex(char *out, size_t cap, double _Complex z) {
  double re = creal(z);
  double im = cimag(z);
  char reBuf[64], imBuf[64];
  if (im == 0.0) {
    mtoc2_format_double(reBuf, sizeof(reBuf), re);
    return snprintf(out, cap, "%s", reBuf);
  }
  if (re == 0.0) {
    mtoc2_format_double(imBuf, sizeof(imBuf), im);
    return snprintf(out, cap, "%si", imBuf);
  }
  if (im < 0.0) {
    mtoc2_format_double(reBuf, sizeof(reBuf), re);
    mtoc2_format_double(imBuf, sizeof(imBuf), -im);
    return snprintf(out, cap, "%s - %si", reBuf, imBuf);
  }
  mtoc2_format_double(reBuf, sizeof(reBuf), re);
  mtoc2_format_double(imBuf, sizeof(imBuf), im);
  return snprintf(out, cap, "%s + %si", reBuf, imBuf);
}
