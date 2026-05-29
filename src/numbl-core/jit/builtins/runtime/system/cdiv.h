/* mtoc2 runtime helper: scalar complex division `a / b` matching
 * numbl's signed-Inf-on-zero-divisor behavior.
 *
 * C99's `_Complex` division is unspecified at the signs of NaN /
 * Inf results when the divisor is zero, so divisions like
 * `(1 + 2i) / 0` can disagree between compilers. Numbl uses Smith's
 * algorithm (which factors out the larger-magnitude divisor
 * component) and explicit ±0 detection to land on the same byte
 * stream regardless of the underlying libc. This helper mirrors
 * that path so cross-runner output stays aligned.
 */

#include <complex.h>
#include <math.h>

static double _Complex mtoc2_cdiv(double _Complex a, double _Complex b) {
  double ar = creal(a), ai = cimag(a);
  double br = creal(b), bi = cimag(b);
  /* Standard Smith's: pick the scaling that puts the larger-magnitude
   * divisor component in the denominator. */
  if (fabs(br) >= fabs(bi)) {
    double r = bi / br;
    double den = br + r * bi;
    return (ar + ai * r) / den + ((ai - ar * r) / den) * I;
  }
  double r = br / bi;
  double den = bi + r * br;
  return (ar * r + ai) / den + ((ai * r - ar) / den) * I;
}
