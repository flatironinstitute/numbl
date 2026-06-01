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

static double mtoc2_signed_inf(double x) {
  return x > 0 ? INFINITY : (x < 0 ? -INFINITY : 0.0);
}

/* Pack re/im into a complex without the `im * I` trap: `INFINITY * I`
 * is `NaN + INFINITY*i` (0*Inf in the real lane), which would corrupt a
 * signed-Inf result. __builtin_complex (gcc/clang) sets the components
 * directly; CMPLX is the C11 fallback. */
static double _Complex mtoc2_cpack(double re, double im) {
#if defined(__clang__) || defined(__GNUC__)
  return __builtin_complex(re, im);
#else
  return CMPLX(re, im);
#endif
}

static double _Complex mtoc2_cdiv(double _Complex a, double _Complex b) {
  double ar = creal(a), ai = cimag(a);
  double br = creal(b), bi = cimag(b);
  /* Zero divisor: match the interpreter (helpers/arithmetic.ts) — 0/0 is
   * NaN, a nonzero numerator yields a signed Inf per component. Smith's
   * algorithm below would otherwise produce NaN+NaNi here. */
  if (br == 0.0 && bi == 0.0) {
    if (ar == 0.0 && ai == 0.0) return mtoc2_cpack(NAN, 0.0);
    return mtoc2_cpack(mtoc2_signed_inf(ar), mtoc2_signed_inf(ai));
  }
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
