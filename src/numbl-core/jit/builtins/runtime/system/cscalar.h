/* mtoc2 runtime helpers: scalar complex operations as explicit
 * function calls.
 *
 * Every `double _Complex` operation mtoc2 emits into user code routes
 * through one of these helpers instead of relying on C99's operator
 * overloading or the `I` macro. On the C path each is `static inline`
 * and the compiler folds back to the same instructions C99 would have
 * generated. The win is parallelism with the js-aot path: the JS
 * sibling (`cscalar.js`) ships a matching set of helpers that operate
 * on a `{re, im}` JS representation, so user code emitted by `emitJs`
 * uses the same call shape the C emitter would have produced.
 *
 * Division and pow are NOT defined here:
 *   - `mtoc2_cdiv` lives in `cdiv.h` because it needs Smith's
 *     algorithm + signed-zero detection to match numbl byte-for-byte.
 *   - `cpow` is emitted directly today (still a libm call, not an
 *     operator); a `mtoc2_cpow` wrapper lands when the JS side grows
 *     a power helper.
 */

#include <complex.h>
#include <math.h>

static inline double _Complex mtoc2_cmake(double re, double im) {
  return re + im * I;
}
static inline double mtoc2_creal(double _Complex z) { return creal(z); }
static inline double mtoc2_cimag(double _Complex z) { return cimag(z); }
static inline double _Complex mtoc2_cadd(double _Complex a, double _Complex b) { return a + b; }
static inline double _Complex mtoc2_csub(double _Complex a, double _Complex b) { return a - b; }
static inline double _Complex mtoc2_cmul(double _Complex a, double _Complex b) { return a * b; }
static inline double _Complex mtoc2_cneg(double _Complex z) { return -z; }
static inline double _Complex mtoc2_cconj(double _Complex z) { return conj(z); }
static inline double mtoc2_cabs(double _Complex z) {
  return hypot(creal(z), cimag(z));
}
static inline double mtoc2_cangle(double _Complex z) {
  return atan2(cimag(z), creal(z));
}
static inline int mtoc2_cnonzero(double _Complex z) {
  return creal(z) != 0.0 || cimag(z) != 0.0;
}
static inline int mtoc2_ceq(double _Complex a, double _Complex b) {
  return creal(a) == creal(b) && cimag(a) == cimag(b);
}
static inline int mtoc2_cne(double _Complex a, double _Complex b) {
  return creal(a) != creal(b) || cimag(a) != cimag(b);
}
static inline double _Complex mtoc2_cpow(double _Complex a, double _Complex b) {
  return cpow(a, b);
}

/* Unary complex math — `<complex.h>` wrappers + a handful of
 * derived helpers C99 doesn't ship (log2/log10, MATLAB-style
 * componentwise rounding, signum). Same `mtoc2_c*` shape as the
 * core scalar helpers above so the c2js backend can substitute
 * its own `{re, im}` JS implementations.
 *
 * The rounding helpers (floor/ceil/round/fix) match MATLAB by
 * applying the rounding mode to each component independently and
 * returning a complex result; the magnitude is NOT preserved
 * (MATLAB's `floor([1.5 + 1.5i])` returns `1 + 1i`, not the
 * floor of the magnitude).
 *
 * `mtoc2_csign(z)` matches MATLAB: `z/|z|` for nonzero, exactly
 * `0 + 0i` for the zero scalar. */
static inline double _Complex mtoc2_csqrt(double _Complex z) { return csqrt(z); }
static inline double _Complex mtoc2_cexp(double _Complex z) { return cexp(z); }
static inline double _Complex mtoc2_clog(double _Complex z) { return clog(z); }
static inline double _Complex mtoc2_clog2(double _Complex z) {
  return clog(z) / log(2.0);
}
static inline double _Complex mtoc2_clog10(double _Complex z) {
  return clog(z) / log(10.0);
}
static inline double _Complex mtoc2_csin(double _Complex z) { return csin(z); }
static inline double _Complex mtoc2_ccos(double _Complex z) { return ccos(z); }
static inline double _Complex mtoc2_ctan(double _Complex z) { return ctan(z); }
static inline double _Complex mtoc2_catan(double _Complex z) { return catan(z); }
static inline double _Complex mtoc2_cfloor(double _Complex z) {
  return floor(creal(z)) + floor(cimag(z)) * I;
}
static inline double _Complex mtoc2_cceil(double _Complex z) {
  return ceil(creal(z)) + ceil(cimag(z)) * I;
}
static inline double _Complex mtoc2_cround(double _Complex z) {
  return round(creal(z)) + round(cimag(z)) * I;
}
static inline double _Complex mtoc2_cfix(double _Complex z) {
  return trunc(creal(z)) + trunc(cimag(z)) * I;
}
static inline double _Complex mtoc2_csign(double _Complex z) {
  double re = creal(z), im = cimag(z);
  if (re == 0.0 && im == 0.0) return 0.0;
  double m = hypot(re, im);
  return (re / m) + (im / m) * I;
}
