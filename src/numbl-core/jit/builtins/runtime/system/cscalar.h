/* @snippet name=jit_cscalar */
/* JIT runtime helpers: scalar complex operations as explicit
 * function calls.
 *
 * Every `double _Complex` operation the JIT emits into user code routes
 * through one of these helpers instead of relying on C99's operator
 * overloading or the `I` macro. On the C path each is `static inline`
 * and the compiler folds back to the same instructions C99 would have
 * generated. The win is parallelism with the JS-JIT path: the JS
 * sibling (`cscalar.js`) ships a matching set of helpers that operate
 * on a `{re, im}` JS representation, so user code emitted by `emitJs`
 * uses the same call shape the C emitter would have produced.
 *
 * Division and pow are NOT defined here:
 *   - `jit_cdiv` lives in `cdiv.h` because it needs Smith's
 *     algorithm + signed-zero detection to match numbl byte-for-byte.
 *   - `cpow` is emitted directly today (still a libm call, not an
 *     operator); a `jit_cpow` wrapper lands when the JS side grows
 *     a power helper.
 */

#include <complex.h>
#include <math.h>

static inline double _Complex jit_cmake(double re, double im) {
  return re + im * I;
}
/* Pack re/im into a complex without the `im * I` trap: `INFINITY * I`
 * is `NaN + INFINITY*i` (0*Inf in the real lane). __builtin_complex
 * (gcc/clang) sets the components directly; CMPLX is the C11 fallback. */
static inline double _Complex jit_cpack(double re, double im) {
#if defined(__clang__) || defined(__GNUC__)
  return __builtin_complex(re, im);
#else
  return CMPLX(re, im);
#endif
}
static inline double jit_creal(double _Complex z) { return creal(z); }
static inline double jit_cimag(double _Complex z) { return cimag(z); }
static inline double _Complex jit_cadd(double _Complex a, double _Complex b) { return a + b; }
static inline double _Complex jit_csub(double _Complex a, double _Complex b) { return a - b; }
static inline double _Complex jit_cmul(double _Complex a, double _Complex b) { return a * b; }
/* Scalar `*` with MATLAB's mixed real/complex rule (as the interpreter's
 * `mixedRealComplex` in helpers/arithmetic.ts): an operand whose imag part
 * is 0 acts as real, and a real operand scales each part of a complex one
 * without promotion, so the real part stays 0 where the complex operand's
 * is 0 (NaN or Inf in the real operand would otherwise give NaN*0 = NaN).
 * `jit_cmul` stays the plain complex product. */
static inline double _Complex jit_cmul_mixed(double _Complex a, double _Complex b) {
  double ar = creal(a), ai = cimag(a), br = creal(b), bi = cimag(b);
  if (ai == 0.0 && bi == 0.0) return jit_cpack(ar * br, 0.0);
  if (bi == 0.0) return jit_cpack(ar == 0.0 ? 0.0 : ar * br, ai * br);
  if (ai == 0.0) return jit_cpack(br == 0.0 ? 0.0 : br * ar, bi * ar);
  return jit_cmul(a, b);
}
static inline double _Complex jit_cneg(double _Complex z) { return -z; }
static inline double _Complex jit_cconj(double _Complex z) { return conj(z); }
static inline double jit_cabs(double _Complex z) {
  return hypot(creal(z), cimag(z));
}
static inline double jit_cangle(double _Complex z) {
  return atan2(cimag(z), creal(z));
}
static inline int jit_cnonzero(double _Complex z) {
  return creal(z) != 0.0 || cimag(z) != 0.0;
}
static inline int jit_ceq(double _Complex a, double _Complex b) {
  return creal(a) == creal(b) && cimag(a) == cimag(b);
}
static inline int jit_cne(double _Complex a, double _Complex b) {
  return creal(a) != creal(b) || cimag(a) != cimag(b);
}
static inline double _Complex jit_cpow(double _Complex a, double _Complex b) {
  return cpow(a, b);
}

/* Unary complex math — `<complex.h>` wrappers + a handful of
 * derived helpers C99 doesn't ship (log2/log10, MATLAB-style
 * componentwise rounding, signum). Same `jit_c*` shape as the
 * core scalar helpers above, with `{re, im}` JS siblings in
 * cscalar.js.
 *
 * The rounding helpers (floor/ceil/round/fix) match MATLAB by
 * applying the rounding mode to each component independently and
 * returning a complex result; the magnitude is NOT preserved
 * (MATLAB's `floor([1.5 + 1.5i])` returns `1 + 1i`, not the
 * floor of the magnitude).
 *
 * `jit_csign(z)` matches MATLAB: `z/|z|` for nonzero, exactly
 * `0 + 0i` for the zero scalar. */
/* True when z is a NaN on the real axis. The JIT lifts a real argument whose
 * sign it cannot prove onto these complex helpers, and MATLAB's sqrt(NaN) /
 * log(NaN) are a real NaN — so a lifted NaN must come back as NaN + 0i, not
 * 0 + NaN i. The lift is a compilation detail and must not be observable. */
static inline int jit_nan_on_real_axis(double _Complex z) {
  return cimag(z) == 0.0 && isnan(creal(z));
}
static inline double _Complex jit_csqrt(double _Complex z) {
  if (jit_nan_on_real_axis(z)) return jit_cmake(creal(z), 0.0);
  return csqrt(z);
}
static inline double _Complex jit_cexp(double _Complex z) { return cexp(z); }
static inline double _Complex jit_clog(double _Complex z) {
  if (jit_nan_on_real_axis(z)) return jit_cmake(creal(z), 0.0);
  return clog(z);
}
static inline double _Complex jit_clog2(double _Complex z) {
  if (jit_nan_on_real_axis(z)) return jit_cmake(creal(z), 0.0);
  return clog(z) / log(2.0);
}
static inline double _Complex jit_clog10(double _Complex z) {
  if (jit_nan_on_real_axis(z)) return jit_cmake(creal(z), 0.0);
  return clog(z) / log(10.0);
}
static inline double _Complex jit_csin(double _Complex z) { return csin(z); }
static inline double _Complex jit_ccos(double _Complex z) { return ccos(z); }
static inline double _Complex jit_ctan(double _Complex z) { return ctan(z); }
static inline double _Complex jit_catan(double _Complex z) {
  /* Real-valued input: clean real atan (no spurious imaginary residue,
     no Inf/Inf NaN at large |re|). */
  if (cimag(z) == 0.0) return jit_cmake(atan(creal(z)), 0.0);
  /* atan(z) = (i/2)*log((1 - iz)/(1 + iz)). MATLAB's branch for a
     pure-imaginary z with |z| > 1 is -pi/2 + ... (atan(2i) = -1.5708 +
     0.5493i); libc catan uses the opposite +pi/2 Annex-G branch. Compute
     the formula here so the C path matches MATLAB and the JS runtime.
     Normalize a -0 imaginary part of the quotient to +0 (via +0.0) so
     clog picks the +pi branch — a -0 would flip it to -pi -> +pi/2. */
  double _Complex iz = jit_cmake(-cimag(z), creal(z));
  double _Complex num = jit_cmake(1.0 - creal(iz), -cimag(iz));
  double _Complex den = jit_cmake(1.0 + creal(iz), cimag(iz));
  double _Complex q = num / den;
  double _Complex l = clog(jit_cmake(creal(q), cimag(q) + 0.0));
  return jit_cmake(-cimag(l) / 2.0, creal(l) / 2.0);
}
/* Hyperbolic sinh/cosh/tanh are entire functions (no branch cuts), so
 * C99's csinh/ccosh/ctanh agree with the JS componentwise formulas in
 * `cscalar.js` to within floating-point ULPs everywhere. */
static inline double _Complex jit_csinh(double _Complex z) { return csinh(z); }
static inline double _Complex jit_ccosh(double _Complex z) { return ccosh(z); }
static inline double _Complex jit_ctanh(double _Complex z) { return ctanh(z); }
static inline double _Complex jit_cfloor(double _Complex z) {
  return floor(creal(z)) + floor(cimag(z)) * I;
}
static inline double _Complex jit_cceil(double _Complex z) {
  return ceil(creal(z)) + ceil(cimag(z)) * I;
}
static inline double _Complex jit_cround(double _Complex z) {
  return round(creal(z)) + round(cimag(z)) * I;
}
static inline double _Complex jit_cfix(double _Complex z) {
  return trunc(creal(z)) + trunc(cimag(z)) * I;
}
static inline double _Complex jit_csign(double _Complex z) {
  double re = creal(z), im = cimag(z);
  if (re == 0.0 && im == 0.0) return 0.0;
  double m = hypot(re, im);
  return (re / m) + (im / m) * I;
}
