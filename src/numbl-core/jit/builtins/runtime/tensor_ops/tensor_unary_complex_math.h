/* mtoc2 runtime helpers: elementwise unary math on complex tensors.
 *
 * Sibling of `tensor_unary_real_math.h`. Each helper allocates a
 * freshly-owned complex tensor (both lanes), then loops over every
 * element building a `double _Complex` via `mtoc2_cmake`, applying
 * the per-op `mtoc2_c*` helper, and writing the result lanes via
 * `mtoc2_creal` / `mtoc2_cimag`. Tolerates an input with `imag ==
 * NULL` (a real tensor that flowed in through a complex-typed
 * route) by treating its imag lane as zero.
 *
 * Routing through `mtoc2_c*` rather than bare `<complex.h>` calls
 * keeps the body c2js-translatable — `mtoc2_csqrt` and friends
 * resolve to JS implementations on `{re, im}` objects at execution
 * time.
 */

#include <stdlib.h>

#define MTOC2_DEFINE_UNARY_CMATH(name, FN)                                  \
  static mtoc2_tensor_t name(mtoc2_tensor_t a) {                            \
    long n = 1;                                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                        \
    mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);       \
    for (long i = 0; i < n; i++) {                                          \
      double aim = (a.imag != NULL) ? a.imag[i] : 0.0;                      \
      double _Complex av = mtoc2_cmake(a.real[i], aim);                     \
      double _Complex rv = FN(av);                                          \
      r.real[i] = mtoc2_creal(rv);                                          \
      r.imag[i] = mtoc2_cimag(rv);                                          \
    }                                                                       \
    return r;                                                               \
  }

MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_sqrt_complex, mtoc2_csqrt)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_exp_complex, mtoc2_cexp)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_log_complex, mtoc2_clog)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_log2_complex, mtoc2_clog2)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_log10_complex, mtoc2_clog10)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_sin_complex, mtoc2_csin)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_cos_complex, mtoc2_ccos)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_tan_complex, mtoc2_ctan)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_atan_complex, mtoc2_catan)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_sinh_complex, mtoc2_csinh)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_cosh_complex, mtoc2_ccosh)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_tanh_complex, mtoc2_ctanh)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_floor_complex, mtoc2_cfloor)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_ceil_complex, mtoc2_cceil)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_round_complex, mtoc2_cround)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_fix_complex, mtoc2_cfix)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_sign_complex, mtoc2_csign)
MTOC2_DEFINE_UNARY_CMATH(mtoc2_tensor_conj_complex, mtoc2_cconj)

/* `imag`, `real`, `angle` on a complex tensor — all return a REAL
 * tensor. Mirror the `abs` shape (alloc via `_nd`, not `_nd_complex`).
 * For a real input flowing through with `a.imag == NULL`, imag
 * returns zeros, real returns a copy, angle returns zeros (since
 * `atan2(0, re) == 0` for nonneg re; the sign-of-re matters but for
 * the cleaner path we just zero).
 */
static mtoc2_tensor_t mtoc2_tensor_imag_complex(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r = mtoc2_tensor_alloc_nd(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    r.real[i] = (a.imag != NULL) ? a.imag[i] : 0.0;
  }
  return r;
}

static mtoc2_tensor_t mtoc2_tensor_real_complex(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r = mtoc2_tensor_alloc_nd(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    r.real[i] = a.real[i];
  }
  return r;
}

static mtoc2_tensor_t mtoc2_tensor_angle_complex(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r = mtoc2_tensor_alloc_nd(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    double aim = (a.imag != NULL) ? a.imag[i] : 0.0;
    r.real[i] = atan2(aim, a.real[i]);
  }
  return r;
}

/* `abs` on a complex tensor returns a REAL tensor (the magnitude).
 * Different shape from the rest of the family — `r.imag` stays NULL
 * because the result is statically real. The codegen routes this
 * specially via `abs.ts` since the result type is real, not complex. */
static mtoc2_tensor_t mtoc2_tensor_abs_complex(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r = mtoc2_tensor_alloc_nd(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    double aim = (a.imag != NULL) ? a.imag[i] : 0.0;
    r.real[i] = hypot(a.real[i], aim);
  }
  return r;
}
