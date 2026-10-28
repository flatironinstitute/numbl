/* JIT runtime helpers: elementwise unary math on complex tensors.
 *
 * Sibling of `tensor_unary_real_math.h`. Each helper allocates a
 * freshly-owned complex tensor (both lanes), then loops over every
 * element building a `double _Complex` via `jit_cmake`, applying
 * the per-op `jit_c*` helper, and writing the result lanes via
 * `jit_creal` / `jit_cimag`. Tolerates an input with `imag ==
 * NULL` (a real tensor that flowed in through a complex-typed
 * route) by treating its imag lane as zero.
 *
 * Per-element math routes through the `jit_c*` helpers rather than
 * bare `<complex.h>` calls, matching the JS sibling.
 */

#include <stdlib.h>

#define JIT_DEFINE_UNARY_CMATH(name, FN)                                    \
  static jit_tensor_t name(jit_tensor_t a) {                                \
    long n = 1;                                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                        \
    jit_tensor_t r = jit_tensor_alloc_nd_complex(a.ndim, a.dims);           \
    for (long i = 0; i < n; i++) {                                          \
      double aim = (a.imag != NULL) ? a.imag[i] : 0.0;                      \
      double _Complex av = jit_cmake(a.real[i], aim);                       \
      double _Complex rv = FN(av);                                          \
      r.real[i] = jit_creal(rv);                                            \
      r.imag[i] = jit_cimag(rv);                                            \
    }                                                                       \
    return r;                                                               \
  }

JIT_DEFINE_UNARY_CMATH(jit_tensor_sqrt_complex, jit_csqrt)
JIT_DEFINE_UNARY_CMATH(jit_tensor_exp_complex, jit_cexp)
JIT_DEFINE_UNARY_CMATH(jit_tensor_log_complex, jit_clog)
JIT_DEFINE_UNARY_CMATH(jit_tensor_log2_complex, jit_clog2)
JIT_DEFINE_UNARY_CMATH(jit_tensor_log10_complex, jit_clog10)
JIT_DEFINE_UNARY_CMATH(jit_tensor_sin_complex, jit_csin)
JIT_DEFINE_UNARY_CMATH(jit_tensor_cos_complex, jit_ccos)
JIT_DEFINE_UNARY_CMATH(jit_tensor_tan_complex, jit_ctan)
JIT_DEFINE_UNARY_CMATH(jit_tensor_atan_complex, jit_catan)
JIT_DEFINE_UNARY_CMATH(jit_tensor_sinh_complex, jit_csinh)
JIT_DEFINE_UNARY_CMATH(jit_tensor_cosh_complex, jit_ccosh)
JIT_DEFINE_UNARY_CMATH(jit_tensor_tanh_complex, jit_ctanh)
JIT_DEFINE_UNARY_CMATH(jit_tensor_floor_complex, jit_cfloor)
JIT_DEFINE_UNARY_CMATH(jit_tensor_ceil_complex, jit_cceil)
JIT_DEFINE_UNARY_CMATH(jit_tensor_round_complex, jit_cround)
JIT_DEFINE_UNARY_CMATH(jit_tensor_fix_complex, jit_cfix)
JIT_DEFINE_UNARY_CMATH(jit_tensor_sign_complex, jit_csign)
JIT_DEFINE_UNARY_CMATH(jit_tensor_conj_complex, jit_cconj)

/* `imag`, `real`, `angle` on a complex tensor — all return a REAL
 * tensor. Mirror the `abs` shape (alloc via `_nd`, not `_nd_complex`).
 * For a real input flowing through with `a.imag == NULL`, imag
 * returns zeros, real returns a copy, angle returns zeros (since
 * `atan2(0, re) == 0` for nonneg re; the sign-of-re matters but for
 * the cleaner path we just zero).
 */
static jit_tensor_t jit_tensor_imag_complex(jit_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t r = jit_tensor_alloc_nd(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    r.real[i] = (a.imag != NULL) ? a.imag[i] : 0.0;
  }
  return r;
}

static jit_tensor_t jit_tensor_real_complex(jit_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t r = jit_tensor_alloc_nd(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    r.real[i] = a.real[i];
  }
  return r;
}

static jit_tensor_t jit_tensor_angle_complex(jit_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t r = jit_tensor_alloc_nd(a.ndim, a.dims);
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
static jit_tensor_t jit_tensor_abs_complex(jit_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t r = jit_tensor_alloc_nd(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    double aim = (a.imag != NULL) ? a.imag[i] : 0.0;
    r.real[i] = hypot(a.real[i], aim);
  }
  return r;
}
