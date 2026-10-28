/* JIT runtime helpers: elementwise unary math on real tensors.
 *
 * Allocate-and-fill pattern matching `tensor_elemwise_real.h`. The
 * fresh tensor inherits the input's shape; the loop applies the C
 * <math.h> function elementwise.
 *
 * `sign` has no direct <math.h> equivalent — we provide
 * `jit_signum` inline. `round` uses C99's `round()` (half-away-
 * from-zero, matching MATLAB) via the `jit_round_half_away`
 * wrapper.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* MATLAB convention: round(-1.5) = -2 (away from zero). C99's round()
 * is half-away-from-zero per the spec, which matches. Wrapped for
 * symmetry with the rest of the helpers (and to give the codegen a
 * single C symbol to call regardless of compiler quirks). */
static double jit_round_half_away(double x) {
  return round(x);
}

/* signum(x) ∈ {-1, 0, 1, NaN}. Matches JS `Math.sign`:
 *   - NaN → NaN (any comparison with NaN is false, so a naive
 *     positive/negative test would silently return 0).
 *   - -0 → 0 (handled by the equality-to-zero fall-through). */
static double jit_signum(double x) {
  if (isnan(x)) return x;
  if (x > 0) return 1.0;
  if (x < 0) return -1.0;
  return 0.0;
}

/* realsqrt(x): MATLAB raises an error rather than returning a complex
 * result for x < 0. NaN passes through as NaN. */
static double jit_realsqrt(double x) {
  if (x < 0) {
    jit_raise_error("MATLAB:realsqrt:complexResult", "Realsqrt produced complex result.");
  }
  return sqrt(x);
}

#define JIT_DEFINE_UNARY_MATH(name, FN)                                     \
  static jit_tensor_t name(jit_tensor_t a) {                                \
    long n = 1;                                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                        \
    jit_tensor_t r;                                                         \
    r.real = jit_alloc((size_t)n * sizeof(double));                         \
    r.imag = NULL;                                                          \
    r.ndim = a.ndim;                                                        \
    for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];                 \
    JIT_OMP_PARFOR_N                                                        \
    for (long i = 0; i < n; i++) r.real[i] = FN(a.real[i]);                 \
    return r;                                                               \
  }

JIT_DEFINE_UNARY_MATH(jit_tensor_cos,   cos)
JIT_DEFINE_UNARY_MATH(jit_tensor_sin,   sin)
JIT_DEFINE_UNARY_MATH(jit_tensor_tan,   tan)
JIT_DEFINE_UNARY_MATH(jit_tensor_atan,  atan)
JIT_DEFINE_UNARY_MATH(jit_tensor_sinh,  sinh)
JIT_DEFINE_UNARY_MATH(jit_tensor_cosh,  cosh)
JIT_DEFINE_UNARY_MATH(jit_tensor_tanh,  tanh)
JIT_DEFINE_UNARY_MATH(jit_tensor_asinh, asinh)
JIT_DEFINE_UNARY_MATH(jit_tensor_exp,   exp)
JIT_DEFINE_UNARY_MATH(jit_tensor_log,   log)
JIT_DEFINE_UNARY_MATH(jit_tensor_log2,  log2)
JIT_DEFINE_UNARY_MATH(jit_tensor_log10, log10)
JIT_DEFINE_UNARY_MATH(jit_tensor_sqrt,  sqrt)
JIT_DEFINE_UNARY_MATH(jit_tensor_realsqrt, jit_realsqrt)
JIT_DEFINE_UNARY_MATH(jit_tensor_abs,   fabs)
JIT_DEFINE_UNARY_MATH(jit_tensor_floor, floor)
JIT_DEFINE_UNARY_MATH(jit_tensor_ceil,  ceil)
JIT_DEFINE_UNARY_MATH(jit_tensor_fix,   trunc)
JIT_DEFINE_UNARY_MATH(jit_tensor_round, jit_round_half_away)
JIT_DEFINE_UNARY_MATH(jit_tensor_sign,  jit_signum)
