/* mtoc2 runtime helpers: elementwise unary math on real tensors.
 *
 * Allocate-and-fill pattern matching `tensor_elemwise_real.h`. The
 * fresh tensor inherits the input's shape; the loop applies the C
 * <math.h> function elementwise.
 *
 * `sign` has no direct <math.h> equivalent — we provide
 * `mtoc2_signum` inline. `round` uses C99's `round()` (half-away-
 * from-zero, matching MATLAB) via the `mtoc2_round_half_away`
 * wrapper.
 */
#include <math.h>
#include <stdlib.h>

/* MATLAB convention: round(-1.5) = -2 (away from zero). C99's round()
 * is half-away-from-zero per the spec, which matches. Wrapped for
 * symmetry with the rest of the helpers (and to give the codegen a
 * single C symbol to call regardless of compiler quirks). */
static double mtoc2_round_half_away(double x) {
  return round(x);
}

/* signum(x) ∈ {-1, 0, 1, NaN}. Matches JS `Math.sign`:
 *   - NaN → NaN (any comparison with NaN is false, so a naive
 *     positive/negative test would silently return 0).
 *   - -0 → 0 (handled by the equality-to-zero fall-through). */
static double mtoc2_signum(double x) {
  if (isnan(x)) return x;
  if (x > 0) return 1.0;
  if (x < 0) return -1.0;
  return 0.0;
}

#define MTOC2_DEFINE_UNARY_MATH(name, FN)                                   \
  static mtoc2_tensor_t name(mtoc2_tensor_t a) {                            \
    long n = 1;                                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                        \
    mtoc2_tensor_t r;                                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));                       \
    r.imag = NULL;                                                          \
    r.ndim = a.ndim;                                                        \
    for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];                 \
    MTOC2_OMP_PARFOR_N                                                      \
    for (long i = 0; i < n; i++) r.real[i] = FN(a.real[i]);                 \
    return r;                                                               \
  }

MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_cos,   cos)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_sin,   sin)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_tan,   tan)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_atan,  atan)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_sinh,  sinh)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_cosh,  cosh)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_tanh,  tanh)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_asinh, asinh)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_exp,   exp)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_log,   log)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_log2,  log2)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_log10, log10)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_sqrt,  sqrt)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_abs,   fabs)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_floor, floor)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_ceil,  ceil)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_fix,   trunc)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_round, mtoc2_round_half_away)
MTOC2_DEFINE_UNARY_MATH(mtoc2_tensor_sign,  mtoc2_signum)
