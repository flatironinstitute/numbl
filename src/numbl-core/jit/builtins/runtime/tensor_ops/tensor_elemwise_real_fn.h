/* mtoc2 runtime helpers: elementwise binary functions on real
 * tensors (`mod`, `rem`, `atan2`, `hypot`, `power`). Sibling of
 * `tensor_elemwise_real.h`; same broadcast rules (`_tt` same-shape
 * fast path plus `_bcast_tt` for MATLAB-style implicit expansion).
 *
 * Each helper returns a freshly-owned tensor. The differences from
 * `tensor_elemwise_real.h`:
 *   - The kernel is `FN(a, b)` (a C function call) rather than
 *     `a OP b` (an infix operator).
 *   - `mod` needs `mtoc2_mod_real` (MATLAB convention: result sign
 *     tracks `b`'s sign), distinct from C's `fmod` (result sign
 *     tracks `a`'s sign).
 *
 * The scalar paths (mod/rem/atan2/hypot on two scalars) call the
 * underlying C function directly via the builtin's `codegenC`; this
 * header only carries the tensor helpers.
 */
#include <math.h>
#include <stdlib.h>

/* MATLAB `mod(a, b)`: result has the sign of `b`. `mod(a, 0) = a`. */
static double mtoc2_mod_real(double a, double b) {
  if (b == 0.0) return a;
  double r = fmod(a, b);
  if (r != 0.0 && ((r < 0) != (b < 0))) r += b;
  return r;
}

/* Power matching ECMAScript `Math.pow` (which numbl's interpreter and
 * JS-JIT both use), NOT C99 `pow`. They differ only when |base| == 1 and
 * the exponent is non-finite: C99 `pow(±1, ±inf) == pow(1, nan) == 1`,
 * but `Math.pow` returns NaN there. Using bare C `pow` here would make
 * the C-JIT (`--opt 2`) disagree with `--opt 0/1` for e.g. `1 .^ NaN`. */
static double mtoc2_pow_real(double a, double b) {
  if ((a == 1.0 || a == -1.0) && !isfinite(b)) return NAN;
  return pow(a, b);
}

#define MTOC2_DEFINE_ELEMWISE_TT_FN(name, FN)                               \
  static mtoc2_tensor_t name(mtoc2_tensor_t a, mtoc2_tensor_t b) {          \
    long n = 1;                                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                        \
    mtoc2_tensor_t r;                                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));                       \
    r.imag = NULL;                                                          \
    r.ndim = a.ndim;                                                        \
    for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];                 \
    MTOC2_OMP_PARFOR_N                                                      \
    for (long i = 0; i < n; i++) r.real[i] = FN(a.real[i], b.real[i]);      \
    return r;                                                               \
  }

#define MTOC2_DEFINE_ELEMWISE_TS_FN(name, FN)                               \
  static mtoc2_tensor_t name(mtoc2_tensor_t a, double s) {                  \
    long n = 1;                                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                        \
    mtoc2_tensor_t r;                                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));                       \
    r.imag = NULL;                                                          \
    r.ndim = a.ndim;                                                        \
    for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];                 \
    MTOC2_OMP_PARFOR_N                                                      \
    for (long i = 0; i < n; i++) r.real[i] = FN(a.real[i], s);              \
    return r;                                                               \
  }

#define MTOC2_DEFINE_ELEMWISE_ST_FN(name, FN)                               \
  static mtoc2_tensor_t name(double s, mtoc2_tensor_t b) {                  \
    long n = 1;                                                             \
    for (int i = 0; i < b.ndim; i++) n *= b.dims[i];                        \
    mtoc2_tensor_t r;                                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));                       \
    r.imag = NULL;                                                          \
    r.ndim = b.ndim;                                                        \
    for (int i = 0; i < b.ndim; i++) r.dims[i] = b.dims[i];                 \
    MTOC2_OMP_PARFOR_N                                                      \
    for (long i = 0; i < n; i++) r.real[i] = FN(s, b.real[i]);              \
    return r;                                                               \
  }

/* Broadcasting tensor-tensor helper for FN-kernel ops. See the
 * `MTOC2_DEFINE_ELEMWISE_BCAST_TT` doc in `tensor_elemwise_real.h` for
 * the broadcast rule. */
#define MTOC2_DEFINE_ELEMWISE_BCAST_TT_FN(name, FN)                         \
  static mtoc2_tensor_t name(mtoc2_tensor_t a, mtoc2_tensor_t b) {          \
    int rnd = a.ndim > b.ndim ? a.ndim : b.ndim;                            \
    long adim[MTOC2_MAX_NDIM], bdim[MTOC2_MAX_NDIM];                        \
    long rdim[MTOC2_MAX_NDIM];                                              \
    long astride[MTOC2_MAX_NDIM], bstride[MTOC2_MAX_NDIM];                  \
    long aacc = 1, bacc = 1;                                                \
    long n = 1;                                                             \
    for (int i = 0; i < rnd; i++) {                                         \
      adim[i] = (i < a.ndim) ? a.dims[i] : 1;                               \
      bdim[i] = (i < b.ndim) ? b.dims[i] : 1;                               \
      rdim[i] = (adim[i] == 1) ? bdim[i] : adim[i];                         \
      astride[i] = (adim[i] == 1) ? 0 : aacc;                               \
      bstride[i] = (bdim[i] == 1) ? 0 : bacc;                               \
      aacc *= adim[i];                                                      \
      bacc *= bdim[i];                                                      \
      n *= rdim[i];                                                         \
    }                                                                       \
    mtoc2_tensor_t r;                                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));                       \
    r.imag = NULL;                                                          \
    r.ndim = rnd;                                                           \
    for (int i = 0; i < rnd; i++) r.dims[i] = rdim[i];                      \
    long ix[MTOC2_MAX_NDIM] = {0};                                          \
    for (long k = 0; k < n; k++) {                                          \
      long ai = 0, bi = 0;                                                  \
      for (int i = 0; i < rnd; i++) {                                       \
        ai += ix[i] * astride[i];                                           \
        bi += ix[i] * bstride[i];                                           \
      }                                                                     \
      r.real[k] = FN(a.real[ai], b.real[bi]);                               \
      for (int i = 0; i < rnd; i++) {                                       \
        ix[i]++;                                                            \
        if (ix[i] < rdim[i]) break;                                         \
        ix[i] = 0;                                                          \
      }                                                                     \
    }                                                                       \
    return r;                                                               \
  }

MTOC2_DEFINE_ELEMWISE_TT_FN(mtoc2_tensor_mod_tt,   mtoc2_mod_real)
MTOC2_DEFINE_ELEMWISE_TT_FN(mtoc2_tensor_rem_tt,   fmod)
MTOC2_DEFINE_ELEMWISE_TT_FN(mtoc2_tensor_atan2_tt, atan2)
MTOC2_DEFINE_ELEMWISE_TT_FN(mtoc2_tensor_hypot_tt, hypot)
MTOC2_DEFINE_ELEMWISE_TT_FN(mtoc2_tensor_power_tt, mtoc2_pow_real)

MTOC2_DEFINE_ELEMWISE_TS_FN(mtoc2_tensor_mod_ts,   mtoc2_mod_real)
MTOC2_DEFINE_ELEMWISE_TS_FN(mtoc2_tensor_rem_ts,   fmod)
MTOC2_DEFINE_ELEMWISE_TS_FN(mtoc2_tensor_atan2_ts, atan2)
MTOC2_DEFINE_ELEMWISE_TS_FN(mtoc2_tensor_hypot_ts, hypot)
MTOC2_DEFINE_ELEMWISE_TS_FN(mtoc2_tensor_power_ts, mtoc2_pow_real)

/* Non-commutative non-fn ops (mod, rem, atan2, power) need scalar-
 * first variants. `hypot` is commutative; the builtin emits `_ts`
 * with swapped operands at scalar-OP-tensor sites. */
MTOC2_DEFINE_ELEMWISE_ST_FN(mtoc2_tensor_mod_st,   mtoc2_mod_real)
MTOC2_DEFINE_ELEMWISE_ST_FN(mtoc2_tensor_rem_st,   fmod)
MTOC2_DEFINE_ELEMWISE_ST_FN(mtoc2_tensor_atan2_st, atan2)
MTOC2_DEFINE_ELEMWISE_ST_FN(mtoc2_tensor_power_st, mtoc2_pow_real)

MTOC2_DEFINE_ELEMWISE_BCAST_TT_FN(mtoc2_tensor_mod_bcast_tt,   mtoc2_mod_real)
MTOC2_DEFINE_ELEMWISE_BCAST_TT_FN(mtoc2_tensor_rem_bcast_tt,   fmod)
MTOC2_DEFINE_ELEMWISE_BCAST_TT_FN(mtoc2_tensor_atan2_bcast_tt, atan2)
MTOC2_DEFINE_ELEMWISE_BCAST_TT_FN(mtoc2_tensor_hypot_bcast_tt, hypot)
MTOC2_DEFINE_ELEMWISE_BCAST_TT_FN(mtoc2_tensor_power_bcast_tt, mtoc2_pow_real)
