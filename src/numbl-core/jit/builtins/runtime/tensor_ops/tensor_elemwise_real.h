/* mtoc2 runtime helpers: elementwise binary + unary ops on real
 * tensors.
 *
 * Convention in the function names:
 *   `_tt`       — both args are tensors of the same statically-known
 *                 shape (the codegen fast path)
 *   `_ts`       — tensor + scalar (broadcast scalar across all elements)
 *   `_st`       — scalar + tensor (only emitted for non-commutative ops)
 *   `_bcast_tt` — both args are tensors but at least one axis needs
 *                 MATLAB-style implicit expansion. Pads the shorter
 *                 shape with trailing 1s and uses stride=0 on any
 *                 singleton axis. Allocates a result tensor of the
 *                 broadcast output shape and walks it column-major.
 *
 * Every helper returns a freshly-owned tensor — the codegen
 * invariant. The caller hands the result to `mtoc2_tensor_assign`
 * (or to a hoisted temp Assign for in-line uses). No shape check
 * in the same-shape paths: that requirement is enforced statically
 * at lowering time, so the runtime trusts its inputs. The bcast
 * helper similarly trusts that each axis pair is either equal or
 * has 1 on at least one side.
 */

#include <stdlib.h>

#define MTOC2_DEFINE_ELEMWISE_TT(name, OP)                                  \
  static mtoc2_tensor_t name(mtoc2_tensor_t a, mtoc2_tensor_t b) {          \
    long n = 1;                                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                        \
    mtoc2_tensor_t r;                                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));                       \
    r.imag = NULL;                                                          \
    r.ndim = a.ndim;                                                        \
    for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];                 \
    MTOC2_OMP_PARFOR_N                                                      \
    for (long i = 0; i < n; i++) r.real[i] = a.real[i] OP b.real[i];        \
    return r;                                                               \
  }

#define MTOC2_DEFINE_ELEMWISE_TS(name, OP)                                  \
  static mtoc2_tensor_t name(mtoc2_tensor_t a, double s) {                  \
    long n = 1;                                                             \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                        \
    mtoc2_tensor_t r;                                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));                       \
    r.imag = NULL;                                                          \
    r.ndim = a.ndim;                                                        \
    for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];                 \
    MTOC2_OMP_PARFOR_N                                                      \
    for (long i = 0; i < n; i++) r.real[i] = a.real[i] OP s;                \
    return r;                                                               \
  }

#define MTOC2_DEFINE_ELEMWISE_ST(name, OP)                                  \
  static mtoc2_tensor_t name(double s, mtoc2_tensor_t b) {                  \
    long n = 1;                                                             \
    for (int i = 0; i < b.ndim; i++) n *= b.dims[i];                        \
    mtoc2_tensor_t r;                                                       \
    r.real = mtoc2_alloc((size_t)n * sizeof(double));                       \
    r.imag = NULL;                                                          \
    r.ndim = b.ndim;                                                        \
    for (int i = 0; i < b.ndim; i++) r.dims[i] = b.dims[i];                 \
    MTOC2_OMP_PARFOR_N                                                      \
    for (long i = 0; i < n; i++) r.real[i] = s OP b.real[i];                \
    return r;                                                               \
  }

/* Broadcasting tensor-tensor helper. MATLAB-style implicit expansion:
 * pad the shorter shape with trailing 1s, then for each axis either
 * the dims match or one of them is 1 (replicated to the other's size).
 * The codegen path emits this only when at least one axis statically
 * needs broadcasting; same-shape calls still go through `_tt`. */
#define MTOC2_DEFINE_ELEMWISE_BCAST_TT(name, OP)                            \
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
    /* 2-D fast path: avoids the per-element ndim-D index update and the   \
     * per-element stride-vector dot product. Most MATLAB bcast uses are   \
     * 2-D (row-vector OP column-vector style), and on the JS target this  \
     * specialization removes the dominant tensor-binop cost. */           \
    if (rnd == 2) {                                                         \
      long d0 = rdim[0], d1 = rdim[1];                                      \
      long as0 = astride[0], as1 = astride[1];                              \
      long bs0 = bstride[0], bs1 = bstride[1];                              \
      /* Compute each row's `k` offset (`i1 * d0`) per outer iteration so   \
       * the outer loop has no carried state — required for the OpenMP     \
       * parallel-for below to be safe. */                                  \
      MTOC2_OMP_PARFOR_N                                                    \
      for (long i1 = 0; i1 < d1; i1++) {                                    \
        long aoff1 = i1 * as1, boff1 = i1 * bs1, koff1 = i1 * d0;           \
        for (long i0 = 0; i0 < d0; i0++) {                                  \
          r.real[koff1 + i0] = a.real[aoff1 + i0 * as0] OP b.real[boff1 + i0 * bs0]; \
        }                                                                   \
      }                                                                     \
      return r;                                                             \
    }                                                                       \
    long ix[MTOC2_MAX_NDIM] = {0};                                          \
    for (long k = 0; k < n; k++) {                                          \
      long ai = 0, bi = 0;                                                  \
      for (int i = 0; i < rnd; i++) {                                       \
        ai += ix[i] * astride[i];                                           \
        bi += ix[i] * bstride[i];                                           \
      }                                                                     \
      r.real[k] = a.real[ai] OP b.real[bi];                                 \
      for (int i = 0; i < rnd; i++) {                                       \
        ix[i]++;                                                            \
        if (ix[i] < rdim[i]) break;                                         \
        ix[i] = 0;                                                          \
      }                                                                     \
    }                                                                       \
    return r;                                                               \
  }

MTOC2_DEFINE_ELEMWISE_TT(mtoc2_tensor_plus_tt, +)
MTOC2_DEFINE_ELEMWISE_TT(mtoc2_tensor_minus_tt, -)
MTOC2_DEFINE_ELEMWISE_TT(mtoc2_tensor_times_tt, *)
MTOC2_DEFINE_ELEMWISE_TT(mtoc2_tensor_rdivide_tt, /)

MTOC2_DEFINE_ELEMWISE_TS(mtoc2_tensor_plus_ts, +)
MTOC2_DEFINE_ELEMWISE_TS(mtoc2_tensor_minus_ts, -)
MTOC2_DEFINE_ELEMWISE_TS(mtoc2_tensor_times_ts, *)
MTOC2_DEFINE_ELEMWISE_TS(mtoc2_tensor_rdivide_ts, /)

/* Commutative ops (plus, times) reuse `_ts` for `scalar OP tensor`.
 * Non-commutative ops need their own scalar-first flavor. */
MTOC2_DEFINE_ELEMWISE_ST(mtoc2_tensor_minus_st, -)
MTOC2_DEFINE_ELEMWISE_ST(mtoc2_tensor_rdivide_st, /)

MTOC2_DEFINE_ELEMWISE_BCAST_TT(mtoc2_tensor_plus_bcast_tt, +)
MTOC2_DEFINE_ELEMWISE_BCAST_TT(mtoc2_tensor_minus_bcast_tt, -)
MTOC2_DEFINE_ELEMWISE_BCAST_TT(mtoc2_tensor_times_bcast_tt, *)
MTOC2_DEFINE_ELEMWISE_BCAST_TT(mtoc2_tensor_rdivide_bcast_tt, /)

/* Unary minus — negate every element. */
static mtoc2_tensor_t mtoc2_tensor_uminus(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)n * sizeof(double));
  r.imag = NULL;
  r.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];
  MTOC2_OMP_PARFOR_N
  for (long i = 0; i < n; i++) r.real[i] = -a.real[i];
  return r;
}
