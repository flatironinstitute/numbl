/* mtoc2 runtime helpers: cumulative (prefix-scan) ops on tensors.
 *
 * One macro per op generates a `_dim` helper that scans along the
 * 1-based axis `dim`, returning a freshly-owned tensor of the SAME
 * shape as the input (no squeeze, no collapse — `cumsum`/`cumprod`
 * are shape-preserving by definition). `*_complex_dim` siblings walk
 * both lanes (cumsum adds component-wise; cumprod is a per-step
 * complex multiplication).
 *
 * Layout mirrors numbl's `cumOp` (helpers/reduction/cumulative.ts):
 * column-major (before × axis × after) traversal with stride `before`
 * between elements along the scanned axis. Per-fiber accumulator
 * resets at each (outer, inner) pair.
 *
 * If `dim > a.ndim`, the helper emits a fresh copy (matches numbl's
 * branch returning `RTV.tensor(allocFloat64Array(v.data), shape)`).
 *
 * NaN propagation is the natural C behaviour — once a NaN enters the
 * accumulator, every subsequent output along that fiber is NaN.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MTOC2_DEFINE_CUMULATIVE(name, INIT, ACCUM)                            \
  static mtoc2_tensor_t mtoc2_tensor_##name##_dim(mtoc2_tensor_t a, int dim) {\
    if (dim < 1) {                                                            \
      fprintf(stderr, "mtoc2: " #name "_dim: dim must be >= 1 (got %d)\n",    \
              dim);                                                           \
      abort();                                                                \
    }                                                                         \
    long total = 1;                                                           \
    for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                      \
    long out_dims[MTOC2_MAX_NDIM];                                            \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(a.ndim, out_dims);             \
    if (dim > a.ndim) {                                                       \
      /* No-op axis: output is a fresh copy of the input. */                  \
      memcpy(out.real, a.real, (size_t)total * sizeof(double));               \
      return out;                                                             \
    }                                                                         \
    int dimIdx = dim - 1;                                                     \
    long axis = a.dims[dimIdx];                                               \
    long before = 1;                                                          \
    for (int i = 0; i < dimIdx; i++) before *= a.dims[i];                     \
    long after = 1;                                                           \
    for (int i = dimIdx + 1; i < a.ndim; i++) after *= a.dims[i];             \
    long slab = before * axis;                                                \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        double acc = (INIT);                                                  \
        for (long k = 0; k < axis; k++) {                                     \
          long idx = slabBase + inner + k * before;                           \
          double x = a.real[idx];                                             \
          ACCUM(acc, x);                                                      \
          out.real[idx] = acc;                                                \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    return out;                                                               \
  }

#define MTOC2_CUM_ACC_SUM(acc, x)  do { (acc) += (x); } while (0)
#define MTOC2_CUM_ACC_PROD(acc, x) do { (acc) *= (x); } while (0)

MTOC2_DEFINE_CUMULATIVE(cumsum,  0.0, MTOC2_CUM_ACC_SUM)
MTOC2_DEFINE_CUMULATIVE(cumprod, 1.0, MTOC2_CUM_ACC_PROD)

#define MTOC2_DEFINE_CUMULATIVE_COMPLEX(name, INIT_RE, INIT_IM, ACCUM)        \
  static mtoc2_tensor_t                                                       \
  mtoc2_tensor_##name##_complex_dim(mtoc2_tensor_t a, int dim) {              \
    if (dim < 1) {                                                            \
      fprintf(stderr, "mtoc2: " #name "_complex_dim: dim must be >= 1 (got %d)\n", \
              dim);                                                           \
      abort();                                                                \
    }                                                                         \
    long total = 1;                                                           \
    for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                      \
    long out_dims[MTOC2_MAX_NDIM];                                            \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(a.ndim, out_dims);     \
    int srcHasImag = (a.imag != NULL);                                        \
    if (dim > a.ndim) {                                                       \
      memcpy(out.real, a.real, (size_t)total * sizeof(double));               \
      if (srcHasImag) {                                                       \
        memcpy(out.imag, a.imag, (size_t)total * sizeof(double));             \
      } else {                                                                \
        memset(out.imag, 0, (size_t)total * sizeof(double));                  \
      }                                                                       \
      return out;                                                             \
    }                                                                         \
    int dimIdx = dim - 1;                                                     \
    long axis = a.dims[dimIdx];                                               \
    long before = 1;                                                          \
    for (int i = 0; i < dimIdx; i++) before *= a.dims[i];                     \
    long after = 1;                                                           \
    for (int i = dimIdx + 1; i < a.ndim; i++) after *= a.dims[i];             \
    long slab = before * axis;                                                \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        double accRe = (INIT_RE);                                             \
        double accIm = (INIT_IM);                                             \
        for (long k = 0; k < axis; k++) {                                     \
          long idx = slabBase + inner + k * before;                           \
          double xRe = a.real[idx];                                           \
          double xIm = srcHasImag ? a.imag[idx] : 0.0;                        \
          ACCUM(accRe, accIm, xRe, xIm);                                      \
          out.real[idx] = accRe;                                              \
          out.imag[idx] = accIm;                                              \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    return out;                                                               \
  }

#define MTOC2_CUM_ACC_SUM_COMPLEX(aRe, aIm, xRe, xIm)                         \
  do { (aRe) += (xRe); (aIm) += (xIm); } while (0)

#define MTOC2_CUM_ACC_PROD_COMPLEX(aRe, aIm, xRe, xIm)                        \
  do {                                                                        \
    double _t_re = (aRe) * (xRe) - (aIm) * (xIm);                             \
    double _t_im = (aRe) * (xIm) + (aIm) * (xRe);                             \
    (aRe) = _t_re;                                                            \
    (aIm) = _t_im;                                                            \
  } while (0)

MTOC2_DEFINE_CUMULATIVE_COMPLEX(cumsum,  0.0, 0.0, MTOC2_CUM_ACC_SUM_COMPLEX)
MTOC2_DEFINE_CUMULATIVE_COMPLEX(cumprod, 1.0, 0.0, MTOC2_CUM_ACC_PROD_COMPLEX)
