/* mtoc2 runtime helpers: complex-tensor reductions.
 *
 * Sibling of `tensor_reduce_real.h`. Same `_all` / `_dim` shape per
 * op; each kernel walks both lanes and builds intermediate
 * `double _Complex` values through `mtoc2_cmake` / `mtoc2_c*` so the
 * c2js backend can translate the bodies straight (no bare
 * `<complex.h>` operators).
 *
 * Result types per op:
 *   sum / prod / mean         → complex (lane-pair accumulator)
 *   min / max                 → complex (magnitude compare, atan2
 *                                tiebreak — matches numbl's
 *                                `complexIsBetter`)
 *   any / all                 → real (toBool: `re != 0 || im != 0`,
 *                                then aggregate via OR/AND)
 *
 * Input tolerance: `imag == NULL` (a real tensor flowing in through
 * a complex-typed route) is treated as zero on every cell.
 *
 * `_all` returns a `double _Complex` for the numeric reducers and a
 * `double` for the logical reducers. `_dim` returns a freshly-owned
 * complex tensor (numeric reducers) or a real tensor (logical
 * reducers) of the reduced shape — same trailing-singleton squeeze
 * rule as the real helper.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Shared with the real helper; defined inline here so this file is
 * standalone-includable (the runtime activator may pull the complex
 * snippet in independently of the real one). */
static void mtoc2__squeeze_trailing_c(int *ndim, long *dims) {
  while (*ndim > 2 && dims[*ndim - 1] == 1) {
    (*ndim)--;
  }
}

/* Numeric (sum/prod/mean) reduction template — complex accumulator. */
#define MTOC2_DEFINE_CACCUM_REDUCTION(name, INIT, ACCUM, FINALIZE)            \
  static double _Complex mtoc2_##name##_complex_all(mtoc2_tensor_t a) {       \
    long n = 1;                                                               \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                          \
    double _Complex acc = (INIT);                                             \
    for (long i = 0; i < n; i++) {                                            \
      double aim = (a.imag != NULL) ? a.imag[i] : 0.0;                        \
      double _Complex x = mtoc2_cmake(a.real[i], aim);                        \
      acc = ACCUM(acc, x);                                                    \
    }                                                                         \
    return FINALIZE(acc, n);                                                  \
  }                                                                           \
                                                                              \
  static mtoc2_tensor_t mtoc2_##name##_complex_dim(                           \
      mtoc2_tensor_t a, int dim) {                                            \
    if (dim < 1) {                                                            \
      fprintf(stderr, "mtoc2: " #name "_complex_dim: dim must be >= 1 (got %d)\n", dim); \
      abort();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);     \
      memcpy(out.real, a.real, (size_t)total * sizeof(double));               \
      if (a.imag != NULL) {                                                   \
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
    long out_dims[MTOC2_MAX_NDIM];                                            \
    int out_ndim = a.ndim;                                                    \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    out_dims[dimIdx] = 1;                                                     \
    mtoc2__squeeze_trailing_c(&out_ndim, out_dims);                           \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(out_ndim, out_dims);   \
    long slab = before * axis;                                                \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        double _Complex acc = (INIT);                                         \
        for (long k = 0; k < axis; k++) {                                     \
          long off = slabBase + inner + k * before;                           \
          double aim = (a.imag != NULL) ? a.imag[off] : 0.0;                  \
          double _Complex x = mtoc2_cmake(a.real[off], aim);                  \
          acc = ACCUM(acc, x);                                                \
        }                                                                     \
        double _Complex fin = FINALIZE(acc, axis);                            \
        long dst = outer * before + inner;                                    \
        out.real[dst] = mtoc2_creal(fin);                                     \
        out.imag[dst] = mtoc2_cimag(fin);                                     \
      }                                                                       \
    }                                                                         \
    return out;                                                               \
  }

/* True when the tensor carries no imaginary content (NULL lane or every
 * element zero). Such a tensor is real in value — min/max must order by
 * value, not magnitude, to match the interpreter and MATLAB on real data. */
static int mtoc2__creduce_all_imag_zero(mtoc2_tensor_t a) {
  if (a.imag == NULL) return 1;
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  for (long i = 0; i < n; i++) {
    if (a.imag[i] != 0.0) return 0;
  }
  return 1;
}

/* min/max template — complex compare via magnitude + atan2 tiebreak.
 * When the whole tensor is real (all-zero imag) we order by value
 * instead. NaN-skip on either lane. Accumulator seed is NaN+NaN; first
 * non-NaN element captures, later non-NaN elements compare. */
#define MTOC2_DEFINE_CMINMAX_REDUCTION(name, CMP)                              \
  static int mtoc2__##name##_complex_better(                                  \
      double aRe, double aIm, double bRe, double bIm) {                       \
    double absA = hypot(aRe, aIm);                                            \
    double absB = hypot(bRe, bIm);                                            \
    if (absA != absB) return absA CMP absB;                                   \
    return atan2(aIm, aRe) CMP atan2(bIm, bRe);                               \
  }                                                                           \
  static double _Complex mtoc2_##name##_complex_all(mtoc2_tensor_t a) {       \
    long n = 1;                                                               \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                          \
    int realMode = mtoc2__creduce_all_imag_zero(a);                           \
    int found = 0;                                                            \
    double mRe = NAN, mIm = 0.0;                                              \
    for (long i = 0; i < n; i++) {                                            \
      double xr = a.real[i];                                                  \
      double xi = (a.imag != NULL) ? a.imag[i] : 0.0;                         \
      if (xr != xr || xi != xi) continue;                                     \
      int better = realMode ? (xr CMP mRe)                                    \
                            : mtoc2__##name##_complex_better(xr, xi, mRe, mIm); \
      if (!found || better) {                                                 \
        mRe = xr;                                                             \
        mIm = xi;                                                             \
        found = 1;                                                            \
      }                                                                       \
    }                                                                         \
    return mtoc2_cmake(mRe, mIm);                                             \
  }                                                                           \
                                                                              \
  static mtoc2_tensor_t mtoc2_##name##_complex_dim(                           \
      mtoc2_tensor_t a, int dim) {                                            \
    if (dim < 1) {                                                            \
      fprintf(stderr, "mtoc2: " #name "_complex_dim: dim must be >= 1 (got %d)\n", dim); \
      abort();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);     \
      memcpy(out.real, a.real, (size_t)total * sizeof(double));               \
      if (a.imag != NULL) {                                                   \
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
    long out_dims[MTOC2_MAX_NDIM];                                            \
    int out_ndim = a.ndim;                                                    \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    out_dims[dimIdx] = 1;                                                     \
    mtoc2__squeeze_trailing_c(&out_ndim, out_dims);                           \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(out_ndim, out_dims);   \
    long slab = before * axis;                                                \
    int realMode = mtoc2__creduce_all_imag_zero(a);                           \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        int found = 0;                                                        \
        double mRe = NAN, mIm = 0.0;                                          \
        for (long k = 0; k < axis; k++) {                                     \
          long off = slabBase + inner + k * before;                           \
          double xr = a.real[off];                                            \
          double xi = (a.imag != NULL) ? a.imag[off] : 0.0;                   \
          if (xr != xr || xi != xi) continue;                                 \
          int better = realMode ? (xr CMP mRe)                                \
                                : mtoc2__##name##_complex_better(xr, xi, mRe, mIm); \
          if (!found || better) {                                             \
            mRe = xr;                                                         \
            mIm = xi;                                                         \
            found = 1;                                                        \
          }                                                                   \
        }                                                                     \
        long dst = outer * before + inner;                                    \
        out.real[dst] = mRe;                                                  \
        out.imag[dst] = mIm;                                                  \
      }                                                                       \
    }                                                                         \
    return out;                                                               \
  }

/* any/all template — real result; toBool per element (either lane
 * nonzero). Mirrors `MTOC2_DEFINE_LOGICAL_REDUCTION` shape. */
#define MTOC2_DEFINE_CLOGICAL_REDUCTION(name, EMPTY_RESULT, SHORT_BODY)        \
  static double mtoc2_##name##_complex_all(mtoc2_tensor_t a) {                \
    long n = 1;                                                               \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                          \
    if (n == 0) return (double)(EMPTY_RESULT);                                \
    double acc = (double)(EMPTY_RESULT);                                      \
    for (long i = 0; i < n; i++) {                                            \
      double xr = a.real[i];                                                  \
      double xi = (a.imag != NULL) ? a.imag[i] : 0.0;                         \
      int x = (xr != 0.0 || xi != 0.0);                                       \
      SHORT_BODY;                                                             \
    }                                                                         \
    return acc;                                                               \
  }                                                                           \
                                                                              \
  static mtoc2_tensor_t mtoc2_##name##_complex_dim(                           \
      mtoc2_tensor_t a, int dim) {                                            \
    if (dim < 1) {                                                            \
      fprintf(stderr, "mtoc2: " #name "_complex_dim: dim must be >= 1 (got %d)\n", dim); \
      abort();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      long out_dims[MTOC2_MAX_NDIM];                                          \
      int out_ndim = a.ndim;                                                  \
      for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];               \
      mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(out_ndim, out_dims);         \
      for (long i = 0; i < total; i++) {                                      \
        double xr = a.real[i];                                                \
        double xi = (a.imag != NULL) ? a.imag[i] : 0.0;                       \
        out.real[i] = (xr != 0.0 || xi != 0.0) ? 1.0 : 0.0;                   \
      }                                                                       \
      return out;                                                             \
    }                                                                         \
    int dimIdx = dim - 1;                                                     \
    long axis = a.dims[dimIdx];                                               \
    long before = 1;                                                          \
    for (int i = 0; i < dimIdx; i++) before *= a.dims[i];                     \
    long after = 1;                                                           \
    for (int i = dimIdx + 1; i < a.ndim; i++) after *= a.dims[i];             \
    long out_dims[MTOC2_MAX_NDIM];                                            \
    int out_ndim = a.ndim;                                                    \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    out_dims[dimIdx] = 1;                                                     \
    mtoc2__squeeze_trailing_c(&out_ndim, out_dims);                           \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(out_ndim, out_dims);           \
    long slab = before * axis;                                                \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        double acc = (double)(EMPTY_RESULT);                                  \
        for (long k = 0; k < axis; k++) {                                     \
          long off = slabBase + inner + k * before;                           \
          double xr = a.real[off];                                            \
          double xi = (a.imag != NULL) ? a.imag[off] : 0.0;                   \
          int x = (xr != 0.0 || xi != 0.0);                                   \
          SHORT_BODY;                                                         \
        }                                                                     \
        out.real[outer * before + inner] = acc;                               \
      }                                                                       \
    }                                                                         \
    return out;                                                               \
  }

/* Accumulator-statement macros. */
#define MTOC2_CACC_SUM(acc, x) mtoc2_cadd((acc), (x))
#define MTOC2_CACC_PROD(acc, x) mtoc2_cmul((acc), (x))
#define MTOC2_CFIN_ID(acc, n) (acc)
#define MTOC2_CFIN_MEAN(acc, n) mtoc2_cdiv((acc), mtoc2_cmake((double)(n), 0.0))

MTOC2_DEFINE_CACCUM_REDUCTION(sum, mtoc2_cmake(0.0, 0.0), MTOC2_CACC_SUM, MTOC2_CFIN_ID)
MTOC2_DEFINE_CACCUM_REDUCTION(prod, mtoc2_cmake(1.0, 0.0), MTOC2_CACC_PROD, MTOC2_CFIN_ID)
MTOC2_DEFINE_CACCUM_REDUCTION(mean, mtoc2_cmake(0.0, 0.0), MTOC2_CACC_SUM, MTOC2_CFIN_MEAN)

MTOC2_DEFINE_CMINMAX_REDUCTION(min, <)
MTOC2_DEFINE_CMINMAX_REDUCTION(max, >)

MTOC2_DEFINE_CLOGICAL_REDUCTION(any, 0,
  if (x) { acc = 1.0; break; })
MTOC2_DEFINE_CLOGICAL_REDUCTION(all, 1,
  if (!x) { acc = 0.0; break; })
