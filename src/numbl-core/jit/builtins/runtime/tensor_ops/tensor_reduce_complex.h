/* JIT runtime helpers: complex-tensor reductions.
 *
 * Sibling of `tensor_reduce_real.h`. Same `_all` / `_dim` shape per
 * op; each kernel walks both lanes and builds intermediate
 * `double _Complex` values through `jit_cmake` / `jit_c*` (no
 * bare `<complex.h>` operators).
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
static void jit__squeeze_trailing_c(int *ndim, long *dims) {
  while (*ndim > 2 && dims[*ndim - 1] == 1) {
    (*ndim)--;
  }
}

/* Numeric (sum/prod/mean) reduction template — complex accumulator. */
#define JIT_DEFINE_CACCUM_REDUCTION(name, INIT, ACCUM, FINALIZE)              \
  static double _Complex jit_##name##_complex_all(jit_tensor_t a) {           \
    long n = 1;                                                               \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                          \
    double _Complex acc = (INIT);                                             \
    for (long i = 0; i < n; i++) {                                            \
      double aim = (a.imag != NULL) ? a.imag[i] : 0.0;                        \
      double _Complex x = jit_cmake(a.real[i], aim);                          \
      acc = ACCUM(acc, x);                                                    \
    }                                                                         \
    return FINALIZE(acc, n);                                                  \
  }                                                                           \
                                                                              \
  static jit_tensor_t jit_##name##_complex_dim(                               \
      jit_tensor_t a, int dim) {                                              \
    if (dim < 1) {                                                            \
      jit_raise_bad_dim();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      jit_tensor_t out = jit_tensor_alloc_nd_complex(a.ndim, a.dims);         \
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
    long out_dims[JIT_MAX_NDIM];                                              \
    int out_ndim = a.ndim;                                                    \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    out_dims[dimIdx] = 1;                                                     \
    jit__squeeze_trailing_c(&out_ndim, out_dims);                             \
    jit_tensor_t out = jit_tensor_alloc_nd_complex(out_ndim, out_dims);       \
    long slab = before * axis;                                                \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        double _Complex acc = (INIT);                                         \
        for (long k = 0; k < axis; k++) {                                     \
          long off = slabBase + inner + k * before;                           \
          double aim = (a.imag != NULL) ? a.imag[off] : 0.0;                  \
          double _Complex x = jit_cmake(a.real[off], aim);                    \
          acc = ACCUM(acc, x);                                                \
        }                                                                     \
        double _Complex fin = FINALIZE(acc, axis);                            \
        long dst = outer * before + inner;                                    \
        out.real[dst] = jit_creal(fin);                                       \
        out.imag[dst] = jit_cimag(fin);                                       \
      }                                                                       \
    }                                                                         \
    return out;                                                               \
  }

/* True when the tensor carries no imaginary content (NULL lane or every
 * element zero). Such a tensor is real in value — min/max must order by
 * value, not magnitude, to match the interpreter and MATLAB on real data. */
static int jit__creduce_all_imag_zero(jit_tensor_t a) {
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
#define JIT_DEFINE_CMINMAX_REDUCTION(name, CMP)                                \
  static int jit__##name##_complex_better(                                    \
      double aRe, double aIm, double bRe, double bIm) {                       \
    double absA = hypot(aRe, aIm);                                            \
    double absB = hypot(bRe, bIm);                                            \
    if (absA != absB) return absA CMP absB;                                   \
    return atan2(aIm, aRe) CMP atan2(bIm, bRe);                               \
  }                                                                           \
  static double _Complex jit_##name##_complex_all(jit_tensor_t a) {           \
    long n = 1;                                                               \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                          \
    int realMode = jit__creduce_all_imag_zero(a);                             \
    int found = 0;                                                            \
    double mRe = NAN, mIm = 0.0;                                              \
    for (long i = 0; i < n; i++) {                                            \
      double xr = a.real[i];                                                  \
      double xi = (a.imag != NULL) ? a.imag[i] : 0.0;                         \
      if (xr != xr || xi != xi) continue;                                     \
      int better = realMode ? (xr CMP mRe)                                    \
                            : jit__##name##_complex_better(xr, xi, mRe, mIm);   \
      if (!found || better) {                                                 \
        mRe = xr;                                                             \
        mIm = xi;                                                             \
        found = 1;                                                            \
      }                                                                       \
    }                                                                         \
    return jit_cmake(mRe, mIm);                                               \
  }                                                                           \
                                                                              \
  static jit_tensor_t jit_##name##_complex_dim(                               \
      jit_tensor_t a, int dim) {                                              \
    if (dim < 1) {                                                            \
      jit_raise_bad_dim();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      jit_tensor_t out = jit_tensor_alloc_nd_complex(a.ndim, a.dims);         \
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
    long out_dims[JIT_MAX_NDIM];                                              \
    int out_ndim = a.ndim;                                                    \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    out_dims[dimIdx] = 1;                                                     \
    jit__squeeze_trailing_c(&out_ndim, out_dims);                             \
    jit_tensor_t out = jit_tensor_alloc_nd_complex(out_ndim, out_dims);       \
    long slab = before * axis;                                                \
    int realMode = jit__creduce_all_imag_zero(a);                             \
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
                                : jit__##name##_complex_better(xr, xi, mRe, mIm);   \
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
 * nonzero). Mirrors `JIT_DEFINE_LOGICAL_REDUCTION` shape. */
#define JIT_DEFINE_CLOGICAL_REDUCTION(name, EMPTY_RESULT, SHORT_BODY)          \
  static double jit_##name##_complex_all(jit_tensor_t a) {                    \
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
  static jit_tensor_t jit_##name##_complex_dim(                               \
      jit_tensor_t a, int dim) {                                              \
    if (dim < 1) {                                                            \
      jit_raise_bad_dim();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      long out_dims[JIT_MAX_NDIM];                                            \
      int out_ndim = a.ndim;                                                  \
      for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];               \
      jit_tensor_t out = jit_tensor_alloc_nd(out_ndim, out_dims);             \
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
    long out_dims[JIT_MAX_NDIM];                                              \
    int out_ndim = a.ndim;                                                    \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    out_dims[dimIdx] = 1;                                                     \
    jit__squeeze_trailing_c(&out_ndim, out_dims);                             \
    jit_tensor_t out = jit_tensor_alloc_nd(out_ndim, out_dims);               \
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
#define JIT_CACC_SUM(acc, x) jit_cadd((acc), (x))
#define JIT_CACC_PROD(acc, x) jit_cmul((acc), (x))
#define JIT_CFIN_ID(acc, n) (acc)
#define JIT_CFIN_MEAN(acc, n) jit_cdiv((acc), jit_cmake((double)(n), 0.0))

JIT_DEFINE_CACCUM_REDUCTION(sum, jit_cmake(0.0, 0.0), JIT_CACC_SUM, JIT_CFIN_ID)
JIT_DEFINE_CACCUM_REDUCTION(prod, jit_cmake(1.0, 0.0), JIT_CACC_PROD, JIT_CFIN_ID)
JIT_DEFINE_CACCUM_REDUCTION(mean, jit_cmake(0.0, 0.0), JIT_CACC_SUM, JIT_CFIN_MEAN)

JIT_DEFINE_CMINMAX_REDUCTION(min, <)
JIT_DEFINE_CMINMAX_REDUCTION(max, >)

JIT_DEFINE_CLOGICAL_REDUCTION(any, 0,
  if (x) { acc = 1.0; break; })
JIT_DEFINE_CLOGICAL_REDUCTION(all, 1,
  if (!x) { acc = 0.0; break; })
