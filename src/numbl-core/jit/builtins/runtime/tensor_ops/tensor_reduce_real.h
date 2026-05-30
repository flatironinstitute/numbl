/* mtoc2 runtime helpers: real-tensor reductions.
 *
 * One macro per op generates two helpers:
 *
 *   mtoc2_<name>_all(a)       — reduce every element to a scalar.
 *   mtoc2_<name>_dim(a, dim)  — reduce along the 1-based axis `dim`,
 *                               returning a freshly-owned tensor.
 *
 * The `_dim` template mirrors numbl's `forEachSlice`: compute
 * `before = prod(dims[0..dim-2])`, `axis = dims[dim-1]`,
 * `after = prod(dims[dim..ndim-1])`. Walk the column-major buffer in
 * `(after × before)` fiber order with stride `before` between
 * elements along the reduced axis. Output dims are the input dims
 * with `dims[dim-1] = 1`, then trailing singletons stripped subject
 * to a 2-axis floor (matches the type system's
 * `tensorDoubleFromDims` rule).
 *
 * If `dim > a.ndim` the runtime emits a per-op no-op: every reducer
 * (sum/prod/mean/min/max) copies `a` as-is; the logical reducers
 * (any/all) emit an elementwise cast to {0, 1}. The transfer step
 * already proved the output shape, so this branch only fires when
 * the type-side dim/shape analysis can't fold to AxisAll.
 *
 * Real-only — complex is out of scope. Same-shape and column-major
 * conventions match the rest of mtoc2's tensor runtime.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Strip trailing singleton axes down to a 2-axis floor. Updates
 * `*ndim` in place; `dims` is the row buffer. */
static void mtoc2__squeeze_trailing(int *ndim, long *dims) {
  while (*ndim > 2 && dims[*ndim - 1] == 1) {
    (*ndim)--;
  }
}

/* Helper: reduce-all loop for accumulator-based reducers
 * (sum, prod, mean). `INIT` seeds the accumulator; `ACCUM(acc, x)`
 * is a C statement updating `acc`; `FINALIZE(acc, n)` is the final
 * transformation given the count. */
#define MTOC2_DEFINE_ACCUM_REDUCTION(name, INIT, ACCUM, FINALIZE)             \
  static double mtoc2_##name##_all(mtoc2_tensor_t a) {                        \
    long n = 1;                                                               \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                          \
    double acc = (INIT);                                                      \
    for (long i = 0; i < n; i++) {                                            \
      double x = a.real[i];                                                   \
      ACCUM(acc, x);                                                          \
    }                                                                         \
    return FINALIZE(acc, n);                                                  \
  }                                                                           \
                                                                              \
  static mtoc2_tensor_t mtoc2_##name##_dim(mtoc2_tensor_t a, int dim) {       \
    if (dim < 1) {                                                            \
      fprintf(stderr, "mtoc2: " #name "_dim: dim must be >= 1 (got %d)\n",    \
              dim);                                                           \
      abort();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      /* No-op axis: output is same shape/data as input (fresh copy). */      \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      mtoc2_tensor_t out;                                                     \
      out.ndim = a.ndim;                                                      \
      for (int i = 0; i < a.ndim; i++) out.dims[i] = a.dims[i];               \
      out.real = mtoc2_alloc((size_t)total * sizeof(double));                 \
      out.imag = NULL;                                                        \
      memcpy(out.real, a.real, (size_t)total * sizeof(double));               \
      return out;                                                             \
    }                                                                         \
    int dimIdx = dim - 1;                                                     \
    long axis = a.dims[dimIdx];                                               \
    long before = 1;                                                          \
    for (int i = 0; i < dimIdx; i++) before *= a.dims[i];                     \
    long after = 1;                                                           \
    for (int i = dimIdx + 1; i < a.ndim; i++) after *= a.dims[i];             \
    long out_total = before * after;                                          \
    long out_dims[MTOC2_MAX_NDIM];                                            \
    int out_ndim = a.ndim;                                                    \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                 \
    out_dims[dimIdx] = 1;                                                     \
    mtoc2__squeeze_trailing(&out_ndim, out_dims);                             \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(out_ndim, out_dims);           \
    long slab = before * axis;                                                \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        double acc = (INIT);                                                  \
        for (long k = 0; k < axis; k++) {                                     \
          double x = a.real[slabBase + inner + k * before];                   \
          ACCUM(acc, x);                                                      \
        }                                                                     \
        out.real[outer * before + inner] = FINALIZE(acc, axis);               \
      }                                                                       \
    }                                                                         \
    (void)out_total;                                                          \
    return out;                                                               \
  }

/* Helper: reduce-all loop for min/max. Seed is NaN; first non-NaN
 * element captures, later non-NaN elements compare via CMP.
 * Mirrors numbl's NaN-skip convention. */
#define MTOC2_DEFINE_MINMAX_REDUCTION(name, CMP)                              \
  static double mtoc2_##name##_all(mtoc2_tensor_t a) {                        \
    long n = 1;                                                               \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                          \
    double acc = NAN;                                                         \
    for (long i = 0; i < n; i++) {                                            \
      double x = a.real[i];                                                   \
      if (x != x) continue; /* skip NaN */                                    \
      if (acc != acc || (x CMP acc)) acc = x;                                 \
    }                                                                         \
    return acc;                                                               \
  }                                                                           \
                                                                              \
  static mtoc2_tensor_t mtoc2_##name##_dim(mtoc2_tensor_t a, int dim) {       \
    if (dim < 1) {                                                            \
      fprintf(stderr, "mtoc2: " #name "_dim: dim must be >= 1 (got %d)\n",    \
              dim);                                                           \
      abort();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      mtoc2_tensor_t out;                                                     \
      out.ndim = a.ndim;                                                      \
      for (int i = 0; i < a.ndim; i++) out.dims[i] = a.dims[i];               \
      out.real = mtoc2_alloc((size_t)total * sizeof(double));                 \
      out.imag = NULL;                                                        \
      memcpy(out.real, a.real, (size_t)total * sizeof(double));               \
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
    mtoc2__squeeze_trailing(&out_ndim, out_dims);                             \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(out_ndim, out_dims);           \
    long slab = before * axis;                                                \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        double acc = NAN;                                                     \
        for (long k = 0; k < axis; k++) {                                     \
          double x = a.real[slabBase + inner + k * before];                   \
          if (x != x) continue;                                               \
          if (acc != acc || (x CMP acc)) acc = x;                             \
        }                                                                     \
        out.real[outer * before + inner] = acc;                               \
      }                                                                       \
    }                                                                         \
    return out;                                                               \
  }

/* Helper: any/all reduction. Short-circuits per fiber.
 * `EMPTY_RESULT` is the value when the reduced fiber is empty:
 *  - any: 0 (no element is nonzero in an empty set)
 *  - all: 1 (vacuously true)
 * `SHORT(acc, x)` updates `acc` if `x` triggers the short-circuit;
 * `done` short-circuits the inner loop once `acc` settles. */
#define MTOC2_DEFINE_LOGICAL_REDUCTION(name, EMPTY_RESULT, SHORT_BODY)        \
  static double mtoc2_##name##_all(mtoc2_tensor_t a) {                        \
    long n = 1;                                                               \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                          \
    if (n == 0) return (double)(EMPTY_RESULT);                                \
    double acc = (double)(EMPTY_RESULT);                                      \
    for (long i = 0; i < n; i++) {                                            \
      double x = a.real[i];                                                   \
      SHORT_BODY;                                                             \
    }                                                                         \
    return acc;                                                               \
  }                                                                           \
                                                                              \
  static mtoc2_tensor_t mtoc2_##name##_dim(mtoc2_tensor_t a, int dim) {       \
    if (dim < 1) {                                                            \
      fprintf(stderr, "mtoc2: " #name "_dim: dim must be >= 1 (got %d)\n",    \
              dim);                                                           \
      abort();                                                                \
    }                                                                         \
    if (dim > a.ndim) {                                                       \
      /* Numbl's `logicalAlongDim` with `dim > ndims` does an elementwise   \
       * cast to logical: each element becomes 1.0 if nonzero else 0.0. */   \
      long total = 1;                                                         \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                    \
      long out_dims[MTOC2_MAX_NDIM];                                          \
      int out_ndim = a.ndim;                                                  \
      for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];               \
      mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(out_ndim, out_dims);         \
      for (long i = 0; i < total; i++) {                                      \
        out.real[i] = (a.real[i] != 0.0) ? 1.0 : 0.0;                         \
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
    mtoc2__squeeze_trailing(&out_ndim, out_dims);                             \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(out_ndim, out_dims);           \
    long slab = before * axis;                                                \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                           \
      for (long inner = 0; inner < before; inner++) {                         \
        double acc = (double)(EMPTY_RESULT);                                  \
        for (long k = 0; k < axis; k++) {                                     \
          double x = a.real[slabBase + inner + k * before];                   \
          SHORT_BODY;                                                         \
        }                                                                     \
        out.real[outer * before + inner] = acc;                               \
      }                                                                       \
    }                                                                         \
    return out;                                                               \
  }

/* Identity finalizer (sum, prod): pass the accumulator through. */
#define MTOC2_FIN_ID(acc, n) (acc)
/* Mean finalizer: divide by element count. Empty fiber → 0/0 = NaN. */
#define MTOC2_FIN_MEAN(acc, n) ((double)(acc) / (double)(n))

/* Accumulator-statement macros. Wrapped in `do {} while(0)` to keep
 * them safe inside any single-statement context the templates use. */
#define MTOC2_ACC_SUM(acc, x) do { (acc) += (x); } while (0)
#define MTOC2_ACC_PROD(acc, x) do { (acc) *= (x); } while (0)

MTOC2_DEFINE_ACCUM_REDUCTION(sum, 0.0, MTOC2_ACC_SUM, MTOC2_FIN_ID)
MTOC2_DEFINE_ACCUM_REDUCTION(prod, 1.0, MTOC2_ACC_PROD, MTOC2_FIN_ID)
MTOC2_DEFINE_ACCUM_REDUCTION(mean, 0.0, MTOC2_ACC_SUM, MTOC2_FIN_MEAN)

MTOC2_DEFINE_MINMAX_REDUCTION(min, <)
MTOC2_DEFINE_MINMAX_REDUCTION(max, >)

/* `any` ignores NaN (MATLAB: any(NaN) is 0); `x == x` excludes it so a
 * NaN doesn't wrongly short-circuit to true. `all` tests `x == 0.0`,
 * which NaN already fails, so it needs no guard. */
MTOC2_DEFINE_LOGICAL_REDUCTION(any, 0,
  if (x != 0.0 && x == x) { acc = 1.0; break; })
MTOC2_DEFINE_LOGICAL_REDUCTION(all, 1,
  if (x == 0.0) { acc = 0.0; break; })
