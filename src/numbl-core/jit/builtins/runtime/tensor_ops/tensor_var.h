/* mtoc2 runtime helpers: variance / standard deviation reductions.
 *
 * Two-pass per fiber (mean, then sum of squared deviations), matching
 * numbl's `varianceOf` (interpreter/builtins/reductions.ts):
 *
 *   n == 0            -> NaN
 *   n <= 1 && w == 0  -> 0           (sample variance of one point)
 *   else              -> SS / (w == 1 ? n : n - 1)
 *
 * `w` is the normalization flag: 0 (default) divides by n-1, 1 divides
 * by n. `std` applies sqrt to the variance.
 *
 * `_all(a, w)` reduces every element to a scalar. `_dim(a, dim, w)`
 * reduces along the 1-based axis `dim`, returning a freshly-owned
 * tensor — same `before / axis / after` column-major fiber walk as
 * `tensor_reduce_real.h`. A `dim > ndim` axis makes every fiber a
 * single element, whose variance is 0, so the result is zeros shaped
 * like the input.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Strip trailing singleton axes down to a 2-axis floor. Unique name to
 * avoid colliding with tensor_reduce_real.h's identical helper when both
 * snippets are inlined into one translation unit. */
static void mtoc2__var_squeeze_trailing(int *ndim, long *dims) {
  while (*ndim > 2 && dims[*ndim - 1] == 1) {
    (*ndim)--;
  }
}

/* Variance of the length-`n` fiber starting at `base` with `stride`
 * between consecutive elements. */
static double mtoc2__variance_fiber(const double *data, long base, long n,
                                    long stride, int w) {
  if (n == 0) return NAN;
  if (n <= 1 && w == 0) return 0.0;
  double s = 0.0;
  for (long k = 0; k < n; k++) s += data[base + k * stride];
  double m = s / (double)n;
  double ss = 0.0;
  for (long k = 0; k < n; k++) {
    double d = data[base + k * stride] - m;
    ss += d * d;
  }
  return ss / (double)(w == 1 ? n : n - 1);
}

#define MTOC2_DEFINE_VARIANCE_REDUCTION(name, TRANSFORM)                       \
  static double mtoc2_##name##_all(mtoc2_tensor_t a, int w) {                  \
    long n = 1;                                                                \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                           \
    return TRANSFORM(mtoc2__variance_fiber(a.real, 0, n, 1, w));               \
  }                                                                            \
                                                                              \
  static mtoc2_tensor_t mtoc2_##name##_dim(mtoc2_tensor_t a, int dim,          \
                                           int w) {                            \
    if (dim < 1) {                                                             \
      fprintf(stderr, "mtoc2: " #name "_dim: dim must be >= 1 (got %d)\n",     \
              dim);                                                            \
      abort();                                                                 \
    }                                                                          \
    if (dim > a.ndim) {                                                        \
      /* No-op axis: numbl's reduceDim copies the input verbatim when     \
       * dim > ndim (forEachSlice returns null), regardless of reducer,   \
       * so var/std return the input unchanged here — not zeros. */       \
      long total = 1;                                                          \
      for (int i = 0; i < a.ndim; i++) total *= a.dims[i];                     \
      mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(a.ndim, a.dims);              \
      memcpy(out.real, a.real, (size_t)total * sizeof(double));               \
      return out;                                                              \
    }                                                                          \
    int dimIdx = dim - 1;                                                      \
    long axis = a.dims[dimIdx];                                                \
    long before = 1;                                                          \
    for (int i = 0; i < dimIdx; i++) before *= a.dims[i];                      \
    long after = 1;                                                            \
    for (int i = dimIdx + 1; i < a.ndim; i++) after *= a.dims[i];             \
    long out_dims[MTOC2_MAX_NDIM];                                             \
    int out_ndim = a.ndim;                                                     \
    for (int i = 0; i < a.ndim; i++) out_dims[i] = a.dims[i];                  \
    out_dims[dimIdx] = 1;                                                      \
    mtoc2__var_squeeze_trailing(&out_ndim, out_dims);                          \
    mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(out_ndim, out_dims);            \
    long slab = before * axis;                                                 \
    for (long outer = 0; outer < after; outer++) {                            \
      long slabBase = outer * slab;                                            \
      for (long inner = 0; inner < before; inner++) {                          \
        double v =                                                             \
            mtoc2__variance_fiber(a.real, slabBase + inner, axis, before, w);  \
        out.real[outer * before + inner] = TRANSFORM(v);                       \
      }                                                                        \
    }                                                                          \
    return out;                                                                \
  }

#define MTOC2_VAR_IDENTITY(x) (x)
MTOC2_DEFINE_VARIANCE_REDUCTION(var, MTOC2_VAR_IDENTITY)
MTOC2_DEFINE_VARIANCE_REDUCTION(std, sqrt)
