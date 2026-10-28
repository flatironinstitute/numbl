/* JIT runtime helper: stable sort on a tensor.
 *
 *   jit_sort_real(a, descending)
 *     `b = sort(a)` / `sort(a, 'ascend'|'descend')` — returns a
 *     freshly-owned tensor of the same shape as `a`, with the flat
 *     (column-major) entries sorted in the requested direction.
 *
 *   jit_sort_real_2(a, descending, &out_v, &out_i)
 *     `[v, i] = sort(...)` — fills `*out_v` with the sorted values
 *     and `*out_i` with 1-based original positions.
 *
 *   jit_sort_complex / jit_sort_complex_2
 *     Complex-input siblings. Numbl / MATLAB sort complex by
 *     magnitude (hypot), tiebreak by phase (atan2). Tolerates
 *     `a.imag == NULL` (real-input flowed through a complex route)
 *     by treating imag as zero.
 *
 * Sort is stable in both directions: ties resolve by ascending
 * original index, matching numbl's behaviour (verified against
 * `sort([5 2 8 1 2], 'descend')` → indices `3 1 2 5 4`).
 *
 * The lowering layer restricts the input to a 1×N row vector or N×1
 * column vector for v1; the helper itself walks the column-major
 * flat buffer and would handle any rank, but the type system rejects
 * the higher-rank cases until the per-axis form is plumbed through.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  double v;
  long ix;
} jit_sort_pair_t;

/* NaN ranks as the maximum (MATLAB): last when ascending, first when
 * descending. Without this, NaN compares false both ways and falls to
 * the index tie-break, leaving the comparator non-transitive — which is
 * undefined behavior for qsort and corrupts the array. */
static int jit_sort_cmp_asc(const void *pa, const void *pb) {
  const jit_sort_pair_t *a = (const jit_sort_pair_t *)pa;
  const jit_sort_pair_t *b = (const jit_sort_pair_t *)pb;
  int an = a->v != a->v, bn = b->v != b->v;
  if (an || bn) {
    if (!(an && bn)) return an ? 1 : -1; /* NaN sorts last */
  } else {
    if (a->v < b->v) return -1;
    if (a->v > b->v) return 1;
  }
  if (a->ix < b->ix) return -1;
  if (a->ix > b->ix) return 1;
  return 0;
}

static int jit_sort_cmp_desc(const void *pa, const void *pb) {
  const jit_sort_pair_t *a = (const jit_sort_pair_t *)pa;
  const jit_sort_pair_t *b = (const jit_sort_pair_t *)pb;
  int an = a->v != a->v, bn = b->v != b->v;
  if (an || bn) {
    if (!(an && bn)) return an ? -1 : 1; /* NaN sorts first */
  } else {
    if (a->v > b->v) return -1;
    if (a->v < b->v) return 1;
  }
  /* Tie-break still by ascending original index — both numbl and
   * MATLAB keep ties in original order in either direction. */
  if (a->ix < b->ix) return -1;
  if (a->ix > b->ix) return 1;
  return 0;
}

static jit_tensor_t jit_sort_real(jit_tensor_t a, int descending) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t r;
  r.real = jit_alloc((size_t)n * sizeof(double));
  r.imag = NULL;
  r.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];
  if (n == 0) return r;
  jit_sort_pair_t *buf =
    (jit_sort_pair_t *)malloc((size_t)n * sizeof(jit_sort_pair_t));
  if (!buf) {
    jit_raise_bailf("out of memory (sort buffer)");
  }
  for (long i = 0; i < n; i++) {
    buf[i].v = a.real[i];
    buf[i].ix = i;
  }
  qsort(buf, (size_t)n, sizeof(jit_sort_pair_t),
        descending ? jit_sort_cmp_desc : jit_sort_cmp_asc);
  for (long i = 0; i < n; i++) r.real[i] = buf[i].v;
  free(buf);
  return r;
}

static void jit_sort_real_2(jit_tensor_t a, int descending,
                              jit_tensor_t *out_v, jit_tensor_t *out_i) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t v;
  jit_tensor_t ix;
  v.real = jit_alloc((size_t)n * sizeof(double));
  v.imag = NULL;
  v.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) v.dims[i] = a.dims[i];
  ix.real = jit_alloc((size_t)n * sizeof(double));
  ix.imag = NULL;
  ix.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) ix.dims[i] = a.dims[i];
  if (n > 0) {
    jit_sort_pair_t *buf =
      (jit_sort_pair_t *)malloc((size_t)n * sizeof(jit_sort_pair_t));
    if (!buf) {
      jit_raise_bailf("out of memory (sort buffer)");
    }
    for (long i = 0; i < n; i++) {
      buf[i].v = a.real[i];
      buf[i].ix = i;
    }
    qsort(buf, (size_t)n, sizeof(jit_sort_pair_t),
          descending ? jit_sort_cmp_desc : jit_sort_cmp_asc);
    for (long i = 0; i < n; i++) {
      v.real[i] = buf[i].v;
      ix.real[i] = (double)(buf[i].ix + 1);
    }
    free(buf);
  }
  jit_tensor_assign(out_v, v);
  jit_tensor_assign(out_i, ix);
}

typedef struct {
  double mag;
  double phase;
  long ix;
} jit_sort_complex_pair_t;

static int jit_sort_cmp_complex_asc(const void *pa, const void *pb) {
  const jit_sort_complex_pair_t *a = (const jit_sort_complex_pair_t *)pa;
  const jit_sort_complex_pair_t *b = (const jit_sort_complex_pair_t *)pb;
  if (a->mag < b->mag) return -1;
  if (a->mag > b->mag) return 1;
  if (a->phase < b->phase) return -1;
  if (a->phase > b->phase) return 1;
  if (a->ix < b->ix) return -1;
  if (a->ix > b->ix) return 1;
  return 0;
}

static int jit_sort_cmp_complex_desc(const void *pa, const void *pb) {
  const jit_sort_complex_pair_t *a = (const jit_sort_complex_pair_t *)pa;
  const jit_sort_complex_pair_t *b = (const jit_sort_complex_pair_t *)pb;
  if (a->mag > b->mag) return -1;
  if (a->mag < b->mag) return 1;
  if (a->phase > b->phase) return -1;
  if (a->phase < b->phase) return 1;
  if (a->ix < b->ix) return -1;
  if (a->ix > b->ix) return 1;
  return 0;
}

/* Real-value comparators used when the input's imaginary lane is all
 * zero: order by signed value (the `mag` field holds the signed real
 * part in that mode), NaNs last (asc) / first (desc), tiebreak by index.
 * Matches the interpreter and MATLAB on real data. */
static int jit_sort_cmp_real_asc(const void *pa, const void *pb) {
  const jit_sort_complex_pair_t *a = (const jit_sort_complex_pair_t *)pa;
  const jit_sort_complex_pair_t *b = (const jit_sort_complex_pair_t *)pb;
  int aNaN = a->mag != a->mag;
  int bNaN = b->mag != b->mag;
  if (!(aNaN && bNaN)) {
    if (aNaN) return 1;
    if (bNaN) return -1;
    if (a->mag < b->mag) return -1;
    if (a->mag > b->mag) return 1;
  }
  if (a->ix < b->ix) return -1;
  if (a->ix > b->ix) return 1;
  return 0;
}

static int jit_sort_cmp_real_desc(const void *pa, const void *pb) {
  const jit_sort_complex_pair_t *a = (const jit_sort_complex_pair_t *)pa;
  const jit_sort_complex_pair_t *b = (const jit_sort_complex_pair_t *)pb;
  int aNaN = a->mag != a->mag;
  int bNaN = b->mag != b->mag;
  if (!(aNaN && bNaN)) {
    if (aNaN) return -1;
    if (bNaN) return 1;
    if (a->mag > b->mag) return -1;
    if (a->mag < b->mag) return 1;
  }
  if (a->ix < b->ix) return -1;
  if (a->ix > b->ix) return 1;
  return 0;
}

/* True when the tensor carries no imaginary content (NULL lane or all
 * elements zero) — then sort orders by signed real value. */
static int jit_sort_all_imag_zero(jit_tensor_t a) {
  if (a.imag == NULL) return 1;
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  for (long i = 0; i < n; i++) {
    if (a.imag[i] != 0.0) return 0;
  }
  return 1;
}

static jit_tensor_t jit_sort_complex(jit_tensor_t a, int descending) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t r = jit_tensor_alloc_nd_complex(a.ndim, a.dims);
  if (n == 0) return r;
  int srcHasImag = (a.imag != NULL);
  if (!srcHasImag) memset(r.imag, 0, (size_t)n * sizeof(double));
  int realMode = jit_sort_all_imag_zero(a);
  jit_sort_complex_pair_t *buf =
    (jit_sort_complex_pair_t *)malloc(
      (size_t)n * sizeof(jit_sort_complex_pair_t));
  if (!buf) {
    jit_raise_bailf("out of memory (sort complex buffer)");
  }
  for (long i = 0; i < n; i++) {
    double re = a.real[i];
    double im = srcHasImag ? a.imag[i] : 0.0;
    buf[i].mag = realMode ? re : hypot(re, im);
    buf[i].phase = realMode ? 0.0 : atan2(im, re);
    buf[i].ix = i;
  }
  qsort(buf, (size_t)n, sizeof(jit_sort_complex_pair_t),
        realMode ? (descending ? jit_sort_cmp_real_desc
                               : jit_sort_cmp_real_asc)
                 : (descending ? jit_sort_cmp_complex_desc
                               : jit_sort_cmp_complex_asc));
  for (long i = 0; i < n; i++) {
    r.real[i] = a.real[buf[i].ix];
    r.imag[i] = srcHasImag ? a.imag[buf[i].ix] : 0.0;
  }
  free(buf);
  return r;
}

static void jit_sort_complex_2(jit_tensor_t a, int descending,
                                 jit_tensor_t *out_v, jit_tensor_t *out_i) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t v = jit_tensor_alloc_nd_complex(a.ndim, a.dims);
  jit_tensor_t ix;
  ix.real = jit_alloc((size_t)n * sizeof(double));
  ix.imag = NULL;
  ix.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) ix.dims[i] = a.dims[i];
  if (n > 0) {
    int srcHasImag = (a.imag != NULL);
    if (!srcHasImag) memset(v.imag, 0, (size_t)n * sizeof(double));
    int realMode = jit_sort_all_imag_zero(a);
    jit_sort_complex_pair_t *buf =
      (jit_sort_complex_pair_t *)malloc(
        (size_t)n * sizeof(jit_sort_complex_pair_t));
    if (!buf) {
      jit_raise_bailf("out of memory (sort complex buffer)");
    }
    for (long i = 0; i < n; i++) {
      double re = a.real[i];
      double im = srcHasImag ? a.imag[i] : 0.0;
      buf[i].mag = realMode ? re : hypot(re, im);
      buf[i].phase = realMode ? 0.0 : atan2(im, re);
      buf[i].ix = i;
    }
    qsort(buf, (size_t)n, sizeof(jit_sort_complex_pair_t),
          realMode ? (descending ? jit_sort_cmp_real_desc
                                 : jit_sort_cmp_real_asc)
                   : (descending ? jit_sort_cmp_complex_desc
                                 : jit_sort_cmp_complex_asc));
    for (long i = 0; i < n; i++) {
      v.real[i] = a.real[buf[i].ix];
      v.imag[i] = srcHasImag ? a.imag[buf[i].ix] : 0.0;
      ix.real[i] = (double)(buf[i].ix + 1);
    }
    free(buf);
  }
  jit_tensor_assign(out_v, v);
  jit_tensor_assign(out_i, ix);
}
