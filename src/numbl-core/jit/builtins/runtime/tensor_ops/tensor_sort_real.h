/* mtoc2 runtime helper: stable sort on a tensor.
 *
 *   mtoc2_sort_real(a, descending)
 *     `b = sort(a)` / `sort(a, 'ascend'|'descend')` — returns a
 *     freshly-owned tensor of the same shape as `a`, with the flat
 *     (column-major) entries sorted in the requested direction.
 *
 *   mtoc2_sort_real_2(a, descending, &out_v, &out_i)
 *     `[v, i] = sort(...)` — fills `*out_v` with the sorted values
 *     and `*out_i` with 1-based original positions.
 *
 *   mtoc2_sort_complex / mtoc2_sort_complex_2
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
} mtoc2_sort_pair_t;

/* NaN ranks as the maximum (MATLAB): last when ascending, first when
 * descending. Without this, NaN compares false both ways and falls to
 * the index tie-break, leaving the comparator non-transitive — which is
 * undefined behavior for qsort and corrupts the array. */
static int mtoc2_sort_cmp_asc(const void *pa, const void *pb) {
  const mtoc2_sort_pair_t *a = (const mtoc2_sort_pair_t *)pa;
  const mtoc2_sort_pair_t *b = (const mtoc2_sort_pair_t *)pb;
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

static int mtoc2_sort_cmp_desc(const void *pa, const void *pb) {
  const mtoc2_sort_pair_t *a = (const mtoc2_sort_pair_t *)pa;
  const mtoc2_sort_pair_t *b = (const mtoc2_sort_pair_t *)pb;
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

static mtoc2_tensor_t mtoc2_sort_real(mtoc2_tensor_t a, int descending) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)n * sizeof(double));
  r.imag = NULL;
  r.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];
  if (n == 0) return r;
  mtoc2_sort_pair_t *buf =
    (mtoc2_sort_pair_t *)malloc((size_t)n * sizeof(mtoc2_sort_pair_t));
  if (!buf) {
    fprintf(stderr, "mtoc2: out of memory (sort buffer)\n");
    abort();
  }
  for (long i = 0; i < n; i++) {
    buf[i].v = a.real[i];
    buf[i].ix = i;
  }
  qsort(buf, (size_t)n, sizeof(mtoc2_sort_pair_t),
        descending ? mtoc2_sort_cmp_desc : mtoc2_sort_cmp_asc);
  for (long i = 0; i < n; i++) r.real[i] = buf[i].v;
  free(buf);
  return r;
}

static void mtoc2_sort_real_2(mtoc2_tensor_t a, int descending,
                              mtoc2_tensor_t *out_v, mtoc2_tensor_t *out_i) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t v;
  mtoc2_tensor_t ix;
  v.real = mtoc2_alloc((size_t)n * sizeof(double));
  v.imag = NULL;
  v.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) v.dims[i] = a.dims[i];
  ix.real = mtoc2_alloc((size_t)n * sizeof(double));
  ix.imag = NULL;
  ix.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) ix.dims[i] = a.dims[i];
  if (n > 0) {
    mtoc2_sort_pair_t *buf =
      (mtoc2_sort_pair_t *)malloc((size_t)n * sizeof(mtoc2_sort_pair_t));
    if (!buf) {
      fprintf(stderr, "mtoc2: out of memory (sort buffer)\n");
      abort();
    }
    for (long i = 0; i < n; i++) {
      buf[i].v = a.real[i];
      buf[i].ix = i;
    }
    qsort(buf, (size_t)n, sizeof(mtoc2_sort_pair_t),
          descending ? mtoc2_sort_cmp_desc : mtoc2_sort_cmp_asc);
    for (long i = 0; i < n; i++) {
      v.real[i] = buf[i].v;
      ix.real[i] = (double)(buf[i].ix + 1);
    }
    free(buf);
  }
  mtoc2_tensor_assign(out_v, v);
  mtoc2_tensor_assign(out_i, ix);
}

typedef struct {
  double mag;
  double phase;
  long ix;
} mtoc2_sort_complex_pair_t;

static int mtoc2_sort_cmp_complex_asc(const void *pa, const void *pb) {
  const mtoc2_sort_complex_pair_t *a = (const mtoc2_sort_complex_pair_t *)pa;
  const mtoc2_sort_complex_pair_t *b = (const mtoc2_sort_complex_pair_t *)pb;
  if (a->mag < b->mag) return -1;
  if (a->mag > b->mag) return 1;
  if (a->phase < b->phase) return -1;
  if (a->phase > b->phase) return 1;
  if (a->ix < b->ix) return -1;
  if (a->ix > b->ix) return 1;
  return 0;
}

static int mtoc2_sort_cmp_complex_desc(const void *pa, const void *pb) {
  const mtoc2_sort_complex_pair_t *a = (const mtoc2_sort_complex_pair_t *)pa;
  const mtoc2_sort_complex_pair_t *b = (const mtoc2_sort_complex_pair_t *)pb;
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
static int mtoc2_sort_cmp_real_asc(const void *pa, const void *pb) {
  const mtoc2_sort_complex_pair_t *a = (const mtoc2_sort_complex_pair_t *)pa;
  const mtoc2_sort_complex_pair_t *b = (const mtoc2_sort_complex_pair_t *)pb;
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

static int mtoc2_sort_cmp_real_desc(const void *pa, const void *pb) {
  const mtoc2_sort_complex_pair_t *a = (const mtoc2_sort_complex_pair_t *)pa;
  const mtoc2_sort_complex_pair_t *b = (const mtoc2_sort_complex_pair_t *)pb;
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
static int mtoc2_sort_all_imag_zero(mtoc2_tensor_t a) {
  if (a.imag == NULL) return 1;
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  for (long i = 0; i < n; i++) {
    if (a.imag[i] != 0.0) return 0;
  }
  return 1;
}

static mtoc2_tensor_t mtoc2_sort_complex(mtoc2_tensor_t a, int descending) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);
  if (n == 0) return r;
  int srcHasImag = (a.imag != NULL);
  if (!srcHasImag) memset(r.imag, 0, (size_t)n * sizeof(double));
  int realMode = mtoc2_sort_all_imag_zero(a);
  mtoc2_sort_complex_pair_t *buf =
    (mtoc2_sort_complex_pair_t *)malloc(
      (size_t)n * sizeof(mtoc2_sort_complex_pair_t));
  if (!buf) {
    fprintf(stderr, "mtoc2: out of memory (sort complex buffer)\n");
    abort();
  }
  for (long i = 0; i < n; i++) {
    double re = a.real[i];
    double im = srcHasImag ? a.imag[i] : 0.0;
    buf[i].mag = realMode ? re : hypot(re, im);
    buf[i].phase = realMode ? 0.0 : atan2(im, re);
    buf[i].ix = i;
  }
  qsort(buf, (size_t)n, sizeof(mtoc2_sort_complex_pair_t),
        realMode ? (descending ? mtoc2_sort_cmp_real_desc
                               : mtoc2_sort_cmp_real_asc)
                 : (descending ? mtoc2_sort_cmp_complex_desc
                               : mtoc2_sort_cmp_complex_asc));
  for (long i = 0; i < n; i++) {
    r.real[i] = a.real[buf[i].ix];
    r.imag[i] = srcHasImag ? a.imag[buf[i].ix] : 0.0;
  }
  free(buf);
  return r;
}

static void mtoc2_sort_complex_2(mtoc2_tensor_t a, int descending,
                                 mtoc2_tensor_t *out_v, mtoc2_tensor_t *out_i) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t v = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);
  mtoc2_tensor_t ix;
  ix.real = mtoc2_alloc((size_t)n * sizeof(double));
  ix.imag = NULL;
  ix.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) ix.dims[i] = a.dims[i];
  if (n > 0) {
    int srcHasImag = (a.imag != NULL);
    if (!srcHasImag) memset(v.imag, 0, (size_t)n * sizeof(double));
    int realMode = mtoc2_sort_all_imag_zero(a);
    mtoc2_sort_complex_pair_t *buf =
      (mtoc2_sort_complex_pair_t *)malloc(
        (size_t)n * sizeof(mtoc2_sort_complex_pair_t));
    if (!buf) {
      fprintf(stderr, "mtoc2: out of memory (sort complex buffer)\n");
      abort();
    }
    for (long i = 0; i < n; i++) {
      double re = a.real[i];
      double im = srcHasImag ? a.imag[i] : 0.0;
      buf[i].mag = realMode ? re : hypot(re, im);
      buf[i].phase = realMode ? 0.0 : atan2(im, re);
      buf[i].ix = i;
    }
    qsort(buf, (size_t)n, sizeof(mtoc2_sort_complex_pair_t),
          realMode ? (descending ? mtoc2_sort_cmp_real_desc
                                 : mtoc2_sort_cmp_real_asc)
                   : (descending ? mtoc2_sort_cmp_complex_desc
                                 : mtoc2_sort_cmp_complex_asc));
    for (long i = 0; i < n; i++) {
      v.real[i] = a.real[buf[i].ix];
      v.imag[i] = srcHasImag ? a.imag[buf[i].ix] : 0.0;
      ix.real[i] = (double)(buf[i].ix + 1);
    }
    free(buf);
  }
  mtoc2_tensor_assign(out_v, v);
  mtoc2_tensor_assign(out_i, ix);
}
