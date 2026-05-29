/* mtoc2 runtime helper: `cat(dim, A, B, ...)` — concatenate N tensors
 * along axis `dim` (1-based, like MATLAB / numbl). dim==1 means
 * vertical (rows), dim==2 means horizontal (cols), dim>=3 grows a new
 * outer axis.
 *
 * Mirrors numbl's `catAlongDim` in
 * `numbl/runtime/tensor-construction.ts`. Empty inputs are dropped
 * before the shape check, including the asymmetric MATLAB rule: a
 * zero-element input keeps its slot only if its non-cat dims match the
 * first non-empty input's shape; otherwise it's silently dropped.
 *
 * Args travel through a small tagged-arg struct so a single helper can
 * accept a mix of scalars and tensors. Codegen builds the array of
 * `mtoc2_cat_arg_t` values as a C99 compound literal at the call site.
 *
 * Result is freshly owned. The `*_complex` sibling carries both
 * lanes (scalars split into `(re, im)`; tensors with `imag == NULL`
 * are treated as zero on the imag side).
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  /* 0 = tensor, 1 = scalar. */
  int kind;
  /* Valid when kind == 1. */
  double scalar;
  /* Valid when kind == 0. */
  mtoc2_tensor_t tensor;
} mtoc2_cat_arg_t;

static mtoc2_tensor_t mtoc2_tensor_cat(long dim, int nin,
                                       const mtoc2_cat_arg_t *xs) {
  if (dim < 1) {
    fprintf(stderr, "mtoc2: cat dim %ld must be >= 1\n", dim);
    abort();
  }
  if (nin < 0 || nin > 256) {
    fprintf(stderr, "mtoc2: cat nin %d out of range\n", nin);
    abort();
  }
  long dimIdx = dim - 1;

  /* Step 1 — normalize each input to an (ndim, dims, real) tuple. A
   * scalar maps to shape {1, 1}. We also right-pad shapes to a common
   * ndim = max(2, dim, max(input ndims)). */
  int maxIn = 2;
  if ((long)maxIn < dim) maxIn = (int)dim;
  for (int i = 0; i < nin; i++) {
    if (xs[i].kind == 0) {
      if (xs[i].tensor.ndim > maxIn) maxIn = xs[i].tensor.ndim;
    }
  }
  int ndim = maxIn;
  if (ndim > MTOC2_MAX_NDIM) {
    fprintf(stderr,
      "mtoc2: cat output ndim %d exceeds %d\n", ndim, MTOC2_MAX_NDIM);
    abort();
  }

  long padDims[256][MTOC2_MAX_NDIM];
  long padTotal[256];
  const double *padReal[256];
  int keep[256];
  for (int i = 0; i < nin; i++) {
    if (xs[i].kind == 1) {
      for (int d = 0; d < ndim; d++) padDims[i][d] = 1;
      padReal[i] = &xs[i].scalar;
      padTotal[i] = 1;
    } else {
      const mtoc2_tensor_t *t = &xs[i].tensor;
      long tot = 1;
      for (int d = 0; d < ndim; d++) {
        long s = d < t->ndim ? t->dims[d] : 1;
        padDims[i][d] = s;
        tot *= s;
      }
      padReal[i] = t->real;
      padTotal[i] = tot;
    }
    /* Default-keep tensors whose flat element count is > 0. Zero-element
     * inputs are filtered below (their non-cat dims may or may not be
     * compatible with the reference). */
    keep[i] = padTotal[i] > 0 ? 1 : -1;
  }

  /* Step 2 — find first non-empty input to use as the reference shape. */
  int refIdx = -1;
  for (int i = 0; i < nin; i++) {
    if (keep[i] == 1) { refIdx = i; break; }
  }

  if (refIdx == -1) {
    /* All inputs are empty (or there are no inputs at all). Return the
     * canonical empty `[0, 0]` tensor. */
    long zeros[2] = {0, 0};
    return mtoc2_tensor_alloc_nd(2, zeros);
  }

  /* Step 3 — decide which empty inputs to keep (those whose non-cat
   * dims match the ref). Validate non-cat dims of non-empty inputs. */
  for (int i = 0; i < nin; i++) {
    if (keep[i] == 1) {
      if (i == refIdx) continue;
      for (int d = 0; d < ndim; d++) {
        if (d == dimIdx) continue;
        if (padDims[i][d] != padDims[refIdx][d]) {
          fprintf(stderr,
            "mtoc2: cat dimension mismatch on dimension %d\n", d + 1);
          abort();
        }
      }
    } else {
      /* Empty input: keep iff non-cat dims match reference. */
      int compat = 1;
      for (int d = 0; d < ndim; d++) {
        if (d == dimIdx) continue;
        if (padDims[i][d] != padDims[refIdx][d]) { compat = 0; break; }
      }
      keep[i] = compat ? 1 : 0;
    }
  }

  /* Step 4 — compute result shape. */
  long resultDims[MTOC2_MAX_NDIM];
  for (int d = 0; d < ndim; d++) resultDims[d] = padDims[refIdx][d];
  long catSum = 0;
  for (int i = 0; i < nin; i++) {
    if (keep[i] == 1) catSum += padDims[i][dimIdx];
  }
  resultDims[dimIdx] = catSum;

  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(ndim, resultDims);
  long resultTotal = 1;
  for (int d = 0; d < ndim; d++) resultTotal *= resultDims[d];
  if (resultTotal == 0) return out;

  /* Step 5 — column-major slab copies. stride = product of dims below
   * dimIdx; numOuter = product of dims above. For each outer slab,
   * copy each input's contiguous block-along-dim into the result. */
  long strideDim = 1;
  for (long d = 0; d < dimIdx; d++) strideDim *= resultDims[d];
  long numOuter = 1;
  for (long d = dimIdx + 1; d < ndim; d++) numOuter *= resultDims[d];

  for (long outer = 0; outer < numOuter; outer++) {
    long dstOff = outer * strideDim * resultDims[dimIdx];
    for (int i = 0; i < nin; i++) {
      if (!keep[i]) continue;
      long srcDimSize = padDims[i][dimIdx];
      long blockSize = strideDim * srcDimSize;
      long srcOff = outer * blockSize;
      if (blockSize > 0) {
        memcpy(out.real + dstOff, padReal[i] + srcOff,
               (size_t)blockSize * sizeof(double));
      }
      dstOff += blockSize;
    }
  }
  return out;
}

typedef struct {
  /* 0 = tensor, 1 = scalar. */
  int kind;
  /* Valid when kind == 1: a single complex scalar split into two
   * doubles to match the rest of mtoc2's complex-arg encoding (real
   * args at the source level get `im = 0` at the codegen-emit site). */
  double scalar_re;
  double scalar_im;
  /* Valid when kind == 0. */
  mtoc2_tensor_t tensor;
} mtoc2_cat_complex_arg_t;

static mtoc2_tensor_t mtoc2_tensor_cat_complex(
    long dim, int nin, const mtoc2_cat_complex_arg_t *xs) {
  if (dim < 1) {
    fprintf(stderr, "mtoc2: cat_complex dim %ld must be >= 1\n", dim);
    abort();
  }
  if (nin < 0 || nin > 256) {
    fprintf(stderr, "mtoc2: cat_complex nin %d out of range\n", nin);
    abort();
  }
  long dimIdx = dim - 1;

  int maxIn = 2;
  if ((long)maxIn < dim) maxIn = (int)dim;
  for (int i = 0; i < nin; i++) {
    if (xs[i].kind == 0) {
      if (xs[i].tensor.ndim > maxIn) maxIn = xs[i].tensor.ndim;
    }
  }
  int ndim = maxIn;
  if (ndim > MTOC2_MAX_NDIM) {
    fprintf(stderr,
      "mtoc2: cat_complex output ndim %d exceeds %d\n", ndim, MTOC2_MAX_NDIM);
    abort();
  }

  long padDims[256][MTOC2_MAX_NDIM];
  long padTotal[256];
  const double *padReal[256];
  const double *padImag[256];
  int keep[256];
  for (int i = 0; i < nin; i++) {
    if (xs[i].kind == 1) {
      for (int d = 0; d < ndim; d++) padDims[i][d] = 1;
      padReal[i] = &xs[i].scalar_re;
      padImag[i] = &xs[i].scalar_im;
      padTotal[i] = 1;
    } else {
      const mtoc2_tensor_t *t = &xs[i].tensor;
      long tot = 1;
      for (int d = 0; d < ndim; d++) {
        long s = d < t->ndim ? t->dims[d] : 1;
        padDims[i][d] = s;
        tot *= s;
      }
      padReal[i] = t->real;
      padImag[i] = t->imag; /* may be NULL for a real tensor */
      padTotal[i] = tot;
    }
    keep[i] = padTotal[i] > 0 ? 1 : -1;
  }

  int refIdx = -1;
  for (int i = 0; i < nin; i++) {
    if (keep[i] == 1) { refIdx = i; break; }
  }

  if (refIdx == -1) {
    long zeros[2] = {0, 0};
    return mtoc2_tensor_alloc_nd_complex(2, zeros);
  }

  for (int i = 0; i < nin; i++) {
    if (keep[i] == 1) {
      if (i == refIdx) continue;
      for (int d = 0; d < ndim; d++) {
        if (d == dimIdx) continue;
        if (padDims[i][d] != padDims[refIdx][d]) {
          fprintf(stderr,
            "mtoc2: cat_complex dimension mismatch on dimension %d\n", d + 1);
          abort();
        }
      }
    } else {
      int compat = 1;
      for (int d = 0; d < ndim; d++) {
        if (d == dimIdx) continue;
        if (padDims[i][d] != padDims[refIdx][d]) { compat = 0; break; }
      }
      keep[i] = compat ? 1 : 0;
    }
  }

  long resultDims[MTOC2_MAX_NDIM];
  for (int d = 0; d < ndim; d++) resultDims[d] = padDims[refIdx][d];
  long catSum = 0;
  for (int i = 0; i < nin; i++) {
    if (keep[i] == 1) catSum += padDims[i][dimIdx];
  }
  resultDims[dimIdx] = catSum;

  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(ndim, resultDims);
  long resultTotal = 1;
  for (int d = 0; d < ndim; d++) resultTotal *= resultDims[d];
  if (resultTotal == 0) return out;

  long strideDim = 1;
  for (long d = 0; d < dimIdx; d++) strideDim *= resultDims[d];
  long numOuter = 1;
  for (long d = dimIdx + 1; d < ndim; d++) numOuter *= resultDims[d];

  for (long outer = 0; outer < numOuter; outer++) {
    long dstOff = outer * strideDim * resultDims[dimIdx];
    for (int i = 0; i < nin; i++) {
      if (!keep[i]) continue;
      long srcDimSize = padDims[i][dimIdx];
      long blockSize = strideDim * srcDimSize;
      long srcOff = outer * blockSize;
      if (blockSize > 0) {
        memcpy(out.real + dstOff, padReal[i] + srcOff,
               (size_t)blockSize * sizeof(double));
        if (padImag[i] != NULL) {
          memcpy(out.imag + dstOff, padImag[i] + srcOff,
                 (size_t)blockSize * sizeof(double));
        } else {
          memset(out.imag + dstOff, 0, (size_t)blockSize * sizeof(double));
        }
      }
      dstOff += blockSize;
    }
  }
  return out;
}
