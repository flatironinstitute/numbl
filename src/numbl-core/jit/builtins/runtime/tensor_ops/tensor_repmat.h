/* mtoc2 runtime helper: `repmat(A, reps)` — tile a tensor by
 * replicating it along each axis.
 *
 * Numbl's reference is the `repmat` builtin in
 * `interpreter/builtins/array-manipulation.ts` (the tensor branch).
 *
 * Contract:
 *   - `in` is the source tensor (real, owned-value invariant unchanged).
 *   - `nreps` is the number of replication factors supplied (1..MTOC2_MAX_NDIM).
 *   - `reps_in[i]` is the per-axis replication factor; negative values
 *     clamp to 0 (yielding an empty axis), matching numbl/MATLAB.
 *
 * Output shape is `padShape[i] * padReps[i]` where the input's shape
 * and the reps vector are both right-padded with 1s to a common rank
 * `max(in.ndim, nreps)`. Result is freshly owned; `imag` is NULL.
 *
 * Algorithm: copy the input data into the start of the output buffer
 * (column-major flat layout is preserved when trailing dims are 1),
 * then iteratively expand along each axis. For axis `d` with rep > 1,
 * we walk the existing blocks of size `blockSize = prod(curShape[0..d])`
 * in reverse order and replicate each block `rep` times consecutively.
 * Reverse order avoids overwriting source data; `memmove` covers the
 * b=0 in-place case where the block stays at its original offset.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static mtoc2_tensor_t mtoc2_tensor_repmat(mtoc2_tensor_t in, int nreps,
                                          const long *reps_in) {
  if (nreps < 1 || nreps > MTOC2_MAX_NDIM) {
    fprintf(stderr,
      "mtoc2: repmat nreps %d out of range [1, %d]\n", nreps, MTOC2_MAX_NDIM);
    abort();
  }
  long reps[MTOC2_MAX_NDIM];
  for (int i = 0; i < nreps; i++) reps[i] = reps_in[i] < 0 ? 0 : reps_in[i];

  int in_ndim = in.ndim;
  int out_ndim = nreps > in_ndim ? nreps : in_ndim;
  if (out_ndim > MTOC2_MAX_NDIM) {
    fprintf(stderr,
      "mtoc2: repmat output ndim %d exceeds %d\n", out_ndim, MTOC2_MAX_NDIM);
    abort();
  }

  long padShape[MTOC2_MAX_NDIM];
  long padReps[MTOC2_MAX_NDIM];
  long outDims[MTOC2_MAX_NDIM];
  for (int i = 0; i < out_ndim; i++) {
    padShape[i] = i < in_ndim ? in.dims[i] : 1;
    padReps[i] = i < nreps ? reps[i] : 1;
    outDims[i] = padShape[i] * padReps[i];
  }

  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(out_ndim, outDims);

  size_t outTotal = 1;
  for (int i = 0; i < out_ndim; i++) outTotal *= (size_t)outDims[i];
  if (outTotal == 0) return out;

  size_t inTotal = 1;
  for (int i = 0; i < in_ndim; i++) inTotal *= (size_t)in.dims[i];
  if (inTotal == 0) return out;

  /* Initial copy: input's data laid out in column-major with shape
   * `in.dims` matches the same flat layout under `padShape` (trailing
   * 1s don't change flat indexing). */
  memcpy(out.real, in.real, inTotal * sizeof(double));

  long curShape[MTOC2_MAX_NDIM];
  for (int i = 0; i < out_ndim; i++) curShape[i] = padShape[i];
  size_t curTotal = inTotal;

  for (int d = 0; d < out_ndim; d++) {
    long rep = padReps[d];
    if (rep == 1) continue;

    size_t blockSize = 1;
    for (int i = 0; i <= d; i++) blockSize *= (size_t)curShape[i];

    if (rep == 0 || blockSize == 0) {
      curShape[d] *= rep;
      curTotal = 0;
      /* Once curTotal is 0, no further work needed — outTotal is also
       * 0 (because outDims[d] = padShape[d] * 0 = 0). The alloc above
       * already produced a zero-element tensor; bail out. */
      return out;
    }

    size_t numBlocks = curTotal / blockSize;
    /* Walk blocks in reverse so writes don't clobber as-yet-unread
     * source blocks. Each block of `blockSize` doubles to `blockSize
     * * rep` consecutive slots at offset `b * blockSize * rep`. */
    for (size_t b = numBlocks; b > 0;) {
      b--;
      size_t srcOff = b * blockSize;
      size_t dstBase = b * blockSize * (size_t)rep;
      if (dstBase != srcOff) {
        memmove(out.real + dstBase, out.real + srcOff,
                blockSize * sizeof(double));
      }
      for (long r = 1; r < rep; r++) {
        memcpy(out.real + dstBase + (size_t)r * blockSize,
               out.real + dstBase,
               blockSize * sizeof(double));
      }
    }

    curShape[d] *= rep;
    curTotal *= (size_t)rep;
  }

  return out;
}

/* Complex-input sibling: tiles both lanes. Tolerates `in.imag == NULL`
 * (a real tensor that flowed in via a complex-typed route) by zero-
 * filling the output imag lane. */
static mtoc2_tensor_t mtoc2_tensor_repmat_complex(mtoc2_tensor_t in,
                                                  int nreps,
                                                  const long *reps_in) {
  if (nreps < 1 || nreps > MTOC2_MAX_NDIM) {
    fprintf(stderr,
      "mtoc2: repmat_complex nreps %d out of range [1, %d]\n",
      nreps, MTOC2_MAX_NDIM);
    abort();
  }
  long reps[MTOC2_MAX_NDIM];
  for (int i = 0; i < nreps; i++) reps[i] = reps_in[i] < 0 ? 0 : reps_in[i];

  int in_ndim = in.ndim;
  int out_ndim = nreps > in_ndim ? nreps : in_ndim;
  if (out_ndim > MTOC2_MAX_NDIM) {
    fprintf(stderr,
      "mtoc2: repmat_complex output ndim %d exceeds %d\n",
      out_ndim, MTOC2_MAX_NDIM);
    abort();
  }

  long padShape[MTOC2_MAX_NDIM];
  long padReps[MTOC2_MAX_NDIM];
  long outDims[MTOC2_MAX_NDIM];
  for (int i = 0; i < out_ndim; i++) {
    padShape[i] = i < in_ndim ? in.dims[i] : 1;
    padReps[i] = i < nreps ? reps[i] : 1;
    outDims[i] = padShape[i] * padReps[i];
  }

  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(out_ndim, outDims);

  size_t outTotal = 1;
  for (int i = 0; i < out_ndim; i++) outTotal *= (size_t)outDims[i];
  if (outTotal == 0) return out;

  size_t inTotal = 1;
  for (int i = 0; i < in_ndim; i++) inTotal *= (size_t)in.dims[i];
  if (inTotal == 0) {
    memset(out.imag, 0, outTotal * sizeof(double));
    return out;
  }

  int srcHasImag = (in.imag != NULL);

  memcpy(out.real, in.real, inTotal * sizeof(double));
  if (srcHasImag) {
    memcpy(out.imag, in.imag, inTotal * sizeof(double));
  } else {
    memset(out.imag, 0, inTotal * sizeof(double));
  }

  long curShape[MTOC2_MAX_NDIM];
  for (int i = 0; i < out_ndim; i++) curShape[i] = padShape[i];
  size_t curTotal = inTotal;

  for (int d = 0; d < out_ndim; d++) {
    long rep = padReps[d];
    if (rep == 1) continue;

    size_t blockSize = 1;
    for (int i = 0; i <= d; i++) blockSize *= (size_t)curShape[i];

    if (rep == 0 || blockSize == 0) {
      curShape[d] *= rep;
      curTotal = 0;
      return out;
    }

    size_t numBlocks = curTotal / blockSize;
    for (size_t b = numBlocks; b > 0;) {
      b--;
      size_t srcOff = b * blockSize;
      size_t dstBase = b * blockSize * (size_t)rep;
      if (dstBase != srcOff) {
        memmove(out.real + dstBase, out.real + srcOff,
                blockSize * sizeof(double));
        memmove(out.imag + dstBase, out.imag + srcOff,
                blockSize * sizeof(double));
      }
      for (long r = 1; r < rep; r++) {
        memcpy(out.real + dstBase + (size_t)r * blockSize,
               out.real + dstBase,
               blockSize * sizeof(double));
        memcpy(out.imag + dstBase + (size_t)r * blockSize,
               out.imag + dstBase,
               blockSize * sizeof(double));
      }
    }

    curShape[d] *= rep;
    curTotal *= (size_t)rep;
  }

  return out;
}
