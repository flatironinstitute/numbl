/* mtoc2 runtime helper: `flip(t, dimIdx)` — return a freshly-owned
 * tensor with `t.real` mirrored along axis `dimIdx` (0-based).
 *
 * Numbl's reference is `flipAlongDim` in
 * `interpreter/builtins/array-manipulation.ts` line ~41. Same
 * column-major slab math: stride = product of dims below the axis,
 * outer = product of dims above. For each outer slab, we walk the
 * axis backwards on the source and forward on the destination,
 * copying `strideDim`-element contiguous blocks.
 *
 * `flipud(t)` lowers to `mtoc2_tensor_flip(t, 0)`; `fliplr(t)` to
 * `mtoc2_tensor_flip(t, 1)`; `flip(t, k)` to
 * `mtoc2_tensor_flip(t, k - 1)`. The `dimIdx` is 0-based at the C
 * boundary so the runtime stays uniform across the three source-
 * level builtins. mtoc2 codegen converts MATLAB's 1-based `k` to
 * 0-based at the call site.
 *
 * Out-of-range `dimIdx` (≥ ndim) is a no-op flip — numbl returns
 * the input unchanged in that case (the "axis is size 1" rule). We
 * still allocate a fresh copy so the owned-value invariant holds.
 *
 * `mtoc2_tensor_flip_complex` is the sibling that walks both lanes;
 * it tolerates `a.imag == NULL` (a real tensor that flowed in via a
 * complex-typed route) by zero-filling the output imag lane.
 */

#include <string.h>
#include <stdlib.h>

static mtoc2_tensor_t mtoc2_tensor_flip(mtoc2_tensor_t a, long dimIdx) {
  long total = 1;
  for (int i = 0; i < a.ndim; i++) total *= a.dims[i];
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)total * sizeof(double));
  r.imag = NULL;
  r.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];

  long axisSize = (dimIdx >= 0 && dimIdx < (long)a.ndim) ? a.dims[dimIdx] : 1;
  if (axisSize <= 1) {
    // Nothing to flip — just deep-copy the buffer.
    if (total > 0) memcpy(r.real, a.real, (size_t)total * sizeof(double));
    return r;
  }

  long strideDim = 1;
  for (long d = 0; d < dimIdx; d++) strideDim *= a.dims[d];
  long slabSize = strideDim * axisSize;
  long numOuter = total / slabSize;

  for (long outer = 0; outer < numOuter; outer++) {
    long base = outer * slabSize;
    for (long k = 0; k < axisSize; k++) {
      long srcOff = base + k * strideDim;
      long dstOff = base + (axisSize - 1 - k) * strideDim;
      memcpy(
        r.real + dstOff,
        a.real + srcOff,
        (size_t)strideDim * sizeof(double)
      );
    }
  }
  return r;
}

static mtoc2_tensor_t mtoc2_tensor_flip_complex(mtoc2_tensor_t a, long dimIdx) {
  long total = 1;
  for (int i = 0; i < a.ndim; i++) total *= a.dims[i];
  mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);
  int srcHasImag = (a.imag != NULL);

  long axisSize = (dimIdx >= 0 && dimIdx < (long)a.ndim) ? a.dims[dimIdx] : 1;
  if (axisSize <= 1) {
    if (total > 0) {
      memcpy(r.real, a.real, (size_t)total * sizeof(double));
      if (srcHasImag) {
        memcpy(r.imag, a.imag, (size_t)total * sizeof(double));
      } else {
        memset(r.imag, 0, (size_t)total * sizeof(double));
      }
    }
    return r;
  }

  if (!srcHasImag && total > 0) {
    memset(r.imag, 0, (size_t)total * sizeof(double));
  }

  long strideDim = 1;
  for (long d = 0; d < dimIdx; d++) strideDim *= a.dims[d];
  long slabSize = strideDim * axisSize;
  long numOuter = total / slabSize;

  for (long outer = 0; outer < numOuter; outer++) {
    long base = outer * slabSize;
    for (long k = 0; k < axisSize; k++) {
      long srcOff = base + k * strideDim;
      long dstOff = base + (axisSize - 1 - k) * strideDim;
      memcpy(r.real + dstOff, a.real + srcOff,
             (size_t)strideDim * sizeof(double));
      if (srcHasImag) {
        memcpy(r.imag + dstOff, a.imag + srcOff,
               (size_t)strideDim * sizeof(double));
      }
    }
  }
  return r;
}
