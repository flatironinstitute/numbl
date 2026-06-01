/* mtoc2 runtime helper: first-order difference `diff(A)` / `diff(A, 1, dim)`.
 *
 * Mirrors numbl's `diffOnce` (helpers/reduction/cumulative.ts): a
 * forward difference `out[i] = in[i+1] - in[i]` along the operating
 * axis. The result is shorter by one along that axis; if the axis has
 * <= 1 element the result is empty (size 0 along the axis).
 *
 * `dim == 0` selects the default axis (numbl's rule: row vector → dim
 * 2, else the first dim); `dim > 0` is an explicit 1-based axis. Output
 * dims drop trailing singletons to a 2-D floor, matching RTV.tensor.
 *
 * Only first-order (n == 1) real diff is JIT-compiled here; the def
 * (`diff.ts`) declines n != 1, complex, scalar, and out-of-range dim to
 * the interpreter.
 */

#include <stdlib.h>

static mtoc2_tensor_t mtoc2_tensor_diff(mtoc2_tensor_t a, int dim) {
  int opDim;
  if (dim > 0) {
    opDim = dim - 1;
  } else if (a.ndim <= 1 || (a.ndim == 2 && a.dims[0] == 1)) {
    opDim = (a.ndim == 2 && a.dims[0] == 1) ? 1 : 0;
  } else {
    opDim = 0;
  }
  long dimSize = (opDim < a.ndim) ? a.dims[opDim] : 1;

  long newDims[MTOC2_MAX_NDIM];
  int newNdim = a.ndim;
  for (int i = 0; i < a.ndim; i++) newDims[i] = a.dims[i];

  if (dimSize <= 1) {
    if (opDim < newNdim) newDims[opDim] = 0;
    while (newNdim > 2 && newDims[newNdim - 1] == 1) newNdim--;
    return mtoc2_tensor_alloc_nd(newNdim, newDims);
  }

  newDims[opDim] = dimSize - 1;
  while (newNdim > 2 && newDims[newNdim - 1] == 1) newNdim--;
  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(newNdim, newDims);

  long innerCount = 1;
  for (int d = 0; d < opDim; d++) innerCount *= a.dims[d];
  long outerCount = 1;
  for (int d = opDim + 1; d < a.ndim; d++) outerCount *= a.dims[d];

  long outIdx = 0;
  for (long outer = 0; outer < outerCount; outer++) {
    for (long k = 0; k < dimSize - 1; k++) {
      for (long inner = 0; inner < innerCount; inner++) {
        long base = outer * (dimSize * innerCount) + k * innerCount + inner;
        out.real[outIdx++] = a.real[base + innerCount] - a.real[base];
      }
    }
  }
  return out;
}
