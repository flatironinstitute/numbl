/* mtoc2 runtime helper: build a real N-D tensor filled with zeros.
 *
 * Allocates via `mtoc2_tensor_alloc_nd`, then zero-fills the `real`
 * buffer with `memset`. The returned tensor is freshly owned;
 * `imag` is NULL.
 */

#include <string.h>

static mtoc2_tensor_t mtoc2_tensor_zeros_nd(int ndim, const long *dims) {
  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(ndim, dims);
  size_t n = 1;
  for (int i = 0; i < ndim; i++) n *= (size_t)out.dims[i];
  memset(out.real, 0, n * sizeof(double));
  return out;
}
