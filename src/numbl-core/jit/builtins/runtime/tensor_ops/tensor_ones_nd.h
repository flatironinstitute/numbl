/* mtoc2 runtime helper: build a real N-D tensor filled with ones.
 *
 * Allocates via `mtoc2_tensor_alloc_nd`, then fills the `real` buffer
 * with `1.0` via a plain element loop (`memset` only works for byte
 * patterns; `1.0` is not such a pattern). The returned tensor is
 * freshly owned; `imag` is NULL.
 */

static mtoc2_tensor_t mtoc2_tensor_ones_nd(int ndim, const long *dims) {
  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(ndim, dims);
  size_t n = 1;
  for (int i = 0; i < ndim; i++) n *= (size_t)out.dims[i];
  for (size_t i = 0; i < n; i++) out.real[i] = 1.0;
  return out;
}
