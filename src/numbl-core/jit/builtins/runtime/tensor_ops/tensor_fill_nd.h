/* mtoc2 runtime helper: build an N-D tensor filled with `v`. Real
 * variant allocates a real tensor; complex variant takes `(re, im)`
 * and fills both lanes.
 *
 * Parameterized companion to `mtoc2_tensor_zeros_nd` / `_ones_nd`;
 * activated by the `nan` / `NaN` / `Inf` / `inf` shape-constructor
 * builtins (which would otherwise need their own per-constant fill
 * helpers) and by `repmat(scalar, ...)`. The returned tensor is
 * freshly owned.
 */

static mtoc2_tensor_t mtoc2_tensor_fill_nd(double v, int ndim,
                                           const long *dims) {
  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd(ndim, dims);
  size_t n = 1;
  for (int i = 0; i < ndim; i++) n *= (size_t)out.dims[i];
  for (size_t i = 0; i < n; i++) out.real[i] = v;
  return out;
}

static mtoc2_tensor_t mtoc2_tensor_fill_nd_complex(double re, double im,
                                                   int ndim,
                                                   const long *dims) {
  mtoc2_tensor_t out = mtoc2_tensor_alloc_nd_complex(ndim, dims);
  size_t n = 1;
  for (int i = 0; i < ndim; i++) n *= (size_t)out.dims[i];
  for (size_t i = 0; i < n; i++) {
    out.real[i] = re;
    out.imag[i] = im;
  }
  return out;
}
