/* mtoc runtime helper: MATLAB's `numel(t)` on a multi-element tensor.
 *
 * Returns the product of the axis sizes as a double. Scalar args don't
 * reach this helper — codegen emits the literal `1.0` directly because
 * the C arg is a bare `double`, not an `mtoc2_tensor_t`.
 */

static double mtoc2_numel(mtoc2_tensor_t t) {
  long n = 1;
  for (int i = 0; i < t.ndim; i++) {
    n *= t.dims[i];
  }
  return (double)n;
}
