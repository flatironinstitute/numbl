/* mtoc runtime helper: MATLAB's `length(t)` on a multi-element tensor.
 *
 * Returns the size of the largest axis as a double, or 0.0 if any axis
 * has length 0. Mirrors MATLAB/numbl semantics. Scalar args don't reach
 * this helper — codegen emits the literal `1.0` directly because the C
 * arg is a bare `double`, not an `mtoc2_tensor_t`.
 */

static double mtoc2_length(mtoc2_tensor_t t) {
  long max = 0;
  for (int i = 0; i < t.ndim; i++) {
    if (t.dims[i] == 0) return 0.0;
    if (t.dims[i] > max) max = t.dims[i];
  }
  return (double)max;
}
