/* mtoc2 runtime helper: build an n×n real tensor filled with `v`.
 *
 * Single-eval companion to `mtoc2_tensor_fill_nd` for the MATLAB
 * `nan(n)` / `Inf(n)` shorthand when `n` is a runtime expression.
 * See `mtoc2_tensor_zeros_square` for the rationale.
 */

static mtoc2_tensor_t mtoc2_tensor_fill_square(double v, long n) {
  long dims[2] = {n, n};
  return mtoc2_tensor_fill_nd(v, 2, dims);
}
