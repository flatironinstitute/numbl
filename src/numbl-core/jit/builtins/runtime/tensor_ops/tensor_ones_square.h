/* mtoc2 runtime helper: build an n×n real tensor filled with ones.
 *
 * Single-eval companion to `mtoc2_tensor_ones_nd` for the MATLAB
 * `ones(n)` shorthand when `n` is a runtime expression. See the
 * `mtoc2_tensor_zeros_square` header for the rationale.
 */

static mtoc2_tensor_t mtoc2_tensor_ones_square(long n) {
  long dims[2] = {n, n};
  return mtoc2_tensor_ones_nd(2, dims);
}
