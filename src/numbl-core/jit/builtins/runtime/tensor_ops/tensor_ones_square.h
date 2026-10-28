/* JIT runtime helper: build an n×n real tensor filled with ones.
 *
 * Single-eval companion to `jit_tensor_ones_nd` for the MATLAB
 * `ones(n)` shorthand when `n` is a runtime expression. See the
 * `jit_tensor_zeros_square` header for the rationale.
 */

static jit_tensor_t jit_tensor_ones_square(long n) {
  long dims[2] = {n, n};
  return jit_tensor_ones_nd(2, dims);
}
