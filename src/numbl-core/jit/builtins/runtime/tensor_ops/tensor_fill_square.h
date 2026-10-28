/* @snippet name=jit_tensor_fill_square deps=jit_tensor_fill_nd */
/* JIT runtime helper: build an n×n real tensor filled with `v`.
 *
 * Single-eval companion to `jit_tensor_fill_nd` for the MATLAB
 * `nan(n)` / `Inf(n)` shorthand when `n` is a runtime expression.
 * See `jit_tensor_zeros_square` for the rationale.
 */

static jit_tensor_t jit_tensor_fill_square(double v, long n) {
  long dims[2] = {n, n};
  return jit_tensor_fill_nd(v, 2, dims);
}
