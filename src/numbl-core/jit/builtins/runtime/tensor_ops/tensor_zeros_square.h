/* mtoc2 runtime helper: build an n×n real tensor filled with zeros.
 *
 * Single-eval companion to `mtoc2_tensor_zeros_nd` for the MATLAB
 * `zeros(n)` shorthand when `n` is a runtime expression. Taking the
 * dim in a function parameter guarantees the caller-side expression
 * is evaluated exactly once — placing the same C expression in both
 * slots of a compound literal would re-evaluate it (broken for any
 * side-effecting source like `zeros(myfun())`).
 */

static mtoc2_tensor_t mtoc2_tensor_zeros_square(long n) {
  long dims[2] = {n, n};
  return mtoc2_tensor_zeros_nd(2, dims);
}
