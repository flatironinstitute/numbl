/* @snippet name=jit_tensor_make_range deps=jit_tensor_t,jit_tensor_alloc_nd,jit_loop_count,jit_range_value */
/* JIT runtime helper: build a freshly-owned 1×N row tensor for a
 * `start : step : end` range used as a value (outside an index slot
 * and outside a for-loop iterable).
 *
 * The count is computed via `jit_loop_count`; the result tensor is
 * allocated via `jit_tensor_alloc_nd` so it lives in the standard
 * owned-tensor lifecycle (assign / copy / free). Caller is expected
 * to consume the result via `jit_tensor_assign(&v, ...)` (or as a
 * fresh function-call arg).
 */

static jit_tensor_t jit_tensor_make_range(double start, double step, double end) {
  long n = jit_loop_count(start, end, step);
  jit_tensor_t t = jit_tensor_alloc_nd(2, (long[]){1, n});
  for (long k = 0; k < n; k++) {
    t.real[k] = jit_range_value(start, step, end, n, k);
  }
  return t;
}
