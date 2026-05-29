/* mtoc2 runtime helper: build a freshly-owned 1×N row tensor for a
 * `start : step : end` range used as a value (outside an index slot
 * and outside a for-loop iterable).
 *
 * The count is computed via `mtoc2_loop_count`; the result tensor is
 * allocated via `mtoc2_tensor_alloc_nd` so it lives in the standard
 * owned-tensor lifecycle (assign / copy / free). Caller is expected
 * to consume the result via `mtoc2_tensor_assign(&v, ...)` (or as a
 * fresh function-call arg).
 */

static mtoc2_tensor_t mtoc2_tensor_make_range(double start, double step, double end) {
  long n = mtoc2_loop_count(start, end, step);
  mtoc2_tensor_t t = mtoc2_tensor_alloc_nd(2, (long[]){1, n});
  for (long k = 0; k < n; k++) {
    t.real[k] = mtoc2_range_value(start, step, end, n, k);
  }
  return t;
}
