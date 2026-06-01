/* mtoc2 runtime helper: build a real 2-D identity matrix.
 *
 * Two entry points share one snippet:
 *   - `mtoc2_eye_rect(m, n)` returns an m×n real tensor with 1s on
 *     the main diagonal (positions `(i, i)` for `i` in `0..min(m,n)`).
 *     Negative dims clamp to 0 via `mtoc2_tensor_alloc`.
 *   - `mtoc2_eye_square(n)` is the single-eval companion for the
 *     MATLAB `eye(n)` shorthand when `n` is a runtime expression
 *     (taking the dim by parameter avoids duplicating the source
 *     expression in both slots).
 *
 * Storage is column-major to match `mtoc2_tensor_t`'s layout, so the
 * diagonal entry `(i, i)` sits at flat offset `i + i*rows` in `real`.
 * The returned tensor is freshly owned; `imag` is NULL.
 */

#include <string.h>

static mtoc2_tensor_t mtoc2_eye_rect(long rows, long cols) {
  mtoc2_tensor_t out = mtoc2_tensor_alloc(rows, cols);
  long r = out.dims[0];
  long c = out.dims[1];
  size_t n = (size_t)r * (size_t)c;
  if (n > 0) memset(out.real, 0, n * sizeof(double));
  long m = r < c ? r : c;
  for (long i = 0; i < m; i++) {
    out.real[i + i * r] = 1.0;
  }
  return out;
}

static mtoc2_tensor_t mtoc2_eye_square(long n) {
  return mtoc2_eye_rect(n, n);
}
