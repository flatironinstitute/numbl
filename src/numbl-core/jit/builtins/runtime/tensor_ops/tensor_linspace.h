/* mtoc2 runtime helper: build a 1×n row tensor of n linearly-spaced
 * values from `a` to `b` inclusive. Matches numbl's `linspace` byte-
 * for-byte:
 *
 *   - n <= 0  → 1×0 empty tensor.
 *   - n == 1  → just `[b]` (matches MATLAB; not the midpoint).
 *   - n  > 1  → first/last slots pinned at `a`/`b` exactly so a NaN
 *               or Inf endpoint doesn't contaminate the other end;
 *               inner values are `a + (b - a) * i / (n - 1)`.
 *
 * Opposite-sign infinite endpoints place 0 at the exact center for
 * odd n (e.g. `linspace(-Inf, Inf, 5)` → `[-Inf, ?, 0, ?, Inf]`).
 */

#include <math.h>

static mtoc2_tensor_t mtoc2_tensor_linspace(double a, double b, long n) {
  if (n < 0) n = 0;
  mtoc2_tensor_t out = mtoc2_tensor_alloc(1, n);
  if (n == 0) return out;
  if (n == 1) {
    out.real[0] = b;
    return out;
  }
  out.real[0] = a;
  out.real[n - 1] = b;
  for (long i = 1; i < n - 1; i++) {
    out.real[i] = a + (b - a) * (double)i / (double)(n - 1);
  }
  if ((n & 1) == 1 && !isfinite(a) && !isfinite(b)) {
    double sa = (a > 0) - (a < 0);
    double sb = (b > 0) - (b < 0);
    if (sa != sb) {
      out.real[(n - 1) / 2] = 0.0;
    }
  }
  return out;
}
