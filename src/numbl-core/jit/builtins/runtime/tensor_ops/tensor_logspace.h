/* mtoc2 runtime helper: build a 1×n row tensor of n logarithmically-
 * spaced values from 10^a to 10^b. Matches numbl's `logspace`:
 *
 *   - n <= 0  → 1×0 empty tensor.
 *   - n == 1  → just `[endVal]` (1×1), where endVal is `pi` when the
 *               upper limit is exactly pi, else `10^b`.
 *   - n  > 1  → `10^t` for `t` stepped linearly from `a` to `b`.
 *
 * MATLAB special case: an upper limit of exactly `pi` makes the last
 * point `pi` (not 10^pi) — the loop steps `t` over `log10(10^a)` ..
 * `log10(pi)` so the endpoint lands on `pi`.
 */

#include <math.h>

static mtoc2_tensor_t mtoc2_tensor_logspace(double a, double b, long n) {
  if (n <= 0) return mtoc2_tensor_alloc(1, 0);
  int isPi = (b == 3.141592653589793);
  double endVal = isPi ? 3.141592653589793 : pow(10.0, b);
  mtoc2_tensor_t out = mtoc2_tensor_alloc(1, n);
  if (n == 1) {
    out.real[0] = endVal;
    return out;
  }
  if (isPi) {
    double logStart = log10(pow(10.0, a));
    double logEnd = log10(3.141592653589793);
    for (long i = 0; i < n; i++) {
      double t = logStart + (logEnd - logStart) * (double)i / (double)(n - 1);
      out.real[i] = pow(10.0, t);
    }
  } else {
    for (long i = 0; i < n; i++) {
      double t = a + (b - a) * (double)i / (double)(n - 1);
      out.real[i] = pow(10.0, t);
    }
  }
  return out;
}
