/* mtoc2 runtime helper: uniquetol for real-double tensors.
 *
 * Numbl's elementwise `uniquetol(x, tol)` scans column-major and
 * keeps the first occurrence of each new value — a value is "new"
 * if it differs from every previous unique by more than `tol` in
 * absolute value. NaN is never within tol of anything (including
 * itself), so each NaN survives as its own entry.
 *
 * The scan is O(n × n_unique). Numbl uses the same naive
 * pairwise-against-running-list algorithm; this implementation
 * mirrors it (NOT a sort+adjacent-dedup, which would change the
 * transitive-chaining behaviour, e.g.
 * `uniquetol([0 0.6 1.2 1.8], 0.7)` collapses 1.8 into the same
 * class as 0 through the intermediate 0.6 / 1.2 chain).
 *
 * Output shape:
 *   - `row_out=1` (input was 1×N) → freshly-owned `[1, n_unique]`
 *   - `row_out=0`                 → freshly-owned `[n_unique, 1]`
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static mtoc2_tensor_t mtoc2_uniquetol_real(mtoc2_tensor_t a, double tol,
                                           int row_out) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  double *uniques = NULL;
  long nu = 0;
  if (n > 0) {
    uniques = (double *)malloc((size_t)n * sizeof(double));
    if (!uniques) {
      fprintf(stderr, "mtoc2: out of memory (uniquetol)\n");
      abort();
    }
    for (long i = 0; i < n; i++) {
      double x = a.real[i];
      int found = 0;
      for (long k = 0; k < nu; k++) {
        if (isnan(uniques[k]) || isnan(x)) continue;
        if (fabs(x - uniques[k]) <= tol) {
          found = 1;
          break;
        }
      }
      if (!found) uniques[nu++] = x;
    }
  }
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)nu * sizeof(double));
  r.imag = NULL;
  r.ndim = 2;
  if (row_out) {
    r.dims[0] = 1;
    r.dims[1] = nu;
  } else {
    r.dims[0] = nu;
    r.dims[1] = 1;
  }
  for (long i = 0; i < nu; i++) r.real[i] = uniques[i];
  if (uniques) free(uniques);
  return r;
}
