/* mtoc2 runtime helper: real-tensor non-conjugate transpose for 2-D
 * inputs. Returns a freshly-owned tensor with `dims` swapped. The
 * 2-D restriction is enforced at lowering; by the time this helper
 * runs, `a.ndim` is always 2.
 *
 * Column-major in, column-major out. Source `a` has shape (m × n);
 * destination has shape (n × m). Source element at (sr, sc) lives at
 * `a.real[sr + sc*m]`; destination element at (sc, sr) — the transpose
 * mapping — lives at `out.real[sc + sr*n]`. The inner loop walks the
 * source's column-major buffer linearly to keep the read stride
 * unit-stride.
 *
 * For complex support (not yet a thing in mtoc2), the conjugate
 * variant would negate `a.imag` while copying; the non-conjugate
 * variant just copies. Mirrors numbl's `transposeCore` in
 * `helpers/arithmetic.ts`.
 */

#include <stdlib.h>

static mtoc2_tensor_t mtoc2_tensor_transpose(mtoc2_tensor_t a) {
  long m = a.dims[0];
  long n = a.dims[1];
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)m * (size_t)n * sizeof(double));
  r.imag = NULL;
  r.ndim = 2;
  r.dims[0] = n;
  r.dims[1] = m;
  for (long sc = 0; sc < n; sc++) {
    for (long sr = 0; sr < m; sr++) {
      r.real[sc + sr * n] = a.real[sr + sc * m];
    }
  }
  return r;
}
