/* @snippet name=jit_tensor_transpose_complex deps=jit_tensor_t,jit_tensor_alloc_nd_complex */
/* JIT runtime helper: complex-tensor non-conjugate transpose for
 * 2-D inputs. Sibling of `jit_tensor_transpose` — same shape
 * permutation, but copies BOTH lanes (no conjugation, the `.'`
 * operator). The `'` (conjugate transpose) operator lowers to
 * `transpose(conj(z))` at the lowering layer, so this helper only
 * sees the non-conjugating case.
 *
 * Tolerates `a.imag == NULL` (real-tensor flowing through a
 * complex-typed transpose route) by zeroing the result's imag lane.
 */

#include <stdlib.h>
#include <string.h>

static jit_tensor_t jit_tensor_transpose_complex(jit_tensor_t a) {
  long m = a.dims[0];
  long n = a.dims[1];
  long dims[2];
  dims[0] = n;
  dims[1] = m;
  jit_tensor_t r = jit_tensor_alloc_nd_complex(2, dims);
  if (a.imag == NULL) {
    /* Defensive: zero the imag lane so the transposed result is a
     * well-formed complex tensor with re-only content. */
    memset(r.imag, 0, (size_t)m * (size_t)n * sizeof(double));
  }
  for (long sc = 0; sc < n; sc++) {
    for (long sr = 0; sr < m; sr++) {
      r.real[sc + sr * n] = a.real[sr + sc * m];
      if (a.imag != NULL) {
        r.imag[sc + sr * n] = a.imag[sr + sc * m];
      }
    }
  }
  return r;
}
