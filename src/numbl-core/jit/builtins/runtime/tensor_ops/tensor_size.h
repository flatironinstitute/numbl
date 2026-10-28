/* @snippet name=jit_tensor_size_row deps=jit_tensor_t,jit_alloc */
/* JIT runtime helper: `size(t)` — returns a freshly-owned 1×ndim
 * row tensor whose elements are the input's dim sizes as doubles.
 * MATLAB / numbl semantics: scalars and vectors return at least a
 * 2-element row (the type system already pads to ndim ≥ 2; this
 * helper just copies dims[] into a row vector).
 *
 * For `size(t, k)` the JIT emits a scalar `(double)t.dims[k-1]` inline
 * — no runtime helper needed for that form.
 */

#include <stdlib.h>

static jit_tensor_t jit_tensor_size_row(jit_tensor_t a) {
  long n = a.ndim;
  jit_tensor_t r;
  r.real = jit_alloc((size_t)n * sizeof(double));
  r.imag = NULL;
  r.ndim = 2;
  r.dims[0] = 1;
  r.dims[1] = n;
  for (long i = 0; i < n; i++) r.real[i] = (double)a.dims[i];
  return r;
}
