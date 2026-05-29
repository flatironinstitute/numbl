/* mtoc2 runtime helpers: elementwise logical ops on tensors.
 *
 * Same allocate-and-fill pattern as tensor_unary_real_math.h, but the
 * per-element kernel is a logical predicate that returns 0.0 or 1.0.
 *
 * `mtoc2_tensor_not` mirrors numbl's `not(v)` (runtimeOperators.ts):
 * a real-tensor input produces a freshly-owned logical-typed tensor
 * of the same shape with `out[i] = (in[i] == 0.0) ? 1.0 : 0.0`. The
 * complex sibling fires "true" iff both lanes are exactly zero. The
 * result is a real (logical-typed) tensor in both cases — the type
 * system records `elem: "logical"` so disp / downstream consumers
 * know how to interpret the doubles.
 */
#include <stdlib.h>

static mtoc2_tensor_t mtoc2_tensor_not(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)n * sizeof(double));
  r.imag = NULL;
  r.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];
  for (long i = 0; i < n; i++) r.real[i] = (a.real[i] == 0.0) ? 1.0 : 0.0;
  return r;
}

static mtoc2_tensor_t mtoc2_tensor_not_complex(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)n * sizeof(double));
  r.imag = NULL;
  r.ndim = a.ndim;
  for (int i = 0; i < a.ndim; i++) r.dims[i] = a.dims[i];
  int srcHasImag = (a.imag != NULL);
  for (long i = 0; i < n; i++) {
    double re = a.real[i];
    double im = srcHasImag ? a.imag[i] : 0.0;
    r.real[i] = (re == 0.0 && im == 0.0) ? 1.0 : 0.0;
  }
  return r;
}
