/* mtoc2 runtime helper: 2-D matrix multiplication producing a complex
 * result.
 *
 * Handles all three mixed cases (complex × complex, complex × real,
 * real × complex) in one helper by treating a NULL imag pointer as
 * an implicit zero imag lane. The branch on `a.imag` / `b.imag` is
 * loop-invariant, so the C compiler hoists it out of the inner
 * accumulator.
 *
 * The translate-time check enforces 2-D inputs with matching inner
 * dimension when shapes are known. Runtime aborts on mismatch
 * otherwise (no exception channel in mtoc2 today).
 *
 * Allocates both real and imag lanes of the result. There is a
 * separate scalar-return variant for the 1×k * k×1 inner-product
 * case so consumers expecting a `double _Complex` (e.g. scalar
 * disp or scalar arithmetic) don't pay the tensor-alloc tax.
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

static double _Complex mtoc2_tensor_mtimes_complex_scalar(mtoc2_tensor_t a,
                                                          mtoc2_tensor_t b) {
  if (a.ndim != 2 || b.ndim != 2) {
    fprintf(stderr,
            "mtoc2_tensor_mtimes_complex_scalar: inputs must be 2-D\n");
    abort();
  }
  if (a.dims[0] != 1 || b.dims[1] != 1) {
    fprintf(stderr,
            "mtoc2_tensor_mtimes_complex_scalar: requires 1×k * k×1\n");
    abort();
  }
  if (a.dims[1] != b.dims[0]) {
    fprintf(stderr,
            "mtoc2_tensor_mtimes_complex_scalar: inner-dim mismatch\n");
    abort();
  }
  long k = a.dims[1];
  double accR = 0.0;
  double accI = 0.0;
  int aHasI = a.imag != NULL;
  int bHasI = b.imag != NULL;
  for (long p = 0; p < k; p++) {
    double ar = a.real[p];
    double ai = aHasI ? a.imag[p] : 0.0;
    double br = b.real[p];
    double bi = bHasI ? b.imag[p] : 0.0;
    accR += ar * br - ai * bi;
    accI += ar * bi + ai * br;
  }
  return accR + accI * I;
}

static mtoc2_tensor_t mtoc2_tensor_mtimes_complex(mtoc2_tensor_t a,
                                                   mtoc2_tensor_t b) {
  if (a.ndim != 2 || b.ndim != 2) {
    fprintf(stderr,
            "mtoc2_tensor_mtimes_complex: inputs must be 2-D (got %d-D, %d-D)\n",
            a.ndim, b.ndim);
    abort();
  }
  long m = a.dims[0];
  long k = a.dims[1];
  long k2 = b.dims[0];
  long n = b.dims[1];
  if (k != k2) {
    fprintf(stderr,
            "mtoc2_tensor_mtimes_complex: inner-dim mismatch (%ld×%ld * %ld×%ld)\n",
            m, k, k2, n);
    abort();
  }
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)m * (size_t)n * sizeof(double));
  r.imag = mtoc2_alloc((size_t)m * (size_t)n * sizeof(double));
  r.ndim = 2;
  r.dims[0] = m;
  r.dims[1] = n;
  int aHasI = a.imag != NULL;
  int bHasI = b.imag != NULL;
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i++) {
      double accR = 0.0;
      double accI = 0.0;
      for (long p = 0; p < k; p++) {
        long aoff = i + p * m;
        long boff = p + j * k;
        double ar = a.real[aoff];
        double ai = aHasI ? a.imag[aoff] : 0.0;
        double br = b.real[boff];
        double bi = bHasI ? b.imag[boff] : 0.0;
        accR += ar * br - ai * bi;
        accI += ar * bi + ai * br;
      }
      r.real[i + j * m] = accR;
      r.imag[i + j * m] = accI;
    }
  }
  return r;
}
