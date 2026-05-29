/* mtoc2 runtime helpers: dot product on real and complex tensors.
 *
 *   mtoc2_dot_real(a, b) → double
 *     Vector form on real inputs: `sum_i a.real[i] * b.real[i]`.
 *     Length mismatch aborts (the static type system rejects the
 *     case when both lengths are known at compile time, so a
 *     mismatch only reaches the helper through dim-unknown shapes).
 *
 *   mtoc2_dot_real_matrix(a, b) → mtoc2_tensor_t
 *     Matrix form on real inputs: both args same shape MxN, returns
 *     a freshly-owned `[1, N]` row vector whose j-th entry is
 *     `sum_i a[i,j] * b[i,j]`.
 *
 *   mtoc2_dot_complex(a, b) → double _Complex
 *     Vector form on complex (or mixed) inputs. Matches numbl's
 *     `sum(conj(a) .* b)`:
 *         re = sum_i  a.re*b.re + a.im*b.im
 *         im = sum_i  a.re*b.im - a.im*b.re
 *     A NULL imag lane on either side is treated as zero (a real
 *     tensor that flowed in via a complex-typed route).
 *
 *   mtoc2_dot_complex_matrix(a, b) → mtoc2_tensor_t (1×N complex)
 *     Matrix form on complex (or mixed) inputs. Same column-major
 *     fiber-wise dot.
 *
 * Storage column-major in both lanes to match `mtoc2_tensor_t`.
 */

#include <stdio.h>
#include <stdlib.h>

static double mtoc2_dot_real(mtoc2_tensor_t a, mtoc2_tensor_t b) {
  long na = 1, nb = 1;
  for (int i = 0; i < a.ndim; i++) na *= a.dims[i];
  for (int i = 0; i < b.ndim; i++) nb *= b.dims[i];
  if (na != nb) {
    fprintf(stderr, "mtoc2: dot: vectors must be same length\n");
    abort();
  }
  double acc = 0.0;
  for (long i = 0; i < na; i++) acc += a.real[i] * b.real[i];
  return acc;
}

static mtoc2_tensor_t mtoc2_dot_real_matrix(mtoc2_tensor_t a,
                                            mtoc2_tensor_t b) {
  long rows = a.dims[0];
  long cols = a.dims[1];
  mtoc2_tensor_t r;
  r.real = mtoc2_alloc((size_t)cols * sizeof(double));
  r.imag = NULL;
  r.ndim = 2;
  r.dims[0] = 1;
  r.dims[1] = cols;
  for (long j = 0; j < cols; j++) {
    double acc = 0.0;
    for (long i = 0; i < rows; i++) {
      long off = j * rows + i;
      acc += a.real[off] * b.real[off];
    }
    r.real[j] = acc;
  }
  return r;
}

static double _Complex mtoc2_dot_complex(mtoc2_tensor_t a, mtoc2_tensor_t b) {
  long na = 1, nb = 1;
  for (int i = 0; i < a.ndim; i++) na *= a.dims[i];
  for (int i = 0; i < b.ndim; i++) nb *= b.dims[i];
  if (na != nb) {
    fprintf(stderr, "mtoc2: dot_complex: vectors must be same length\n");
    abort();
  }
  int aHasImag = (a.imag != NULL);
  int bHasImag = (b.imag != NULL);
  double accRe = 0.0;
  double accIm = 0.0;
  for (long i = 0; i < na; i++) {
    double aRe = a.real[i];
    double aIm = aHasImag ? a.imag[i] : 0.0;
    double bRe = b.real[i];
    double bIm = bHasImag ? b.imag[i] : 0.0;
    /* conj(a) * b = (aRe - i*aIm) * (bRe + i*bIm)
     *             = (aRe*bRe + aIm*bIm) + i*(aRe*bIm - aIm*bRe) */
    accRe += aRe * bRe + aIm * bIm;
    accIm += aRe * bIm - aIm * bRe;
  }
  return mtoc2_cmake(accRe, accIm);
}

static mtoc2_tensor_t mtoc2_dot_complex_matrix(mtoc2_tensor_t a,
                                               mtoc2_tensor_t b) {
  long rows = a.dims[0];
  long cols = a.dims[1];
  long dims2[2] = {1, cols};
  mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(2, dims2);
  int aHasImag = (a.imag != NULL);
  int bHasImag = (b.imag != NULL);
  for (long j = 0; j < cols; j++) {
    double accRe = 0.0;
    double accIm = 0.0;
    for (long i = 0; i < rows; i++) {
      long off = j * rows + i;
      double aRe = a.real[off];
      double aIm = aHasImag ? a.imag[off] : 0.0;
      double bRe = b.real[off];
      double bIm = bHasImag ? b.imag[off] : 0.0;
      accRe += aRe * bRe + aIm * bIm;
      accIm += aRe * bIm - aIm * bRe;
    }
    r.real[j] = accRe;
    r.imag[j] = accIm;
  }
  return r;
}
