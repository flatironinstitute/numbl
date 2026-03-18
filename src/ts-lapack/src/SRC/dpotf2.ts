// Translated from LAPACK/SRC/dpotf2.f
// DPOTF2 computes the Cholesky factorization of a real symmetric
// positive definite matrix A (unblocked algorithm, Level 2 BLAS).
//
//   A = U**T * U,  if UPLO = UPPER, or
//   A = L  * L**T,  if UPLO = LOWER,
//
// where U is upper triangular and L is lower triangular.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)

import { ddot } from "../BLAS/ddot.js";
import { dgemv } from "../BLAS/dgemv.js";
import { dscal } from "../BLAS/dscal.js";
import { xerbla } from "../utils/xerbla.js";
import { UPPER, LOWER, TRANS, NOTRANS } from "../utils/constants.js";

export function dpotf2(
  uplo: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  const upper = uplo === UPPER;

  // Test the input parameters
  let info = 0;
  if (!upper && uplo !== LOWER) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Math.max(1, n)) {
    info = -5;
  }
  if (info !== 0) {
    xerbla("DPOTF2", -info);
    return info;
  }

  // Quick return if possible
  if (n === 0) return 0;

  if (upper) {
    // Compute the Cholesky factorization A = U**T * U.
    for (let j = 1; j <= n; j++) {
      // Compute U(J,J) and test for non-positive-definiteness.
      let ajj =
        a[aOff + (j - 1) + (j - 1) * lda] -
        ddot(j - 1, a, aOff + (j - 1) * lda, 1, a, aOff + (j - 1) * lda, 1);
      if (ajj <= 0 || Number.isNaN(ajj)) {
        a[aOff + (j - 1) + (j - 1) * lda] = ajj;
        return j;
      }
      ajj = Math.sqrt(ajj);
      a[aOff + (j - 1) + (j - 1) * lda] = ajj;

      // Compute elements J+1:N of row J.
      if (j < n) {
        dgemv(
          TRANS,
          j - 1,
          n - j,
          -1.0,
          a,
          aOff + j * lda,
          lda,
          a,
          aOff + (j - 1) * lda,
          1,
          1.0,
          a,
          aOff + (j - 1) + j * lda,
          lda
        );
        dscal(n - j, 1.0 / ajj, a, aOff + (j - 1) + j * lda, lda);
      }
    }
  } else {
    // Compute the Cholesky factorization A = L * L**T.
    for (let j = 1; j <= n; j++) {
      // Compute L(J,J) and test for non-positive-definiteness.
      let ajj =
        a[aOff + (j - 1) + (j - 1) * lda] -
        ddot(j - 1, a, aOff + (j - 1), lda, a, aOff + (j - 1), lda);
      if (ajj <= 0 || Number.isNaN(ajj)) {
        a[aOff + (j - 1) + (j - 1) * lda] = ajj;
        return j;
      }
      ajj = Math.sqrt(ajj);
      a[aOff + (j - 1) + (j - 1) * lda] = ajj;

      // Compute elements J+1:N of column J.
      if (j < n) {
        dgemv(
          NOTRANS,
          n - j,
          j - 1,
          -1.0,
          a,
          aOff + j,
          lda,
          a,
          aOff + (j - 1),
          lda,
          1.0,
          a,
          aOff + j + (j - 1) * lda,
          1
        );
        dscal(n - j, 1.0 / ajj, a, aOff + j + (j - 1) * lda, 1);
      }
    }
  }

  return 0;
}
