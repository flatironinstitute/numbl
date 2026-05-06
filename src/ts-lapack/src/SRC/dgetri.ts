// Translated from SRC/dgetri.f
// DGETRI computes the inverse of a matrix using the LU factorization
// computed by DGETRF.
//
// This method inverts U and then computes inv(A) by solving the system
//   inv(A)*L = inv(U)  for inv(A).
//
// The WORK array from the original Fortran is allocated internally here.
// The public API matches the Fortran in all other respects.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   IPIV(I)   =>  ipiv[ipivOff + (I-1)]           (I is 1-based)
//   WORK(I)   =>  work[I-1]                        (I is 1-based)
//
// Parameters:
//   n    - order of the matrix (>= 0)
//   a    - Float64Array containing the LU factors from dgetrf; overwritten
//          with inv(A) on exit
//   lda  - leading dimension of a (>= max(1,n))
//   ipiv - Int32Array, pivot indices from dgetrf (1-based, length n)
//
// Returns INFO:
//   0   => successful exit
//   < 0 => -INFO-th argument had an illegal value (thrown as error)
//   > 0 => U(INFO,INFO) is exactly zero; singular, no inverse

import { dgemm } from "../BLAS/dgemm.js";
import { dgemv } from "../BLAS/dgemv.js";
import { dswap } from "../BLAS/dswap.js";
import { dtrsm } from "../BLAS/dtrsm.js";
import { dtrtri } from "./dtrtri.js";
import { xerbla } from "../utils/xerbla.js";
import { ilaenv } from "../utils/ilaenv.js";
import {
  UPPER,
  LOWER,
  NOTRANS,
  UNIT,
  NONUNIT,
  RIGHT,
} from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/executors/jsJit/helpers/alloc.js";

export function dgetri(
  n: number,
  a: Float64Array,
  lda: number,
  ipiv: Int32Array
): number {
  const aOff = 0;
  const ipivOff = 0;

  // Determine block size and optimal workspace size
  const nb = ilaenv(1, "DGETRI", " ", n, -1, -1, -1);
  const lwkopt = Math.max(1, n * nb);

  // Test the input parameters
  let info = 0;
  if (n < 0) {
    info = -1;
  } else if (lda < Math.max(1, n)) {
    info = -3;
  }
  if (info !== 0) {
    xerbla("DGETRI", -info);
    return info;
  }

  if (n === 0) return 0;

  // Form inv(U). If singular, return immediately.
  info = dtrtri(UPPER, NONUNIT, n, a, aOff, lda);
  if (info > 0) return info;

  // Allocate workspace internally (size = lwkopt)
  const lwork = lwkopt;
  const work = allocFloat64Array(lwork);
  const ldwork = n; // leading dimension of WORK-as-matrix

  // Decide effective block size
  let nbEff = nb;
  let nbmin = 2;
  let iws: number;
  if (nbEff > 1 && nbEff < n) {
    iws = Math.max(ldwork * nbEff, 1);
    if (lwork < iws) {
      nbEff = Math.floor(lwork / ldwork);
      nbmin = Math.max(2, ilaenv(2, "DGETRI", " ", n, -1, -1, -1));
    }
  } else {
    iws = n;
  }

  // Solve the equation  inv(A)*L = inv(U)  for inv(A).
  if (nbEff < nbmin || nbEff >= n) {
    // Use unblocked code
    for (let j = n; j >= 1; j--) {
      // Copy current column of L to WORK and replace with zeros
      for (let i = j + 1; i <= n; i++) {
        work[i - 1] = a[aOff + (i - 1) + (j - 1) * lda]; // WORK(I)=A(I,J)
        a[aOff + (i - 1) + (j - 1) * lda] = 0.0;
      }

      // Compute current column of inv(A)
      // DGEMV('N', N, N-J, -ONE, A(1,J+1), LDA, WORK(J+1), 1, ONE, A(1,J), 1)
      if (j < n) {
        dgemv(
          NOTRANS,
          n,
          n - j,
          -1.0,
          a,
          aOff + j * lda,
          lda, // A(1,J+1)
          work,
          j,
          1, // WORK(J+1) = work[J]
          1.0,
          a,
          aOff + (j - 1) * lda,
          1 // A(1,J)
        );
      }
    }
  } else {
    // Use blocked code
    const nn = Math.floor((n - 1) / nbEff) * nbEff + 1;
    for (let j = nn; j >= 1; j -= nbEff) {
      const jb = Math.min(nbEff, n - j + 1);

      // Copy current block column of L to WORK and replace with zeros
      for (let jj = j; jj <= j + jb - 1; jj++) {
        for (let i = jj + 1; i <= n; i++) {
          // WORK(I+(JJ-J)*LDWORK) = A(I,JJ)
          work[i - 1 + (jj - j) * ldwork] = a[aOff + (i - 1) + (jj - 1) * lda];
          a[aOff + (i - 1) + (jj - 1) * lda] = 0.0;
        }
      }

      // Compute current block column of inv(A)
      if (j + jb <= n) {
        // DGEMM('N','N', N, JB, N-J-JB+1, -ONE, A(1,J+JB), LDA,
        //        WORK(J+JB), LDWORK, ONE, A(1,J), LDA)
        // A(1,J+JB)  => aOff + 0 + (j+jb-1)*lda
        // WORK(J+JB) => work[j+jb-1]  (bOff for WORK-as-matrix)
        // A(1,J)     => aOff + 0 + (j-1)*lda
        dgemm(
          NOTRANS,
          NOTRANS,
          n,
          jb,
          n - j - jb + 1,
          -1.0,
          a,
          aOff + (j + jb - 1) * lda,
          lda,
          work,
          j + jb - 1,
          ldwork,
          1.0,
          a,
          aOff + (j - 1) * lda,
          lda
        );
      }

      // DTRSM('Right','Lower','No transpose','Unit', N, JB, ONE,
      //        WORK(J), LDWORK, A(1,J), LDA)
      // WORK(J) => work[j-1]  (aOff for WORK-as-matrix)
      // A(1,J)  => aOff + 0 + (j-1)*lda
      dtrsm(
        RIGHT,
        LOWER,
        NOTRANS,
        UNIT,
        n,
        jb,
        1.0,
        work,
        j - 1,
        ldwork,
        a,
        aOff + (j - 1) * lda,
        lda
      );
    }
  }

  // Apply column interchanges
  // DO J = N-1, 1, -1:  if IPIV(J) != J, swap columns J and IPIV(J)
  for (let j = n - 1; j >= 1; j--) {
    const jp = ipiv[ipivOff + (j - 1)]; // IPIV(J), 1-based
    if (jp !== j) {
      // DSWAP(N, A(1,J), 1, A(1,JP), 1)
      // A(1,J)  => aOff + 0 + (j-1)*lda
      // A(1,JP) => aOff + 0 + (jp-1)*lda
      dswap(n, a, aOff + (j - 1) * lda, 1, a, aOff + (jp - 1) * lda, 1);
    }
  }

  return 0;
}
