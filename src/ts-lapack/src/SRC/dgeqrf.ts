// Translated from SRC/dgeqrf.f
// DGEQRF computes a QR factorization of a real M-by-N matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where Q is a M-by-M orthogonal matrix and R is an upper-triangular
// min(M,N)-by-N matrix (upper triangular if M >= N).
//
// The matrix Q is represented as a product of elementary reflectors
//    Q = H(1) H(2) . . . H(k), where k = min(m,n).
// Each H(i) has the form H(i) = I - tau * v * v**T where tau is a real
// scalar and v is a real vector with v(1:i-1) = 0 and v(i) = 1;
// v(i+1:m) is stored on exit in A(i+1:m,i), and tau in TAU(i).
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)    =>  tau[tauOff + (I-1)]              (I is 1-based)
//
// Parameters:
//   m       - number of rows    (>= 0)
//   n       - number of columns (>= 0)
//   a       - Float64Array of the matrix (modified in place)
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,m))
//   tau     - Float64Array of length min(m,n); output scalar factors
//   tauOff  - offset into tau for TAU(1)
//
// Returns INFO (0 = success, < 0 = illegal argument)
//
// Note: ilaenv returns blocksize 1 for DGEQRF, so the blocked path is
// never taken and this always calls the unblocked DGEQR2.

import { dgeqr2 } from "./dgeqr2.js";
import { ilaenv } from "../utils/ilaenv.js";
import { xerbla } from "../utils/xerbla.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

export function dgeqrf(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number
): number {
  const k = Math.min(m, n);

  // Test the input arguments
  let info = 0;
  const nb = ilaenv(1, "DGEQRF", " ", m, n, -1, -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Math.max(1, m)) {
    info = -4;
  }
  if (info !== 0) {
    xerbla("DGEQRF", -info);
    return info;
  }

  // Quick return if possible
  if (k === 0) return 0;

  const nbmin = 2;
  let nx = 0;
  let iws = n;

  if (nb > 1 && nb < k) {
    // Determine when to cross over from blocked to unblocked code
    nx = Math.max(0, ilaenv(3, "DGEQRF", " ", m, n, -1, -1));
    if (nx < k) {
      const ldwork = n;
      iws = ldwork * nb;
    }
  }

  // Use unblocked code to factor the last or only block.
  // (With ilaenv returning 1, nb=1 so nb > 1 is false; I always falls to 1.)
  let i = 1;

  if (nb >= nbmin && nb < k && nx < k) {
    // Blocked code path (in practice skipped since ilaenv returns 1 for DGEQRF)
    const work = allocFloat64Array(iws);
    for (i = 1; i <= k - nx; i += nb) {
      const ib = Math.min(k - i + 1, nb);
      // Factor block A(i:m, i:i+ib-1)
      dgeqr2(
        m - i + 1,
        ib,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        lda,
        tau,
        tauOff + (i - 1),
        work,
        0
      );
      // (dlarft + dlarfb omitted; would only be reached with NB > 1)
    }
  }

  // Use unblocked code for the last or only block
  if (i <= k) {
    const work = allocFloat64Array(n);
    dgeqr2(
      m - i + 1,
      n - i + 1,
      a,
      aOff + (i - 1) + (i - 1) * lda,
      lda,
      tau,
      tauOff + (i - 1),
      work,
      0
    );
  }

  return 0;
}
