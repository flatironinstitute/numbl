// Translated from SRC/dorgtr.f
// DORGTR generates a real orthogonal matrix Q which is defined as the product
// of n-1 elementary reflectors of order n, as returned by DSYTRD/DSYTD2.
//
// Only uplo = LOWER is implemented: the callers in this tree (dsyev) always
// reduce the lower triangle, and the LOWER branch reuses the existing
// DORGQR/DORG2R, whereas the UPPER branch would need a DORGQL that has no
// other user here.  Passing UPPER returns INFO = -1.
//
// Array indexing convention (column-major):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//
// Returns INFO (0 = success, < 0 = illegal argument).

import { LOWER } from "../utils/constants.js";
import { dorgqr } from "./dorgqr.js";
import { xerbla } from "../utils/xerbla.js";

export function dorgtr(
  uplo: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number
): number {
  const ONE = 1.0;
  const ZERO = 0.0;

  // Test the input arguments
  let info = 0;
  if (uplo !== LOWER) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Math.max(1, n)) {
    info = -4;
  }
  if (info !== 0) {
    xerbla("DORGTR", -info);
    return info;
  }

  // Quick return if possible
  if (n === 0) return 0;

  const A = (i: number, j: number) => aOff + (i - 1) + (j - 1) * lda;

  // Q was determined by a call to DSYTRD with UPLO = 'L'.
  // Shift the vectors which define the elementary reflectors one column to
  // the right, and set the first row and column of Q to those of the unit
  // matrix.
  for (let j = n; j >= 2; j--) {
    a[A(1, j)] = ZERO;
    for (let i = j + 1; i <= n; i++) {
      a[A(i, j)] = a[A(i, j - 1)];
    }
  }
  a[A(1, 1)] = ONE;
  for (let i = 2; i <= n; i++) {
    a[A(i, 1)] = ZERO;
  }

  if (n > 1) {
    // Generate Q(2:n,2:n)
    info = dorgqr(n - 1, n - 1, n - 1, a, A(2, 2), lda, tau, tauOff);
  }

  return info;
}
