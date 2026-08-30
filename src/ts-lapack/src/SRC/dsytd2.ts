// Translated from SRC/dsytd2.f
// DSYTD2 reduces a real symmetric matrix A to symmetric tridiagonal form T by
// an orthogonal similarity transformation: Q**T * A * Q = T.
//
// On exit, if uplo = LOWER, the diagonal and first subdiagonal of A are
// overwritten by the corresponding elements of the tridiagonal matrix T, and
// the elements below the first subdiagonal, with the array TAU, represent the
// orthogonal matrix Q as a product of elementary reflectors.  If uplo = UPPER,
// the same holds for the diagonal and first superdiagonal.
//
// Array indexing convention (column-major):
//   A(I,J)  => a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   D(I)    => d[dOff + (I-1)]
//   E(I)    => e[eOff + (I-1)]
//   TAU(I)  => tau[tauOff + (I-1)]
//
// Returns INFO (0 = success, < 0 = illegal argument).

import { UPPER, LOWER } from "../utils/constants.js";
import { dlarfg } from "./dlarfg.js";
import { dsymv } from "../BLAS/dsymv.js";
import { dsyr2 } from "../BLAS/dsyr2.js";
import { ddot } from "../BLAS/ddot.js";
import { daxpy } from "../BLAS/daxpy.js";
import { xerbla } from "../utils/xerbla.js";

export function dsytd2(
  uplo: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  d: Float64Array,
  dOff: number,
  e: Float64Array,
  eOff: number,
  tau: Float64Array,
  tauOff: number
): number {
  const ONE = 1.0;
  const ZERO = 0.0;
  const HALF = 0.5;

  // Test the input parameters
  let info = 0;
  const upper = uplo === UPPER;
  if (!upper && uplo !== LOWER) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Math.max(1, n)) {
    info = -4;
  }
  if (info !== 0) {
    xerbla("DSYTD2", -info);
    return info;
  }

  // Quick return if possible
  if (n <= 0) return 0;

  const A = (i: number, j: number) => aOff + (i - 1) + (j - 1) * lda;

  if (upper) {
    // Reduce the upper triangle of A
    for (let i = n - 1; i >= 1; i--) {
      // Generate elementary reflector H(i) = I - tau * v * v**T
      // to annihilate A(1:i-1,i+1)
      const r = dlarfg(i, a[A(i, i + 1)], a, A(1, i + 1), 1);
      a[A(i, i + 1)] = r.alpha;
      const taui = r.tau;
      e[eOff + (i - 1)] = a[A(i, i + 1)];

      if (taui !== ZERO) {
        // Apply H(i) from both sides to A(1:i,1:i)
        a[A(i, i + 1)] = ONE;

        // Compute  x := tau * A * v  storing x in TAU(1:i)
        dsymv(
          uplo,
          i,
          taui,
          a,
          aOff,
          lda,
          a,
          A(1, i + 1),
          1,
          ZERO,
          tau,
          tauOff,
          1
        );

        // Compute  w := x - 1/2 * tau * (x**T * v) * v
        const alpha = -HALF * taui * ddot(i, tau, tauOff, 1, a, A(1, i + 1), 1);
        daxpy(i, alpha, a, A(1, i + 1), 1, tau, tauOff, 1);

        // Apply the transformation as a rank-2 update:
        //    A := A - v * w**T - w * v**T
        dsyr2(uplo, i, -ONE, a, A(1, i + 1), 1, tau, tauOff, 1, a, aOff, lda);

        a[A(i, i + 1)] = e[eOff + (i - 1)];
      }
      d[dOff + i] = a[A(i + 1, i + 1)];
      tau[tauOff + (i - 1)] = taui;
    }
    d[dOff] = a[A(1, 1)];
  } else {
    // Reduce the lower triangle of A
    for (let i = 1; i <= n - 1; i++) {
      // Generate elementary reflector H(i) = I - tau * v * v**T
      // to annihilate A(i+2:n,i)
      const r = dlarfg(n - i, a[A(i + 1, i)], a, A(Math.min(i + 2, n), i), 1);
      a[A(i + 1, i)] = r.alpha;
      const taui = r.tau;
      e[eOff + (i - 1)] = a[A(i + 1, i)];

      if (taui !== ZERO) {
        // Apply H(i) from both sides to A(i+1:n,i+1:n)
        a[A(i + 1, i)] = ONE;

        // Compute  x := tau * A * v  storing y in TAU(i:n-1)
        dsymv(
          uplo,
          n - i,
          taui,
          a,
          A(i + 1, i + 1),
          lda,
          a,
          A(i + 1, i),
          1,
          ZERO,
          tau,
          tauOff + (i - 1),
          1
        );

        // Compute  w := x - 1/2 * tau * (x**T * v) * v
        const alpha =
          -HALF *
          taui *
          ddot(n - i, tau, tauOff + (i - 1), 1, a, A(i + 1, i), 1);
        daxpy(n - i, alpha, a, A(i + 1, i), 1, tau, tauOff + (i - 1), 1);

        // Apply the transformation as a rank-2 update:
        //    A := A - v * w**T - w * v**T
        dsyr2(
          uplo,
          n - i,
          -ONE,
          a,
          A(i + 1, i),
          1,
          tau,
          tauOff + (i - 1),
          1,
          a,
          A(i + 1, i + 1),
          lda
        );

        a[A(i + 1, i)] = e[eOff + (i - 1)];
      }
      d[dOff + (i - 1)] = a[A(i, i)];
      tau[tauOff + (i - 1)] = taui;
    }
    d[dOff + (n - 1)] = a[A(n, n)];
  }

  return 0;
}
