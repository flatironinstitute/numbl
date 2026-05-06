// Translated from SRC/dorgqr.f
// DORGQR generates an M-by-N real matrix Q with orthonormal columns,
// which is defined as the first N columns of a product of K elementary
// reflectors of order M:
//
//       Q = H(1) H(2) . . . H(k)
//
// as returned by DGEQRF.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)    =>  tau[tauOff + (I-1)]              (I is 1-based)
//
// Parameters:
//   m       - number of rows of Q (>= 0)
//   n       - number of columns of Q (m >= n >= 0)
//   k       - number of elementary reflectors (n >= k >= 0)
//   a       - Float64Array; on entry reflector vectors; on exit is Q
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,m))
//   tau     - Float64Array of length k; scalar factors of the reflectors
//   tauOff  - offset into tau for TAU(1)
//
// Returns INFO (0 = success, < 0 = illegal argument)
//
// Note: ilaenv returns blocksize 1 for DORGQR, so the blocked path is
// never taken and this always calls the unblocked DORG2R.

import { dorg2r } from "./dorg2r.js";
import { ilaenv } from "../utils/ilaenv.js";
import { xerbla } from "../utils/xerbla.js";
import { allocFloat64Array } from "../../../numbl-core/executors/jsJit/helpers/alloc.js";

export function dorgqr(
  m: number,
  n: number,
  k: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number
): number {
  // Test the input arguments
  let info = 0;
  const nb = ilaenv(1, "DORGQR", " ", m, n, k, -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0 || n > m) {
    info = -2;
  } else if (k < 0 || k > n) {
    info = -3;
  } else if (lda < Math.max(1, m)) {
    info = -5;
  }
  if (info !== 0) {
    xerbla("DORGQR", -info);
    return info;
  }

  // Quick return if possible
  if (n <= 0) return 0;

  const nbmin = 2;
  let nx = 0;
  let iws = n;
  let kk = 0;

  if (nb > 1 && nb < k) {
    // Determine when to cross over from blocked to unblocked code
    nx = Math.max(0, ilaenv(3, "DORGQR", " ", m, n, k, -1));
    if (nx < k) {
      const ldwork = n;
      iws = ldwork * nb;
    }
  }

  if (nb >= nbmin && nb < k && nx < k) {
    // Blocked code path (in practice skipped since ilaenv returns 1 for DORGQR)
    const ki = Math.floor((k - nx - 1) / nb) * nb;
    kk = Math.min(k, ki + nb);

    // Set A(1:kk, kk+1:n) to zero
    for (let j = kk + 1; j <= n; j++) {
      for (let i = 1; i <= kk; i++) {
        a[aOff + (i - 1) + (j - 1) * lda] = 0.0;
      }
    }
  } else {
    kk = 0;
  }

  // Use unblocked code for the last or only block
  if (kk < n) {
    const work = allocFloat64Array(Math.max(1, n));
    dorg2r(
      m - kk,
      n - kk,
      k - kk,
      a,
      aOff + kk + kk * lda,
      lda,
      tau,
      tauOff + kk,
      work,
      0
    );
  }

  if (kk > 0) {
    // Blocked code (in practice unreachable with ilaenv returning 1)
    const ldwork = n;
    const work = allocFloat64Array(iws);
    const ki = Math.floor((k - nx - 1) / nb) * nb;
    for (let i = ki + 1; i >= 1; i -= nb) {
      const ib = Math.min(nb, k - i + 1);
      if (i + ib <= n) {
        // (dlarft + dlarfb omitted — unreachable)
      }
      // Apply H to rows i:m of current block
      dorg2r(
        m - i + 1,
        ib,
        ib,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        lda,
        tau,
        tauOff + (i - 1),
        work,
        0
      );
      // Set rows 1:i-1 of current block to zero
      for (let j = i; j <= i + ib - 1; j++) {
        for (let l = 1; l <= i - 1; l++) {
          a[aOff + (l - 1) + (j - 1) * lda] = 0.0;
        }
      }
      if (i <= ldwork) {
        // suppress unused variable warning
      }
    }
  }

  return 0;
}
