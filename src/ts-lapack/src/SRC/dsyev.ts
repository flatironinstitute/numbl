// Translated from SRC/dsyev.f
// DSYEV computes all eigenvalues and, optionally, eigenvectors of a real
// symmetric matrix A.  Eigenvalues are returned in ascending order and the
// eigenvectors form an orthonormal basis.
//
// Two deviations from the reference, both harmless at the sizes numbl sees:
//   - the unblocked DSYTD2 is called directly instead of the blocked DSYTRD
//     (ilaenv returns blocksize 1 for the other reductions in this tree too);
//   - the eigenvalues-only path uses DSTEQR with COMPZ = 'N' rather than
//     DSTERF.  Both are QL/QR on the tridiagonal; DSTERF is only the faster
//     root-free variant.
//
// Only uplo = LOWER is supported, because DORGTR here implements only that
// branch.  Callers hold a full symmetric matrix anyway.
//
// Array indexing convention (column-major):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   W(I)   => w[wOff + (I-1)]
//
// On exit, if jobz = 'V' (wantz), A holds the orthonormal eigenvectors.
//
// Returns INFO: 0 = success, < 0 = illegal argument, > 0 = the number of
// off-diagonal elements that failed to converge to zero.

import { LOWER } from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";
import { dlamch } from "./dlamch.js";
import { MACH_SFMIN, MACH_PREC } from "../utils/constants.js";
import { dlansy } from "./dlansy.js";
import { dlascl } from "./dlascl.js";
import { dsytd2 } from "./dsytd2.js";
import { dorgtr } from "./dorgtr.js";
import { dsteqr, COMPZ_N, COMPZ_V } from "./dsteqr.js";
import { dscal } from "../BLAS/dscal.js";
import { xerbla } from "../utils/xerbla.js";

const NORM_MAX = 0; // dlansy 'M'
const TYPE_L = 1; // dlascl 'L' (lower triangular)

export function dsyev(
  wantz: boolean,
  uplo: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  w: Float64Array,
  wOff: number
): number {
  const ONE = 1.0;
  const ZERO = 0.0;

  // Test the input parameters
  let info = 0;
  if (uplo !== LOWER) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Math.max(1, n)) {
    info = -5;
  }
  if (info !== 0) {
    xerbla("DSYEV ", -info);
    return info;
  }

  // Quick return if possible
  if (n === 0) return 0;
  if (n === 1) {
    w[wOff] = a[aOff];
    if (wantz) a[aOff] = ONE;
    return 0;
  }

  // Get machine constants
  const safmin = dlamch(MACH_SFMIN);
  const eps = dlamch(MACH_PREC);
  const smlnum = safmin / eps;
  const bignum = ONE / smlnum;
  const rmin = Math.sqrt(smlnum);
  const rmax = Math.sqrt(bignum);

  const work = allocFloat64Array(Math.max(1, 3 * n));

  // Scale matrix to allowable range, if necessary
  const anrm = dlansy(NORM_MAX, uplo, n, a, aOff, lda, work, 0);
  let iscale = 0;
  let sigma = ONE;
  if (anrm > ZERO && anrm < rmin) {
    iscale = 1;
    sigma = rmin / anrm;
  } else if (anrm > rmax) {
    iscale = 1;
    sigma = rmax / anrm;
  }
  if (iscale === 1) dlascl(TYPE_L, 0, 0, ONE, sigma, n, n, a, aOff, lda);

  // Reduce the symmetric matrix to tridiagonal form
  const e = allocFloat64Array(Math.max(1, n - 1));
  const tau = allocFloat64Array(Math.max(1, n - 1));
  dsytd2(uplo, n, a, aOff, lda, w, wOff, e, 0, tau, 0);

  if (!wantz) {
    info = dsteqr(COMPZ_N, n, w, wOff, e, 0, a, aOff, lda, work, 0);
  } else {
    // Generate the orthogonal matrix from the reflectors, then run the
    // implicit QL/QR iteration on the tridiagonal, accumulating into it.
    const iinfo = dorgtr(uplo, n, a, aOff, lda, tau, 0);
    if (iinfo !== 0) return iinfo;
    info = dsteqr(COMPZ_V, n, w, wOff, e, 0, a, aOff, lda, work, 0);
  }

  // If the matrix was scaled, rescale the eigenvalues appropriately
  if (iscale === 1) {
    const imax = info === 0 ? n : info - 1;
    dscal(imax, ONE / sigma, w, wOff, 1);
  }

  return info;
}
