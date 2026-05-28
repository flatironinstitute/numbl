// Translated from SRC/dgeev.f
// DGEEV computes for an N-by-N real nonsymmetric matrix A, the
// eigenvalues and, optionally, the left and/or right eigenvectors.
//
// The right eigenvector v(j) of A satisfies
//                  A * v(j) = lambda(j) * v(j)
// where lambda(j) is its eigenvalue.
// The left eigenvector u(j) of A satisfies
//               u(j)**H * A = lambda(j) * u(j)**H
// where u(j)**H denotes the conjugate-transpose of u(j).
//
// The computed eigenvectors are normalized to have Euclidean norm
// equal to 1 and largest component real.
//
// Algorithm:
//   1. Balance matrix (dgebal)
//   2. Reduce to upper Hessenberg form (dgehrd)
//   3. Optionally generate orthogonal matrix (dorghr)
//   4. Compute Schur form (dhseqr)
//   5. Compute eigenvectors from Schur form (dtrevc3)
//   6. Back-transform eigenvectors (dgebak)
//   7. Normalize eigenvectors
//   8. Sort complex conjugate eigenvalue pairs
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J)    => a[aOff + (I-1) + (J-1)*lda]     (I,J are 1-based)
//   WR(I)     => wr[wrOff + (I-1)]                (I is 1-based)
//   WI(I)     => wi[wiOff + (I-1)]                (I is 1-based)
//   VL(I,J)   => vl[vlOff + (I-1) + (J-1)*ldvl]  (I,J are 1-based)
//   VR(I,J)   => vr[vrOff + (I-1) + (J-1)*ldvr]  (I,J are 1-based)
//   WORK(I)   => work[workOff + (I-1)]            (I is 1-based)
//
// Parameters:
//   jobvl   - 0='N' (don't compute left eigenvectors), 1='V' (compute)
//   jobvr   - 0='N' (don't compute right eigenvectors), 1='V' (compute)
//   n       - order of the matrix A (n >= 0)
//   a       - Float64Array; on entry the N-by-N matrix A; on exit overwritten
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,n))
//   wr      - Float64Array of length n; real parts of eigenvalues
//   wrOff   - offset into wr
//   wi      - Float64Array of length n; imaginary parts of eigenvalues
//   wiOff   - offset into wi
//   vl      - Float64Array; left eigenvectors if jobvl=1
//   vlOff   - offset into vl
//   ldvl    - leading dimension of vl (>= 1; if jobvl=1, >= n)
//   vr      - Float64Array; right eigenvectors if jobvr=1
//   vrOff   - offset into vr
//   ldvr    - leading dimension of vr (>= 1; if jobvr=1, >= n)
//   work    - Float64Array workspace of length max(1, lwork)
//   workOff - offset into work
//   lwork   - length of work array (>= max(1,3*N), or 4*N if eigenvectors);
//             if lwork=-1, workspace query
//   balance - optional (default true); false = nobalance (dgebal uses job=0 'N')
//
// Returns INFO:
//   = 0:  successful exit
//   < 0:  if INFO = -i, the i-th argument had an illegal value
//   > 0:  if INFO = i, the QR algorithm failed to compute all the
//         eigenvalues, and no eigenvectors have been computed;
//         elements i+1:N of WR and WI contain eigenvalues which
//         have converged.

import { dgebal } from "./dgebal.js";
import { dgehrd } from "./dgehrd.js";
import { dorghr } from "./dorghr.js";
import { dhseqr } from "./dhseqr.js";
import { dtrevc3 } from "./dtrevc3.js";
import { dgebak } from "./dgebak.js";
import { dlacpy } from "./dlacpy.js";
import { dlascl } from "./dlascl.js";
import { dlange } from "./dlange.js";
import { dlartg } from "./dlartg.js";
import { dlapy2 } from "./dlapy2.js";
import { dlamch } from "./dlamch.js";
import { drot } from "../BLAS/drot.js";
import { dscal } from "../BLAS/dscal.js";
import { dnrm2 } from "../BLAS/dnrm2.js";
import { idamax } from "../BLAS/idamax.js";
import { ilaenv } from "../utils/ilaenv.js";
import {
  MACH_PREC,
  MACH_SFMIN,
  LEFT,
  RIGHT,
  LOWER,
} from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

const ZERO = 0.0;
const ONE = 1.0;

// SIDE constants for dtrevc3
const SIDE_LEFT = LEFT; // 0
const SIDE_RIGHT = RIGHT; // 1
const SIDE_BOTH = 2; // 'B'

// HOWMNY constant for dtrevc3: backtransform
const HOWMNY_BACKTRANSFORM = 1; // 'B'

// DLASCL type: general
const TYPE_G = 0; // 'G'

// DLANGE norm: max abs element
const NORM_MAX = 0; // 'M'

// DLACPY uplo: full matrix
const UPLO_FULL = -1; // anything other than UPPER(0) or LOWER(1)

// DHSEQR job constants
const DHSEQR_E = 0; // 'E' eigenvalues only
const DHSEQR_S = 1; // 'S' Schur form

// DHSEQR compz constants
const DHSEQR_N = 0; // 'N' no Schur vectors
// DHSEQR_I = 1 ('I' initialize to identity) - not used directly in dgeev
const DHSEQR_V = 2; // 'V' use input matrix

// DGEBAL job constants
const GEBAL_NONE = 0; // 'N'
const GEBAL_BOTH = 3; // 'B'

export function dgeev(
  jobvl: number, // 0='N', 1='V'
  jobvr: number, // 0='N', 1='V'
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  wr: Float64Array,
  wrOff: number,
  wi: Float64Array,
  wiOff: number,
  vl: Float64Array,
  vlOff: number,
  ldvl: number,
  vr: Float64Array,
  vrOff: number,
  ldvr: number,
  work: Float64Array,
  workOff: number,
  lwork: number,
  balance?: boolean // default true; false = nobalance
): number {
  // Default balance to true
  const doBalance = balance !== false;

  // Test the input arguments
  let info = 0;
  const lquery = lwork === -1;
  const wantvl = jobvl === 1;
  const wantvr = jobvr === 1;

  if (jobvl !== 0 && jobvl !== 1) {
    info = -1;
  } else if (jobvr !== 0 && jobvr !== 1) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Math.max(1, n)) {
    info = -5;
  } else if (ldvl < 1 || (wantvl && ldvl < n)) {
    info = -9;
  } else if (ldvr < 1 || (wantvr && ldvr < n)) {
    info = -11;
  }

  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  //  minimal amount of workspace needed at that point in the code,
  //  as well as the preferred amount for good performance.
  //  NB refers to the optimal block size for the immediately
  //  following subroutine, as returned by ILAENV.
  //  HSWORK refers to the workspace preferred by DHSEQR, as
  //  calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  //  the worst case.)

  let maxwrk = 0;
  let minwrk = 0;

  if (info === 0) {
    if (n === 0) {
      minwrk = 1;
      maxwrk = 1;
    } else {
      maxwrk = 2 * n + n * ilaenv(1, "DGEHRD", " ", n, 1, n, 0);

      if (wantvl) {
        minwrk = 4 * n;
        maxwrk = Math.max(
          maxwrk,
          2 * n + (n - 1) * ilaenv(1, "DORGHR", " ", n, 1, n, -1)
        );

        // Workspace query for DHSEQR
        dhseqr(
          DHSEQR_S,
          DHSEQR_V,
          n,
          1,
          n,
          a,
          aOff,
          lda,
          wr,
          wrOff,
          wi,
          wiOff,
          vl,
          vlOff,
          ldvl,
          work,
          workOff,
          -1
        );
        const hswork = Math.trunc(work[workOff]);
        maxwrk = Math.max(maxwrk, n + 1, n + hswork);

        // Workspace query for DTREVC3
        dtrevc3(
          SIDE_LEFT,
          HOWMNY_BACKTRANSFORM,
          [false],
          n,
          a,
          aOff,
          lda,
          vl,
          vlOff,
          ldvl,
          vr,
          vrOff,
          ldvr,
          n,
          work,
          workOff,
          -1
        );
        const lwork_trevc = Math.trunc(work[workOff]);
        maxwrk = Math.max(maxwrk, n + lwork_trevc);
        maxwrk = Math.max(maxwrk, 4 * n);
      } else if (wantvr) {
        minwrk = 4 * n;
        maxwrk = Math.max(
          maxwrk,
          2 * n + (n - 1) * ilaenv(1, "DORGHR", " ", n, 1, n, -1)
        );

        // Workspace query for DHSEQR
        dhseqr(
          DHSEQR_S,
          DHSEQR_V,
          n,
          1,
          n,
          a,
          aOff,
          lda,
          wr,
          wrOff,
          wi,
          wiOff,
          vr,
          vrOff,
          ldvr,
          work,
          workOff,
          -1
        );
        const hswork = Math.trunc(work[workOff]);
        maxwrk = Math.max(maxwrk, n + 1, n + hswork);

        // Workspace query for DTREVC3
        dtrevc3(
          SIDE_RIGHT,
          HOWMNY_BACKTRANSFORM,
          [false],
          n,
          a,
          aOff,
          lda,
          vl,
          vlOff,
          ldvl,
          vr,
          vrOff,
          ldvr,
          n,
          work,
          workOff,
          -1
        );
        const lwork_trevc = Math.trunc(work[workOff]);
        maxwrk = Math.max(maxwrk, n + lwork_trevc);
        maxwrk = Math.max(maxwrk, 4 * n);
      } else {
        minwrk = 3 * n;

        // Workspace query for DHSEQR
        dhseqr(
          DHSEQR_E,
          DHSEQR_N,
          n,
          1,
          n,
          a,
          aOff,
          lda,
          wr,
          wrOff,
          wi,
          wiOff,
          vr,
          vrOff,
          ldvr,
          work,
          workOff,
          -1
        );
        const hswork = Math.trunc(work[workOff]);
        maxwrk = Math.max(maxwrk, n + 1, n + hswork);
      }

      maxwrk = Math.max(maxwrk, minwrk);
    }

    work[workOff] = maxwrk;

    if (lwork < minwrk && !lquery) {
      info = -13;
    }
  }

  if (info !== 0) {
    return info;
  } else if (lquery) {
    return 0;
  }

  // Quick return if possible
  if (n === 0) {
    return 0;
  }

  // Get machine constants
  const eps = dlamch(MACH_PREC);
  let smlnum = dlamch(MACH_SFMIN);
  let bignum = ONE / smlnum;
  smlnum = Math.sqrt(smlnum) / eps;
  bignum = ONE / smlnum;

  // Scale A if max element outside range [SMLNUM,BIGNUM]
  const dum = allocFloat64Array(1);
  const anrm = dlange(NORM_MAX, n, n, a, aOff, lda, dum, 0);
  let scalea = false;
  let cscale = 0.0;
  if (anrm > ZERO && anrm < smlnum) {
    scalea = true;
    cscale = smlnum;
  } else if (anrm > bignum) {
    scalea = true;
    cscale = bignum;
  }
  if (scalea) {
    dlascl(TYPE_G, 0, 0, anrm, cscale, n, n, a, aOff, lda);
  }

  // Balance the matrix
  // (Workspace: need N)
  const ibal = 0; // 0-based offset into work for balance scale factors
  const gebalJob = doBalance ? GEBAL_BOTH : GEBAL_NONE;
  const balResult = dgebal(gebalJob, n, a, aOff, lda, work, workOff + ibal);
  const ilo = balResult.ilo;
  const ihi = balResult.ihi;

  // Reduce to upper Hessenberg form
  // (Workspace: need 3*N, prefer 2*N+N*NB)
  const itau = ibal + n; // 0-based offset into work for tau
  const iwrk = itau + n; // 0-based offset into work for workspace

  dgehrd(
    n,
    ilo,
    ihi,
    a,
    aOff,
    lda,
    work,
    workOff + itau, // TAU
    work,
    workOff + iwrk, // WORK for dgehrd
    lwork - iwrk // remaining workspace
  );

  // SIDE variable for dtrevc3
  let side: number;

  if (wantvl) {
    // Want left eigenvectors
    // Copy Householder vectors to VL
    side = SIDE_LEFT;
    dlacpy(LOWER, n, n, a, aOff, lda, vl, vlOff, ldvl);

    // Generate orthogonal matrix in VL
    // (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
    dorghr(
      n,
      ilo,
      ihi,
      vl,
      vlOff,
      ldvl,
      work,
      workOff + itau, // TAU
      work,
      workOff + iwrk, // WORK for dorghr
      lwork - iwrk
    );

    // Perform QR iteration, accumulating Schur vectors in VL
    // (Workspace: need N+1, prefer N+HSWORK)
    const iwrk2 = itau; // reuse tau workspace
    info = dhseqr(
      DHSEQR_S,
      DHSEQR_V,
      n,
      ilo,
      ihi,
      a,
      aOff,
      lda,
      wr,
      wrOff,
      wi,
      wiOff,
      vl,
      vlOff,
      ldvl,
      work,
      workOff + iwrk2,
      lwork - iwrk2
    );

    if (wantvr) {
      // Want left and right eigenvectors
      // Copy Schur vectors to VR
      side = SIDE_BOTH;
      dlacpy(UPLO_FULL, n, n, vl, vlOff, ldvl, vr, vrOff, ldvr);
    }
  } else if (wantvr) {
    // Want right eigenvectors
    // Copy Householder vectors to VR
    side = SIDE_RIGHT;
    dlacpy(LOWER, n, n, a, aOff, lda, vr, vrOff, ldvr);

    // Generate orthogonal matrix in VR
    // (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
    dorghr(
      n,
      ilo,
      ihi,
      vr,
      vrOff,
      ldvr,
      work,
      workOff + itau, // TAU
      work,
      workOff + iwrk, // WORK for dorghr
      lwork - iwrk
    );

    // Perform QR iteration, accumulating Schur vectors in VR
    // (Workspace: need N+1, prefer N+HSWORK)
    const iwrk2 = itau;
    info = dhseqr(
      DHSEQR_S,
      DHSEQR_V,
      n,
      ilo,
      ihi,
      a,
      aOff,
      lda,
      wr,
      wrOff,
      wi,
      wiOff,
      vr,
      vrOff,
      ldvr,
      work,
      workOff + iwrk2,
      lwork - iwrk2
    );
  } else {
    // Compute eigenvalues only
    // (Workspace: need N+1, prefer N+HSWORK)
    side = SIDE_RIGHT; // not used, but initialize
    const iwrk2 = itau;
    info = dhseqr(
      DHSEQR_E,
      DHSEQR_N,
      n,
      ilo,
      ihi,
      a,
      aOff,
      lda,
      wr,
      wrOff,
      wi,
      wiOff,
      vr,
      vrOff,
      ldvr,
      work,
      workOff + iwrk2,
      lwork - iwrk2
    );
  }

  // If INFO != 0 from DHSEQR, then quit
  if (info !== 0) {
    // Jump to undo-scaling (label 50 in Fortran)
    return dgeevFinish(
      info,
      scalea,
      cscale,
      anrm,
      n,
      ilo,
      wr,
      wrOff,
      wi,
      wiOff,
      work,
      workOff,
      maxwrk
    );
  }

  if (wantvl || wantvr) {
    // Compute left and/or right eigenvectors
    // (Workspace: need 4*N, prefer N + N + 2*N*NB)
    const iwrk2 = itau;
    dtrevc3(
      side,
      HOWMNY_BACKTRANSFORM,
      [false], // SELECT not referenced for howmny='B'
      n,
      a,
      aOff,
      lda,
      vl,
      vlOff,
      ldvl,
      vr,
      vrOff,
      ldvr,
      n,
      work,
      workOff + iwrk2,
      lwork - iwrk2
    );
  }

  if (wantvl) {
    // Undo balancing of left eigenvectors
    // (Workspace: need N)
    const bakJob = doBalance ? GEBAL_BOTH : GEBAL_NONE;
    dgebak(bakJob, LEFT, n, ilo, ihi, work, workOff + ibal, n, vl, vlOff, ldvl);

    // Normalize left eigenvectors and make largest component real
    for (let i = 1; i <= n; i++) {
      if (wi[wiOff + (i - 1)] === ZERO) {
        // Real eigenvalue
        const scl = ONE / dnrm2(n, vl, vlOff + (i - 1) * ldvl, 1);
        dscal(n, scl, vl, vlOff + (i - 1) * ldvl, 1);
      } else if (wi[wiOff + (i - 1)] > ZERO) {
        // Complex conjugate pair: columns i and i+1
        const scl =
          ONE /
          dlapy2(
            dnrm2(n, vl, vlOff + (i - 1) * ldvl, 1),
            dnrm2(n, vl, vlOff + i * ldvl, 1)
          );
        dscal(n, scl, vl, vlOff + (i - 1) * ldvl, 1);
        dscal(n, scl, vl, vlOff + i * ldvl, 1);

        // Compute |VL(k,i)|^2 + |VL(k,i+1)|^2 for each row k
        const iwrk2 = itau;
        for (let k = 1; k <= n; k++) {
          const vk_re = vl[vlOff + (k - 1) + (i - 1) * ldvl];
          const vk_im = vl[vlOff + (k - 1) + i * ldvl];
          work[workOff + iwrk2 + (k - 1)] = vk_re * vk_re + vk_im * vk_im;
        }
        // Find row with largest magnitude
        const k = idamax(n, work, workOff + iwrk2, 1);
        // Compute Givens rotation to make VL(k,i) real
        const givens = dlartg(
          vl[vlOff + (k - 1) + (i - 1) * ldvl],
          vl[vlOff + (k - 1) + i * ldvl]
        );
        drot(
          n,
          vl,
          vlOff + (i - 1) * ldvl,
          1,
          vl,
          vlOff + i * ldvl,
          1,
          givens.cs,
          givens.sn
        );
        vl[vlOff + (k - 1) + i * ldvl] = ZERO;
      }
    }
  }

  if (wantvr) {
    // Undo balancing of right eigenvectors
    // (Workspace: need N)
    const bakJob = doBalance ? GEBAL_BOTH : GEBAL_NONE;
    dgebak(
      bakJob,
      RIGHT,
      n,
      ilo,
      ihi,
      work,
      workOff + ibal,
      n,
      vr,
      vrOff,
      ldvr
    );

    // Normalize right eigenvectors and make largest component real
    for (let i = 1; i <= n; i++) {
      if (wi[wiOff + (i - 1)] === ZERO) {
        // Real eigenvalue
        const scl = ONE / dnrm2(n, vr, vrOff + (i - 1) * ldvr, 1);
        dscal(n, scl, vr, vrOff + (i - 1) * ldvr, 1);
      } else if (wi[wiOff + (i - 1)] > ZERO) {
        // Complex conjugate pair: columns i and i+1
        const scl =
          ONE /
          dlapy2(
            dnrm2(n, vr, vrOff + (i - 1) * ldvr, 1),
            dnrm2(n, vr, vrOff + i * ldvr, 1)
          );
        dscal(n, scl, vr, vrOff + (i - 1) * ldvr, 1);
        dscal(n, scl, vr, vrOff + i * ldvr, 1);

        // Compute |VR(k,i)|^2 + |VR(k,i+1)|^2 for each row k
        const iwrk2 = itau;
        for (let k = 1; k <= n; k++) {
          const vk_re = vr[vrOff + (k - 1) + (i - 1) * ldvr];
          const vk_im = vr[vrOff + (k - 1) + i * ldvr];
          work[workOff + iwrk2 + (k - 1)] = vk_re * vk_re + vk_im * vk_im;
        }
        // Find row with largest magnitude
        const k = idamax(n, work, workOff + iwrk2, 1);
        // Compute Givens rotation to make VR(k,i) real
        const givens = dlartg(
          vr[vrOff + (k - 1) + (i - 1) * ldvr],
          vr[vrOff + (k - 1) + i * ldvr]
        );
        drot(
          n,
          vr,
          vrOff + (i - 1) * ldvr,
          1,
          vr,
          vrOff + i * ldvr,
          1,
          givens.cs,
          givens.sn
        );
        vr[vrOff + (k - 1) + i * ldvr] = ZERO;
      }
    }
  }

  // Undo scaling if necessary (label 50 in Fortran)
  return dgeevFinish(
    info,
    scalea,
    cscale,
    anrm,
    n,
    ilo,
    wr,
    wrOff,
    wi,
    wiOff,
    work,
    workOff,
    maxwrk
  );
}

/**
 * Finish routine: undo scaling and write optimal workspace.
 * Corresponds to label 50 and the code after it in the Fortran DGEEV.
 */
function dgeevFinish(
  info: number,
  scalea: boolean,
  cscale: number,
  anrm: number,
  n: number,
  ilo: number,
  wr: Float64Array,
  wrOff: number,
  wi: Float64Array,
  wiOff: number,
  work: Float64Array,
  workOff: number,
  maxwrk: number
): number {
  if (scalea) {
    // DLASCL('G', 0, 0, CSCALE, ANRM, N-INFO, 1, WR(INFO+1), MAX(N-INFO,1), IERR)
    dlascl(
      TYPE_G,
      0,
      0,
      cscale,
      anrm,
      n - info,
      1,
      wr,
      wrOff + info,
      Math.max(n - info, 1)
    );
    // DLASCL('G', 0, 0, CSCALE, ANRM, N-INFO, 1, WI(INFO+1), MAX(N-INFO,1), IERR)
    dlascl(
      TYPE_G,
      0,
      0,
      cscale,
      anrm,
      n - info,
      1,
      wi,
      wiOff + info,
      Math.max(n - info, 1)
    );
    if (info > 0) {
      // DLASCL('G', 0, 0, CSCALE, ANRM, ILO-1, 1, WR, N, IERR)
      dlascl(TYPE_G, 0, 0, cscale, anrm, ilo - 1, 1, wr, wrOff, n);
      // DLASCL('G', 0, 0, CSCALE, ANRM, ILO-1, 1, WI, N, IERR)
      dlascl(TYPE_G, 0, 0, cscale, anrm, ilo - 1, 1, wi, wiOff, n);
    }
  }

  work[workOff] = maxwrk;
  return info;
}
