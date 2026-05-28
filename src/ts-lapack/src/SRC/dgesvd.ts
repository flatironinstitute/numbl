// Translated from LAPACK/SRC/dgesvd.f
// DGESVD computes the singular value decomposition (SVD) of a real
// M-by-N matrix A, optionally computing the left and/or right singular
// vectors. The SVD is written
//
//      A = U * SIGMA * transpose(V)
//
// where SIGMA is an M-by-N matrix which is zero except for its
// min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
// V is an N-by-N orthogonal matrix. The diagonal elements of SIGMA
// are the singular values of A; they are real and non-negative, and
// are returned in descending order. The first min(m,n) columns of
// U and V are the left and right singular vectors of A.
//
// Note that the routine returns V**T, not V.

import { dgebrd } from "./dgebrd.js";
import { dgelqf } from "./dgelqf.js";
import { dgeqrf } from "./dgeqrf.js";
import { dorgbr } from "./dorgbr.js";
import { dorglq } from "./dorglq.js";
import { dorgqr } from "./dorgqr.js";
// dormbr not used in this simplified translation (uses dorgbr instead)
// import { dormbr } from "./dormbr.js";
import { dbdsqr } from "./dbdsqr.js";
import { dlacpy } from "./dlacpy.js";
import { dlascl } from "./dlascl.js";
import { dlaset } from "./dlaset.js";
import { dlange } from "./dlange.js";
import { dlamch } from "./dlamch.js";
import { dgemm } from "../BLAS/dgemm.js";
import { ilaenv } from "../utils/ilaenv.js";
import { xerbla } from "../utils/xerbla.js";
import {
  UPPER,
  LOWER,
  NOTRANS,
  VECT_Q,
  VECT_P,
  MACH_EPS,
  MACH_SFMIN,
} from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

// JOBU encoding
const JOBU_A = 0;
const JOBU_S = 1;
const JOBU_O = 2;
const JOBU_N = 3;
// JOBVT encoding
const JOBVT_A = 0;
const JOBVT_S = 1;
const JOBVT_O = 2;
const JOBVT_N = 3;

// dlascl type for general matrix
const DLASCL_G = 0;

// dlacpy/dlaset uplo for full matrix
const FULL = -1;

// dlange norm type for max abs value
const NORM_MAX = 0;

const ZERO = 0.0;
const ONE = 1.0;

/**
 * DGESVD computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors.
 *
 * @param jobu  - 0='A' all M columns of U, 1='S' first min(m,n) columns,
 *                2='O' overwrite A, 3='N' no left vectors
 * @param jobvt - 0='A' all N rows of V**T, 1='S' first min(m,n) rows,
 *                2='O' overwrite A, 3='N' no right vectors
 * @param m     - Number of rows of A (>= 0)
 * @param n     - Number of columns of A (>= 0)
 * @param a     - Matrix A, dimension (lda, n), modified on exit
 * @param aOff  - Offset into a
 * @param lda   - Leading dimension of a (>= max(1, m))
 * @param s     - Singular values, dimension min(m,n), descending order
 * @param sOff  - Offset into s
 * @param u     - Left singular vectors, dimension (ldu, ucol)
 * @param uOff  - Offset into u
 * @param ldu   - Leading dimension of u
 * @param vt    - Right singular vectors, dimension (ldvt, n)
 * @param vtOff - Offset into vt
 * @param ldvt  - Leading dimension of vt
 * @param work  - Workspace array
 * @param workOff - Offset into work
 * @param lwork - Workspace size; -1 for query
 * @returns info: 0 success, <0 illegal arg, >0 DBDSQR did not converge
 */
export function dgesvd(
  jobu: number,
  jobvt: number,
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  s: Float64Array,
  sOff: number,
  u: Float64Array,
  uOff: number,
  ldu: number,
  vt: Float64Array,
  vtOff: number,
  ldvt: number,
  work: Float64Array,
  workOff: number,
  lwork: number
): number {
  let info = 0;
  const minmn = Math.min(m, n);

  const wntua = jobu === JOBU_A;
  const wntus = jobu === JOBU_S;
  const wntuas = wntua || wntus;
  const wntuo = jobu === JOBU_O;
  const wntun = jobu === JOBU_N;
  const wntva = jobvt === JOBVT_A;
  const wntvs = jobvt === JOBVT_S;
  const wntvas = wntva || wntvs;
  const wntvo = jobvt === JOBVT_O;
  const wntvn = jobvt === JOBVT_N;
  const lquery = lwork === -1;

  // Test the input arguments
  if (!(wntua || wntus || wntuo || wntun)) {
    info = -1;
  } else if (!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo)) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (lda < Math.max(1, m)) {
    info = -6;
  } else if (ldu < 1 || (wntuas && ldu < m)) {
    info = -9;
  } else if (ldvt < 1 || (wntva && ldvt < n) || (wntvs && ldvt < minmn)) {
    info = -11;
  }

  // Compute workspace
  let minwrk = 1;
  let maxwrk = 1;
  let mnthr = 0;
  let wrkbl = 0;
  let bdspac = 0;

  if (info === 0) {
    if (m >= n && minmn > 0) {
      // M >= N case
      mnthr = ilaenv(6, "DGESVD", "  ", m, n, 0, 0);
      bdspac = 5 * n;

      // Workspace sizes for sub-routines (simplified - use generous estimates)
      const lwork_dgeqrf = n + n * 32;
      const lwork_dorgqr_n = n + n * 32;
      const lwork_dorgqr_m = m + m * 32;
      const lwork_dgebrd = 3 * n + 2 * n * 32;
      const lwork_dorgbr_p = n + n * 32;
      const lwork_dorgbr_q = n + n * 32;

      if (m >= mnthr) {
        if (wntun) {
          // Path 1
          maxwrk = n + lwork_dgeqrf;
          maxwrk = Math.max(maxwrk, 3 * n + lwork_dgebrd);
          if (wntvo || wntvas)
            maxwrk = Math.max(maxwrk, 3 * n + lwork_dorgbr_p);
          maxwrk = Math.max(maxwrk, bdspac);
          minwrk = Math.max(4 * n, bdspac);
        } else if (wntuo && wntvn) {
          // Path 2
          wrkbl = n + lwork_dgeqrf;
          wrkbl = Math.max(wrkbl, n + lwork_dorgqr_n);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = Math.max(n * n + wrkbl, n * n + m * n + n);
          minwrk = Math.max(3 * n + m, bdspac);
        } else if (wntuo && wntvas) {
          // Path 3
          wrkbl = n + lwork_dgeqrf;
          wrkbl = Math.max(wrkbl, n + lwork_dorgqr_n);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = Math.max(n * n + wrkbl, n * n + m * n + n);
          minwrk = Math.max(3 * n + m, bdspac);
        } else if (wntus && wntvn) {
          // Path 4
          wrkbl = n + lwork_dgeqrf;
          wrkbl = Math.max(wrkbl, n + lwork_dorgqr_n);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = n * n + wrkbl;
          minwrk = Math.max(3 * n + m, bdspac);
        } else if (wntus && wntvo) {
          // Path 5
          wrkbl = n + lwork_dgeqrf;
          wrkbl = Math.max(wrkbl, n + lwork_dorgqr_n);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = 2 * n * n + wrkbl;
          minwrk = Math.max(3 * n + m, bdspac);
        } else if (wntus && wntvas) {
          // Path 6
          wrkbl = n + lwork_dgeqrf;
          wrkbl = Math.max(wrkbl, n + lwork_dorgqr_n);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = n * n + wrkbl;
          minwrk = Math.max(3 * n + m, bdspac);
        } else if (wntua && wntvn) {
          // Path 7
          wrkbl = n + lwork_dgeqrf;
          wrkbl = Math.max(wrkbl, n + lwork_dorgqr_m);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = n * n + wrkbl;
          minwrk = Math.max(3 * n + m, bdspac);
        } else if (wntua && wntvo) {
          // Path 8
          wrkbl = n + lwork_dgeqrf;
          wrkbl = Math.max(wrkbl, n + lwork_dorgqr_m);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = 2 * n * n + wrkbl;
          minwrk = Math.max(3 * n + m, bdspac);
        } else if (wntua && wntvas) {
          // Path 9
          wrkbl = n + lwork_dgeqrf;
          wrkbl = Math.max(wrkbl, n + lwork_dorgqr_m);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = n * n + wrkbl;
          minwrk = Math.max(3 * n + m, bdspac);
        }
      } else {
        // Path 10 (M at least N, but not much larger)
        maxwrk = 3 * n + (m + n) * 32; // generous estimate for dgebrd
        if (wntus || wntuo) {
          maxwrk = Math.max(maxwrk, 3 * n + n * 32);
        }
        if (wntua) {
          maxwrk = Math.max(maxwrk, 3 * n + m * 32);
        }
        if (!wntvn) {
          maxwrk = Math.max(maxwrk, 3 * n + n * 32);
        }
        maxwrk = Math.max(maxwrk, bdspac);
        minwrk = Math.max(3 * n + m, bdspac);
      }
    } else if (minmn > 0) {
      // M < N case
      mnthr = ilaenv(6, "DGESVD", "  ", m, n, 0, 0);
      bdspac = 5 * m;

      const lwork_dgelqf = m + m * 32;
      const lwork_dorglq_n = n + n * 32;
      const lwork_dorglq_m = m + m * 32;
      const lwork_dgebrd = 3 * m + 2 * m * 32;
      const lwork_dorgbr_p = m + m * 32;
      const lwork_dorgbr_q = m + m * 32;

      if (n >= mnthr) {
        if (wntvn) {
          // Path 1t
          maxwrk = m + lwork_dgelqf;
          maxwrk = Math.max(maxwrk, 3 * m + lwork_dgebrd);
          if (wntuo || wntuas)
            maxwrk = Math.max(maxwrk, 3 * m + lwork_dorgbr_q);
          maxwrk = Math.max(maxwrk, bdspac);
          minwrk = Math.max(4 * m, bdspac);
        } else if (wntvo && wntun) {
          // Path 2t
          wrkbl = m + lwork_dgelqf;
          wrkbl = Math.max(wrkbl, m + lwork_dorglq_m);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = Math.max(m * m + wrkbl, m * m + m * n + m);
          minwrk = Math.max(3 * m + n, bdspac);
        } else if (wntvo && wntuas) {
          // Path 3t
          wrkbl = m + lwork_dgelqf;
          wrkbl = Math.max(wrkbl, m + lwork_dorglq_m);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = Math.max(m * m + wrkbl, m * m + m * n + m);
          minwrk = Math.max(3 * m + n, bdspac);
        } else if (wntvs && wntun) {
          // Path 4t
          wrkbl = m + lwork_dgelqf;
          wrkbl = Math.max(wrkbl, m + lwork_dorglq_m);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = m * m + wrkbl;
          minwrk = Math.max(3 * m + n, bdspac);
        } else if (wntvs && wntuo) {
          // Path 5t
          wrkbl = m + lwork_dgelqf;
          wrkbl = Math.max(wrkbl, m + lwork_dorglq_m);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = 2 * m * m + wrkbl;
          minwrk = Math.max(3 * m + n, bdspac);
        } else if (wntvs && wntuas) {
          // Path 6t
          wrkbl = m + lwork_dgelqf;
          wrkbl = Math.max(wrkbl, m + lwork_dorglq_m);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = m * m + wrkbl;
          minwrk = Math.max(3 * m + n, bdspac);
        } else if (wntva && wntun) {
          // Path 7t
          wrkbl = m + lwork_dgelqf;
          wrkbl = Math.max(wrkbl, m + lwork_dorglq_n);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = m * m + wrkbl;
          minwrk = Math.max(3 * m + n, bdspac);
        } else if (wntva && wntuo) {
          // Path 8t
          wrkbl = m + lwork_dgelqf;
          wrkbl = Math.max(wrkbl, m + lwork_dorglq_n);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = 2 * m * m + wrkbl;
          minwrk = Math.max(3 * m + n, bdspac);
        } else if (wntva && wntuas) {
          // Path 9t
          wrkbl = m + lwork_dgelqf;
          wrkbl = Math.max(wrkbl, m + lwork_dorglq_n);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = Math.max(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = Math.max(wrkbl, bdspac);
          maxwrk = m * m + wrkbl;
          minwrk = Math.max(3 * m + n, bdspac);
        }
      } else {
        // Path 10t (N greater than M, but not much larger)
        maxwrk = 3 * m + (m + n) * 32;
        if (wntvs || wntvo) {
          maxwrk = Math.max(maxwrk, 3 * m + n * 32);
        }
        if (wntva) {
          maxwrk = Math.max(maxwrk, 3 * m + n * 32);
        }
        if (!wntun) {
          maxwrk = Math.max(maxwrk, 3 * m + m * 32);
        }
        maxwrk = Math.max(maxwrk, bdspac);
        minwrk = Math.max(3 * m + n, bdspac);
      }
    }
    maxwrk = Math.max(maxwrk, minwrk);
    work[workOff] = maxwrk;

    if (lwork < minwrk && !lquery) {
      info = -13;
    }
  }

  if (info !== 0) {
    xerbla("DGESVD", -info);
    return info;
  } else if (lquery) {
    return 0;
  }

  // Quick return if possible
  if (m === 0 || n === 0) {
    return 0;
  }

  // Get machine constants
  const eps = dlamch(MACH_EPS);
  const smlnum = Math.sqrt(dlamch(MACH_SFMIN)) / eps;
  const bignum = ONE / smlnum;

  // Scale A if max element outside range [SMLNUM, BIGNUM]
  const dum = allocFloat64Array(1);
  const anrm = dlange(NORM_MAX, m, n, a, aOff, lda, dum, 0);
  let iscl = 0;
  if (anrm > ZERO && anrm < smlnum) {
    iscl = 1;
    dlascl(DLASCL_G, 0, 0, anrm, smlnum, m, n, a, aOff, lda);
  } else if (anrm > bignum) {
    iscl = 1;
    dlascl(DLASCL_G, 0, 0, anrm, bignum, m, n, a, aOff, lda);
  }

  // Allocate generous internal workspace
  const iwrkSize = Math.max(maxwrk, 1);
  const w = allocFloat64Array(iwrkSize + 1); // 1-based indexing workspace
  // We'll use 0-based offset = 0 for w, and use wOff = 0
  // Fortran WORK(I) => w[I-1] with 1-based index I => w[(I-1)]
  // But we'll track offsets as 0-based into w

  let ie = 0;
  let itauq: number;
  let itaup: number;
  let itau: number;
  let iwork: number;
  let ir: number;
  let iu: number;
  let ncvt: number;
  let nru: number;
  let ncu: number;
  let nrvt: number;
  let ldwrkr: number;
  let ldwrku: number;
  let chunk: number;
  let blk: number;
  if (m >= n) {
    // A has at least as many rows as columns
    if (m >= mnthr) {
      if (wntun) {
        // Path 1 (M much larger than N, JOBU='N')
        // No left singular vectors to be computed

        itau = 0;
        iwork = itau + n;

        // Compute A=Q*R
        dgeqrf(m, n, a, aOff, lda, w, itau);

        // Zero out below R
        if (n > 1) {
          dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, a, aOff + 1, lda);
        }
        ie = 0;
        itauq = ie + n;
        itaup = itauq + n;
        iwork = itaup + n;

        // Bidiagonalize R in A
        dgebrd(
          n,
          n,
          a,
          aOff,
          lda,
          s,
          sOff,
          w,
          ie,
          w,
          itauq,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
        ncvt = 0;
        if (wntvo || wntvas) {
          // Generate P'
          dorgbr(
            VECT_P,
            n,
            n,
            n,
            a,
            aOff,
            lda,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          ncvt = n;
        }
        iwork = ie + n;

        // Perform bidiagonal QR iteration
        info = dbdsqr(
          UPPER,
          n,
          ncvt,
          0,
          0,
          s,
          sOff,
          w,
          ie,
          a,
          aOff,
          lda,
          dum,
          0,
          1,
          dum,
          0,
          1,
          w,
          iwork
        );

        // If right singular vectors desired in VT, copy them there
        if (wntvas) {
          dlacpy(FULL, n, n, a, aOff, lda, vt, vtOff, ldvt);
        }
      } else if (wntuo && wntvn) {
        // Path 2 (M much larger than N, JOBU='O', JOBVT='N')

        // Always use fast algorithm since we allocate workspace freely
        ir = 0;
        ldwrkr = n;
        ldwrku = Math.max(n, Math.min(m, Math.floor(iwrkSize / n)));
        itau = ir + ldwrkr * n;
        iwork = itau + n;

        // Compute A=Q*R
        dgeqrf(m, n, a, aOff, lda, w, itau);

        // Copy R to WORK(IR) and zero out below it
        dlacpy(UPPER, n, n, a, aOff, lda, w, ir, ldwrkr);
        dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, w, ir + 1, ldwrkr);

        // Generate Q in A
        dorgqr(m, n, n, a, aOff, lda, w, itau);
        ie = itau;
        itauq = ie + n;
        itaup = itauq + n;
        iwork = itaup + n;

        // Bidiagonalize R in WORK(IR)
        dgebrd(
          n,
          n,
          w,
          ir,
          ldwrkr,
          s,
          sOff,
          w,
          ie,
          w,
          itauq,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );

        // Generate left vectors bidiagonalizing R
        dorgbr(
          VECT_Q,
          n,
          n,
          n,
          w,
          ir,
          ldwrkr,
          w,
          itauq,
          w,
          iwork,
          iwrkSize - iwork
        );
        iwork = ie + n;

        // Perform bidiagonal QR iteration
        info = dbdsqr(
          UPPER,
          n,
          0,
          n,
          0,
          s,
          sOff,
          w,
          ie,
          dum,
          0,
          1,
          w,
          ir,
          ldwrkr,
          dum,
          0,
          1,
          w,
          iwork
        );
        iu = ie + n;

        // Multiply Q in A by left singular vectors of R in WORK(IR)
        for (let i = 0; i < m; i += ldwrku) {
          chunk = Math.min(m - i, ldwrku);
          dgemm(
            NOTRANS,
            NOTRANS,
            chunk,
            n,
            n,
            ONE,
            a,
            aOff + i,
            lda,
            w,
            ir,
            ldwrkr,
            ZERO,
            w,
            iu,
            ldwrku
          );
          dlacpy(FULL, chunk, n, w, iu, ldwrku, a, aOff + i, lda);
        }
      } else if (wntuo && wntvas) {
        // Path 3 (M much larger than N, JOBU='O', JOBVT='S' or 'A')

        ir = 0;
        ldwrkr = n;
        ldwrku = Math.max(n, Math.min(m, Math.floor(iwrkSize / n)));
        itau = ir + ldwrkr * n;
        iwork = itau + n;

        // Compute A=Q*R
        dgeqrf(m, n, a, aOff, lda, w, itau);

        // Copy R to VT, zeroing out below it
        dlacpy(UPPER, n, n, a, aOff, lda, vt, vtOff, ldvt);
        if (n > 1) {
          dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, vt, vtOff + 1, ldvt);
        }

        // Generate Q in A
        dorgqr(m, n, n, a, aOff, lda, w, itau);
        ie = itau;
        itauq = ie + n;
        itaup = itauq + n;
        iwork = itaup + n;

        // Bidiagonalize R in VT, copying result to WORK(IR)
        dgebrd(
          n,
          n,
          vt,
          vtOff,
          ldvt,
          s,
          sOff,
          w,
          ie,
          w,
          itauq,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
        dlacpy(LOWER, n, n, vt, vtOff, ldvt, w, ir, ldwrkr);

        // Generate left vectors bidiagonalizing R in WORK(IR)
        dorgbr(
          VECT_Q,
          n,
          n,
          n,
          w,
          ir,
          ldwrkr,
          w,
          itauq,
          w,
          iwork,
          iwrkSize - iwork
        );

        // Generate right vectors bidiagonalizing R in VT
        dorgbr(
          VECT_P,
          n,
          n,
          n,
          vt,
          vtOff,
          ldvt,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
        iwork = ie + n;

        // Perform bidiagonal QR iteration
        info = dbdsqr(
          UPPER,
          n,
          n,
          n,
          0,
          s,
          sOff,
          w,
          ie,
          vt,
          vtOff,
          ldvt,
          w,
          ir,
          ldwrkr,
          dum,
          0,
          1,
          w,
          iwork
        );
        iu = ie + n;

        // Multiply Q in A by left singular vectors of R in WORK(IR)
        for (let i = 0; i < m; i += ldwrku) {
          chunk = Math.min(m - i, ldwrku);
          dgemm(
            NOTRANS,
            NOTRANS,
            chunk,
            n,
            n,
            ONE,
            a,
            aOff + i,
            lda,
            w,
            ir,
            ldwrkr,
            ZERO,
            w,
            iu,
            ldwrku
          );
          dlacpy(FULL, chunk, n, w, iu, ldwrku, a, aOff + i, lda);
        }
      } else if (wntus) {
        if (wntvn) {
          // Path 4 (M much larger than N, JOBU='S', JOBVT='N')

          ir = 0;
          ldwrkr = n;
          itau = ir + ldwrkr * n;
          iwork = itau + n;

          // Compute A=Q*R
          dgeqrf(m, n, a, aOff, lda, w, itau);

          // Copy R to WORK(IR), zeroing out below it
          dlacpy(UPPER, n, n, a, aOff, lda, w, ir, ldwrkr);
          dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, w, ir + 1, ldwrkr);

          // Generate Q in A
          dorgqr(m, n, n, a, aOff, lda, w, itau);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IR)
          dgebrd(
            n,
            n,
            w,
            ir,
            ldwrkr,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate left vectors bidiagonalizing R in WORK(IR)
          dorgbr(
            VECT_Q,
            n,
            n,
            n,
            w,
            ir,
            ldwrkr,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + n;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            n,
            0,
            n,
            0,
            s,
            sOff,
            w,
            ie,
            dum,
            0,
            1,
            w,
            ir,
            ldwrkr,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply Q in A by left singular vectors of R in WORK(IR)
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            n,
            ONE,
            a,
            aOff,
            lda,
            w,
            ir,
            ldwrkr,
            ZERO,
            u,
            uOff,
            ldu
          );
        } else if (wntvo) {
          // Path 5 (M much larger than N, JOBU='S', JOBVT='O')

          iu = 0;
          ldwrku = n;
          ir = iu + ldwrku * n;
          ldwrkr = n;
          itau = ir + ldwrkr * n;
          iwork = itau + n;

          // Compute A=Q*R
          dgeqrf(m, n, a, aOff, lda, w, itau);

          // Copy R to WORK(IU), zeroing out below it
          dlacpy(UPPER, n, n, a, aOff, lda, w, iu, ldwrku);
          dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, w, iu + 1, ldwrku);

          // Generate Q in A
          dorgqr(m, n, n, a, aOff, lda, w, itau);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IU), copying result to WORK(IR)
          dgebrd(
            n,
            n,
            w,
            iu,
            ldwrku,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          dlacpy(UPPER, n, n, w, iu, ldwrku, w, ir, ldwrkr);

          // Generate left bidiagonalizing vectors in WORK(IU)
          dorgbr(
            VECT_Q,
            n,
            n,
            n,
            w,
            iu,
            ldwrku,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate right bidiagonalizing vectors in WORK(IR)
          dorgbr(
            VECT_P,
            n,
            n,
            n,
            w,
            ir,
            ldwrkr,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + n;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            n,
            n,
            n,
            0,
            s,
            sOff,
            w,
            ie,
            w,
            ir,
            ldwrkr,
            w,
            iu,
            ldwrku,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply Q in A by left singular vectors of R in WORK(IU)
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            n,
            ONE,
            a,
            aOff,
            lda,
            w,
            iu,
            ldwrku,
            ZERO,
            u,
            uOff,
            ldu
          );

          // Copy right singular vectors of R to A
          dlacpy(FULL, n, n, w, ir, ldwrkr, a, aOff, lda);
        } else if (wntvas) {
          // Path 6 (M much larger than N, JOBU='S', JOBVT='S' or 'A')

          iu = 0;
          ldwrku = n;
          itau = iu + ldwrku * n;
          iwork = itau + n;

          // Compute A=Q*R
          dgeqrf(m, n, a, aOff, lda, w, itau);

          // Copy R to WORK(IU), zeroing out below it
          dlacpy(UPPER, n, n, a, aOff, lda, w, iu, ldwrku);
          dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, w, iu + 1, ldwrku);

          // Generate Q in A
          dorgqr(m, n, n, a, aOff, lda, w, itau);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IU), copying result to VT
          dgebrd(
            n,
            n,
            w,
            iu,
            ldwrku,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          dlacpy(UPPER, n, n, w, iu, ldwrku, vt, vtOff, ldvt);

          // Generate left bidiagonalizing vectors in WORK(IU)
          dorgbr(
            VECT_Q,
            n,
            n,
            n,
            w,
            iu,
            ldwrku,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate right bidiagonalizing vectors in VT
          dorgbr(
            VECT_P,
            n,
            n,
            n,
            vt,
            vtOff,
            ldvt,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + n;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            n,
            n,
            n,
            0,
            s,
            sOff,
            w,
            ie,
            vt,
            vtOff,
            ldvt,
            w,
            iu,
            ldwrku,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply Q in A by left singular vectors of R in WORK(IU)
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            n,
            ONE,
            a,
            aOff,
            lda,
            w,
            iu,
            ldwrku,
            ZERO,
            u,
            uOff,
            ldu
          );
        }
      } else if (wntua) {
        if (wntvn) {
          // Path 7 (M much larger than N, JOBU='A', JOBVT='N')

          ir = 0;
          ldwrkr = n;
          itau = ir + ldwrkr * n;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          dgeqrf(m, n, a, aOff, lda, w, itau);
          dlacpy(LOWER, m, n, a, aOff, lda, u, uOff, ldu);

          // Copy R to WORK(IR), zeroing out below it
          dlacpy(UPPER, n, n, a, aOff, lda, w, ir, ldwrkr);
          dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, w, ir + 1, ldwrkr);

          // Generate Q in U
          dorgqr(m, m, n, u, uOff, ldu, w, itau);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IR)
          dgebrd(
            n,
            n,
            w,
            ir,
            ldwrkr,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate left bidiagonalizing vectors in WORK(IR)
          dorgbr(
            VECT_Q,
            n,
            n,
            n,
            w,
            ir,
            ldwrkr,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + n;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            n,
            0,
            n,
            0,
            s,
            sOff,
            w,
            ie,
            dum,
            0,
            1,
            w,
            ir,
            ldwrkr,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply Q in U by left singular vectors of R in WORK(IR), storing in A
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            n,
            ONE,
            u,
            uOff,
            ldu,
            w,
            ir,
            ldwrkr,
            ZERO,
            a,
            aOff,
            lda
          );

          // Copy left singular vectors of A from A to U
          dlacpy(FULL, m, n, a, aOff, lda, u, uOff, ldu);
        } else if (wntvo) {
          // Path 8 (M much larger than N, JOBU='A', JOBVT='O')

          iu = 0;
          ldwrku = n;
          ir = iu + ldwrku * n;
          ldwrkr = n;
          itau = ir + ldwrkr * n;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          dgeqrf(m, n, a, aOff, lda, w, itau);
          dlacpy(LOWER, m, n, a, aOff, lda, u, uOff, ldu);

          // Generate Q in U
          dorgqr(m, m, n, u, uOff, ldu, w, itau);

          // Copy R to WORK(IU), zeroing out below it
          dlacpy(UPPER, n, n, a, aOff, lda, w, iu, ldwrku);
          dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, w, iu + 1, ldwrku);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IU), copying result to WORK(IR)
          dgebrd(
            n,
            n,
            w,
            iu,
            ldwrku,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          dlacpy(UPPER, n, n, w, iu, ldwrku, w, ir, ldwrkr);

          // Generate left bidiagonalizing vectors in WORK(IU)
          dorgbr(
            VECT_Q,
            n,
            n,
            n,
            w,
            iu,
            ldwrku,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate right bidiagonalizing vectors in WORK(IR)
          dorgbr(
            VECT_P,
            n,
            n,
            n,
            w,
            ir,
            ldwrkr,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + n;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            n,
            n,
            n,
            0,
            s,
            sOff,
            w,
            ie,
            w,
            ir,
            ldwrkr,
            w,
            iu,
            ldwrku,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply Q in U by left singular vectors of R in WORK(IU), storing in A
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            n,
            ONE,
            u,
            uOff,
            ldu,
            w,
            iu,
            ldwrku,
            ZERO,
            a,
            aOff,
            lda
          );

          // Copy left singular vectors of A from A to U
          dlacpy(FULL, m, n, a, aOff, lda, u, uOff, ldu);

          // Copy right singular vectors of R from WORK(IR) to A
          dlacpy(FULL, n, n, w, ir, ldwrkr, a, aOff, lda);
        } else if (wntvas) {
          // Path 9 (M much larger than N, JOBU='A', JOBVT='S' or 'A')

          iu = 0;
          ldwrku = n;
          itau = iu + ldwrku * n;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          dgeqrf(m, n, a, aOff, lda, w, itau);
          dlacpy(LOWER, m, n, a, aOff, lda, u, uOff, ldu);

          // Generate Q in U
          dorgqr(m, m, n, u, uOff, ldu, w, itau);

          // Copy R to WORK(IU), zeroing out below it
          dlacpy(UPPER, n, n, a, aOff, lda, w, iu, ldwrku);
          dlaset(LOWER, n - 1, n - 1, ZERO, ZERO, w, iu + 1, ldwrku);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IU), copying result to VT
          dgebrd(
            n,
            n,
            w,
            iu,
            ldwrku,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          dlacpy(UPPER, n, n, w, iu, ldwrku, vt, vtOff, ldvt);

          // Generate left bidiagonalizing vectors in WORK(IU)
          dorgbr(
            VECT_Q,
            n,
            n,
            n,
            w,
            iu,
            ldwrku,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate right bidiagonalizing vectors in VT
          dorgbr(
            VECT_P,
            n,
            n,
            n,
            vt,
            vtOff,
            ldvt,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + n;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            n,
            n,
            n,
            0,
            s,
            sOff,
            w,
            ie,
            vt,
            vtOff,
            ldvt,
            w,
            iu,
            ldwrku,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply Q in U by left singular vectors of R in WORK(IU), storing in A
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            n,
            ONE,
            u,
            uOff,
            ldu,
            w,
            iu,
            ldwrku,
            ZERO,
            a,
            aOff,
            lda
          );

          // Copy left singular vectors of A from A to U
          dlacpy(FULL, m, n, a, aOff, lda, u, uOff, ldu);
        }
      }
    } else {
      // Path 10 (M at least N, but not much larger)
      // Reduce to bidiagonal form without QR decomposition

      ie = 0;
      itauq = ie + n;
      itaup = itauq + n;
      iwork = itaup + n;

      // Bidiagonalize A
      dgebrd(
        m,
        n,
        a,
        aOff,
        lda,
        s,
        sOff,
        w,
        ie,
        w,
        itauq,
        w,
        itaup,
        w,
        iwork,
        iwrkSize - iwork
      );
      if (wntuas) {
        // Copy result to U and generate left bidiagonalizing vectors in U
        dlacpy(LOWER, m, n, a, aOff, lda, u, uOff, ldu);
        ncu = wntus ? n : m;
        dorgbr(
          VECT_Q,
          m,
          ncu,
          n,
          u,
          uOff,
          ldu,
          w,
          itauq,
          w,
          iwork,
          iwrkSize - iwork
        );
      }
      if (wntvas) {
        // Copy result to VT and generate right bidiagonalizing vectors in VT
        dlacpy(UPPER, n, n, a, aOff, lda, vt, vtOff, ldvt);
        dorgbr(
          VECT_P,
          n,
          n,
          n,
          vt,
          vtOff,
          ldvt,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
      }
      if (wntuo) {
        // Generate left bidiagonalizing vectors in A
        dorgbr(
          VECT_Q,
          m,
          n,
          n,
          a,
          aOff,
          lda,
          w,
          itauq,
          w,
          iwork,
          iwrkSize - iwork
        );
      }
      if (wntvo) {
        // Generate right bidiagonalizing vectors in A
        dorgbr(
          VECT_P,
          n,
          n,
          n,
          a,
          aOff,
          lda,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
      }
      iwork = ie + n;

      nru = 0;
      if (wntuas || wntuo) nru = m;
      if (wntun) nru = 0;
      ncvt = 0;
      if (wntvas || wntvo) ncvt = n;
      if (wntvn) ncvt = 0;

      if (!wntuo && !wntvo) {
        // Left sing. vectors in U, right sing. vectors in VT
        info = dbdsqr(
          UPPER,
          n,
          ncvt,
          nru,
          0,
          s,
          sOff,
          w,
          ie,
          vt,
          vtOff,
          ldvt,
          u,
          uOff,
          ldu,
          dum,
          0,
          1,
          w,
          iwork
        );
      } else if (!wntuo && wntvo) {
        // Left sing. vectors in U, right sing. vectors in A
        info = dbdsqr(
          UPPER,
          n,
          ncvt,
          nru,
          0,
          s,
          sOff,
          w,
          ie,
          a,
          aOff,
          lda,
          u,
          uOff,
          ldu,
          dum,
          0,
          1,
          w,
          iwork
        );
      } else {
        // Left sing. vectors in A, right sing. vectors in VT
        info = dbdsqr(
          UPPER,
          n,
          ncvt,
          nru,
          0,
          s,
          sOff,
          w,
          ie,
          vt,
          vtOff,
          ldvt,
          a,
          aOff,
          lda,
          dum,
          0,
          1,
          w,
          iwork
        );
      }
    }
  } else {
    // M < N
    // A has more columns than rows

    if (n >= mnthr) {
      if (wntvn) {
        // Path 1t (N much larger than M, JOBVT='N')

        itau = 0;
        iwork = itau + m;

        // Compute A=L*Q
        dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);

        // Zero out above L
        if (m > 1) {
          dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, a, aOff + lda, lda);
        }
        ie = 0;
        itauq = ie + m;
        itaup = itauq + m;
        iwork = itaup + m;

        // Bidiagonalize L in A
        dgebrd(
          m,
          m,
          a,
          aOff,
          lda,
          s,
          sOff,
          w,
          ie,
          w,
          itauq,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
        if (wntuo || wntuas) {
          // Generate Q
          dorgbr(
            VECT_Q,
            m,
            m,
            m,
            a,
            aOff,
            lda,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );
        }
        iwork = ie + m;
        nru = 0;
        if (wntuo || wntuas) nru = m;

        // Perform bidiagonal QR iteration
        info = dbdsqr(
          UPPER,
          m,
          0,
          nru,
          0,
          s,
          sOff,
          w,
          ie,
          dum,
          0,
          1,
          a,
          aOff,
          lda,
          dum,
          0,
          1,
          w,
          iwork
        );

        // If left singular vectors desired in U, copy them there
        if (wntuas) {
          dlacpy(FULL, m, m, a, aOff, lda, u, uOff, ldu);
        }
      } else if (wntvo && wntun) {
        // Path 2t (N much larger than M, JOBU='N', JOBVT='O')

        ir = 0;
        ldwrkr = m;
        ldwrku = m;
        chunk = Math.max(m, Math.min(n, Math.floor(iwrkSize / m)));
        itau = ir + ldwrkr * m;
        iwork = itau + m;

        // Compute A=L*Q
        dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);

        // Copy L to WORK(IR) and zero out above it
        dlacpy(LOWER, m, m, a, aOff, lda, w, ir, ldwrkr);
        dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, w, ir + ldwrkr, ldwrkr);

        // Generate Q in A
        dorglq(m, n, m, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);
        ie = itau;
        itauq = ie + m;
        itaup = itauq + m;
        iwork = itaup + m;

        // Bidiagonalize L in WORK(IR)
        dgebrd(
          m,
          m,
          w,
          ir,
          ldwrkr,
          s,
          sOff,
          w,
          ie,
          w,
          itauq,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );

        // Generate right vectors bidiagonalizing L
        dorgbr(
          VECT_P,
          m,
          m,
          m,
          w,
          ir,
          ldwrkr,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
        iwork = ie + m;

        // Perform bidiagonal QR iteration
        info = dbdsqr(
          UPPER,
          m,
          m,
          0,
          0,
          s,
          sOff,
          w,
          ie,
          w,
          ir,
          ldwrkr,
          dum,
          0,
          1,
          dum,
          0,
          1,
          w,
          iwork
        );
        iu = ie + m;

        // Multiply right singular vectors of L in WORK(IR) by Q in A
        for (let i = 0; i < n; i += chunk) {
          blk = Math.min(n - i, chunk);
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            blk,
            m,
            ONE,
            w,
            ir,
            ldwrkr,
            a,
            aOff + i * lda,
            lda,
            ZERO,
            w,
            iu,
            ldwrku
          );
          dlacpy(FULL, m, blk, w, iu, ldwrku, a, aOff + i * lda, lda);
        }
      } else if (wntvo && wntuas) {
        // Path 3t (N much larger than M, JOBU='S' or 'A', JOBVT='O')

        ir = 0;
        ldwrkr = m;
        ldwrku = m;
        chunk = Math.max(m, Math.min(n, Math.floor(iwrkSize / m)));
        itau = ir + ldwrkr * m;
        iwork = itau + m;

        // Compute A=L*Q
        dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);

        // Copy L to U, zeroing out above it
        dlacpy(LOWER, m, m, a, aOff, lda, u, uOff, ldu);
        if (m > 1) {
          dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, u, uOff + ldu, ldu);
        }

        // Generate Q in A
        dorglq(m, n, m, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);
        ie = itau;
        itauq = ie + m;
        itaup = itauq + m;
        iwork = itaup + m;

        // Bidiagonalize L in U, copying result to WORK(IR)
        dgebrd(
          m,
          m,
          u,
          uOff,
          ldu,
          s,
          sOff,
          w,
          ie,
          w,
          itauq,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
        dlacpy(UPPER, m, m, u, uOff, ldu, w, ir, ldwrkr);

        // Generate right vectors bidiagonalizing L in WORK(IR)
        dorgbr(
          VECT_P,
          m,
          m,
          m,
          w,
          ir,
          ldwrkr,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );

        // Generate left vectors bidiagonalizing L in U
        dorgbr(
          VECT_Q,
          m,
          m,
          m,
          u,
          uOff,
          ldu,
          w,
          itauq,
          w,
          iwork,
          iwrkSize - iwork
        );
        iwork = ie + m;

        // Perform bidiagonal QR iteration
        info = dbdsqr(
          UPPER,
          m,
          m,
          m,
          0,
          s,
          sOff,
          w,
          ie,
          w,
          ir,
          ldwrkr,
          u,
          uOff,
          ldu,
          dum,
          0,
          1,
          w,
          iwork
        );
        iu = ie + m;

        // Multiply right singular vectors of L in WORK(IR) by Q in A
        for (let i = 0; i < n; i += chunk) {
          blk = Math.min(n - i, chunk);
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            blk,
            m,
            ONE,
            w,
            ir,
            ldwrkr,
            a,
            aOff + i * lda,
            lda,
            ZERO,
            w,
            iu,
            ldwrku
          );
          dlacpy(FULL, m, blk, w, iu, ldwrku, a, aOff + i * lda, lda);
        }
      } else if (wntvs) {
        if (wntun) {
          // Path 4t (N much larger than M, JOBU='N', JOBVT='S')

          ir = 0;
          ldwrkr = m;
          itau = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q
          dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);

          // Copy L to WORK(IR), zeroing out above it
          dlacpy(LOWER, m, m, a, aOff, lda, w, ir, ldwrkr);
          dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, w, ir + ldwrkr, ldwrkr);

          // Generate Q in A
          dorglq(m, n, m, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IR)
          dgebrd(
            m,
            m,
            w,
            ir,
            ldwrkr,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate right vectors bidiagonalizing L in WORK(IR)
          dorgbr(
            VECT_P,
            m,
            m,
            m,
            w,
            ir,
            ldwrkr,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + m;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            m,
            m,
            0,
            0,
            s,
            sOff,
            w,
            ie,
            w,
            ir,
            ldwrkr,
            dum,
            0,
            1,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply right singular vectors of L in WORK(IR) by Q in A, storing in VT
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            m,
            ONE,
            w,
            ir,
            ldwrkr,
            a,
            aOff,
            lda,
            ZERO,
            vt,
            vtOff,
            ldvt
          );
        } else if (wntuo) {
          // Path 5t (N much larger than M, JOBU='O', JOBVT='S')

          iu = 0;
          ldwrku = m;
          ir = iu + ldwrku * m;
          ldwrkr = m;
          itau = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q
          dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);

          // Copy L to WORK(IU), zeroing out above it
          dlacpy(LOWER, m, m, a, aOff, lda, w, iu, ldwrku);
          dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, w, iu + ldwrku, ldwrku);

          // Generate Q in A
          dorglq(m, n, m, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IU), copying result to WORK(IR)
          dgebrd(
            m,
            m,
            w,
            iu,
            ldwrku,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          dlacpy(LOWER, m, m, w, iu, ldwrku, w, ir, ldwrkr);

          // Generate right bidiagonalizing vectors in WORK(IU)
          dorgbr(
            VECT_P,
            m,
            m,
            m,
            w,
            iu,
            ldwrku,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate left bidiagonalizing vectors in WORK(IR)
          dorgbr(
            VECT_Q,
            m,
            m,
            m,
            w,
            ir,
            ldwrkr,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + m;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            m,
            m,
            m,
            0,
            s,
            sOff,
            w,
            ie,
            w,
            iu,
            ldwrku,
            w,
            ir,
            ldwrkr,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply right singular vectors of L in WORK(IU) by Q in A, storing in VT
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            m,
            ONE,
            w,
            iu,
            ldwrku,
            a,
            aOff,
            lda,
            ZERO,
            vt,
            vtOff,
            ldvt
          );

          // Copy left singular vectors of L to A
          dlacpy(FULL, m, m, w, ir, ldwrkr, a, aOff, lda);
        } else if (wntuas) {
          // Path 6t (N much larger than M, JOBU='S' or 'A', JOBVT='S')

          iu = 0;
          ldwrku = m;
          itau = iu + ldwrku * m;
          iwork = itau + m;

          // Compute A=L*Q
          dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);

          // Copy L to WORK(IU), zeroing out above it
          dlacpy(LOWER, m, m, a, aOff, lda, w, iu, ldwrku);
          dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, w, iu + ldwrku, ldwrku);

          // Generate Q in A
          dorglq(m, n, m, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IU), copying result to U
          dgebrd(
            m,
            m,
            w,
            iu,
            ldwrku,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          dlacpy(LOWER, m, m, w, iu, ldwrku, u, uOff, ldu);

          // Generate right bidiagonalizing vectors in WORK(IU)
          dorgbr(
            VECT_P,
            m,
            m,
            m,
            w,
            iu,
            ldwrku,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate left bidiagonalizing vectors in U
          dorgbr(
            VECT_Q,
            m,
            m,
            m,
            u,
            uOff,
            ldu,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + m;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            m,
            m,
            m,
            0,
            s,
            sOff,
            w,
            ie,
            w,
            iu,
            ldwrku,
            u,
            uOff,
            ldu,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply right singular vectors of L in WORK(IU) by Q in A, storing in VT
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            m,
            ONE,
            w,
            iu,
            ldwrku,
            a,
            aOff,
            lda,
            ZERO,
            vt,
            vtOff,
            ldvt
          );
        }
      } else if (wntva) {
        if (wntun) {
          // Path 7t (N much larger than M, JOBU='N', JOBVT='A')

          ir = 0;
          ldwrkr = m;
          itau = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);
          dlacpy(UPPER, m, n, a, aOff, lda, vt, vtOff, ldvt);

          // Copy L to WORK(IR), zeroing out above it
          dlacpy(LOWER, m, m, a, aOff, lda, w, ir, ldwrkr);
          dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, w, ir + ldwrkr, ldwrkr);

          // Generate Q in VT
          dorglq(n, n, m, vt, vtOff, ldvt, w, itau, w, iwork, iwrkSize - iwork);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IR)
          dgebrd(
            m,
            m,
            w,
            ir,
            ldwrkr,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate right bidiagonalizing vectors in WORK(IR)
          dorgbr(
            VECT_P,
            m,
            m,
            m,
            w,
            ir,
            ldwrkr,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + m;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            m,
            m,
            0,
            0,
            s,
            sOff,
            w,
            ie,
            w,
            ir,
            ldwrkr,
            dum,
            0,
            1,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply right singular vectors of L in WORK(IR) by Q in VT, storing in A
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            m,
            ONE,
            w,
            ir,
            ldwrkr,
            vt,
            vtOff,
            ldvt,
            ZERO,
            a,
            aOff,
            lda
          );

          // Copy right singular vectors of A from A to VT
          dlacpy(FULL, m, n, a, aOff, lda, vt, vtOff, ldvt);
        } else if (wntuo) {
          // Path 8t (N much larger than M, JOBU='O', JOBVT='A')

          iu = 0;
          ldwrku = m;
          ir = iu + ldwrku * m;
          ldwrkr = m;
          itau = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);
          dlacpy(UPPER, m, n, a, aOff, lda, vt, vtOff, ldvt);

          // Generate Q in VT
          dorglq(n, n, m, vt, vtOff, ldvt, w, itau, w, iwork, iwrkSize - iwork);

          // Copy L to WORK(IU), zeroing out above it
          dlacpy(LOWER, m, m, a, aOff, lda, w, iu, ldwrku);
          dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, w, iu + ldwrku, ldwrku);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IU), copying result to WORK(IR)
          dgebrd(
            m,
            m,
            w,
            iu,
            ldwrku,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          dlacpy(LOWER, m, m, w, iu, ldwrku, w, ir, ldwrkr);

          // Generate right bidiagonalizing vectors in WORK(IU)
          dorgbr(
            VECT_P,
            m,
            m,
            m,
            w,
            iu,
            ldwrku,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate left bidiagonalizing vectors in WORK(IR)
          dorgbr(
            VECT_Q,
            m,
            m,
            m,
            w,
            ir,
            ldwrkr,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + m;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            m,
            m,
            m,
            0,
            s,
            sOff,
            w,
            ie,
            w,
            iu,
            ldwrku,
            w,
            ir,
            ldwrkr,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply right singular vectors of L in WORK(IU) by Q in VT, storing in A
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            m,
            ONE,
            w,
            iu,
            ldwrku,
            vt,
            vtOff,
            ldvt,
            ZERO,
            a,
            aOff,
            lda
          );

          // Copy right singular vectors of A from A to VT
          dlacpy(FULL, m, n, a, aOff, lda, vt, vtOff, ldvt);

          // Copy left singular vectors of A from WORK(IR) to A
          dlacpy(FULL, m, m, w, ir, ldwrkr, a, aOff, lda);
        } else if (wntuas) {
          // Path 9t (N much larger than M, JOBU='S' or 'A', JOBVT='A')

          iu = 0;
          ldwrku = m;
          itau = iu + ldwrku * m;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          dgelqf(m, n, a, aOff, lda, w, itau, w, iwork, iwrkSize - iwork);
          dlacpy(UPPER, m, n, a, aOff, lda, vt, vtOff, ldvt);

          // Generate Q in VT
          dorglq(n, n, m, vt, vtOff, ldvt, w, itau, w, iwork, iwrkSize - iwork);

          // Copy L to WORK(IU), zeroing out above it
          dlacpy(LOWER, m, m, a, aOff, lda, w, iu, ldwrku);
          dlaset(UPPER, m - 1, m - 1, ZERO, ZERO, w, iu + ldwrku, ldwrku);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IU), copying result to U
          dgebrd(
            m,
            m,
            w,
            iu,
            ldwrku,
            s,
            sOff,
            w,
            ie,
            w,
            itauq,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );
          dlacpy(LOWER, m, m, w, iu, ldwrku, u, uOff, ldu);

          // Generate right bidiagonalizing vectors in WORK(IU)
          dorgbr(
            VECT_P,
            m,
            m,
            m,
            w,
            iu,
            ldwrku,
            w,
            itaup,
            w,
            iwork,
            iwrkSize - iwork
          );

          // Generate left bidiagonalizing vectors in U
          dorgbr(
            VECT_Q,
            m,
            m,
            m,
            u,
            uOff,
            ldu,
            w,
            itauq,
            w,
            iwork,
            iwrkSize - iwork
          );
          iwork = ie + m;

          // Perform bidiagonal QR iteration
          info = dbdsqr(
            UPPER,
            m,
            m,
            m,
            0,
            s,
            sOff,
            w,
            ie,
            w,
            iu,
            ldwrku,
            u,
            uOff,
            ldu,
            dum,
            0,
            1,
            w,
            iwork
          );

          // Multiply right singular vectors of L in WORK(IU) by Q in VT, storing in A
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n,
            m,
            ONE,
            w,
            iu,
            ldwrku,
            vt,
            vtOff,
            ldvt,
            ZERO,
            a,
            aOff,
            lda
          );

          // Copy right singular vectors of A from A to VT
          dlacpy(FULL, m, n, a, aOff, lda, vt, vtOff, ldvt);
        }
      }
    } else {
      // Path 10t (N greater than M, but not much larger)
      // Reduce to bidiagonal form without LQ decomposition

      ie = 0;
      itauq = ie + m;
      itaup = itauq + m;
      iwork = itaup + m;

      // Bidiagonalize A
      dgebrd(
        m,
        n,
        a,
        aOff,
        lda,
        s,
        sOff,
        w,
        ie,
        w,
        itauq,
        w,
        itaup,
        w,
        iwork,
        iwrkSize - iwork
      );
      if (wntuas) {
        // Copy result to U and generate left bidiagonalizing vectors in U
        dlacpy(LOWER, m, m, a, aOff, lda, u, uOff, ldu);
        dorgbr(
          VECT_Q,
          m,
          m,
          n,
          u,
          uOff,
          ldu,
          w,
          itauq,
          w,
          iwork,
          iwrkSize - iwork
        );
      }
      if (wntvas) {
        // Copy result to VT and generate right bidiagonalizing vectors in VT
        dlacpy(UPPER, m, n, a, aOff, lda, vt, vtOff, ldvt);
        nrvt = wntva ? n : m;
        dorgbr(
          VECT_P,
          nrvt,
          n,
          m,
          vt,
          vtOff,
          ldvt,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
      }
      if (wntuo) {
        // Generate left bidiagonalizing vectors in A
        dorgbr(
          VECT_Q,
          m,
          m,
          n,
          a,
          aOff,
          lda,
          w,
          itauq,
          w,
          iwork,
          iwrkSize - iwork
        );
      }
      if (wntvo) {
        // Generate right bidiagonalizing vectors in A
        dorgbr(
          VECT_P,
          m,
          n,
          m,
          a,
          aOff,
          lda,
          w,
          itaup,
          w,
          iwork,
          iwrkSize - iwork
        );
      }
      iwork = ie + m;

      nru = 0;
      if (wntuas || wntuo) nru = m;
      if (wntun) nru = 0;
      ncvt = 0;
      if (wntvas || wntvo) ncvt = n;
      if (wntvn) ncvt = 0;

      if (!wntuo && !wntvo) {
        // Left sing. vectors in U, right sing. vectors in VT
        info = dbdsqr(
          LOWER,
          m,
          ncvt,
          nru,
          0,
          s,
          sOff,
          w,
          ie,
          vt,
          vtOff,
          ldvt,
          u,
          uOff,
          ldu,
          dum,
          0,
          1,
          w,
          iwork
        );
      } else if (!wntuo && wntvo) {
        // Left sing. vectors in U, right sing. vectors in A
        info = dbdsqr(
          LOWER,
          m,
          ncvt,
          nru,
          0,
          s,
          sOff,
          w,
          ie,
          a,
          aOff,
          lda,
          u,
          uOff,
          ldu,
          dum,
          0,
          1,
          w,
          iwork
        );
      } else {
        // Left sing. vectors in A, right sing. vectors in VT
        info = dbdsqr(
          LOWER,
          m,
          ncvt,
          nru,
          0,
          s,
          sOff,
          w,
          ie,
          vt,
          vtOff,
          ldvt,
          a,
          aOff,
          lda,
          dum,
          0,
          1,
          w,
          iwork
        );
      }
    }
  }

  // If DBDSQR failed to converge, copy unconverged superdiagonals to WORK(2:MINMN)
  if (info !== 0) {
    if (ie > 1) {
      for (let i = 0; i < minmn - 1; i++) {
        work[workOff + i + 1] = w[i + ie];
      }
    }
    if (ie < 1) {
      for (let i = minmn - 2; i >= 0; i--) {
        work[workOff + i + 1] = w[i + ie];
      }
    }
  }

  // Undo scaling if necessary
  if (iscl === 1) {
    if (anrm > bignum) {
      dlascl(DLASCL_G, 0, 0, bignum, anrm, minmn, 1, s, sOff, minmn);
    }
    if (info !== 0 && anrm > bignum) {
      dlascl(
        DLASCL_G,
        0,
        0,
        bignum,
        anrm,
        minmn - 1,
        1,
        work,
        workOff + 1,
        minmn
      );
    }
    if (anrm < smlnum) {
      dlascl(DLASCL_G, 0, 0, smlnum, anrm, minmn, 1, s, sOff, minmn);
    }
    if (info !== 0 && anrm < smlnum) {
      dlascl(
        DLASCL_G,
        0,
        0,
        smlnum,
        anrm,
        minmn - 1,
        1,
        work,
        workOff + 1,
        minmn
      );
    }
  }

  // Return optimal workspace in WORK(1)
  work[workOff] = maxwrk;

  return info;
}
