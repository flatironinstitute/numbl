// Translated from SRC/dhseqr.f
// DHSEQR computes the eigenvalues of a Hessenberg matrix H and,
// optionally, the matrices T and Z from the Schur decomposition
// H = Z T Z**T, where T is an upper quasi-triangular matrix (the
// Schur form), and Z is the orthogonal matrix of Schur vectors.
//
// Array indexing convention (column-major, matching Fortran):
//   H(I,J) => h[hOff + (I-1) + (J-1)*ldh]   (I,J are 1-based)
//   Z(I,J) => z[zOff + (I-1) + (J-1)*ldz]
//   WR(I)  => wr[wrOff + (I-1)]
//   WI(I)  => wi[wiOff + (I-1)]

import { dlahqr } from "./dlahqr.js";
import { dlaqr0 } from "./dlaqr30.js";
import { dlacpy } from "./dlacpy.js";
import { dlaset } from "./dlaset.js";
import { ilaenv } from "../utils/ilaenv.js";
import { LOWER } from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

// "ALL" uplo constant for dlacpy/dlaset (anything other than UPPER/LOWER)
const ALL = -1;

/**
 * DHSEQR computes the eigenvalues of a Hessenberg matrix H and,
 * optionally, the matrices T and Z from the Schur decomposition
 * H = Z T Z**T, where T is an upper quasi-triangular matrix (the
 * Schur form), and Z is the orthogonal matrix of Schur vectors.
 *
 * Optionally Z may be postmultiplied into an input orthogonal
 * matrix Q so that this routine can give the Schur factorization
 * of a matrix A which has been reduced to the Hessenberg form H
 * by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.
 *
 * @param job - 0='E' (eigenvalues only), 1='S' (Schur form)
 * @param compz - 0='N' (no Schur vectors), 1='I' (initialize Z to identity), 2='V' (input Z given)
 * @param n - order of the matrix H (>= 0)
 * @param ilo - 1-based lower index of balanced submatrix
 * @param ihi - 1-based upper index of balanced submatrix
 * @param h - Hessenberg matrix, dimension (ldh, n)
 * @param hOff - offset into h
 * @param ldh - leading dimension of h
 * @param wr - real parts of eigenvalues, dimension (n)
 * @param wrOff - offset into wr
 * @param wi - imaginary parts of eigenvalues, dimension (n)
 * @param wiOff - offset into wi
 * @param z - Schur vectors matrix, dimension (ldz, n)
 * @param zOff - offset into z
 * @param ldz - leading dimension of z
 * @param work - workspace array, dimension (lwork)
 * @param workOff - offset into work
 * @param lwork - dimension of workspace (-1 for workspace query)
 * @returns INFO: 0 = success, < 0 = invalid argument, > 0 = convergence failure
 */
export function dhseqr(
  job: number,
  compz: number,
  n: number,
  ilo: number,
  ihi: number,
  h: Float64Array,
  hOff: number,
  ldh: number,
  wr: Float64Array,
  wrOff: number,
  wi: Float64Array,
  wiOff: number,
  z: Float64Array,
  zOff: number,
  ldz: number,
  work: Float64Array,
  workOff: number,
  lwork: number
): number {
  const ZERO = 0.0;
  const ONE = 1.0;

  // ==== Matrices of order NTINY or smaller must be processed by
  //      DLAHQR because of insufficient subdiagonal scratch space.
  //      (This is a hard limit.) ====
  const NTINY = 15;

  // ==== NL allocates some local workspace to help small matrices
  //      through a rare DLAHQR failure.  NL > NTINY = 15 is
  //      required and NL <= NMIN = ILAENV(ISPEC=12,...) is
  //      recommended.  Using NL = 49 allows up to six simultaneous
  //      shifts and a 16-by-16 deflation window. ====
  const NL = 49;

  // Helpers for column-major indexing (1-based I,J)
  const H = (i: number, j: number) => hOff + (i - 1) + (j - 1) * ldh;

  // ==== Decode and check the input parameters. ====

  // job: 0='E' (eigenvalues only), 1='S' (Schur form)
  const wantt = job === 1;
  // compz: 0='N', 1='I', 2='V'
  const initz = compz === 1;
  const wantz = initz || compz === 2;
  const lquery = lwork === -1;

  work[workOff] = Math.max(1, n);

  let info = 0;
  if (job !== 0 && job !== 1) {
    info = -1;
  } else if (compz !== 0 && compz !== 1 && compz !== 2) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ilo < 1 || ilo > Math.max(1, n)) {
    info = -4;
  } else if (ihi < Math.min(ilo, n) || ihi > n) {
    info = -5;
  } else if (ldh < Math.max(1, n)) {
    info = -7;
  } else if (ldz < 1 || (wantz && ldz < Math.max(1, n))) {
    info = -11;
  } else if (lwork < Math.max(1, n) && !lquery) {
    info = -13;
  }

  if (info !== 0) {
    // Quick return in case of invalid argument.
    return info;
  } else if (n === 0) {
    // Quick return in case N = 0; nothing to do.
    return 0;
  } else if (lquery) {
    // Quick return in case of a workspace query.
    // Call dlaqr0 to get its workspace requirement.
    dlaqr0(
      wantt,
      wantz,
      n,
      ilo,
      ihi,
      h,
      hOff,
      ldh,
      wr,
      wrOff,
      wi,
      wiOff,
      ilo,
      ihi,
      z,
      zOff,
      ldz,
      work,
      workOff,
      lwork
    );
    // Ensure reported workspace size is backward-compatible with
    // previous LAPACK versions.
    work[workOff] = Math.max(Math.max(1, n), work[workOff]);
    return 0;
  } else {
    // ==== Copy eigenvalues isolated by DGEBAL ====

    for (let i = 1; i <= ilo - 1; i++) {
      wr[wrOff + (i - 1)] = h[H(i, i)];
      wi[wiOff + (i - 1)] = ZERO;
    }
    for (let i = ihi + 1; i <= n; i++) {
      wr[wrOff + (i - 1)] = h[H(i, i)];
      wi[wiOff + (i - 1)] = ZERO;
    }

    // ==== Initialize Z, if requested ====

    if (initz) {
      dlaset(ALL, n, n, ZERO, ONE, z, zOff, ldz);
    }

    // ==== Quick return if possible ====

    if (ilo === ihi) {
      wr[wrOff + (ilo - 1)] = h[H(ilo, ilo)];
      wi[wiOff + (ilo - 1)] = ZERO;
      return 0;
    }

    // ==== DLAHQR/DLAQR0 crossover point ====

    let nmin = ilaenv(12, "DHSEQR", "EN", n, ilo, ihi, lwork);
    nmin = Math.max(NTINY, nmin);

    // ==== DLAQR0 for big matrices; DLAHQR for small ones ====

    if (n > nmin) {
      info = dlaqr0(
        wantt,
        wantz,
        n,
        ilo,
        ihi,
        h,
        hOff,
        ldh,
        wr,
        wrOff,
        wi,
        wiOff,
        ilo,
        ihi,
        z,
        zOff,
        ldz,
        work,
        workOff,
        lwork
      );
    } else {
      // ==== Small matrix ====

      info = dlahqr(
        wantt,
        wantz,
        n,
        ilo,
        ihi,
        h,
        hOff,
        ldh,
        wr,
        wrOff,
        wi,
        wiOff,
        ilo,
        ihi,
        z,
        zOff,
        ldz
      );

      if (info > 0) {
        // ==== A rare DLAHQR failure!  DLAQR0 sometimes succeeds
        //      when DLAHQR fails. ====

        const kbot = info;

        if (n >= NL) {
          // ==== Larger matrices have enough subdiagonal scratch
          //      space to call DLAQR0 directly. ====

          info = dlaqr0(
            wantt,
            wantz,
            n,
            ilo,
            kbot,
            h,
            hOff,
            ldh,
            wr,
            wrOff,
            wi,
            wiOff,
            ilo,
            ihi,
            z,
            zOff,
            ldz,
            work,
            workOff,
            lwork
          );
        } else {
          // ==== Tiny matrices don't have enough subdiagonal
          //      scratch space to benefit from DLAQR0.  Hence,
          //      tiny matrices must be copied into a larger
          //      array before calling DLAQR0. ====

          const hl = allocFloat64Array(NL * NL);
          const workl = allocFloat64Array(NL);

          dlacpy(ALL, n, n, h, hOff, ldh, hl, 0, NL);
          // HL(N+1, N) = ZERO
          hl[n + (n - 1) * NL] = ZERO;
          dlaset(ALL, NL, NL - n, ZERO, ZERO, hl, n * NL, NL);

          info = dlaqr0(
            wantt,
            wantz,
            NL,
            ilo,
            kbot,
            hl,
            0,
            NL,
            wr,
            wrOff,
            wi,
            wiOff,
            ilo,
            ihi,
            z,
            zOff,
            ldz,
            workl,
            0,
            NL
          );

          if (wantt || info !== 0) {
            dlacpy(ALL, n, n, hl, 0, NL, h, hOff, ldh);
          }
        }
      }
    }

    // ==== Clear out the trash, if necessary. ====

    if ((wantt || info !== 0) && n > 2) {
      // H(3,1) is at row 3, col 1 in 1-based => hOff + 2 + 0*ldh
      dlaset(LOWER, n - 2, n - 2, ZERO, ZERO, h, H(3, 1), ldh);
    }

    // ==== Ensure reported workspace size is backward-compatible with
    //      previous LAPACK versions. ====

    work[workOff] = Math.max(Math.max(1, n), work[workOff]);
  }

  return info;
}
