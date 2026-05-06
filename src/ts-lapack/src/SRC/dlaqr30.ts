// Translated from SRC/dlaqr3.f and SRC/dlaqr0.f
//
// Both routines are placed in a single file to break the circular dependency:
//   dlaqr0 calls dlaqr3 (aggressive early deflation)
//   dlaqr3 calls dlaqr0 (recursion on deflation window)
//
// dlaqr3 — Aggressive early deflation: detects and deflates converged
//          eigenvalues from a trailing principal submatrix of H.
//
// dlaqr0 — Top-level QR algorithm driver: computes eigenvalues (and
//          optionally the Schur decomposition) of an upper Hessenberg matrix.
//          Dispatches to dlahqr for small problems, uses dlaqr3 + dlaqr5 for
//          larger ones.
//
// Array indexing convention (column-major, matching Fortran):
//   H(I,J)  => h[hOff + (I-1) + (J-1)*ldh]   (I,J are 1-based)
//   Z(I,J)  => z[zOff + (I-1) + (J-1)*ldz]
//   T(I,J)  => t[tOff + (I-1) + (J-1)*ldt]
//   V(I,J)  => v[vOff + (I-1) + (J-1)*ldv]
//   WV(I,J) => wv[wvOff + (I-1) + (J-1)*ldwv]
//   SR(I)   => sr[srOff + (I-1)]
//   SI(I)   => si[siOff + (I-1)]

import { dlamch } from "./dlamch.js";
import { dlanv2 } from "./dlanv2.js";
import { dlahqr } from "./dlahqr.js";
import { dlacpy } from "./dlacpy.js";
import { dlaset } from "./dlaset.js";
import { dtrexc } from "./dtrexc.js";
import { dormhr } from "./dormhr.js";
import { dgehrd } from "./dgehrd.js";
import { dlarf1f } from "./dlarf1f.js";
import { dlarfg } from "./dlarfg.js";
import { dlaqr5 } from "./dlaqr5.js";
import { dcopy } from "../BLAS/dcopy.js";
import { dgemm } from "../BLAS/dgemm.js";
import { ilaenv } from "../utils/ilaenv.js";
import {
  MACH_SFMIN,
  MACH_PREC,
  UPPER,
  LOWER,
  NOTRANS,
  TRANS,
  LEFT,
  RIGHT,
} from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/executors/jsJit/helpers/alloc.js";

// "ALL" uplo constant for dlacpy/dlaset (anything other than UPPER/LOWER)
const ALL = -1;

// =====================================================================
// dlaqr3 — Aggressive early deflation
// =====================================================================

/**
 * DLAQR3 accepts as input an upper Hessenberg matrix H and performs an
 * orthogonal similarity transformation designed to detect and deflate
 * fully converged eigenvalues from a trailing principal submatrix.
 *
 * This is the "aggressive" variant that calls dlaqr0 recursively for the
 * QR step inside the deflation window (as opposed to dlaqr2 which uses
 * dlahqr only).
 *
 * @param wantt - true: full Schur form required; false: eigenvalues only
 * @param wantz - true: accumulate transformations into Z
 * @param n - order of the matrix H
 * @param ktop - first row/col of isolated block (1-based)
 * @param kbot - last row/col of isolated block (1-based)
 * @param nw - deflation window size
 * @param h - Hessenberg matrix, dimension (ldh, n)
 * @param hOff - offset into h
 * @param ldh - leading dimension of h
 * @param iloz - first row of Z to update (1-based)
 * @param ihiz - last row of Z to update (1-based)
 * @param z - Schur vectors, dimension (ldz, n)
 * @param zOff - offset into z
 * @param ldz - leading dimension of z
 * @param ns - (output) number of unconverged eigenvalues (shifts)
 * @param nd - (output) number of converged (deflated) eigenvalues
 * @param sr - real parts of eigenvalues/shifts, dimension (kbot)
 * @param srOff - offset into sr
 * @param si - imaginary parts of eigenvalues/shifts, dimension (kbot)
 * @param siOff - offset into si
 * @param v - work array, dimension (ldv, nw)
 * @param vOff - offset into v
 * @param ldv - leading dimension of v
 * @param nh - number of columns of t available
 * @param t - work array, dimension (ldt, nw)
 * @param tOff - offset into t
 * @param ldt - leading dimension of t
 * @param nv - number of rows of wv available
 * @param wv - work array, dimension (ldwv, nw)
 * @param wvOff - offset into wv
 * @param ldwv - leading dimension of wv
 * @param work - workspace, dimension (lwork)
 * @param workOff - offset into work
 * @param lwork - workspace size; if -1, workspace query
 */
export function dlaqr3(
  wantt: boolean,
  wantz: boolean,
  n: number,
  ktop: number,
  kbot: number,
  nw: number,
  h: Float64Array,
  hOff: number,
  ldh: number,
  iloz: number,
  ihiz: number,
  z: Float64Array,
  zOff: number,
  ldz: number,
  ns: { val: number },
  nd: { val: number },
  sr: Float64Array,
  srOff: number,
  si: Float64Array,
  siOff: number,
  v: Float64Array,
  vOff: number,
  ldv: number,
  nh: number,
  t: Float64Array,
  tOff: number,
  ldt: number,
  nv: number,
  wv: Float64Array,
  wvOff: number,
  ldwv: number,
  work: Float64Array,
  workOff: number,
  lwork: number
): void {
  const ZERO = 0.0;
  const ONE = 1.0;

  // Helpers for column-major indexing (1-based I,J)
  const H = (i: number, j: number) => hOff + (i - 1) + (j - 1) * ldh;
  const Z = (i: number, j: number) => zOff + (i - 1) + (j - 1) * ldz;
  const T = (i: number, j: number) => tOff + (i - 1) + (j - 1) * ldt;
  const V = (i: number, j: number) => vOff + (i - 1) + (j - 1) * ldv;
  const WORK = (i: number) => workOff + (i - 1);

  // ==== Estimate optimal workspace ====
  let jw = Math.min(nw, kbot - ktop + 1);
  let lwkopt: number;
  if (jw <= 2) {
    lwkopt = 1;
  } else {
    // Workspace query call to DGEHRD
    dgehrd(jw, 1, jw - 1, t, tOff, ldt, work, workOff, work, workOff, -1);
    const lwk1 = Math.trunc(work[workOff]);

    // Workspace query call to DORMHR
    dormhr(
      RIGHT,
      NOTRANS,
      jw,
      jw,
      1,
      jw - 1,
      t,
      tOff,
      ldt,
      work,
      workOff,
      v,
      vOff,
      ldv,
      work,
      workOff,
      -1
    );
    const lwk2 = Math.trunc(work[workOff]);

    // Workspace query call to DLAQR0 (recursive)
    dlaqr0(
      true,
      true,
      jw,
      1,
      jw,
      t,
      tOff,
      ldt,
      sr,
      srOff,
      si,
      siOff,
      1,
      jw,
      v,
      vOff,
      ldv,
      work,
      workOff,
      -1
    );
    const lwk3 = Math.trunc(work[workOff]);

    // Optimal workspace
    lwkopt = Math.max(jw + Math.max(lwk1, lwk2), lwk3);
  }

  // ==== Quick return in case of workspace query ====
  if (lwork === -1) {
    work[workOff] = lwkopt;
    return;
  }

  // ==== Nothing to do ...
  // ... for an empty active block ... ====
  ns.val = 0;
  nd.val = 0;
  work[workOff] = ONE;
  if (ktop > kbot) return;
  // ... nor for an empty deflation window. ====
  if (nw < 1) return;

  // ==== Machine constants ====
  const safmin = dlamch(MACH_SFMIN);
  const ulp = dlamch(MACH_PREC);
  const smlnum = safmin * (n / ulp);

  // ==== Setup deflation window ====
  jw = Math.min(nw, kbot - ktop + 1);
  const kwtop = kbot - jw + 1;
  let s: number;
  if (kwtop === ktop) {
    s = ZERO;
  } else {
    s = h[H(kwtop, kwtop - 1)];
  }

  if (kbot === kwtop) {
    // ==== 1-by-1 deflation window: not much to do ====
    sr[srOff + (kwtop - 1)] = h[H(kwtop, kwtop)];
    si[siOff + (kwtop - 1)] = ZERO;
    ns.val = 1;
    nd.val = 0;
    if (Math.abs(s) <= Math.max(smlnum, ulp * Math.abs(h[H(kwtop, kwtop)]))) {
      ns.val = 0;
      nd.val = 1;
      if (kwtop > ktop) h[H(kwtop, kwtop - 1)] = ZERO;
    }
    work[workOff] = ONE;
    return;
  }

  // ==== Convert to spike-triangular form. ====
  // Copy upper triangle
  dlacpy(UPPER, jw, jw, h, H(kwtop, kwtop), ldh, t, tOff, ldt);
  // Copy subdiagonal
  dcopy(jw - 1, h, H(kwtop + 1, kwtop), ldh + 1, t, T(2, 1), ldt + 1);

  // Set V = I
  dlaset(ALL, jw, jw, ZERO, ONE, v, vOff, ldv);

  // Compute Schur form of the deflation window
  const nmin = ilaenv(12, "DLAQR3", "SV", jw, 1, jw, lwork);
  let infqr: number;
  if (jw > nmin) {
    infqr = dlaqr0(
      true,
      true,
      jw,
      1,
      jw,
      t,
      tOff,
      ldt,
      sr,
      srOff + (kwtop - 1),
      si,
      siOff + (kwtop - 1),
      1,
      jw,
      v,
      vOff,
      ldv,
      work,
      workOff,
      lwork
    );
  } else {
    infqr = dlahqr(
      true,
      true,
      jw,
      1,
      jw,
      t,
      tOff,
      ldt,
      sr,
      srOff + (kwtop - 1),
      si,
      siOff + (kwtop - 1),
      1,
      jw,
      v,
      vOff,
      ldv
    );
  }

  // ==== DTREXC needs a clean margin near the diagonal ====
  for (let j = 1; j <= jw - 3; j++) {
    t[T(j + 2, j)] = ZERO;
    t[T(j + 3, j)] = ZERO;
  }
  if (jw > 2) t[T(jw, jw - 2)] = ZERO;

  // ==== Deflation detection loop ====
  {
    let nsVal = jw;
    let ilst = infqr + 1;
    while (ilst <= nsVal) {
      let bulge: boolean;
      if (nsVal === 1) {
        bulge = false;
      } else {
        bulge = t[T(nsVal, nsVal - 1)] !== ZERO;
      }

      // ==== Small spike tip test for deflation ====
      if (!bulge) {
        // ==== Real eigenvalue ====
        let foo = Math.abs(t[T(nsVal, nsVal)]);
        if (foo === ZERO) foo = Math.abs(s);
        if (Math.abs(s * v[V(1, nsVal)]) <= Math.max(smlnum, ulp * foo)) {
          // ==== Deflatable ====
          nsVal = nsVal - 1;
        } else {
          // ==== Undeflatable. Move it up out of the way. ====
          const ifst = { val: nsVal };
          const ilstRef = { val: ilst };
          dtrexc(
            1,
            jw,
            t,
            tOff,
            ldt,
            v,
            vOff,
            ldv,
            ifst,
            ilstRef,
            work,
            workOff
          );
          ilst = ilstRef.val + 1;
        }
      } else {
        // ==== Complex conjugate pair ====
        let foo =
          Math.abs(t[T(nsVal, nsVal)]) +
          Math.sqrt(Math.abs(t[T(nsVal, nsVal - 1)])) *
            Math.sqrt(Math.abs(t[T(nsVal - 1, nsVal)]));
        if (foo === ZERO) foo = Math.abs(s);
        if (
          Math.max(
            Math.abs(s * v[V(1, nsVal)]),
            Math.abs(s * v[V(1, nsVal - 1)])
          ) <= Math.max(smlnum, ulp * foo)
        ) {
          // ==== Deflatable ====
          nsVal = nsVal - 2;
        } else {
          // ==== Undeflatable. Move them up out of the way. ====
          const ifst = { val: nsVal };
          const ilstRef = { val: ilst };
          dtrexc(
            1,
            jw,
            t,
            tOff,
            ldt,
            v,
            vOff,
            ldv,
            ifst,
            ilstRef,
            work,
            workOff
          );
          ilst = ilstRef.val + 2;
        }
      }
    }

    // ==== Return to Hessenberg form ====
    if (nsVal === 0) s = ZERO;

    if (nsVal < jw) {
      // ==== Sorting diagonal blocks of T improves accuracy for
      //      graded matrices. Bubble sort deals well with exchange failures. ====
      let sorted = false;
      let ii = nsVal + 1;
      while (!sorted) {
        sorted = true;
        const kend = ii - 1;
        ii = infqr + 1;
        let kk: number;
        if (ii === nsVal) {
          kk = ii + 1;
        } else if (t[T(ii + 1, ii)] === ZERO) {
          kk = ii + 1;
        } else {
          kk = ii + 2;
        }
        while (kk <= kend) {
          let evi: number;
          if (kk === ii + 1) {
            evi = Math.abs(t[T(ii, ii)]);
          } else {
            evi =
              Math.abs(t[T(ii, ii)]) +
              Math.sqrt(Math.abs(t[T(ii + 1, ii)])) *
                Math.sqrt(Math.abs(t[T(ii, ii + 1)]));
          }

          let evk: number;
          if (kk === kend) {
            evk = Math.abs(t[T(kk, kk)]);
          } else if (t[T(kk + 1, kk)] === ZERO) {
            evk = Math.abs(t[T(kk, kk)]);
          } else {
            evk =
              Math.abs(t[T(kk, kk)]) +
              Math.sqrt(Math.abs(t[T(kk + 1, kk)])) *
                Math.sqrt(Math.abs(t[T(kk, kk + 1)]));
          }

          if (evi >= evk) {
            ii = kk;
          } else {
            sorted = false;
            const ifst = { val: ii };
            const ilstRef = { val: kk };
            const trexcInfo = dtrexc(
              1,
              jw,
              t,
              tOff,
              ldt,
              v,
              vOff,
              ldv,
              ifst,
              ilstRef,
              work,
              workOff
            );
            if (trexcInfo === 0) {
              ii = ilstRef.val;
            } else {
              ii = kk;
            }
          }
          if (ii === kend) {
            kk = ii + 1;
          } else if (t[T(ii + 1, ii)] === ZERO) {
            kk = ii + 1;
          } else {
            kk = ii + 2;
          }
        }
      }
    }

    // ==== Restore shift/eigenvalue array from T ====
    {
      let ii = jw;
      while (ii >= infqr + 1) {
        if (ii === infqr + 1) {
          sr[srOff + (kwtop + ii - 2)] = t[T(ii, ii)];
          si[siOff + (kwtop + ii - 2)] = ZERO;
          ii = ii - 1;
        } else if (t[T(ii, ii - 1)] === ZERO) {
          sr[srOff + (kwtop + ii - 2)] = t[T(ii, ii)];
          si[siOff + (kwtop + ii - 2)] = ZERO;
          ii = ii - 1;
        } else {
          const aa = t[T(ii - 1, ii - 1)];
          const cc = t[T(ii, ii - 1)];
          const bb = t[T(ii - 1, ii)];
          const dd = t[T(ii, ii)];
          const res = dlanv2(aa, bb, cc, dd);
          sr[srOff + (kwtop + ii - 3)] = res.rt1r;
          si[siOff + (kwtop + ii - 3)] = res.rt1i;
          sr[srOff + (kwtop + ii - 2)] = res.rt2r;
          si[siOff + (kwtop + ii - 2)] = res.rt2i;
          ii = ii - 2;
        }
      }
    }

    if (nsVal < jw || s === ZERO) {
      if (nsVal > 1 && s !== ZERO) {
        // ==== Reflect spike back into lower triangle ====
        dcopy(nsVal, v, vOff, ldv, work, workOff, 1);
        let beta = work[workOff];
        const rfg = dlarfg(nsVal, beta, work, workOff + 1, 1);
        beta = rfg.alpha;
        const tau = rfg.tau;

        dlaset(LOWER, jw - 2, jw - 2, ZERO, ZERO, t, T(3, 1), ldt);

        dlarf1f(
          LEFT,
          nsVal,
          jw,
          work,
          workOff,
          1,
          tau,
          t,
          tOff,
          ldt,
          work,
          WORK(jw + 1)
        );
        dlarf1f(
          RIGHT,
          nsVal,
          nsVal,
          work,
          workOff,
          1,
          tau,
          t,
          tOff,
          ldt,
          work,
          WORK(jw + 1)
        );
        dlarf1f(
          RIGHT,
          jw,
          nsVal,
          work,
          workOff,
          1,
          tau,
          v,
          vOff,
          ldv,
          work,
          WORK(jw + 1)
        );

        dgehrd(
          jw,
          1,
          nsVal,
          t,
          tOff,
          ldt,
          work,
          workOff,
          work,
          WORK(jw + 1),
          lwork - jw
        );
      }

      // ==== Copy updated reduced window into place ====
      if (kwtop > 1) h[H(kwtop, kwtop - 1)] = s * v[V(1, 1)];
      dlacpy(UPPER, jw, jw, t, tOff, ldt, h, H(kwtop, kwtop), ldh);
      dcopy(jw - 1, t, T(2, 1), ldt + 1, h, H(kwtop + 1, kwtop), ldh + 1);

      // ==== Accumulate orthogonal matrix in order to update H and Z ====
      if (nsVal > 1 && s !== ZERO) {
        dormhr(
          RIGHT,
          NOTRANS,
          jw,
          nsVal,
          1,
          nsVal,
          t,
          tOff,
          ldt,
          work,
          workOff,
          v,
          vOff,
          ldv,
          work,
          WORK(jw + 1),
          lwork - jw
        );
      }

      // ==== Update vertical slab in H ====
      let ltop: number;
      if (wantt) {
        ltop = 1;
      } else {
        ltop = ktop;
      }
      for (let krow = ltop; krow <= kwtop - 1; krow += nv) {
        const kln = Math.min(nv, kwtop - krow);
        dgemm(
          NOTRANS,
          NOTRANS,
          kln,
          jw,
          jw,
          ONE,
          h,
          H(krow, kwtop),
          ldh,
          v,
          vOff,
          ldv,
          ZERO,
          wv,
          wvOff,
          ldwv
        );
        dlacpy(ALL, kln, jw, wv, wvOff, ldwv, h, H(krow, kwtop), ldh);
      }

      // ==== Update horizontal slab in H ====
      if (wantt) {
        for (let kcol = kbot + 1; kcol <= n; kcol += nh) {
          const kln = Math.min(nh, n - kcol + 1);
          dgemm(
            TRANS,
            NOTRANS,
            jw,
            kln,
            jw,
            ONE,
            v,
            vOff,
            ldv,
            h,
            H(kwtop, kcol),
            ldh,
            ZERO,
            t,
            tOff,
            ldt
          );
          dlacpy(ALL, jw, kln, t, tOff, ldt, h, H(kwtop, kcol), ldh);
        }
      }

      // ==== Update vertical slab in Z ====
      if (wantz) {
        for (let krow = iloz; krow <= ihiz; krow += nv) {
          const kln = Math.min(nv, ihiz - krow + 1);
          dgemm(
            NOTRANS,
            NOTRANS,
            kln,
            jw,
            jw,
            ONE,
            z,
            Z(krow, kwtop),
            ldz,
            v,
            vOff,
            ldv,
            ZERO,
            wv,
            wvOff,
            ldwv
          );
          dlacpy(ALL, kln, jw, wv, wvOff, ldwv, z, Z(krow, kwtop), ldz);
        }
      }
    }

    // ==== Return the number of deflations ====
    nd.val = jw - nsVal;

    // ==== ... and the number of shifts ====
    ns.val = nsVal - infqr;
  }

  // ==== Return optimal workspace ====
  work[workOff] = lwkopt;
}

// =====================================================================
// dlaqr0 — Top-level QR algorithm driver (aggressive variant)
// =====================================================================

/**
 * DLAQR0 computes the eigenvalues of a Hessenberg matrix H and,
 * optionally, the matrices T and Z from the Schur decomposition
 * H = Z T Z**T.
 *
 * For small matrices (n <= NTINY = 15), it dispatches to DLAHQR.
 * For larger matrices, it uses multi-shift QR with aggressive early
 * deflation (DLAQR3 + DLAQR5).
 *
 * @param wantt - true: compute full Schur form; false: eigenvalues only
 * @param wantz - true: accumulate transformations into Z
 * @param n - order of the matrix H
 * @param ilo - first row/col of active block (1-based)
 * @param ihi - last row/col of active block (1-based)
 * @param h - Hessenberg matrix, dimension (ldh, n)
 * @param hOff - offset into h
 * @param ldh - leading dimension of h
 * @param wr - real parts of eigenvalues, dimension (ihi)
 * @param wrOff - offset into wr
 * @param wi - imaginary parts of eigenvalues, dimension (ihi)
 * @param wiOff - offset into wi
 * @param iloz - first row of Z to update (1-based)
 * @param ihiz - last row of Z to update (1-based)
 * @param z - Schur vectors, dimension (ldz, n)
 * @param zOff - offset into z
 * @param ldz - leading dimension of z
 * @param work - workspace, dimension (lwork)
 * @param workOff - offset into work
 * @param lwork - workspace size; if -1, workspace query
 * @returns INFO: 0 = success, > 0 = convergence failure at index
 */
export function dlaqr0(
  wantt: boolean,
  wantz: boolean,
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
  iloz: number,
  ihiz: number,
  z: Float64Array,
  zOff: number,
  ldz: number,
  work: Float64Array,
  workOff: number,
  lwork: number
): number {
  const ZERO = 0.0;
  const ONE = 1.0;

  // ==== Matrices of order NTINY or smaller must use DLAHQR ====
  const NTINY = 15;

  // ==== Exceptional deflation windows ====
  const KEXNW = 5;

  // ==== Exceptional shifts ====
  const KEXSH = 6;

  // ==== Wilkinson shift constants ====
  const WILK1 = 0.75;
  const WILK2 = -0.4375;

  // Helpers for column-major indexing (1-based I,J)
  const H = (i: number, j: number) => hOff + (i - 1) + (j - 1) * ldh;
  const WR = (i: number) => wrOff + (i - 1);
  const WI = (i: number) => wiOff + (i - 1);

  let info = 0;
  let lwkopt = 1;

  // ==== Quick return for N = 0 ====
  if (n === 0) {
    work[workOff] = ONE;
    return 0;
  }

  if (n <= NTINY) {
    // ==== Tiny matrices must use DLAHQR ====
    lwkopt = 1;
    if (lwork !== -1) {
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
        iloz,
        ihiz,
        z,
        zOff,
        ldz
      );
    }
  } else {
    // ==== Use multi-shift QR with aggressive early deflation ====

    info = 0;

    // ==== Set up job flags for ILAENV ====
    let jbcmpz: string;
    if (wantt) {
      jbcmpz = "S";
    } else {
      jbcmpz = "E";
    }
    if (wantz) {
      jbcmpz += "V";
    } else {
      jbcmpz += "N";
    }

    // ==== NWR = recommended deflation window size ====
    let nwr = ilaenv(13, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
    nwr = Math.max(2, nwr);
    nwr = Math.min(ihi - ilo + 1, Math.trunc((n - 1) / 3), nwr);

    // ==== NSR = recommended number of simultaneous shifts ====
    let nsr = ilaenv(15, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
    nsr = Math.min(nsr, Math.trunc((n - 3) / 6), ihi - ilo);
    nsr = Math.max(2, nsr - (nsr % 2));

    // ==== Estimate optimal workspace ====

    // Workspace query call to DLAQR3
    const nsRef = { val: 0 };
    const ndRef = { val: 0 };
    dlaqr3(
      wantt,
      wantz,
      n,
      ilo,
      ihi,
      nwr + 1,
      h,
      hOff,
      ldh,
      iloz,
      ihiz,
      z,
      zOff,
      ldz,
      nsRef,
      ndRef,
      wr,
      wrOff,
      wi,
      wiOff,
      h,
      hOff,
      ldh,
      n,
      h,
      hOff,
      ldh,
      n,
      h,
      hOff,
      ldh,
      work,
      workOff,
      -1
    );

    // Optimal workspace = MAX(DLAQR5, DLAQR3)
    lwkopt = Math.max(Math.trunc((3 * nsr) / 2), Math.trunc(work[workOff]));

    // ==== Quick return in case of workspace query ====
    if (lwork === -1) {
      work[workOff] = lwkopt;
      return 0;
    }

    // ==== DLAHQR/DLAQR0 crossover point ====
    let nmin = ilaenv(12, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
    nmin = Math.max(NTINY, nmin);

    // ==== Nibble crossover point ====
    let nibble = ilaenv(14, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
    nibble = Math.max(0, nibble);

    // ==== Accumulate reflections during ttswp? Use block 2-by-2? ====
    let kacc22 = ilaenv(16, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
    kacc22 = Math.max(0, kacc22);
    kacc22 = Math.min(2, kacc22);

    // ==== NWMAX = the largest possible deflation window ====
    const nwmax = Math.min(Math.trunc((n - 1) / 3), Math.trunc(lwork / 2));
    let nw = nwmax;

    // ==== NSMAX = the largest number of simultaneous shifts ====
    let nsmax = Math.min(Math.trunc((n - 3) / 6), Math.trunc((2 * lwork) / 3));
    nsmax = nsmax - (nsmax % 2);

    // ==== NDFL: an iteration count restarted at deflation ====
    let ndfl = 1;

    // ==== ITMAX = iteration limit ====
    const itmax = Math.max(30, 2 * KEXSH) * Math.max(10, ihi - ilo + 1);

    // ==== Last row and column in the active block ====
    let kbot = ihi;

    // ==== NDEC: used for window size adjustment ====
    let ndec = -1;

    // Dummy 1x1 Z array for non-wantz recursive calls
    const zdum = allocFloat64Array(1);

    // ==== Main Loop ====
    for (let it = 1; it <= itmax; it++) {
      // ==== Done when KBOT falls below ILO ====
      if (kbot < ilo) {
        // goto 90 — success
        work[workOff] = lwkopt;
        return info;
      }

      // ==== Locate active block ====
      let ktop_: number;
      {
        let k: number;
        for (k = kbot; k >= ilo + 1; k--) {
          if (h[H(k, k - 1)] === ZERO) break;
        }
        if (k < ilo + 1) k = ilo;
        ktop_ = k;
      }

      // ==== Select deflation window size ====
      const nh_ = kbot - ktop_ + 1;
      const nwupbd = Math.min(nh_, nwmax);
      if (ndfl < KEXNW) {
        nw = Math.min(nwupbd, nwr);
      } else {
        nw = Math.min(nwupbd, 2 * nw);
      }
      if (nw < nwmax) {
        if (nw >= nh_ - 1) {
          nw = nh_;
        } else {
          const kwtop = kbot - nw + 1;
          if (
            Math.abs(h[H(kwtop, kwtop - 1)]) >
            Math.abs(h[H(kwtop - 1, kwtop - 2)])
          ) {
            nw = nw + 1;
          }
        }
      }
      if (ndfl < KEXNW) {
        ndec = -1;
      } else if (ndec >= 0 || nw >= nwupbd) {
        ndec = ndec + 1;
        if (nw - ndec < 2) ndec = 0;
        nw = nw - ndec;
      }

      // ==== Aggressive early deflation ====
      // Split workspace under the subdiagonal
      const kv = n - nw + 1;
      const kt = nw + 1;
      const nho = n - nw - 1 - kt + 1;
      const kwv = nw + 2;
      const nve = n - nw - kwv + 1;

      const ls = { val: 0 };
      const ld = { val: 0 };
      dlaqr3(
        wantt,
        wantz,
        n,
        ktop_,
        kbot,
        nw,
        h,
        hOff,
        ldh,
        iloz,
        ihiz,
        z,
        zOff,
        ldz,
        ls,
        ld,
        wr,
        wrOff,
        wi,
        wiOff,
        h,
        H(kv, 1),
        ldh,
        nho,
        h,
        H(kv, kt),
        ldh,
        nve,
        h,
        H(kwv, 1),
        ldh,
        work,
        workOff,
        lwork
      );

      // ==== Adjust KBOT accounting for new deflations ====
      kbot = kbot - ld.val;

      // ==== KS points to the shifts ====
      let ks = kbot - ls.val + 1;

      // ==== Skip an expensive QR sweep if there is a (partly
      //      heuristic) reason to expect many eigenvalues will deflate ====
      if (
        ld.val === 0 ||
        (100 * ld.val <= nw * nibble &&
          kbot - ktop_ + 1 > Math.min(nmin, nwmax))
      ) {
        // ==== NS = nominal number of simultaneous shifts ====
        let ns_ = Math.min(nsmax, nsr, Math.max(2, kbot - ktop_));
        ns_ = ns_ - (ns_ % 2);

        // ==== If there have been no deflations in a multiple of KEXSH
        //      iterations, then try exceptional shifts ====
        if (ndfl % KEXSH === 0) {
          ks = kbot - ns_ + 1;
          for (let i = kbot; i >= Math.max(ks + 1, ktop_ + 2); i -= 2) {
            const ss = Math.abs(h[H(i, i - 1)]) + Math.abs(h[H(i - 1, i - 2)]);
            const aa = WILK1 * ss + h[H(i, i)];
            const bb = ss;
            const cc = WILK2 * ss;
            const dd = aa;
            const res = dlanv2(aa, bb, cc, dd);
            wr[WR(i - 1)] = res.rt1r;
            wi[WI(i - 1)] = res.rt1i;
            wr[WR(i)] = res.rt2r;
            wi[WI(i)] = res.rt2i;
          }
          if (ks === ktop_) {
            wr[WR(ks + 1)] = h[H(ks + 1, ks + 1)];
            wi[WI(ks + 1)] = ZERO;
            wr[WR(ks)] = wr[WR(ks + 1)];
            wi[WI(ks)] = wi[WI(ks + 1)];
          }
        } else {
          // ==== Got NS/2 or fewer shifts? Use DLAQR0 or DLAHQR
          //      on a trailing principal submatrix to get more. ====
          if (kbot - ks + 1 <= Math.trunc(ns_ / 2)) {
            ks = kbot - ns_ + 1;
            const kt2 = n - ns_ + 1;
            dlacpy(ALL, ns_, ns_, h, H(ks, ks), ldh, h, H(kt2, 1), ldh);
            let inf: number;
            if (ns_ > nmin) {
              inf = dlaqr0(
                false,
                false,
                ns_,
                1,
                ns_,
                h,
                H(kt2, 1),
                ldh,
                wr,
                WR(ks),
                wi,
                WI(ks),
                1,
                1,
                zdum,
                0,
                1,
                work,
                workOff,
                lwork
              );
            } else {
              inf = dlahqr(
                false,
                false,
                ns_,
                1,
                ns_,
                h,
                H(kt2, 1),
                ldh,
                wr,
                WR(ks),
                wi,
                WI(ks),
                1,
                1,
                zdum,
                0,
                1
              );
            }
            ks = ks + inf;

            // ==== In case of a rare QR failure use eigenvalues of
            //      the trailing 2-by-2 principal submatrix ====
            if (ks >= kbot) {
              const aa = h[H(kbot - 1, kbot - 1)];
              const cc = h[H(kbot, kbot - 1)];
              const bb = h[H(kbot - 1, kbot)];
              const dd = h[H(kbot, kbot)];
              const res = dlanv2(aa, bb, cc, dd);
              wr[WR(kbot - 1)] = res.rt1r;
              wi[WI(kbot - 1)] = res.rt1i;
              wr[WR(kbot)] = res.rt2r;
              wi[WI(kbot)] = res.rt2i;
              ks = kbot - 1;
            }
          }

          if (kbot - ks + 1 > ns_) {
            // ==== Sort the shifts (helps a little) ====
            let sorted = false;
            for (let k = kbot; k >= ks + 1; k--) {
              if (sorted) break;
              sorted = true;
              for (let i = ks; i <= k - 1; i++) {
                if (
                  Math.abs(wr[WR(i)]) + Math.abs(wi[WI(i)]) <
                  Math.abs(wr[WR(i + 1)]) + Math.abs(wi[WI(i + 1)])
                ) {
                  sorted = false;
                  let swap: number;
                  swap = wr[WR(i)];
                  wr[WR(i)] = wr[WR(i + 1)];
                  wr[WR(i + 1)] = swap;

                  swap = wi[WI(i)];
                  wi[WI(i)] = wi[WI(i + 1)];
                  wi[WI(i + 1)] = swap;
                }
              }
            }
          }

          // ==== Shuffle shifts into pairs of real shifts and pairs
          //      of complex conjugate shifts ====
          for (let i = kbot; i >= ks + 2; i -= 2) {
            if (wi[WI(i)] !== -wi[WI(i - 1)]) {
              let swap: number;
              swap = wr[WR(i)];
              wr[WR(i)] = wr[WR(i - 1)];
              wr[WR(i - 1)] = wr[WR(i - 2)];
              wr[WR(i - 2)] = swap;

              swap = wi[WI(i)];
              wi[WI(i)] = wi[WI(i - 1)];
              wi[WI(i - 1)] = wi[WI(i - 2)];
              wi[WI(i - 2)] = swap;
            }
          }
        }

        // ==== If there are only two shifts and both are real,
        //      then use only one ====
        if (kbot - ks + 1 === 2) {
          if (wi[WI(kbot)] === ZERO) {
            if (
              Math.abs(wr[WR(kbot)] - h[H(kbot, kbot)]) <
              Math.abs(wr[WR(kbot - 1)] - h[H(kbot, kbot)])
            ) {
              wr[WR(kbot - 1)] = wr[WR(kbot)];
            } else {
              wr[WR(kbot)] = wr[WR(kbot - 1)];
            }
          }
        }

        // ==== Use up to NS of the smallest magnitude shifts ====
        ns_ = Math.min(ns_, kbot - ks + 1);
        ns_ = ns_ - (ns_ % 2);
        ks = kbot - ns_ + 1;

        // ==== Small-bulge multi-shift QR sweep ====
        const kdu = 2 * ns_;
        const ku = n - kdu + 1;
        const kwh = kdu + 1;
        const nho2 = n - kdu + 1 - 4 - (kdu + 1) + 1;
        const kwv2 = kdu + 4;
        const nve2 = n - kdu - kwv2 + 1;

        dlaqr5(
          wantt,
          wantz,
          kacc22,
          n,
          ktop_,
          kbot,
          ns_,
          wr,
          WR(ks),
          wi,
          WI(ks),
          h,
          hOff,
          ldh,
          iloz,
          ihiz,
          z,
          zOff,
          ldz,
          work,
          workOff,
          3,
          h,
          H(ku, 1),
          ldh,
          nve2,
          h,
          H(kwv2, 1),
          ldh,
          nho2,
          h,
          H(ku, kwh),
          ldh
        );
      }

      // ==== Note progress (or the lack of it) ====
      if (ld.val > 0) {
        ndfl = 1;
      } else {
        ndfl = ndfl + 1;
      }
    }

    // ==== Iteration limit exceeded ====
    info = kbot;
  }

  // ==== Return the optimal value of LWORK ====
  work[workOff] = lwkopt;
  return info;
}
