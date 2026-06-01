// Translated from SRC/dlaqr5.f
// DLAQR5 performs a single small-bulge multi-shift QR sweep.
//
// Called by DLAQR0, performs the near-the-diagonal part of a small bulge
// multi-shift QR sweep. Each sweep chases chains of NBMPS bulges across
// the Hessenberg matrix.
//
// Array indexing convention (column-major, matching Fortran):
//   H(I,J) => h[hOff + (I-1) + (J-1)*ldh]   (I,J are 1-based)
//   Z(I,J) => z[zOff + (I-1) + (J-1)*ldz]
//   V(I,J) => v[vOff + (I-1) + (J-1)*ldv]
//   U(I,J) => u[uOff + (I-1) + (J-1)*ldu]
//   WV(I,J) => wv[wvOff + (I-1) + (J-1)*ldwv]
//   WH(I,J) => wh[whOff + (I-1) + (J-1)*ldwh]
//   SR(I) => sr[srOff + (I-1)]
//   SI(I) => si[siOff + (I-1)]

import { dlamch } from "./dlamch.js";
import { dgemm } from "../BLAS/dgemm.js";
import { dlacpy } from "./dlacpy.js";
import { dlaqr1 } from "./dlaqr1.js";
import { dlarfg } from "./dlarfg.js";
import { dlaset } from "./dlaset.js";
import { MACH_SFMIN, MACH_PREC, TRANS, NOTRANS } from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

// "ALL" uplo constant for dlacpy/dlaset (anything other than UPPER/LOWER)
const ALL = -1;

/**
 * DLAQR5 performs a single small-bulge multi-shift QR sweep.
 *
 * @param wantt - true if the quasi-triangular Schur factor is being computed
 * @param wantz - true if the orthogonal Schur factor is being computed
 * @param kacc22 - 0, 1, or 2; specifies computation mode of far-from-diagonal updates
 * @param n - order of the Hessenberg matrix H
 * @param ktop - first row/col of isolated diagonal block (1-based)
 * @param kbot - last row/col of isolated diagonal block (1-based)
 * @param nshfts - number of simultaneous shifts (must be positive and even)
 * @param sr - real parts of shifts, dimension (nshfts)
 * @param srOff - offset into sr
 * @param si - imaginary parts of shifts, dimension (nshfts)
 * @param siOff - offset into si
 * @param h - Hessenberg matrix, dimension (ldh, n)
 * @param hOff - offset into h
 * @param ldh - leading dimension of h
 * @param iloz - first row of Z to update (1-based)
 * @param ihiz - last row of Z to update (1-based)
 * @param z - Schur vectors, dimension (ldz, ihiz)
 * @param zOff - offset into z
 * @param ldz - leading dimension of z
 * @param v - workspace for reflectors, dimension (ldv, nshfts/2)
 * @param vOff - offset into v
 * @param ldv - leading dimension of v (>= 3)
 * @param u - workspace for accumulating reflections, dimension (ldu, 2*nshfts)
 * @param uOff - offset into u
 * @param ldu - leading dimension of u
 * @param nv - number of rows in wv available for workspace
 * @param wv - workspace, dimension (ldwv, 2*nshfts)
 * @param wvOff - offset into wv
 * @param ldwv - leading dimension of wv
 * @param nh - number of columns in wh available for workspace
 * @param wh - workspace, dimension (ldwh, nh)
 * @param whOff - offset into wh
 * @param ldwh - leading dimension of wh
 */
export function dlaqr5(
  wantt: boolean,
  wantz: boolean,
  kacc22: number,
  n: number,
  ktop: number,
  kbot: number,
  nshfts: number,
  sr: Float64Array,
  srOff: number,
  si: Float64Array,
  siOff: number,
  h: Float64Array,
  hOff: number,
  ldh: number,
  iloz: number,
  ihiz: number,
  z: Float64Array,
  zOff: number,
  ldz: number,
  v: Float64Array,
  vOff: number,
  ldv: number,
  u: Float64Array,
  uOff: number,
  ldu: number,
  nv: number,
  wv: Float64Array,
  wvOff: number,
  ldwv: number,
  nh: number,
  wh: Float64Array,
  whOff: number,
  ldwh: number
): void {
  const ZERO = 0.0;
  const ONE = 1.0;

  // Helpers for column-major indexing (1-based I,J)
  const H = (i: number, j: number) => hOff + (i - 1) + (j - 1) * ldh;
  const Zi = (i: number, j: number) => zOff + (i - 1) + (j - 1) * ldz;
  const V = (i: number, j: number) => vOff + (i - 1) + (j - 1) * ldv;
  const U = (i: number, j: number) => uOff + (i - 1) + (j - 1) * ldu;

  // If there are no shifts, then there is nothing to do.
  if (nshfts < 2) return;

  // If the active block is empty or 1-by-1, then there is nothing to do.
  if (ktop >= kbot) return;

  // Shuffle shifts into pairs of real shifts and pairs of complex
  // conjugate shifts assuming complex conjugate shifts are already
  // adjacent to one another.
  for (let i = 1; i <= nshfts - 2; i += 2) {
    if (si[siOff + (i - 1)] !== -si[siOff + i]) {
      let swap = sr[srOff + (i - 1)];
      sr[srOff + (i - 1)] = sr[srOff + i];
      sr[srOff + i] = sr[srOff + (i + 1)];
      sr[srOff + (i + 1)] = swap;

      swap = si[siOff + (i - 1)];
      si[siOff + (i - 1)] = si[siOff + i];
      si[siOff + i] = si[siOff + (i + 1)];
      si[siOff + (i + 1)] = swap;
    }
  }

  // NSHFTS is supposed to be even, but if it is odd,
  // then simply reduce it by one.
  const ns = nshfts - (nshfts % 2);

  // Machine constants for deflation
  const safmin = dlamch(MACH_SFMIN);
  // const safmax = ONE / safmin;
  const ulp = dlamch(MACH_PREC);
  const smlnum = safmin * (n / ulp);

  // Use accumulated reflections to update far-from-diagonal entries?
  const accum = kacc22 === 1 || kacc22 === 2;

  // Clear trash
  if (ktop + 2 <= kbot) {
    h[H(ktop + 2, ktop)] = ZERO;
  }

  // NBMPS = number of 2-shift bulges in the chain
  const nbmps = Math.trunc(ns / 2);

  // KDU = width of slab
  const kdu = 4 * nbmps;

  // Local array
  const vt = allocFloat64Array(3);

  // Create and chase chains of NBMPS bulges
  for (
    let incol = ktop - 2 * nbmps + 1;
    incol <= kbot - 2;
    incol += 2 * nbmps
  ) {
    // JTOP = Index from which updates from the right start.
    let jtop: number;
    if (accum) {
      jtop = Math.max(ktop, incol);
    } else if (wantt) {
      jtop = 1;
    } else {
      jtop = ktop;
    }

    const ndcol = incol + kdu;
    if (accum) {
      dlaset(ALL, kdu, kdu, ZERO, ONE, u, uOff, ldu);
    }

    // Near-the-diagonal bulge chase.
    for (
      let krcol = incol;
      krcol <= Math.min(incol + 2 * nbmps - 1, kbot - 2);
      krcol++
    ) {
      // Bulges number MTOP to MBOT are active double implicit shift bulges.
      const mtop = Math.max(1, Math.trunc((ktop - krcol) / 2) + 1);
      const mbot = Math.min(nbmps, Math.trunc((kbot - krcol - 1) / 2));
      const m22 = mbot + 1;
      const bmp22 = mbot < nbmps && krcol + 2 * (m22 - 1) === kbot - 2;

      // Generate reflections to chase the chain right one column.
      if (bmp22) {
        // Special case: 2-by-2 reflection at bottom treated separately
        const k = krcol + 2 * (m22 - 1);
        if (k === ktop - 1) {
          dlaqr1(
            2,
            h,
            H(k + 1, k + 1),
            ldh,
            sr[srOff + (2 * m22 - 2)],
            si[siOff + (2 * m22 - 2)],
            sr[srOff + (2 * m22 - 1)],
            si[siOff + (2 * m22 - 1)],
            v,
            V(1, m22)
          );
          let beta = v[V(1, m22)];
          const rfg = dlarfg(2, beta, v, V(2, m22), 1);
          beta = rfg.alpha;
          v[V(1, m22)] = rfg.tau;
        } else {
          let beta = h[H(k + 1, k)];
          v[V(2, m22)] = h[H(k + 2, k)];
          const rfg = dlarfg(2, beta, v, V(2, m22), 1);
          beta = rfg.alpha;
          v[V(1, m22)] = rfg.tau;
          h[H(k + 1, k)] = beta;
          h[H(k + 2, k)] = ZERO;
        }

        // Perform update from right within computational window.
        let t1 = v[V(1, m22)];
        let t2 = t1 * v[V(2, m22)];
        for (let j = jtop; j <= Math.min(kbot, k + 3); j++) {
          const refsum = h[H(j, k + 1)] + v[V(2, m22)] * h[H(j, k + 2)];
          h[H(j, k + 1)] = h[H(j, k + 1)] - refsum * t1;
          h[H(j, k + 2)] = h[H(j, k + 2)] - refsum * t2;
        }

        // Perform update from left within computational window.
        let jbot: number;
        if (accum) {
          jbot = Math.min(ndcol, kbot);
        } else if (wantt) {
          jbot = n;
        } else {
          jbot = kbot;
        }
        t1 = v[V(1, m22)];
        t2 = t1 * v[V(2, m22)];
        for (let j = k + 1; j <= jbot; j++) {
          const refsum = h[H(k + 1, j)] + v[V(2, m22)] * h[H(k + 2, j)];
          h[H(k + 1, j)] = h[H(k + 1, j)] - refsum * t1;
          h[H(k + 2, j)] = h[H(k + 2, j)] - refsum * t2;
        }

        // Convergence test
        if (k >= ktop) {
          if (h[H(k + 1, k)] !== ZERO) {
            let tst1 = Math.abs(h[H(k, k)]) + Math.abs(h[H(k + 1, k + 1)]);
            if (tst1 === ZERO) {
              if (k >= ktop + 1) tst1 = tst1 + Math.abs(h[H(k, k - 1)]);
              if (k >= ktop + 2) tst1 = tst1 + Math.abs(h[H(k, k - 2)]);
              if (k >= ktop + 3) tst1 = tst1 + Math.abs(h[H(k, k - 3)]);
              if (k <= kbot - 2) tst1 = tst1 + Math.abs(h[H(k + 2, k + 1)]);
              if (k <= kbot - 3) tst1 = tst1 + Math.abs(h[H(k + 3, k + 1)]);
              if (k <= kbot - 4) tst1 = tst1 + Math.abs(h[H(k + 4, k + 1)]);
            }
            if (Math.abs(h[H(k + 1, k)]) <= Math.max(smlnum, ulp * tst1)) {
              const h12 = Math.max(
                Math.abs(h[H(k + 1, k)]),
                Math.abs(h[H(k, k + 1)])
              );
              const h21 = Math.min(
                Math.abs(h[H(k + 1, k)]),
                Math.abs(h[H(k, k + 1)])
              );
              const h11 = Math.max(
                Math.abs(h[H(k + 1, k + 1)]),
                Math.abs(h[H(k, k)] - h[H(k + 1, k + 1)])
              );
              const h22 = Math.min(
                Math.abs(h[H(k + 1, k + 1)]),
                Math.abs(h[H(k, k)] - h[H(k + 1, k + 1)])
              );
              const scl = h11 + h12;
              const tst2 = h22 * (h11 / scl);

              if (
                tst2 === ZERO ||
                h21 * (h12 / scl) <= Math.max(smlnum, ulp * tst2)
              ) {
                h[H(k + 1, k)] = ZERO;
              }
            }
          }
        }

        // Accumulate orthogonal transformations.
        if (accum) {
          const kms = k - incol;
          t1 = v[V(1, m22)];
          t2 = t1 * v[V(2, m22)];
          for (let j = Math.max(1, ktop - incol); j <= kdu; j++) {
            const refsum = u[U(j, kms + 1)] + v[V(2, m22)] * u[U(j, kms + 2)];
            u[U(j, kms + 1)] = u[U(j, kms + 1)] - refsum * t1;
            u[U(j, kms + 2)] = u[U(j, kms + 2)] - refsum * t2;
          }
        } else if (wantz) {
          t1 = v[V(1, m22)];
          t2 = t1 * v[V(2, m22)];
          for (let j = iloz; j <= ihiz; j++) {
            const refsum = z[Zi(j, k + 1)] + v[V(2, m22)] * z[Zi(j, k + 2)];
            z[Zi(j, k + 1)] = z[Zi(j, k + 1)] - refsum * t1;
            z[Zi(j, k + 2)] = z[Zi(j, k + 2)] - refsum * t2;
          }
        }
      }

      // Normal case: Chain of 3-by-3 reflections
      for (let m = mbot; m >= mtop; m--) {
        const k = krcol + 2 * (m - 1);
        if (k === ktop - 1) {
          dlaqr1(
            3,
            h,
            H(ktop, ktop),
            ldh,
            sr[srOff + (2 * m - 2)],
            si[siOff + (2 * m - 2)],
            sr[srOff + (2 * m - 1)],
            si[siOff + (2 * m - 1)],
            v,
            V(1, m)
          );
          let alpha = v[V(1, m)];
          const rfg = dlarfg(3, alpha, v, V(2, m), 1);
          alpha = rfg.alpha;
          v[V(1, m)] = rfg.tau;
        } else {
          // Perform delayed transformation of row below Mth bulge.
          // Exploit fact that first two elements of row are actually zero.
          let t1 = v[V(1, m)];
          let t2 = t1 * v[V(2, m)];
          let t3 = t1 * v[V(3, m)];
          let refsum = v[V(3, m)] * h[H(k + 3, k + 2)];
          h[H(k + 3, k)] = -refsum * t1;
          h[H(k + 3, k + 1)] = -refsum * t2;
          h[H(k + 3, k + 2)] = h[H(k + 3, k + 2)] - refsum * t3;

          // Calculate reflection to move Mth bulge one step.
          let beta = h[H(k + 1, k)];
          v[V(2, m)] = h[H(k + 2, k)];
          v[V(3, m)] = h[H(k + 3, k)];
          const rfg = dlarfg(3, beta, v, V(2, m), 1);
          beta = rfg.alpha;
          v[V(1, m)] = rfg.tau;

          // A Bulge may collapse because of vigilant deflation or
          // destructive underflow. In the underflow case, try the
          // two-small-subdiagonals trick to try to reinflate the bulge.
          if (
            h[H(k + 3, k)] !== ZERO ||
            h[H(k + 3, k + 1)] !== ZERO ||
            h[H(k + 3, k + 2)] === ZERO
          ) {
            // Typical case: not collapsed (yet).
            h[H(k + 1, k)] = beta;
            h[H(k + 2, k)] = ZERO;
            h[H(k + 3, k)] = ZERO;
          } else {
            // Atypical case: collapsed. Attempt to reintroduce ignoring
            // H(K+1,K) and H(K+2,K).
            dlaqr1(
              3,
              h,
              H(k + 1, k + 1),
              ldh,
              sr[srOff + (2 * m - 2)],
              si[siOff + (2 * m - 2)],
              sr[srOff + (2 * m - 1)],
              si[siOff + (2 * m - 1)],
              vt,
              0
            );
            let alpha = vt[0];
            const rfg2 = dlarfg(3, alpha, vt, 1, 1);
            alpha = rfg2.alpha;
            vt[0] = rfg2.tau;
            t1 = vt[0];
            t2 = t1 * vt[1];
            t3 = t1 * vt[2];
            refsum = h[H(k + 1, k)] + vt[1] * h[H(k + 2, k)];

            if (
              Math.abs(h[H(k + 2, k)] - refsum * t2) + Math.abs(refsum * t3) >
              ulp *
                (Math.abs(h[H(k, k)]) +
                  Math.abs(h[H(k + 1, k + 1)]) +
                  Math.abs(h[H(k + 2, k + 2)]))
            ) {
              // Starting a new bulge here would create non-negligible fill.
              // Use the old one with trepidation.
              h[H(k + 1, k)] = beta;
              h[H(k + 2, k)] = ZERO;
              h[H(k + 3, k)] = ZERO;
            } else {
              // Starting a new bulge here would create only negligible fill.
              // Replace the old reflector with the new one.
              h[H(k + 1, k)] = h[H(k + 1, k)] - refsum * t1;
              h[H(k + 2, k)] = ZERO;
              h[H(k + 3, k)] = ZERO;
              v[V(1, m)] = vt[0];
              v[V(2, m)] = vt[1];
              v[V(3, m)] = vt[2];
            }
          }
        }

        // Apply reflection from the right and
        // the first column of update from the left.
        // These updates are required for the vigilant deflation check.
        // We still delay most of the updates from the left for efficiency.
        const t1 = v[V(1, m)];
        const t2 = t1 * v[V(2, m)];
        const t3 = t1 * v[V(3, m)];
        for (let j = jtop; j <= Math.min(kbot, k + 3); j++) {
          const refsum =
            h[H(j, k + 1)] +
            v[V(2, m)] * h[H(j, k + 2)] +
            v[V(3, m)] * h[H(j, k + 3)];
          h[H(j, k + 1)] = h[H(j, k + 1)] - refsum * t1;
          h[H(j, k + 2)] = h[H(j, k + 2)] - refsum * t2;
          h[H(j, k + 3)] = h[H(j, k + 3)] - refsum * t3;
        }

        // Perform update from left for subsequent column.
        {
          const refsum =
            h[H(k + 1, k + 1)] +
            v[V(2, m)] * h[H(k + 2, k + 1)] +
            v[V(3, m)] * h[H(k + 3, k + 1)];
          h[H(k + 1, k + 1)] = h[H(k + 1, k + 1)] - refsum * t1;
          h[H(k + 2, k + 1)] = h[H(k + 2, k + 1)] - refsum * t2;
          h[H(k + 3, k + 1)] = h[H(k + 3, k + 1)] - refsum * t3;
        }

        // Convergence test
        if (k < ktop) continue;
        if (h[H(k + 1, k)] !== ZERO) {
          let tst1 = Math.abs(h[H(k, k)]) + Math.abs(h[H(k + 1, k + 1)]);
          if (tst1 === ZERO) {
            if (k >= ktop + 1) tst1 = tst1 + Math.abs(h[H(k, k - 1)]);
            if (k >= ktop + 2) tst1 = tst1 + Math.abs(h[H(k, k - 2)]);
            if (k >= ktop + 3) tst1 = tst1 + Math.abs(h[H(k, k - 3)]);
            if (k <= kbot - 2) tst1 = tst1 + Math.abs(h[H(k + 2, k + 1)]);
            if (k <= kbot - 3) tst1 = tst1 + Math.abs(h[H(k + 3, k + 1)]);
            if (k <= kbot - 4) tst1 = tst1 + Math.abs(h[H(k + 4, k + 1)]);
          }
          if (Math.abs(h[H(k + 1, k)]) <= Math.max(smlnum, ulp * tst1)) {
            const h12 = Math.max(
              Math.abs(h[H(k + 1, k)]),
              Math.abs(h[H(k, k + 1)])
            );
            const h21 = Math.min(
              Math.abs(h[H(k + 1, k)]),
              Math.abs(h[H(k, k + 1)])
            );
            const h11 = Math.max(
              Math.abs(h[H(k + 1, k + 1)]),
              Math.abs(h[H(k, k)] - h[H(k + 1, k + 1)])
            );
            const h22 = Math.min(
              Math.abs(h[H(k + 1, k + 1)]),
              Math.abs(h[H(k, k)] - h[H(k + 1, k + 1)])
            );
            const scl = h11 + h12;
            const tst2 = h22 * (h11 / scl);

            if (
              tst2 === ZERO ||
              h21 * (h12 / scl) <= Math.max(smlnum, ulp * tst2)
            ) {
              h[H(k + 1, k)] = ZERO;
            }
          }
        }
      } // end m loop (3-by-3 chain)

      // Multiply H by reflections from the left
      let jbot: number;
      if (accum) {
        jbot = Math.min(ndcol, kbot);
      } else if (wantt) {
        jbot = n;
      } else {
        jbot = kbot;
      }

      for (let m = mbot; m >= mtop; m--) {
        const k = krcol + 2 * (m - 1);
        const t1 = v[V(1, m)];
        const t2 = t1 * v[V(2, m)];
        const t3 = t1 * v[V(3, m)];
        for (let j = Math.max(ktop, krcol + 2 * m); j <= jbot; j++) {
          const refsum =
            h[H(k + 1, j)] +
            v[V(2, m)] * h[H(k + 2, j)] +
            v[V(3, m)] * h[H(k + 3, j)];
          h[H(k + 1, j)] = h[H(k + 1, j)] - refsum * t1;
          h[H(k + 2, j)] = h[H(k + 2, j)] - refsum * t2;
          h[H(k + 3, j)] = h[H(k + 3, j)] - refsum * t3;
        }
      }

      // Accumulate orthogonal transformations.
      if (accum) {
        // Accumulate U. (If needed, update Z later with an efficient
        // matrix-matrix multiply.)
        for (let m = mbot; m >= mtop; m--) {
          const k = krcol + 2 * (m - 1);
          const kms = k - incol;
          let i2 = Math.max(1, ktop - incol);
          i2 = Math.max(i2, kms - (krcol - incol) + 1);
          const i4 = Math.min(kdu, krcol + 2 * (mbot - 1) - incol + 5);
          const t1 = v[V(1, m)];
          const t2 = t1 * v[V(2, m)];
          const t3 = t1 * v[V(3, m)];
          for (let j = i2; j <= i4; j++) {
            const refsum =
              u[U(j, kms + 1)] +
              v[V(2, m)] * u[U(j, kms + 2)] +
              v[V(3, m)] * u[U(j, kms + 3)];
            u[U(j, kms + 1)] = u[U(j, kms + 1)] - refsum * t1;
            u[U(j, kms + 2)] = u[U(j, kms + 2)] - refsum * t2;
            u[U(j, kms + 3)] = u[U(j, kms + 3)] - refsum * t3;
          }
        }
      } else if (wantz) {
        // U is not accumulated, so update Z now by multiplying by
        // reflections from the right.
        for (let m = mbot; m >= mtop; m--) {
          const k = krcol + 2 * (m - 1);
          const t1 = v[V(1, m)];
          const t2 = t1 * v[V(2, m)];
          const t3 = t1 * v[V(3, m)];
          for (let j = iloz; j <= ihiz; j++) {
            const refsum =
              z[Zi(j, k + 1)] +
              v[V(2, m)] * z[Zi(j, k + 2)] +
              v[V(3, m)] * z[Zi(j, k + 3)];
            z[Zi(j, k + 1)] = z[Zi(j, k + 1)] - refsum * t1;
            z[Zi(j, k + 2)] = z[Zi(j, k + 2)] - refsum * t2;
            z[Zi(j, k + 3)] = z[Zi(j, k + 3)] - refsum * t3;
          }
        }
      }
    } // end krcol loop (near-the-diagonal bulge chase)

    // Use U (if accumulated) to update far-from-diagonal entries in H.
    // If required, use U to update Z as well.
    if (accum) {
      let jtopFar: number;
      let jbotFar: number;
      if (wantt) {
        jtopFar = 1;
        jbotFar = n;
      } else {
        jtopFar = ktop;
        jbotFar = kbot;
      }
      const k1 = Math.max(1, ktop - incol);
      const nu = kdu - Math.max(0, ndcol - kbot) - k1 + 1;

      // Horizontal Multiply
      for (let jcol = Math.min(ndcol, kbot) + 1; jcol <= jbotFar; jcol += nh) {
        const jlen = Math.min(nh, jbotFar - jcol + 1);
        dgemm(
          TRANS,
          NOTRANS,
          nu,
          jlen,
          nu,
          ONE,
          u,
          U(k1, k1),
          ldu,
          h,
          H(incol + k1, jcol),
          ldh,
          ZERO,
          wh,
          whOff,
          ldwh
        );
        dlacpy(ALL, nu, jlen, wh, whOff, ldwh, h, H(incol + k1, jcol), ldh);
      }

      // Vertical multiply
      for (let jrow = jtopFar; jrow <= Math.max(ktop, incol) - 1; jrow += nv) {
        const jlen = Math.min(nv, Math.max(ktop, incol) - jrow);
        dgemm(
          NOTRANS,
          NOTRANS,
          jlen,
          nu,
          nu,
          ONE,
          h,
          H(jrow, incol + k1),
          ldh,
          u,
          U(k1, k1),
          ldu,
          ZERO,
          wv,
          wvOff,
          ldwv
        );
        dlacpy(ALL, jlen, nu, wv, wvOff, ldwv, h, H(jrow, incol + k1), ldh);
      }

      // Z multiply (also vertical)
      if (wantz) {
        for (let jrow = iloz; jrow <= ihiz; jrow += nv) {
          const jlen = Math.min(nv, ihiz - jrow + 1);
          dgemm(
            NOTRANS,
            NOTRANS,
            jlen,
            nu,
            nu,
            ONE,
            z,
            Zi(jrow, incol + k1),
            ldz,
            u,
            U(k1, k1),
            ldu,
            ZERO,
            wv,
            wvOff,
            ldwv
          );
          dlacpy(ALL, jlen, nu, wv, wvOff, ldwv, z, Zi(jrow, incol + k1), ldz);
        }
      }
    }
  } // end incol loop
}
