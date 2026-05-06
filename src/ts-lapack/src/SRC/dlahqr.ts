// Translated from SRC/dlahqr.f
// DLAHQR computes the eigenvalues and Schur factorization of an upper
// Hessenberg matrix, using the double-shift/single-shift QR algorithm.
//
// DLAHQR is an auxiliary routine called by DHSEQR to update the
// eigenvalues and Schur decomposition already computed by DHSEQR, by
// dealing with the Hessenberg submatrix in rows and columns ILO to IHI.
//
// Array indexing convention (column-major, matching Fortran):
//   H(I,J) => h[hOff + (I-1) + (J-1)*ldh]   (I,J are 1-based)
//   WR(I)  => wr[wrOff + (I-1)]              (I is 1-based)
//   WI(I)  => wi[wiOff + (I-1)]              (I is 1-based)
//   Z(I,J) => z[zOff + (I-1) + (J-1)*ldz]   (I,J are 1-based)

import { dlamch } from "./dlamch.js";
import { dlanv2 } from "./dlanv2.js";
import { dlarfg } from "./dlarfg.js";
import { dcopy } from "../BLAS/dcopy.js";
import { drot } from "../BLAS/drot.js";
import { MACH_SFMIN, MACH_PREC } from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/executors/jsJit/helpers/alloc.js";

/**
 * DLAHQR computes the eigenvalues and optionally the Schur factorization
 * of an upper Hessenberg matrix H.
 *
 * @param wantt - true: compute full Schur form T; false: only eigenvalues
 * @param wantz - true: accumulate transformations into Z; false: no Z update
 * @param n - order of the matrix H (>= 0)
 * @param ilo - 1-based index, first row/col of active block
 * @param ihi - 1-based index, last row/col of active block
 * @param h - Hessenberg matrix, dimension (ldh, n)
 * @param hOff - offset into h
 * @param ldh - leading dimension of h
 * @param wr - real parts of eigenvalues, dimension (n)
 * @param wrOff - offset into wr
 * @param wi - imaginary parts of eigenvalues, dimension (n)
 * @param wiOff - offset into wi
 * @param iloz - first row of Z to which transformations apply (1-based)
 * @param ihiz - last row of Z to which transformations apply (1-based)
 * @param z - Schur vectors matrix, dimension (ldz, n)
 * @param zOff - offset into z
 * @param ldz - leading dimension of z
 * @returns INFO: 0 = success, > 0 = convergence failure at index
 */
export function dlahqr(
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
  ldz: number
): number {
  const ZERO = 0.0;
  const ONE = 1.0;
  const TWO = 2.0;
  const DAT1 = 3.0 / 4.0;
  const DAT2 = -0.4375;
  const KEXSH = 10;

  // Helpers for column-major indexing (1-based I,J)
  const H = (i: number, j: number) => hOff + (i - 1) + (j - 1) * ldh;
  const Z = (i: number, j: number) => zOff + (i - 1) + (j - 1) * ldz;

  let info = 0;

  // Quick return if possible
  if (n === 0) return 0;
  if (ilo === ihi) {
    wr[wrOff + (ilo - 1)] = h[H(ilo, ilo)];
    wi[wiOff + (ilo - 1)] = ZERO;
    return 0;
  }

  // Clear out the trash
  for (let j = ilo; j <= ihi - 3; j++) {
    h[H(j + 2, j)] = ZERO;
    h[H(j + 3, j)] = ZERO;
  }
  if (ilo <= ihi - 2) {
    h[H(ihi, ihi - 2)] = ZERO;
  }

  const nh = ihi - ilo + 1;
  const nz = ihiz - iloz + 1;

  // Set machine-dependent constants for the stopping criterion.
  const safmin = dlamch(MACH_SFMIN);
  // const safmax = ONE / safmin;
  const ulp = dlamch(MACH_PREC);
  const smlnum = safmin * (nh / ulp);

  // I1 and I2 are the indices of the first row and last column of H
  // to which transformations must be applied.
  let i1: number;
  let i2: number;
  if (wantt) {
    i1 = 1;
    i2 = n;
  } else {
    i1 = 0; // Will be set inside the main loop
    i2 = 0;
  }

  // ITMAX is the total number of QR iterations allowed.
  const itmax = 30 * Math.max(10, nh);

  // KDEFL counts the number of iterations since a deflation
  let kdefl = 0;

  // The main loop begins here. i is the loop index and decreases from
  // IHI to ILO in steps of 1 or 2.
  let i = ihi;

  // Local arrays
  const v = allocFloat64Array(3);

  // Outer loop (label 20 in Fortran)
  outer: while (true) {
    let l = ilo;
    if (i < ilo) break; // goto 160

    // Perform QR iterations on rows and columns ILO to I until a
    // submatrix of order 1 or 2 splits off at the bottom.
    for (let its = 0; its <= itmax; its++) {
      // Look for a single small subdiagonal element.
      let k: number;
      for (k = i; k >= l + 1; k--) {
        if (Math.abs(h[H(k, k - 1)]) <= smlnum) break;
        let tst = Math.abs(h[H(k - 1, k - 1)]) + Math.abs(h[H(k, k)]);
        if (tst === ZERO) {
          if (k - 2 >= ilo) tst = tst + Math.abs(h[H(k - 1, k - 2)]);
          if (k + 1 <= ihi) tst = tst + Math.abs(h[H(k + 1, k)]);
        }
        // Conservative small subdiagonal deflation criterion
        // (Ahues & Tisseur, LAWN 122, 1997)
        if (Math.abs(h[H(k, k - 1)]) <= ulp * tst) {
          const ab = Math.max(
            Math.abs(h[H(k, k - 1)]),
            Math.abs(h[H(k - 1, k)])
          );
          const ba = Math.min(
            Math.abs(h[H(k, k - 1)]),
            Math.abs(h[H(k - 1, k)])
          );
          const aa = Math.max(
            Math.abs(h[H(k, k)]),
            Math.abs(h[H(k - 1, k - 1)] - h[H(k, k)])
          );
          const bb = Math.min(
            Math.abs(h[H(k, k)]),
            Math.abs(h[H(k - 1, k - 1)] - h[H(k, k)])
          );
          const s = aa + ab;
          if (ba * (ab / s) <= Math.max(smlnum, ulp * (bb * (aa / s)))) break;
        }
      }
      l = k;
      if (l > ilo) {
        // H(L,L-1) is negligible
        h[H(l, l - 1)] = ZERO;
      }

      // Exit from loop if a submatrix of order 1 or 2 has split off.
      if (l >= i - 1) {
        // goto 150 - deflation
        if (l === i) {
          // One eigenvalue has converged.
          wr[wrOff + (i - 1)] = h[H(i, i)];
          wi[wiOff + (i - 1)] = ZERO;
        } else if (l === i - 1) {
          // A pair of eigenvalues have converged.
          // Transform the 2-by-2 submatrix to standard Schur form.
          const res = dlanv2(
            h[H(i - 1, i - 1)],
            h[H(i - 1, i)],
            h[H(i, i - 1)],
            h[H(i, i)]
          );
          h[H(i - 1, i - 1)] = res.a;
          h[H(i - 1, i)] = res.b;
          h[H(i, i - 1)] = res.c;
          h[H(i, i)] = res.d;
          wr[wrOff + (i - 2)] = res.rt1r;
          wi[wiOff + (i - 2)] = res.rt1i;
          wr[wrOff + (i - 1)] = res.rt2r;
          wi[wiOff + (i - 1)] = res.rt2i;
          const cs = res.cs;
          const sn = res.sn;

          if (wantt) {
            // Apply the transformation to the rest of H.
            if (i2 > i) {
              drot(
                i2 - i,
                h,
                H(i - 1, i + 1),
                ldh,
                h,
                H(i, i + 1),
                ldh,
                cs,
                sn
              );
            }
            drot(i - i1 - 1, h, H(i1, i - 1), 1, h, H(i1, i), 1, cs, sn);
          }
          if (wantz) {
            // Apply the transformation to Z.
            drot(nz, z, Z(iloz, i - 1), 1, z, Z(iloz, i), 1, cs, sn);
          }
        }
        // reset deflation counter
        kdefl = 0;

        // return to start of the main loop with new value of I.
        i = l - 1;
        continue outer;
      }
      kdefl = kdefl + 1;

      // Now the active submatrix is in rows and columns L to I.
      if (!wantt) {
        i1 = l;
        i2 = i;
      }

      let h11: number, h12: number, h21: number, h22: number;
      if (kdefl % (2 * KEXSH) === 0) {
        // Exceptional shift.
        const s = Math.abs(h[H(i, i - 1)]) + Math.abs(h[H(i - 1, i - 2)]);
        h11 = DAT1 * s + h[H(i, i)];
        h12 = DAT2 * s;
        h21 = s;
        h22 = h11;
      } else if (kdefl % KEXSH === 0) {
        // Exceptional shift.
        const s = Math.abs(h[H(l + 1, l)]) + Math.abs(h[H(l + 2, l + 1)]);
        h11 = DAT1 * s + h[H(l, l)];
        h12 = DAT2 * s;
        h21 = s;
        h22 = h11;
      } else {
        // Prepare to use Francis' double shift
        h11 = h[H(i - 1, i - 1)];
        h21 = h[H(i, i - 1)];
        h12 = h[H(i - 1, i)];
        h22 = h[H(i, i)];
      }

      let s = Math.abs(h11) + Math.abs(h12) + Math.abs(h21) + Math.abs(h22);
      let rt1r: number, rt1i: number, rt2r: number, rt2i: number;
      if (s === ZERO) {
        rt1r = ZERO;
        rt1i = ZERO;
        rt2r = ZERO;
        rt2i = ZERO;
      } else {
        h11 = h11 / s;
        h21 = h21 / s;
        h12 = h12 / s;
        h22 = h22 / s;
        const tr = (h11 + h22) / TWO;
        const det = (h11 - tr) * (h22 - tr) - h12 * h21;
        const rtdisc = Math.sqrt(Math.abs(det));
        if (det >= ZERO) {
          // complex conjugate shifts
          rt1r = tr * s;
          rt2r = rt1r;
          rt1i = rtdisc * s;
          rt2i = -rt1i;
        } else {
          // real shifts (use only one of them)
          rt1r = tr + rtdisc;
          rt2r = tr - rtdisc;
          if (Math.abs(rt1r - h22) <= Math.abs(rt2r - h22)) {
            rt1r = rt1r * s;
            rt2r = rt1r;
          } else {
            rt2r = rt2r * s;
            rt1r = rt2r;
          }
          rt1i = ZERO;
          rt2i = ZERO;
        }
      }

      // Look for two consecutive small subdiagonal elements.
      let m: number;
      for (m = i - 2; m >= l; m--) {
        const h21s = h[H(m + 1, m)];
        s = Math.abs(h[H(m, m)] - rt2r) + Math.abs(rt2i) + Math.abs(h21s);
        const h21sNorm = h[H(m + 1, m)] / s;
        v[0] =
          h21sNorm * h[H(m, m + 1)] +
          (h[H(m, m)] - rt1r) * ((h[H(m, m)] - rt2r) / s) -
          rt1i * (rt2i / s);
        v[1] = h21sNorm * (h[H(m, m)] + h[H(m + 1, m + 1)] - rt1r - rt2r);
        v[2] = h21sNorm * h[H(m + 2, m + 1)];
        s = Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]);
        v[0] = v[0] / s;
        v[1] = v[1] / s;
        v[2] = v[2] / s;
        if (m === l) break;
        if (
          Math.abs(h[H(m, m - 1)]) * (Math.abs(v[1]) + Math.abs(v[2])) <=
          ulp *
            Math.abs(v[0]) *
            (Math.abs(h[H(m - 1, m - 1)]) +
              Math.abs(h[H(m, m)]) +
              Math.abs(h[H(m + 1, m + 1)]))
        )
          break;
      }

      // Double-shift QR step
      for (k = m; k <= i - 1; k++) {
        const nr = Math.min(3, i - k + 1);
        if (k > m) {
          dcopy(nr, h, H(k, k - 1), 1, v, 0, 1);
        }
        const rfg = dlarfg(nr, v[0], v, 1, 1);
        v[0] = rfg.alpha;
        const t1 = rfg.tau;
        if (k > m) {
          h[H(k, k - 1)] = v[0];
          h[H(k + 1, k - 1)] = ZERO;
          if (k < i - 1) h[H(k + 2, k - 1)] = ZERO;
        } else if (m > l) {
          // Use the following instead of H(K,K-1) = -H(K,K-1) to
          // avoid a bug when v(2) and v(3) underflow.
          h[H(k, k - 1)] = h[H(k, k - 1)] * (ONE - t1);
        }
        const v2 = v[1];
        const t2 = t1 * v2;
        if (nr === 3) {
          const v3 = v[2];
          const t3 = t1 * v3;

          // Apply G from the left to transform the rows of the matrix
          // in columns K to I2.
          for (let j = k; j <= i2; j++) {
            const sum = h[H(k, j)] + v2 * h[H(k + 1, j)] + v3 * h[H(k + 2, j)];
            h[H(k, j)] = h[H(k, j)] - sum * t1;
            h[H(k + 1, j)] = h[H(k + 1, j)] - sum * t2;
            h[H(k + 2, j)] = h[H(k + 2, j)] - sum * t3;
          }

          // Apply G from the right to transform the columns of the
          // matrix in rows I1 to min(K+3,I).
          for (let j = i1; j <= Math.min(k + 3, i); j++) {
            const sum = h[H(j, k)] + v2 * h[H(j, k + 1)] + v3 * h[H(j, k + 2)];
            h[H(j, k)] = h[H(j, k)] - sum * t1;
            h[H(j, k + 1)] = h[H(j, k + 1)] - sum * t2;
            h[H(j, k + 2)] = h[H(j, k + 2)] - sum * t3;
          }

          if (wantz) {
            // Accumulate transformations in the matrix Z
            for (let j = iloz; j <= ihiz; j++) {
              const sum =
                z[Z(j, k)] + v2 * z[Z(j, k + 1)] + v3 * z[Z(j, k + 2)];
              z[Z(j, k)] = z[Z(j, k)] - sum * t1;
              z[Z(j, k + 1)] = z[Z(j, k + 1)] - sum * t2;
              z[Z(j, k + 2)] = z[Z(j, k + 2)] - sum * t3;
            }
          }
        } else if (nr === 2) {
          // Apply G from the left to transform the rows of the matrix
          // in columns K to I2.
          for (let j = k; j <= i2; j++) {
            const sum = h[H(k, j)] + v2 * h[H(k + 1, j)];
            h[H(k, j)] = h[H(k, j)] - sum * t1;
            h[H(k + 1, j)] = h[H(k + 1, j)] - sum * t2;
          }

          // Apply G from the right to transform the columns of the
          // matrix in rows I1 to I.
          for (let j = i1; j <= i; j++) {
            const sum = h[H(j, k)] + v2 * h[H(j, k + 1)];
            h[H(j, k)] = h[H(j, k)] - sum * t1;
            h[H(j, k + 1)] = h[H(j, k + 1)] - sum * t2;
          }

          if (wantz) {
            // Accumulate transformations in the matrix Z
            for (let j = iloz; j <= ihiz; j++) {
              const sum = z[Z(j, k)] + v2 * z[Z(j, k + 1)];
              z[Z(j, k)] = z[Z(j, k)] - sum * t1;
              z[Z(j, k + 1)] = z[Z(j, k + 1)] - sum * t2;
            }
          }
        }
      }
    }

    // Failure to converge in remaining number of iterations
    info = i;
    return info;
  }

  // label 160
  return 0;
}
