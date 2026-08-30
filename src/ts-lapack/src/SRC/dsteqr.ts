// Translated from SRC/dsteqr.f
// DSTEQR computes all eigenvalues and, optionally, eigenvectors of a
// symmetric tridiagonal matrix using the implicit QL or QR method.
//
// compz codes:
//   0 = 'N' — eigenvalues only
//   1 = 'V' — eigenvalues and eigenvectors of the original symmetric matrix;
//             Z must contain the orthogonal matrix used to reduce it to
//             tridiagonal form (e.g. from DORGTR)
//   2 = 'I' — eigenvalues and eigenvectors of the tridiagonal matrix; Z is
//             initialized to the identity here
//
// On normal exit D holds the eigenvalues in ascending order and E is
// destroyed.  WORK must have length >= max(1, 2*n-2) when compz > 0.
//
// Array indexing convention (column-major):
//   D(I)   => d[dOff + (I-1)]
//   E(I)   => e[eOff + (I-1)]
//   Z(I,J) => z[zOff + (I-1) + (J-1)*ldz]   (I,J are 1-based)
//
// Returns INFO: 0 = success, < 0 = illegal argument, > 0 = that many
// off-diagonal elements failed to converge to zero.

import { dlamch } from "./dlamch.js";
import { dlanst } from "./dlanst.js";
import { dlapy2 } from "./dlapy2.js";
import { dlae2 } from "./dlae2.js";
import { dlaev2 } from "./dlaev2.js";
import { dlartg } from "./dlartg.js";
import { dlascl } from "./dlascl.js";
import { dlaset } from "./dlaset.js";
import { dlasr } from "./dlasr.js";
import { dlasrt } from "./dlasrt.js";
import { dswap } from "../BLAS/dswap.js";
import { xerbla } from "../utils/xerbla.js";
import {
  MACH_EPS,
  MACH_SFMIN,
  RIGHT,
  PIVOT_V,
  DIRECT_F,
  DIRECT_B,
  SORT_INC,
} from "../utils/constants.js";

// compz constants
export const COMPZ_N = 0; // 'N'
export const COMPZ_V = 1; // 'V'
export const COMPZ_I = 2; // 'I'

const NORM_MAX = 0; // dlanst 'M'
const TYPE_G = 0; // dlascl 'G'
const UPLO_FULL = -1; // dlaset 'Full' (neither UPPER nor LOWER)

// Fortran SIGN(a,b): |a| with the sign of b (b = 0 counts as positive).
function fsign(a: number, b: number): number {
  return b >= 0 ? Math.abs(a) : -Math.abs(a);
}

export function dsteqr(
  compz: number,
  n: number,
  d: Float64Array,
  dOff: number,
  e: Float64Array,
  eOff: number,
  z: Float64Array,
  zOff: number,
  ldz: number,
  work: Float64Array,
  workOff: number
): number {
  const ZERO = 0.0;
  const ONE = 1.0;
  const TWO = 2.0;
  const THREE = 3.0;
  const MAXIT = 30;

  let info = 0;
  if (compz !== COMPZ_N && compz !== COMPZ_V && compz !== COMPZ_I) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ldz < 1 || (compz > 0 && ldz < Math.max(1, n))) {
    info = -6;
  }
  if (info !== 0) {
    xerbla("DSTEQR", -info);
    return info;
  }

  // Quick return if possible
  if (n === 0) return 0;
  if (n === 1) {
    if (compz === COMPZ_I) z[zOff] = ONE;
    return 0;
  }

  const D = (i: number) => dOff + (i - 1);
  const E = (i: number) => eOff + (i - 1);
  const Zc = (j: number) => zOff + (j - 1) * ldz; // address of Z(1,j)
  const W = (i: number) => workOff + (i - 1);

  // Determine the unit roundoff and over/underflow thresholds
  const eps = dlamch(MACH_EPS);
  const eps2 = eps * eps;
  const safmin = dlamch(MACH_SFMIN);
  const safmax = ONE / safmin;
  const ssfmax = Math.sqrt(safmax) / THREE;
  const ssfmin = Math.sqrt(safmin) / eps2;

  // Compute the eigenvalues and eigenvectors of the tridiagonal matrix
  if (compz === COMPZ_I) dlaset(UPLO_FULL, n, n, ZERO, ONE, z, zOff, ldz);

  const nmaxit = n * MAXIT;
  let jtot = 0;

  // Determine where the matrix splits and choose QL or QR iteration for each
  // block, according to whether top or bottom diagonal element is smaller.
  let l1 = 1;
  const nm1 = n - 1;

  let ordered = false; // reached the "order eigenvalues" tail (label 160)

  // label 10
  outer: for (;;) {
    if (l1 > n) {
      ordered = true;
      break outer;
    }
    if (l1 > 1) e[E(l1 - 1)] = ZERO;
    let m = n;
    if (l1 <= nm1) {
      let split = false;
      for (m = l1; m <= nm1; m++) {
        const tst = Math.abs(e[E(m)]);
        if (tst === ZERO) {
          split = true;
          break;
        }
        if (
          tst <=
          Math.sqrt(Math.abs(d[D(m)])) * Math.sqrt(Math.abs(d[D(m + 1)])) * eps
        ) {
          e[E(m)] = ZERO;
          split = true;
          break;
        }
      }
      if (!split) m = n;
    }

    // label 30
    let l = l1;
    const lsv = l;
    let lend = m;
    const lendsv = lend;
    l1 = m + 1;
    if (lend === l) continue outer;

    // Scale submatrix in rows and columns L to LEND
    const anorm = dlanst(NORM_MAX, lend - l + 1, d, D(l), e, E(l));
    let iscale = 0;
    if (anorm === ZERO) continue outer;
    if (anorm > ssfmax) {
      iscale = 1;
      dlascl(TYPE_G, 0, 0, anorm, ssfmax, lend - l + 1, 1, d, D(l), n);
      dlascl(TYPE_G, 0, 0, anorm, ssfmax, lend - l, 1, e, E(l), n);
    } else if (anorm < ssfmin) {
      iscale = 2;
      dlascl(TYPE_G, 0, 0, anorm, ssfmin, lend - l + 1, 1, d, D(l), n);
      dlascl(TYPE_G, 0, 0, anorm, ssfmin, lend - l, 1, e, E(l), n);
    }

    // Choose between QL and QR iteration
    if (Math.abs(d[D(lend)]) < Math.abs(d[D(l)])) {
      lend = lsv;
      l = lendsv;
    }

    if (lend > l) {
      // ── QL iteration ──────────────────────────────────────────────────
      // label 40
      ql: for (;;) {
        // Look for small subdiagonal element
        if (l !== lend) {
          let found = false;
          for (m = l; m <= lend - 1; m++) {
            const tst = Math.abs(e[E(m)]) ** 2;
            if (
              tst <=
              eps2 * Math.abs(d[D(m)]) * Math.abs(d[D(m + 1)]) + safmin
            ) {
              found = true;
              break;
            }
          }
          if (!found) m = lend;
        } else {
          m = lend;
        }

        // label 60
        if (m < lend) e[E(m)] = ZERO;
        let p = d[D(l)];
        if (m === l) {
          // label 80: eigenvalue found
          d[D(l)] = p;
          l = l + 1;
          if (l <= lend) continue ql;
          break ql;
        }

        // If remaining matrix is 2-by-2, use DLAE2 or DLAEV2 to compute its
        // eigensystem.
        if (m === l + 1) {
          if (compz > 0) {
            const r = dlaev2(d[D(l)], e[E(l)], d[D(l + 1)]);
            work[W(l)] = r.cs1;
            work[W(n - 1 + l)] = r.sn1;
            dlasr(
              RIGHT,
              PIVOT_V,
              DIRECT_B,
              n,
              2,
              work,
              W(l),
              work,
              W(n - 1 + l),
              z,
              Zc(l),
              ldz
            );
            d[D(l)] = r.rt1;
            d[D(l + 1)] = r.rt2;
          } else {
            const r = dlae2(d[D(l)], e[E(l)], d[D(l + 1)]);
            d[D(l)] = r.rt1;
            d[D(l + 1)] = r.rt2;
          }
          e[E(l)] = ZERO;
          l = l + 2;
          if (l <= lend) continue ql;
          break ql;
        }

        if (jtot === nmaxit) break ql;
        jtot = jtot + 1;

        // Form shift
        let g = (d[D(l + 1)] - p) / (TWO * e[E(l)]);
        let r = dlapy2(g, ONE);
        g = d[D(m)] - p + e[E(l)] / (g + fsign(r, g));

        let s = ONE;
        let c = ONE;
        p = ZERO;

        // Inner loop
        for (let i = m - 1; i >= l; i--) {
          const f = s * e[E(i)];
          const b = c * e[E(i)];
          const rot = dlartg(g, f);
          c = rot.cs;
          s = rot.sn;
          r = rot.r;
          if (i !== m - 1) e[E(i + 1)] = r;
          g = d[D(i + 1)] - p;
          r = (d[D(i)] - g) * s + TWO * c * b;
          p = s * r;
          d[D(i + 1)] = g + p;
          g = c * r - b;
          // If eigenvectors are desired, then save rotations
          if (compz > 0) {
            work[W(i)] = c;
            work[W(n - 1 + i)] = -s;
          }
        }

        // If eigenvectors are desired, then apply saved rotations
        if (compz > 0) {
          const mm = m - l + 1;
          dlasr(
            RIGHT,
            PIVOT_V,
            DIRECT_B,
            n,
            mm,
            work,
            W(l),
            work,
            W(n - 1 + l),
            z,
            Zc(l),
            ldz
          );
        }

        d[D(l)] = d[D(l)] - p;
        e[E(l)] = g;
        continue ql;
      }
    } else {
      // ── QR iteration ──────────────────────────────────────────────────
      // label 90
      qr: for (;;) {
        // Look for small superdiagonal element
        if (l !== lend) {
          let found = false;
          for (m = l; m >= lend + 1; m--) {
            const tst = Math.abs(e[E(m - 1)]) ** 2;
            if (
              tst <=
              eps2 * Math.abs(d[D(m)]) * Math.abs(d[D(m - 1)]) + safmin
            ) {
              found = true;
              break;
            }
          }
          if (!found) m = lend;
        } else {
          m = lend;
        }

        // label 110
        if (m > lend) e[E(m - 1)] = ZERO;
        let p = d[D(l)];
        if (m === l) {
          // label 130: eigenvalue found
          d[D(l)] = p;
          l = l - 1;
          if (l >= lend) continue qr;
          break qr;
        }

        // If remaining matrix is 2-by-2, use DLAE2 or DLAEV2 to compute its
        // eigensystem.
        if (m === l - 1) {
          if (compz > 0) {
            const r = dlaev2(d[D(l - 1)], e[E(l - 1)], d[D(l)]);
            work[W(m)] = r.cs1;
            work[W(n - 1 + m)] = r.sn1;
            dlasr(
              RIGHT,
              PIVOT_V,
              DIRECT_F,
              n,
              2,
              work,
              W(m),
              work,
              W(n - 1 + m),
              z,
              Zc(l - 1),
              ldz
            );
            d[D(l - 1)] = r.rt1;
            d[D(l)] = r.rt2;
          } else {
            const r = dlae2(d[D(l - 1)], e[E(l - 1)], d[D(l)]);
            d[D(l - 1)] = r.rt1;
            d[D(l)] = r.rt2;
          }
          e[E(l - 1)] = ZERO;
          l = l - 2;
          if (l >= lend) continue qr;
          break qr;
        }

        if (jtot === nmaxit) break qr;
        jtot = jtot + 1;

        // Form shift
        let g = (d[D(l - 1)] - p) / (TWO * e[E(l - 1)]);
        let r = dlapy2(g, ONE);
        g = d[D(m)] - p + e[E(l - 1)] / (g + fsign(r, g));

        let s = ONE;
        let c = ONE;
        p = ZERO;

        // Inner loop
        for (let i = m; i <= l - 1; i++) {
          const f = s * e[E(i)];
          const b = c * e[E(i)];
          const rot = dlartg(g, f);
          c = rot.cs;
          s = rot.sn;
          r = rot.r;
          if (i !== m) e[E(i - 1)] = r;
          g = d[D(i)] - p;
          r = (d[D(i + 1)] - g) * s + TWO * c * b;
          p = s * r;
          d[D(i)] = g + p;
          g = c * r - b;
          // If eigenvectors are desired, then save rotations
          if (compz > 0) {
            work[W(i)] = c;
            work[W(n - 1 + i)] = s;
          }
        }

        // If eigenvectors are desired, then apply saved rotations
        if (compz > 0) {
          const mm = l - m + 1;
          dlasr(
            RIGHT,
            PIVOT_V,
            DIRECT_F,
            n,
            mm,
            work,
            W(m),
            work,
            W(n - 1 + m),
            z,
            Zc(m),
            ldz
          );
        }

        d[D(l)] = d[D(l)] - p;
        e[E(l - 1)] = g;
        continue qr;
      }
    }

    // label 140: undo scaling if necessary
    if (iscale === 1) {
      dlascl(TYPE_G, 0, 0, ssfmax, anorm, lendsv - lsv + 1, 1, d, D(lsv), n);
      dlascl(TYPE_G, 0, 0, ssfmax, anorm, lendsv - lsv, 1, e, E(lsv), n);
    } else if (iscale === 2) {
      dlascl(TYPE_G, 0, 0, ssfmin, anorm, lendsv - lsv + 1, 1, d, D(lsv), n);
      dlascl(TYPE_G, 0, 0, ssfmin, anorm, lendsv - lsv, 1, e, E(lsv), n);
    }

    // Check for no convergence to an eigenvalue after a total of N*MAXIT
    // iterations.
    if (jtot < nmaxit) continue outer;
    for (let i = 1; i <= n - 1; i++) {
      if (e[E(i)] !== ZERO) info = info + 1;
    }
    break outer;
  }

  if (!ordered) return info;

  // label 160: order eigenvalues and eigenvectors
  if (compz === COMPZ_N) {
    // Use quick sort
    const sortInfo = dlasrt(SORT_INC, n, d, dOff);
    if (sortInfo !== 0) return sortInfo;
  } else {
    // Use selection sort to minimize swaps of eigenvectors
    for (let ii = 2; ii <= n; ii++) {
      const i = ii - 1;
      let k = i;
      let p = d[D(i)];
      for (let j = ii; j <= n; j++) {
        if (d[D(j)] < p) {
          k = j;
          p = d[D(j)];
        }
      }
      if (k !== i) {
        d[D(k)] = d[D(i)];
        d[D(i)] = p;
        dswap(n, z, Zc(i), 1, z, Zc(k), 1);
      }
    }
  }

  return info;
}
