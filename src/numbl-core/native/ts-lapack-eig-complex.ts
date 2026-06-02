/**
 * Pure-TypeScript complex non-symmetric eigensolver — a zgeev-equivalent for
 * the ts-lapack bridge (browser path, no native LAPACK addon).
 *
 * Ported from the netlib LAPACK reference (zgeev → zgehd2 / zunghr → zlahqr →
 * ztrevc). Simplifications relative to zgeev, valid for the well-scaled inputs
 * numbl produces (the acceptance bar is residual-based, A*V≈V*D / W^H*A≈D*W^H,
 * not bit-matching MKL):
 *   - No balancing (zgebal/zgebak) — ILO=1, IHI=N.
 *   - No norm scaling (zlascl).
 *   - Eigenvector back-substitution uses plain triangular solves with the
 *     ztrevc SMIN diagonal regularization, instead of the overflow-safe zlatrs.
 *
 * Eigenvectors are normalized to unit Euclidean norm with the largest-magnitude
 * component made real (the documented zgeev convention).
 *
 * All matrices are column-major with leading dimension n, split into parallel
 * real/imag Float64Arrays — matching the bridge's eigComplex I/O shape.
 */

import { allocFloat64Array } from "../runtime/alloc.js";

const EPS = 2.220446049250313e-16; // DLAMCH('P') (relative machine precision)
const SAFMIN = 2.2250738585072014e-308; // DLAMCH('S') (smallest normal)

/** |Re| + |Im| (LAPACK CABS1). */
function cabs1(re: number, im: number): number {
  return Math.abs(re) + Math.abs(im);
}

/** Robust complex divide (a/b) via Smith's algorithm (DLADIV basic path). */
function cdiv(
  ar: number,
  ai: number,
  br: number,
  bi: number
): [number, number] {
  if (Math.abs(br) >= Math.abs(bi)) {
    const r = bi / br;
    const t = br + bi * r;
    return [(ar + ai * r) / t, (ai - ar * r) / t];
  } else {
    const r = br / bi;
    const t = br * r + bi;
    return [(ar * r + ai) / t, (ai * r - ar) / t];
  }
}

/** Fortran SIGN(mag, s): |mag| with the sign of s. */
function fsign(mag: number, s: number): number {
  return s >= 0 ? Math.abs(mag) : -Math.abs(mag);
}

/**
 * ZLARFG (no underflow-rescale loop): generate an elementary complex reflector
 * H = I - tau*v*v^H with H^H*[alpha; x] = [beta; 0], beta real. On entry
 * re/im[off] = alpha and the X vector is re/im[off + i*incx], i=1..n-1. On exit
 * re/im[off] = beta and the X part holds v(2:); returns tau.
 */
function zlarfg(
  n: number,
  re: Float64Array,
  im: Float64Array,
  off: number,
  incx: number
): [number, number] {
  if (n <= 0) return [0, 0];
  // xnorm = ||X||_2 over the n-1 trailing elements
  let ssq = 0;
  for (let i = 1; i < n; i++) {
    const o = off + i * incx;
    ssq += re[o] * re[o] + im[o] * im[o];
  }
  const xnorm = Math.sqrt(ssq);
  const alphr = re[off];
  const alphi = im[off];
  if (xnorm === 0 && alphi === 0) return [0, 0];

  const beta = -fsign(Math.hypot(alphr, alphi, xnorm), alphr);
  const tauRe = (beta - alphr) / beta;
  const tauIm = -alphi / beta;
  const [sRe, sIm] = cdiv(1, 0, alphr - beta, alphi); // 1/(alpha-beta)
  for (let i = 1; i < n; i++) {
    const o = off + i * incx;
    const xr = re[o];
    const xi = im[o];
    re[o] = xr * sRe - xi * sIm;
    im[o] = xr * sIm + xi * sRe;
  }
  re[off] = beta;
  im[off] = 0;
  return [tauRe, tauIm];
}

/**
 * Reduce a general complex matrix (in HRe/HIm) to upper Hessenberg form by
 * Householder similarity, accumulating the unitary Q into ZRe/ZIm so that
 * A = Q H Q^H. Mirrors zgehd2 (ILO=1, IHI=N) plus explicit Q accumulation.
 */
function hessenberg(
  n: number,
  HRe: Float64Array,
  HIm: Float64Array,
  ZRe: Float64Array,
  ZIm: Float64Array,
  wantZ: boolean
): void {
  const v = allocFloat64Array(n); // v(1)=1 implicit; store v(2:) here
  const vIm = allocFloat64Array(n);
  for (let i = 1; i <= n - 2; i++) {
    // reflector from column i, rows i+1..n
    const colOff = (i - 1) * n; // H(_, i) base; row r at colOff + (r-1)
    const [tauRe, tauIm] = zlarfg(n - i, HRe, HIm, colOff + i, 1);
    // v(1)=1, v(2..n-i) = H(i+2..n, i)
    v[0] = 1;
    vIm[0] = 0;
    for (let r = i + 2; r <= n; r++) {
      v[r - i - 1] = HRe[colOff + (r - 1)];
      vIm[r - i - 1] = HIm[colOff + (r - 1)];
    }
    if (tauRe !== 0 || tauIm !== 0) {
      // Right: A(1:n, i+1:n) := A (I - tau v v^H)   [v indexed over rows i+1..n]
      applyReflectorRight(HRe, HIm, n, 1, n, i + 1, n, v, vIm, tauRe, tauIm);
      // Left: A(i+1:n, i+1:n) := (I - conj(tau) v v^H) A
      applyReflectorLeft(
        HRe,
        HIm,
        n,
        i + 1,
        n,
        i + 1,
        n,
        v,
        vIm,
        tauRe,
        -tauIm
      );
      if (wantZ) {
        // Q := Q (I - tau v v^H), columns i+1..n
        applyReflectorRight(ZRe, ZIm, n, 1, n, i + 1, n, v, vIm, tauRe, tauIm);
      }
    }
    // finalize Hessenberg column: subdiagonal already = beta in H(i+1,i),
    // zero the rest below.
    for (let r = i + 2; r <= n; r++) {
      HRe[colOff + (r - 1)] = 0;
      HIm[colOff + (r - 1)] = 0;
    }
  }
}

/**
 * Apply C := (I - tau v v^H) C to rows r0..r1, cols c0..c1 of C (col-major, ld=n).
 * The reflector v has length (r1-r0+1) and is read from vRe/vIm[p] for
 * p = 0..(r1-r0), where row r0+p uses vRe/vIm[p] (and v[0] = 1 implicitly).
 */
function applyReflectorLeft(
  CRe: Float64Array,
  CIm: Float64Array,
  n: number,
  r0: number,
  r1: number,
  c0: number,
  c1: number,
  vRe: Float64Array,
  vIm: Float64Array,
  tauRe: number,
  tauIm: number
): void {
  for (let j = c0; j <= c1; j++) {
    const cb = (j - 1) * n;
    // w = v^H * C(r0:r1, j) = sum conj(v_p) C(r0+p, j)
    let wr = 0;
    let wi = 0;
    for (let p = 0; p <= r1 - r0; p++) {
      const o = cb + (r0 - 1 + p);
      const vr = vRe[p];
      const vi = vIm[p];
      // conj(v)*C = (vr - i vi)(cr + i ci)
      wr += vr * CRe[o] + vi * CIm[o];
      wi += vr * CIm[o] - vi * CRe[o];
    }
    // C(r0+p, j) -= tau * v_p * w
    const twr = tauRe * wr - tauIm * wi;
    const twi = tauRe * wi + tauIm * wr;
    for (let p = 0; p <= r1 - r0; p++) {
      const o = cb + (r0 - 1 + p);
      const vr = vRe[p];
      const vi = vIm[p];
      // v_p * (tw)
      CRe[o] -= vr * twr - vi * twi;
      CIm[o] -= vr * twi + vi * twr;
    }
  }
}

/**
 * Apply C := C (I - tau v v^H) to rows r0..r1, cols c0..c1 of C.
 * v indexed over columns c0..c1 (v(1)=1 then v(2:)).
 */
function applyReflectorRight(
  CRe: Float64Array,
  CIm: Float64Array,
  n: number,
  r0: number,
  r1: number,
  c0: number,
  c1: number,
  vRe: Float64Array,
  vIm: Float64Array,
  tauRe: number,
  tauIm: number
): void {
  for (let i = r0; i <= r1; i++) {
    // w = C(i, c0:c1) * v = sum C(i, c0+p) v_p
    let wr = 0;
    let wi = 0;
    for (let p = 0; p <= c1 - c0; p++) {
      const o = (c0 - 1 + p) * n + (i - 1);
      const vr = vRe[p];
      const vi = vIm[p];
      wr += CRe[o] * vr - CIm[o] * vi;
      wi += CRe[o] * vi + CIm[o] * vr;
    }
    // C(i, c0+p) -= tau * w * conj(v_p)
    const twr = tauRe * wr - tauIm * wi;
    const twi = tauRe * wi + tauIm * wr;
    for (let p = 0; p <= c1 - c0; p++) {
      const o = (c0 - 1 + p) * n + (i - 1);
      const vr = vRe[p];
      const vi = vIm[p];
      // tw * conj(v) = (twr + i twi)(vr - i vi)
      CRe[o] -= twr * vr + twi * vi;
      CIm[o] -= twi * vr - twr * vi;
    }
  }
}

/**
 * ZLAHQR: eigenvalues and Schur factorization of the upper-Hessenberg matrix in
 * HRe/HIm (rows/cols ilo..ihi), single-shift complex QR. Faithful port.
 * Writes eigenvalues to wRe/wIm; if wantt, H becomes upper triangular; if wantz,
 * accumulates Schur vectors into ZRe/ZIm. Returns INFO (0 = success).
 */
function zlahqr(
  wantt: boolean,
  wantz: boolean,
  n: number,
  ilo: number,
  ihi: number,
  HRe: Float64Array,
  HIm: Float64Array,
  wRe: Float64Array,
  wIm: Float64Array,
  iloz: number,
  ihiz: number,
  ZRe: Float64Array,
  ZIm: Float64Array
): number {
  const H = (i: number, j: number) => (j - 1) * n + (i - 1);
  const Z = (i: number, j: number) => (j - 1) * n + (i - 1);
  const DAT1 = 0.75;
  const KEXSH = 10;

  if (n === 0) return 0;
  if (ilo === ihi) {
    wRe[ilo - 1] = HRe[H(ilo, ilo)];
    wIm[ilo - 1] = HIm[H(ilo, ilo)];
    return 0;
  }

  // clear out the trash
  for (let j = ilo; j <= ihi - 3; j++) {
    HRe[H(j + 2, j)] = 0;
    HIm[H(j + 2, j)] = 0;
    HRe[H(j + 3, j)] = 0;
    HIm[H(j + 3, j)] = 0;
  }
  if (ilo <= ihi - 2) {
    HRe[H(ihi, ihi - 2)] = 0;
    HIm[H(ihi, ihi - 2)] = 0;
  }

  // ensure subdiagonal entries are real
  const jlo = wantt ? 1 : ilo;
  const jhi = wantt ? n : ihi;
  for (let i = ilo + 1; i <= ihi; i++) {
    if (HIm[H(i, i - 1)] !== 0) {
      const hr = HRe[H(i, i - 1)];
      const hi = HIm[H(i, i - 1)];
      const c1 = cabs1(hr, hi);
      // sc = conj(h/|h|_1) / |h/|h|_1|
      let scRe = hr / c1;
      let scIm = hi / c1;
      const scAbs = Math.hypot(scRe, scIm);
      // conj(sc)/|sc|
      scRe = scRe / scAbs;
      scIm = -scIm / scAbs;
      const habs = Math.hypot(hr, hi);
      HRe[H(i, i - 1)] = habs;
      HIm[H(i, i - 1)] = 0;
      // ZSCAL(jhi-i+1, sc, H(i,i), ldh): row i, columns i..jhi
      zscalStrided(HRe, HIm, scRe, scIm, H(i, i), n, jhi - i + 1);
      // ZSCAL(min(jhi,i+1)-jlo+1, conj(sc), H(jlo,i), 1): col i, rows jlo..min(jhi,i+1)
      zscalStrided(
        HRe,
        HIm,
        scRe,
        -scIm,
        H(jlo, i),
        1,
        Math.min(jhi, i + 1) - jlo + 1
      );
      if (wantz) {
        zscalStrided(ZRe, ZIm, scRe, -scIm, Z(iloz, i), 1, ihiz - iloz + 1);
      }
    }
  }

  const nh = ihi - ilo + 1;
  const nz = ihiz - iloz + 1;
  const ulp = EPS;
  const smlnum = SAFMIN * (nh / ulp);

  let i1 = 1;
  let i2 = n;
  const itmax = 30 * Math.max(10, nh);
  let kdefl = 0;

  const vRe = new Float64Array(2);
  const vIm = new Float64Array(2);

  let i = ihi;
  // main loop (label 30)
  while (i >= ilo) {
    let l = ilo;
    let converged = false;
    for (let its = 0; its <= itmax; its++) {
      // look for a single small subdiagonal element
      let k: number;
      for (k = i; k >= l + 1; k--) {
        if (cabs1(HRe[H(k, k - 1)], HIm[H(k, k - 1)]) <= smlnum) break;
        let tst =
          cabs1(HRe[H(k - 1, k - 1)], HIm[H(k - 1, k - 1)]) +
          cabs1(HRe[H(k, k)], HIm[H(k, k)]);
        if (tst === 0) {
          if (k - 2 >= ilo) tst += Math.abs(HRe[H(k - 1, k - 2)]);
          if (k + 1 <= ihi) tst += Math.abs(HRe[H(k + 1, k)]);
        }
        if (Math.abs(HRe[H(k, k - 1)]) <= ulp * tst) {
          const ab = Math.max(
            cabs1(HRe[H(k, k - 1)], HIm[H(k, k - 1)]),
            cabs1(HRe[H(k - 1, k)], HIm[H(k - 1, k)])
          );
          const ba = Math.min(
            cabs1(HRe[H(k, k - 1)], HIm[H(k, k - 1)]),
            cabs1(HRe[H(k - 1, k)], HIm[H(k - 1, k)])
          );
          const diffRe = HRe[H(k - 1, k - 1)] - HRe[H(k, k)];
          const diffIm = HIm[H(k - 1, k - 1)] - HIm[H(k, k)];
          const aa = Math.max(
            cabs1(HRe[H(k, k)], HIm[H(k, k)]),
            cabs1(diffRe, diffIm)
          );
          const bb = Math.min(
            cabs1(HRe[H(k, k)], HIm[H(k, k)]),
            cabs1(diffRe, diffIm)
          );
          const s = aa + ab;
          if (ba * (ab / s) <= Math.max(smlnum, ulp * (bb * (aa / s)))) break;
        }
      }
      l = k;
      if (l > ilo) {
        HRe[H(l, l - 1)] = 0;
        HIm[H(l, l - 1)] = 0;
      }
      if (l >= i) {
        converged = true;
        break;
      }
      kdefl++;
      if (!wantt) {
        i1 = l;
        i2 = i;
      }

      // shift
      let tRe: number, tIm: number;
      if (kdefl % (2 * KEXSH) === 0) {
        const s = DAT1 * Math.abs(HRe[H(i, i - 1)]);
        tRe = s + HRe[H(i, i)];
        tIm = HIm[H(i, i)];
      } else if (kdefl % KEXSH === 0) {
        const s = DAT1 * Math.abs(HRe[H(l + 1, l)]);
        tRe = s + HRe[H(l, l)];
        tIm = HIm[H(l, l)];
      } else {
        // Wilkinson's shift
        tRe = HRe[H(i, i)];
        tIm = HIm[H(i, i)];
        // u = sqrt(H(i-1,i)) * sqrt(H(i,i-1))
        const [s1r, s1i] = csqrt(HRe[H(i - 1, i)], HIm[H(i - 1, i)]);
        const [s2r, s2i] = csqrt(HRe[H(i, i - 1)], HIm[H(i, i - 1)]);
        const uRe = s1r * s2r - s1i * s2i;
        const uIm = s1r * s2i + s1i * s2r;
        let s = cabs1(uRe, uIm);
        if (s !== 0) {
          const xRe = 0.5 * (HRe[H(i - 1, i - 1)] - tRe);
          const xIm = 0.5 * (HIm[H(i - 1, i - 1)] - tIm);
          const sx = cabs1(xRe, xIm);
          s = Math.max(s, sx);
          // y = s * sqrt((x/s)^2 + (u/s)^2)
          const xsr = xRe / s;
          const xsi = xIm / s;
          const usr = uRe / s;
          const usi = uIm / s;
          // (x/s)^2 + (u/s)^2
          const sumRe = xsr * xsr - xsi * xsi + (usr * usr - usi * usi);
          const sumIm = 2 * xsr * xsi + 2 * usr * usi;
          const [ysr, ysi] = csqrt(sumRe, sumIm);
          let yRe = s * ysr;
          let yIm = s * ysi;
          if (sx > 0) {
            // if Re(x/sx)*Re(y) + Im(x/sx)*Im(y) < 0, y = -y
            const xsxr = xRe / sx;
            const xsxi = xIm / sx;
            if (xsxr * yRe + xsxi * yIm < 0) {
              yRe = -yRe;
              yIm = -yIm;
            }
          }
          // t = t - u * zladiv(u, x+y)
          const [dr, di] = cdiv(uRe, uIm, xRe + yRe, xIm + yIm);
          // u * (dr,di)
          const mr = uRe * dr - uIm * di;
          const mi = uRe * di + uIm * dr;
          tRe -= mr;
          tIm -= mi;
        }
      }

      // look for two consecutive small subdiagonals
      let m: number;
      let h11sRe = 0;
      let h11sIm = 0;
      let h21 = 0;
      for (m = i - 1; m >= l + 1; m--) {
        const h11r = HRe[H(m, m)];
        const h11i = HIm[H(m, m)];
        const h22r = HRe[H(m + 1, m + 1)];
        const h22i = HIm[H(m + 1, m + 1)];
        let s1r = h11r - tRe;
        let s1i = h11i - tIm;
        let h21v = HRe[H(m + 1, m)]; // real part (subdiag is real here)
        const s = cabs1(s1r, s1i) + Math.abs(h21v);
        s1r /= s;
        s1i /= s;
        h21v /= s;
        h11sRe = s1r;
        h11sIm = s1i;
        h21 = h21v;
        const h10 = HRe[H(m, m - 1)];
        if (
          Math.abs(h10) * Math.abs(h21v) <=
          ulp * (cabs1(s1r, s1i) * (cabs1(h11r, h11i) + cabs1(h22r, h22i)))
        ) {
          break;
        }
      }
      if (m < l + 1) {
        // fell through: start at l
        m = l;
        const h11r = HRe[H(l, l)];
        const h11i = HIm[H(l, l)];
        let s1r = h11r - tRe;
        let s1i = h11i - tIm;
        let h21v = HRe[H(l + 1, l)];
        const s = cabs1(s1r, s1i) + Math.abs(h21v);
        s1r /= s;
        s1i /= s;
        h21v /= s;
        h11sRe = s1r;
        h11sIm = s1i;
        h21 = h21v;
      }

      // single-shift QR step
      for (let k = m; k <= i - 1; k++) {
        if (k > m) {
          // V = H(k:k+1, k-1)
          vRe[0] = HRe[H(k, k - 1)];
          vIm[0] = HIm[H(k, k - 1)];
          vRe[1] = HRe[H(k + 1, k - 1)];
          vIm[1] = HIm[H(k + 1, k - 1)];
        } else {
          vRe[0] = h11sRe;
          vIm[0] = h11sIm;
          vRe[1] = h21;
          vIm[1] = 0;
        }
        const [t1r, t1i] = zlarfg(2, vRe, vIm, 0, 1);
        if (k > m) {
          HRe[H(k, k - 1)] = vRe[0];
          HIm[H(k, k - 1)] = vIm[0];
          HRe[H(k + 1, k - 1)] = 0;
          HIm[H(k + 1, k - 1)] = 0;
        }
        const v2r = vRe[1];
        const v2i = vIm[1];
        const t2 = t1r * v2r - t1i * v2i; // Re(T1*V2) (real per LAPACK)

        // left: rows k,k+1 over columns k..i2
        for (let j = k; j <= i2; j++) {
          const ok = H(k, j);
          const ok1 = H(k + 1, j);
          // SUM = conj(T1)*H(k,j) + T2*H(k+1,j)
          const sumRe = t1r * HRe[ok] + t1i * HIm[ok] + t2 * HRe[ok1];
          const sumIm = t1r * HIm[ok] - t1i * HRe[ok] + t2 * HIm[ok1];
          HRe[ok] -= sumRe;
          HIm[ok] -= sumIm;
          // H(k+1,j) -= SUM*V2
          HRe[ok1] -= sumRe * v2r - sumIm * v2i;
          HIm[ok1] -= sumRe * v2i + sumIm * v2r;
        }
        // right: rows i1..min(k+2,i) over columns k,k+1
        const jmax = Math.min(k + 2, i);
        for (let j = i1; j <= jmax; j++) {
          const ojk = H(j, k);
          const ojk1 = H(j, k + 1);
          // SUM = T1*H(j,k) + T2*H(j,k+1)
          const sumRe = t1r * HRe[ojk] - t1i * HIm[ojk] + t2 * HRe[ojk1];
          const sumIm = t1r * HIm[ojk] + t1i * HRe[ojk] + t2 * HIm[ojk1];
          HRe[ojk] -= sumRe;
          HIm[ojk] -= sumIm;
          // H(j,k+1) -= SUM*conj(V2)
          HRe[ojk1] -= sumRe * v2r + sumIm * v2i;
          HIm[ojk1] -= sumIm * v2r - sumRe * v2i;
        }
        if (wantz) {
          for (let j = iloz; j <= ihiz; j++) {
            const ojk = Z(j, k);
            const ojk1 = Z(j, k + 1);
            const sumRe = t1r * ZRe[ojk] - t1i * ZIm[ojk] + t2 * ZRe[ojk1];
            const sumIm = t1r * ZIm[ojk] + t1i * ZRe[ojk] + t2 * ZIm[ojk1];
            ZRe[ojk] -= sumRe;
            ZIm[ojk] -= sumIm;
            ZRe[ojk1] -= sumRe * v2r + sumIm * v2i;
            ZIm[ojk1] -= sumIm * v2r - sumRe * v2i;
          }
        }

        if (k === m && m > l) {
          // extra scaling so H(m,m-1) stays real
          let tempRe = 1 - t1r;
          let tempIm = -t1i;
          const tabs = Math.hypot(tempRe, tempIm);
          tempRe /= tabs;
          tempIm /= tabs;
          // H(m+1,m) *= conj(temp)
          {
            const o = H(m + 1, m);
            const ar = HRe[o];
            const ai = HIm[o];
            HRe[o] = ar * tempRe + ai * tempIm;
            HIm[o] = ai * tempRe - ar * tempIm;
          }
          if (m + 2 <= i) {
            const o = H(m + 2, m + 1);
            const ar = HRe[o];
            const ai = HIm[o];
            HRe[o] = ar * tempRe - ai * tempIm;
            HIm[o] = ar * tempIm + ai * tempRe;
          }
          for (let j = m; j <= i; j++) {
            if (j !== m + 1) {
              if (i2 > j) {
                // H(j, j+1:i2) *= temp  (row j, stride n)
                zscalStrided(HRe, HIm, tempRe, tempIm, H(j, j + 1), n, i2 - j);
              }
              // H(i1:j-1, j) *= conj(temp) (col j, stride 1)
              zscalStrided(HRe, HIm, tempRe, -tempIm, H(i1, j), 1, j - i1);
              if (wantz) {
                zscalStrided(ZRe, ZIm, tempRe, -tempIm, Z(iloz, j), 1, nz);
              }
            }
          }
        }
      }

      // ensure H(i,i-1) is real
      {
        const o = H(i, i - 1);
        if (HIm[o] !== 0) {
          const rtemp = Math.hypot(HRe[o], HIm[o]);
          const tempRe = HRe[o] / rtemp;
          const tempIm = HIm[o] / rtemp;
          HRe[o] = rtemp;
          HIm[o] = 0;
          if (i2 > i) {
            // H(i, i+1:i2) *= conj(temp)
            zscalStrided(HRe, HIm, tempRe, -tempIm, H(i, i + 1), n, i2 - i);
          }
          // H(i1:i-1, i) *= temp
          zscalStrided(HRe, HIm, tempRe, tempIm, H(i1, i), 1, i - i1);
          if (wantz) {
            zscalStrided(ZRe, ZIm, tempRe, tempIm, Z(iloz, i), 1, nz);
          }
        }
      }
    }

    if (!converged) {
      return i; // failed to converge
    }
    // H(i,i-1) negligible: one eigenvalue converged
    wRe[i - 1] = HRe[H(i, i)];
    wIm[i - 1] = HIm[H(i, i)];
    kdefl = 0;
    i = l - 1;
  }
  return 0;
}

/** Complex principal square root. */
function csqrt(re: number, im: number): [number, number] {
  if (re === 0 && im === 0) return [0, 0];
  const r = Math.hypot(re, im);
  const w = Math.sqrt((r + Math.abs(re)) / 2);
  if (re >= 0) {
    return [w, im / (2 * w)];
  } else {
    const wi = im >= 0 ? w : -w;
    return [im / (2 * wi), wi];
  }
}

/** ZSCAL with stride: x[base + k*inc] *= (sr + i si) for k=0..cnt-1. */
function zscalStrided(
  re: Float64Array,
  im: Float64Array,
  sr: number,
  si: number,
  base: number,
  inc: number,
  cnt: number
): void {
  for (let k = 0; k < cnt; k++) {
    const o = base + k * inc;
    const xr = re[o];
    const xi = im[o];
    re[o] = xr * sr - xi * si;
    im[o] = xr * si + xi * sr;
  }
}

/**
 * Right and/or left eigenvectors of A from the Schur form T (HRe/HIm, upper
 * triangular) and Schur vectors (VRe/VIm). Back-substitution port of ztrevc
 * with SMIN regularization (no zlatrs overflow scaling). Output overwrites the
 * provided Schur-vector copy in place (HOWMNY='B').
 */
function trevcRight(
  n: number,
  TRe: Float64Array,
  TIm: Float64Array,
  VRe: Float64Array,
  VIm: Float64Array
): void {
  const T = (i: number, j: number) => (j - 1) * n + (i - 1);
  const ulp = EPS;
  const smlnum = SAFMIN * (n / ulp);
  const workRe = new Float64Array(n);
  const workIm = new Float64Array(n);
  for (let ki = n; ki >= 1; ki--) {
    const smin = Math.max(ulp * cabs1(TRe[T(ki, ki)], TIm[T(ki, ki)]), smlnum);
    workRe[ki - 1] = 1;
    workIm[ki - 1] = 0;
    for (let k = 1; k <= ki - 1; k++) {
      workRe[k - 1] = -TRe[T(k, ki)];
      workIm[k - 1] = -TIm[T(k, ki)];
    }
    // solve (T(1:ki-1,1:ki-1) - T(ki,ki) I) x = work  (upper-tri back-sub)
    const wkkRe = TRe[T(ki, ki)];
    const wkkIm = TIm[T(ki, ki)];
    for (let k = ki - 1; k >= 1; k--) {
      let dRe = TRe[T(k, k)] - wkkRe;
      let dIm = TIm[T(k, k)] - wkkIm;
      if (cabs1(dRe, dIm) < smin) {
        dRe = smin;
        dIm = 0;
      }
      const [qr, qi] = cdiv(workRe[k - 1], workIm[k - 1], dRe, dIm);
      workRe[k - 1] = qr;
      workIm[k - 1] = qi;
      for (let ii = 1; ii <= k - 1; ii++) {
        const o = T(ii, k);
        workRe[ii - 1] -= qr * TRe[o] - qi * TIm[o];
        workIm[ii - 1] -= qr * TIm[o] + qi * TRe[o];
      }
    }
    // VR(:,ki) = Schur(:, 1:ki) * work(1:ki)   (columns 1..ki-1 pristine)
    backTransformColumn(n, VRe, VIm, workRe, workIm, ki);
  }
}

function trevcLeft(
  n: number,
  TRe: Float64Array,
  TIm: Float64Array,
  VRe: Float64Array,
  VIm: Float64Array
): void {
  const T = (i: number, j: number) => (j - 1) * n + (i - 1);
  const ulp = EPS;
  const smlnum = SAFMIN * (n / ulp);
  const workRe = new Float64Array(n);
  const workIm = new Float64Array(n);
  for (let ki = 1; ki <= n; ki++) {
    const smin = Math.max(ulp * cabs1(TRe[T(ki, ki)], TIm[T(ki, ki)]), smlnum);
    workRe[ki - 1] = 1;
    workIm[ki - 1] = 0;
    for (let k = ki + 1; k <= n; k++) {
      // -conj(T(ki,k))
      workRe[k - 1] = -TRe[T(ki, k)];
      workIm[k - 1] = TIm[T(ki, k)];
    }
    // solve (T(ki+1:n,ki+1:n) - T(ki,ki))^H x = work  (forward subst)
    const wkkRe = TRe[T(ki, ki)];
    const wkkIm = TIm[T(ki, ki)];
    for (let k = ki + 1; k <= n; k++) {
      let dRe = TRe[T(k, k)] - wkkRe;
      let dIm = TIm[T(k, k)] - wkkIm;
      if (cabs1(dRe, dIm) < smin) {
        dRe = smin;
        dIm = 0;
      }
      let sRe = workRe[k - 1];
      let sIm = workIm[k - 1];
      for (let j = ki + 1; j <= k - 1; j++) {
        const o = T(j, k);
        // conj(T(j,k)) * work(j)
        const ar = TRe[o];
        const ai = -TIm[o];
        sRe -= ar * workRe[j - 1] - ai * workIm[j - 1];
        sIm -= ar * workIm[j - 1] + ai * workRe[j - 1];
      }
      // divide by conj(d)
      const [qr, qi] = cdiv(sRe, sIm, dRe, -dIm);
      workRe[k - 1] = qr;
      workIm[k - 1] = qi;
    }
    // VL(:,ki) = Schur(:, ki:n) * [1; work(ki+1..n)]  (columns ki..n pristine)
    backTransformColumnLeft(n, VRe, VIm, workRe, workIm, ki);
  }
}

/** VR(:,ki) := sum_{j=1}^{ki} V(:,j)*work(j), then normalize (max |·|_1 = 1). */
function backTransformColumn(
  n: number,
  VRe: Float64Array,
  VIm: Float64Array,
  workRe: Float64Array,
  workIm: Float64Array,
  ki: number
): void {
  const out = new Float64Array(n);
  const outIm = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    let accRe = 0;
    let accIm = 0;
    for (let j = 1; j <= ki; j++) {
      const o = (j - 1) * n + r;
      const wr = workRe[j - 1];
      const wi = workIm[j - 1];
      accRe += VRe[o] * wr - VIm[o] * wi;
      accIm += VRe[o] * wi + VIm[o] * wr;
    }
    out[r] = accRe;
    outIm[r] = accIm;
  }
  storeNormalizedColumn(n, VRe, VIm, out, outIm, ki);
}

/** VL(:,ki) := V(:,ki) + sum_{j=ki+1}^{n} V(:,j)*work(j), then normalize. */
function backTransformColumnLeft(
  n: number,
  VRe: Float64Array,
  VIm: Float64Array,
  workRe: Float64Array,
  workIm: Float64Array,
  ki: number
): void {
  const out = new Float64Array(n);
  const outIm = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    let accRe = VRe[(ki - 1) * n + r];
    let accIm = VIm[(ki - 1) * n + r];
    for (let j = ki + 1; j <= n; j++) {
      const o = (j - 1) * n + r;
      const wr = workRe[j - 1];
      const wi = workIm[j - 1];
      accRe += VRe[o] * wr - VIm[o] * wi;
      accIm += VRe[o] * wi + VIm[o] * wr;
    }
    out[r] = accRe;
    outIm[r] = accIm;
  }
  storeNormalizedColumn(n, VRe, VIm, out, outIm, ki);
}

/** Store column with zgeev normalization: unit 2-norm, largest comp made real. */
function storeNormalizedColumn(
  n: number,
  VRe: Float64Array,
  VIm: Float64Array,
  out: Float64Array,
  outIm: Float64Array,
  ki: number
): void {
  // unit 2-norm
  let nrm = 0;
  for (let r = 0; r < n; r++) nrm += out[r] * out[r] + outIm[r] * outIm[r];
  nrm = Math.sqrt(nrm);
  if (nrm > 0) {
    const scl = 1 / nrm;
    for (let r = 0; r < n; r++) {
      out[r] *= scl;
      outIm[r] *= scl;
    }
  }
  // largest-magnitude component (by |re|^2+|im|^2), made real-positive
  let kmax = 0;
  let best = -1;
  for (let r = 0; r < n; r++) {
    const m = out[r] * out[r] + outIm[r] * outIm[r];
    if (m > best) {
      best = m;
      kmax = r;
    }
  }
  if (best > 0) {
    const mag = Math.sqrt(best);
    // multiply column by conj(out[kmax])/mag
    const cr = out[kmax] / mag;
    const ci = -outIm[kmax] / mag;
    for (let r = 0; r < n; r++) {
      const xr = out[r];
      const xi = outIm[r];
      out[r] = xr * cr - xi * ci;
      outIm[r] = xr * ci + xi * cr;
    }
    out[kmax] = mag;
    outIm[kmax] = 0;
  }
  for (let r = 0; r < n; r++) {
    VRe[(ki - 1) * n + r] = out[r];
    VIm[(ki - 1) * n + r] = outIm[r];
  }
}

/**
 * Complex eigendecomposition (zgeev-equivalent). Matches the LapackBridge
 * eigComplex contract: column-major real/imag inputs, returns eigenvalues and
 * (optionally) left/right eigenvectors.
 */
export function eigComplexTs(
  dataRe: Float64Array,
  dataIm: Float64Array,
  n: number,
  computeVL: boolean,
  computeVR: boolean
): {
  wRe: Float64Array;
  wIm: Float64Array;
  VLRe?: Float64Array;
  VLIm?: Float64Array;
  VRRe?: Float64Array;
  VRIm?: Float64Array;
} {
  const wRe = allocFloat64Array(n);
  const wIm = allocFloat64Array(n);
  if (n === 0) return { wRe, wIm };

  // working copy → becomes Schur form T
  const HRe = allocFloat64Array(dataRe);
  const HIm = allocFloat64Array(dataIm);

  const needVec = computeVL || computeVR;
  // Schur vectors Q
  const QRe = allocFloat64Array(n * n);
  const QIm = allocFloat64Array(n * n);
  if (needVec) {
    for (let i = 0; i < n; i++) QRe[i + i * n] = 1;
  }

  // 1. Hessenberg reduction (accumulate Q)
  hessenberg(n, HRe, HIm, QRe, QIm, needVec);

  // 2. QR iteration → Schur form (wantt = need full T for eigenvectors)
  const info = zlahqr(
    needVec,
    needVec,
    n,
    1,
    n,
    HRe,
    HIm,
    wRe,
    wIm,
    1,
    n,
    QRe,
    QIm
  );
  if (info !== 0) {
    throw new Error(
      `eigComplex: QR iteration failed to converge (info=${info})`
    );
  }

  const result: {
    wRe: Float64Array;
    wIm: Float64Array;
    VLRe?: Float64Array;
    VLIm?: Float64Array;
    VRRe?: Float64Array;
    VRIm?: Float64Array;
  } = { wRe, wIm };

  if (computeVR) {
    const VRRe = allocFloat64Array(QRe);
    const VRIm = allocFloat64Array(QIm);
    trevcRight(n, HRe, HIm, VRRe, VRIm);
    result.VRRe = VRRe;
    result.VRIm = VRIm;
  }
  if (computeVL) {
    const VLRe = allocFloat64Array(QRe);
    const VLIm = allocFloat64Array(QIm);
    trevcLeft(n, HRe, HIm, VLRe, VLIm);
    result.VLRe = VLRe;
    result.VLIm = VLIm;
  }
  return result;
}
