/**
 * Pure-TypeScript Hermitian eigensolver — a zheev-equivalent for the ts-lapack
 * bridge (browser path, no native LAPACK addon).
 *
 * Ported from the netlib LAPACK reference (zheev → zhetd2 / zungtr → zsteqr).
 * One structural deviation: zsteqr accumulates the QL/QR rotations directly
 * into the complex matrix Q, whereas here DSTEQR is run with COMPZ = 'I' to
 * get the real orthogonal eigenvector matrix S of the tridiagonal, and the
 * back-transformation V = Q * S is done as one complex-times-real matrix
 * product afterwards. Both are O(n^3); this way the tridiagonal QL/QR lives in
 * exactly one place (src/ts-lapack/src/SRC/dsteqr.ts) instead of two.
 *
 * Because zlarfg produces a real beta, the tridiagonal form is real symmetric,
 * so DSTEQR applies unchanged.
 *
 * Eigenvalues come back real and in ascending order; eigenvectors form a
 * unitary basis (V^H * V = I), matching dsyev/zheev and MATLAB.
 *
 * All matrices are column-major with leading dimension n, split into parallel
 * real/imag Float64Arrays — matching the bridge's eigHerm I/O shape.
 */

import { allocFloat64Array } from "../runtime/alloc.js";
import { dsteqr, COMPZ_N, COMPZ_I } from "../../ts-lapack/src/SRC/dsteqr.js";

const EPS = 2.220446049250313e-16; // DLAMCH('E')
const SAFMIN = 2.2250738585072014e-308; // DLAMCH('S')

/** sqrt(x^2 + y^2 + z^2) without spurious over/underflow (DLAPY3). */
function dlapy3(x: number, y: number, z: number): number {
  const w = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
  if (w === 0.0 || !Number.isFinite(w))
    return Math.abs(x) + Math.abs(y) + Math.abs(z);
  return w * Math.sqrt((x / w) ** 2 + (y / w) ** 2 + (z / w) ** 2);
}

/** Fortran SIGN(mag, s): |mag| with the sign of s. */
function fsign(mag: number, s: number): number {
  return s >= 0 ? Math.abs(mag) : -Math.abs(mag);
}

/** Robust complex divide (a/b) via Smith's algorithm (ZLADIV). */
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
  }
  const r = br / bi;
  const t = br * r + bi;
  return [(ar * r + ai) / t, (ai * r - ar) / t];
}

/** DZNRM2 over x(off), x(off+incx), ... (n elements). */
function dznrm2(
  n: number,
  re: Float64Array,
  im: Float64Array,
  off: number,
  incx: number
): number {
  let scale = 0.0;
  let ssq = 1.0;
  for (let i = 0, k = off; i < n; i++, k += incx) {
    for (const v of [re[k], im[k]]) {
      if (v !== 0.0) {
        const a = Math.abs(v);
        if (scale < a) {
          ssq = 1.0 + ssq * (scale / a) ** 2;
          scale = a;
        } else {
          ssq += (a / scale) ** 2;
        }
      }
    }
  }
  return scale * Math.sqrt(ssq);
}

/**
 * ZLARFG: generate a complex elementary reflector H = I - tau*v*v^H with
 * H^H * [alpha; x] = [beta; 0] and beta real. On entry re/im[off] = alpha and
 * X is re/im[off + i*incx], i = 1..n-1; on exit re/im[off] = beta and the X
 * part holds v(2:). Returns tau.
 */
function zlarfg(
  n: number,
  re: Float64Array,
  im: Float64Array,
  off: number,
  incx: number
): { tauRe: number; tauIm: number } {
  if (n <= 0) return { tauRe: 0, tauIm: 0 };

  let xnorm = dznrm2(n - 1, re, im, off + incx, incx);
  let alphr = re[off];
  let alphi = im[off];

  if (xnorm === 0.0 && alphi === 0.0) {
    // H = I
    return { tauRe: 0, tauIm: 0 };
  }

  let beta = -fsign(dlapy3(alphr, alphi, xnorm), alphr);
  const safmin = SAFMIN / EPS;
  const rsafmn = 1.0 / safmin;
  let knt = 0;

  if (Math.abs(beta) < safmin) {
    // XNORM, BETA may be inaccurate; scale X and recompute them
    do {
      knt++;
      for (let i = 1, k = off + incx; i < n; i++, k += incx) {
        re[k] *= rsafmn;
        im[k] *= rsafmn;
      }
      beta *= rsafmn;
      alphi *= rsafmn;
      alphr *= rsafmn;
    } while (Math.abs(beta) < safmin && knt < 20);
    xnorm = dznrm2(n - 1, re, im, off + incx, incx);
    beta = -fsign(dlapy3(alphr, alphi, xnorm), alphr);
  }

  const tauRe = (beta - alphr) / beta;
  const tauIm = -alphi / beta;
  const [sr, si] = cdiv(1.0, 0.0, alphr - beta, alphi);
  for (let i = 1, k = off + incx; i < n; i++, k += incx) {
    const xr = re[k];
    const xi = im[k];
    re[k] = xr * sr - xi * si;
    im[k] = xr * si + xi * sr;
  }

  for (let j = 1; j <= knt; j++) beta *= safmin;
  re[off] = beta;
  im[off] = 0.0;

  return { tauRe, tauIm };
}

/**
 * ZHEMV, uplo = 'L', incx = incy = 1, beta = 0:
 *   y := alpha * A * x   with A Hermitian of order n (lower triangle stored).
 */
function zhemvLower(
  n: number,
  alphaRe: number,
  alphaIm: number,
  aRe: Float64Array,
  aIm: Float64Array,
  aOff: number,
  lda: number,
  xRe: Float64Array,
  xIm: Float64Array,
  xOff: number,
  yRe: Float64Array,
  yIm: Float64Array,
  yOff: number
): void {
  for (let i = 0; i < n; i++) {
    yRe[yOff + i] = 0.0;
    yIm[yOff + i] = 0.0;
  }
  for (let j = 0; j < n; j++) {
    const xjr = xRe[xOff + j];
    const xji = xIm[xOff + j];
    const t1r = alphaRe * xjr - alphaIm * xji;
    const t1i = alphaRe * xji + alphaIm * xjr;
    let t2r = 0.0;
    let t2i = 0.0;
    const col = aOff + j * lda;
    // Diagonal entry of a Hermitian matrix is real
    yRe[yOff + j] += t1r * aRe[col + j];
    yIm[yOff + j] += t1i * aRe[col + j];
    for (let i = j + 1; i < n; i++) {
      const ar = aRe[col + i];
      const ai = aIm[col + i];
      yRe[yOff + i] += t1r * ar - t1i * ai;
      yIm[yOff + i] += t1r * ai + t1i * ar;
      // temp2 += conj(A(i,j)) * x(i)
      const xir = xRe[xOff + i];
      const xii = xIm[xOff + i];
      t2r += ar * xir + ai * xii;
      t2i += ar * xii - ai * xir;
    }
    yRe[yOff + j] += alphaRe * t2r - alphaIm * t2i;
    yIm[yOff + j] += alphaRe * t2i + alphaIm * t2r;
  }
}

/**
 * ZHER2, uplo = 'L', incx = incy = 1, alpha = -1:
 *   A := -x*y^H - y*x^H + A   (lower triangle only; diagonal kept real).
 */
function zher2LowerMinus(
  n: number,
  xRe: Float64Array,
  xIm: Float64Array,
  xOff: number,
  yRe: Float64Array,
  yIm: Float64Array,
  yOff: number,
  aRe: Float64Array,
  aIm: Float64Array,
  aOff: number,
  lda: number
): void {
  for (let j = 0; j < n; j++) {
    const xjr = xRe[xOff + j];
    const xji = xIm[xOff + j];
    const yjr = yRe[yOff + j];
    const yji = yIm[yOff + j];
    const col = aOff + j * lda;
    if (xjr !== 0 || xji !== 0 || yjr !== 0 || yji !== 0) {
      // temp1 = alpha*conj(y(j)), temp2 = conj(alpha*x(j)), alpha = -1
      const t1r = -yjr;
      const t1i = yji;
      const t2r = -xjr;
      const t2i = xji;
      // A(j,j) = real(A(j,j)) + real(x(j)*temp1 + y(j)*temp2)
      aRe[col + j] =
        aRe[col + j] + (xjr * t1r - xji * t1i) + (yjr * t2r - yji * t2i);
      aIm[col + j] = 0.0;
      for (let i = j + 1; i < n; i++) {
        const xir = xRe[xOff + i];
        const xii = xIm[xOff + i];
        const yir = yRe[yOff + i];
        const yii = yIm[yOff + i];
        aRe[col + i] += xir * t1r - xii * t1i + (yir * t2r - yii * t2i);
        aIm[col + i] += xir * t1i + xii * t1r + (yir * t2i + yii * t2r);
      }
    } else {
      aIm[col + j] = 0.0;
    }
  }
}

/**
 * ZHETD2, uplo = 'L': reduce the Hermitian matrix A to real symmetric
 * tridiagonal form by a unitary similarity transformation Q^H * A * Q = T.
 * On exit d/e hold T, and the reflectors below the first subdiagonal together
 * with tau represent Q.
 */
function zhetd2Lower(
  n: number,
  aRe: Float64Array,
  aIm: Float64Array,
  lda: number,
  d: Float64Array,
  e: Float64Array,
  tauRe: Float64Array,
  tauIm: Float64Array
): void {
  const wRe = allocFloat64Array(Math.max(1, n));
  const wIm = allocFloat64Array(Math.max(1, n));

  for (let i = 1; i <= n - 1; i++) {
    // Generate the reflector annihilating A(i+2:n,i)
    const off = i + (i - 1) * lda; // A(i+1, i), 0-based
    const t = zlarfg(n - i, aRe, aIm, off, 1);
    e[i - 1] = aRe[off]; // beta is real
    const taur = t.tauRe;
    const taui = t.tauIm;

    if (taur !== 0.0 || taui !== 0.0) {
      aRe[off] = 1.0;
      aIm[off] = 0.0;

      const m = n - i;
      const a11 = i + i * lda; // A(i+1, i+1), 0-based

      // x := tau * A * v, stored in w(0:m-1)
      zhemvLower(m, taur, taui, aRe, aIm, a11, lda, aRe, aIm, off, wRe, wIm, 0);

      // alpha := -1/2 * tau * (x^H * v);  ZDOTC(m, w, v) = sum conj(w)*v
      let dr = 0.0;
      let di = 0.0;
      for (let k = 0; k < m; k++) {
        const vr = aRe[off + k];
        const vi = aIm[off + k];
        dr += wRe[k] * vr + wIm[k] * vi;
        di += wRe[k] * vi - wIm[k] * vr;
      }
      const ar = -0.5 * (taur * dr - taui * di);
      const ai = -0.5 * (taur * di + taui * dr);

      // w := w + alpha * v
      for (let k = 0; k < m; k++) {
        const vr = aRe[off + k];
        const vi = aIm[off + k];
        wRe[k] += ar * vr - ai * vi;
        wIm[k] += ar * vi + ai * vr;
      }

      // A := A - v*w^H - w*v^H
      zher2LowerMinus(m, aRe, aIm, off, wRe, wIm, 0, aRe, aIm, a11, lda);
    } else {
      const a11 = i + i * lda;
      aIm[a11] = 0.0;
    }

    aRe[off] = e[i - 1];
    aIm[off] = 0.0;
    d[i - 1] = aRe[i - 1 + (i - 1) * lda];
    tauRe[i - 1] = taur;
    tauIm[i - 1] = taui;
  }
  d[n - 1] = aRe[n - 1 + (n - 1) * lda];
}

/**
 * ZUNG2R applied by ZUNGTR, uplo = 'L': generate the n×n unitary Q from the
 * reflectors left in A by zhetd2Lower. A is overwritten by Q.
 */
function zungtrLower(
  n: number,
  aRe: Float64Array,
  aIm: Float64Array,
  lda: number,
  tauRe: Float64Array,
  tauIm: Float64Array
): void {
  // Shift the reflector columns one to the right and make the first row and
  // column those of the unit matrix.
  for (let j = n; j >= 2; j--) {
    aRe[(j - 1) * lda] = 0.0;
    aIm[(j - 1) * lda] = 0.0;
    for (let i = j + 1; i <= n; i++) {
      aRe[i - 1 + (j - 1) * lda] = aRe[i - 1 + (j - 2) * lda];
      aIm[i - 1 + (j - 1) * lda] = aIm[i - 1 + (j - 2) * lda];
    }
  }
  aRe[0] = 1.0;
  aIm[0] = 0.0;
  for (let i = 2; i <= n; i++) {
    aRe[i - 1] = 0.0;
    aIm[i - 1] = 0.0;
  }
  if (n <= 1) return;

  // ZUNG2R on the trailing (n-1)×(n-1) block starting at A(2,2)
  const m = n - 1;
  const base = 1 + lda; // 0-based offset of A(2,2)
  const workRe = allocFloat64Array(Math.max(1, m));
  const workIm = allocFloat64Array(Math.max(1, m));

  for (let i = m; i >= 1; i--) {
    const vOff = base + (i - 1) + (i - 1) * lda; // block-local A(i,i)
    const tr = tauRe[i - 1];
    const ti = tauIm[i - 1];
    if (i < m) {
      aRe[vOff] = 1.0;
      aIm[vOff] = 0.0;
      // ZLARF 'Left' on the block A(i:m, i+1:m) with v = A(i:m,i), tau
      const rows = m - i + 1;
      const cols = m - i;
      const cOff = base + (i - 1) + i * lda; // block-local A(i,i+1)
      // work = C^H * v
      for (let j = 0; j < cols; j++) {
        let sr = 0.0;
        let si = 0.0;
        for (let k = 0; k < rows; k++) {
          const cr = aRe[cOff + k + j * lda];
          const ci = aIm[cOff + k + j * lda];
          const vr = aRe[vOff + k];
          const vi = aIm[vOff + k];
          sr += cr * vr + ci * vi;
          si += cr * vi - ci * vr;
        }
        workRe[j] = sr;
        workIm[j] = si;
      }
      // C := C - tau * v * work^H
      for (let j = 0; j < cols; j++) {
        // f = -tau * conj(work(j))
        const wr = workRe[j];
        const wi = -workIm[j];
        const fr = -(tr * wr - ti * wi);
        const fi = -(tr * wi + ti * wr);
        for (let k = 0; k < rows; k++) {
          const vr = aRe[vOff + k];
          const vi = aIm[vOff + k];
          aRe[cOff + k + j * lda] += fr * vr - fi * vi;
          aIm[cOff + k + j * lda] += fr * vi + fi * vr;
        }
      }
    }
    if (i < m) {
      // A(i+1:m, i) := -tau * A(i+1:m, i)
      for (let k = 1; k <= m - i; k++) {
        const vr = aRe[vOff + k];
        const vi = aIm[vOff + k];
        aRe[vOff + k] = -(tr * vr - ti * vi);
        aIm[vOff + k] = -(tr * vi + ti * vr);
      }
    }
    aRe[vOff] = 1.0 - tr;
    aIm[vOff] = -ti;
    // Set A(1:i-1, i) to zero
    for (let k = 1; k <= i - 1; k++) {
      aRe[base + (k - 1) + (i - 1) * lda] = 0.0;
      aIm[base + (k - 1) + (i - 1) * lda] = 0.0;
    }
  }
}

/**
 * ZHEEV-equivalent. `dataRe`/`dataIm` hold the full n×n Hermitian matrix in
 * column-major order (only the lower triangle is read); they are not modified.
 *
 * Returns the eigenvalues in ascending order, plus a unitary eigenvector basis
 * when `computeV` is set.
 */
export function eigHermTs(
  dataRe: Float64Array,
  dataIm: Float64Array,
  n: number,
  computeV: boolean
): { w: Float64Array; VRe?: Float64Array; VIm?: Float64Array } {
  const w = allocFloat64Array(n);
  if (n === 0)
    return { w, VRe: allocFloat64Array(0), VIm: allocFloat64Array(0) };
  if (n === 1) {
    w[0] = dataRe[0];
    if (!computeV) return { w };
    const VRe = allocFloat64Array(1);
    VRe[0] = 1;
    return { w, VRe, VIm: allocFloat64Array(1) };
  }

  const aRe = allocFloat64Array(dataRe);
  const aIm = allocFloat64Array(dataIm);
  const e = allocFloat64Array(Math.max(1, n - 1));
  const tauRe = allocFloat64Array(Math.max(1, n - 1));
  const tauIm = allocFloat64Array(Math.max(1, n - 1));

  zhetd2Lower(n, aRe, aIm, n, w, e, tauRe, tauIm);

  if (!computeV) {
    const work = allocFloat64Array(Math.max(1, 2 * n - 2));
    const dummy = allocFloat64Array(1);
    dsteqr(COMPZ_N, n, w, 0, e, 0, dummy, 0, 1, work, 0);
    return { w };
  }

  // Q (unitary, from the reflectors) and S (real, from the tridiagonal)
  zungtrLower(n, aRe, aIm, n, tauRe, tauIm);
  const S = allocFloat64Array(n * n);
  const work = allocFloat64Array(Math.max(1, 2 * n - 2));
  dsteqr(COMPZ_I, n, w, 0, e, 0, S, 0, n, work, 0);

  // V = Q * S
  const VRe = allocFloat64Array(n * n);
  const VIm = allocFloat64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let k = 0; k < n; k++) {
      const s = S[k + j * n];
      if (s === 0) continue;
      for (let i = 0; i < n; i++) {
        VRe[i + j * n] += aRe[i + k * n] * s;
        VIm[i + j * n] += aIm[i + k * n] * s;
      }
    }
  }

  return { w, VRe, VIm };
}
