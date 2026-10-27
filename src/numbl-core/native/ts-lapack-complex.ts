/**
 * Complex scalar helpers and ZLARFG shared by the ts-lapack complex
 * routines (eig, Hermitian eig, QR). Split real/imag storage throughout.
 */

export const EPS = 2.220446049250313e-16; // DLAMCH('E')
export const SAFMIN = 2.2250738585072014e-308; // DLAMCH('S')

/** |Re| + |Im| (LAPACK CABS1). */
export function cabs1(re: number, im: number): number {
  return Math.abs(re) + Math.abs(im);
}

/** Fortran SIGN(mag, s): |mag| with the sign of s. */
export function fsign(mag: number, s: number): number {
  return s >= 0 ? Math.abs(mag) : -Math.abs(mag);
}

/** Robust complex divide (a/b) via Smith's algorithm (ZLADIV). */
export function cdiv(
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

/** sqrt(x^2 + y^2 + z^2) without spurious over/underflow (DLAPY3). */
export function dlapy3(x: number, y: number, z: number): number {
  const w = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
  if (w === 0.0 || !Number.isFinite(w))
    return Math.abs(x) + Math.abs(y) + Math.abs(z);
  return w * Math.sqrt((x / w) ** 2 + (y / w) ** 2 + (z / w) ** 2);
}

/** DZNRM2 over x(off), x(off+incx), ... (n elements). */
export function dznrm2(
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
export function zlarfg(
  n: number,
  re: Float64Array,
  im: Float64Array,
  off: number,
  incx: number
): [number, number] {
  if (n <= 0) return [0, 0];

  let xnorm = dznrm2(n - 1, re, im, off + incx, incx);
  let alphr = re[off];
  let alphi = im[off];

  if (xnorm === 0.0 && alphi === 0.0) return [0, 0]; // H = I

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

  return [tauRe, tauIm];
}
