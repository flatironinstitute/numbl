/**
 * Complex QR (zgeqr2/zung2r), column-pivoted QR (zlaqp2) and LU (zgetf2)
 * for the ts-lapack bridge (browser path, no native LAPACK addon).
 *
 * Unblocked translations that follow the LAPACK conventions (zlarfg:
 * real diagonal of R with beta = -sign(Re(alpha))*norm; izamax pivoting
 * by |re|+|im|) so results agree with MATLAB up to rounding.
 */

import { allocFloat64Array } from "../runtime/alloc.js";

/**
 * Householder QR in place on a (m×n, column-major split complex).
 * Column j's reflector H_j = I - tau_j v v^H has v(0) = 1 and v(1:) stored
 * below the diagonal. With `jpvt`, does column pivoting (zlaqp2) and
 * fills jpvt with the 1-based permutation.
 */
function zgeqr2(
  aRe: Float64Array,
  aIm: Float64Array,
  m: number,
  n: number,
  tauRe: Float64Array,
  tauIm: Float64Array,
  jpvt?: Int32Array
): void {
  const k = Math.min(m, n);
  let vn1: Float64Array | undefined, vn2: Float64Array | undefined;
  const colNorm = (j: number, i0: number) => {
    let s = 0;
    for (let i = i0; i < m; i++)
      s = Math.hypot(s, aRe[i + j * m], aIm[i + j * m]);
    return s;
  };
  if (jpvt) {
    vn1 = allocFloat64Array(n);
    vn2 = allocFloat64Array(n);
    for (let j = 0; j < n; j++) {
      jpvt[j] = j + 1;
      vn1[j] = vn2[j] = colNorm(j, 0);
    }
  }
  const tol3z = Math.sqrt(Number.EPSILON);

  for (let j = 0; j < k; j++) {
    if (jpvt && vn1 && vn2) {
      let p = j;
      for (let c = j + 1; c < n; c++) if (vn1[c] > vn1[p]) p = c;
      if (p !== j) {
        for (let i = 0; i < m; i++) {
          const a = i + j * m,
            b = i + p * m;
          let t = aRe[a];
          aRe[a] = aRe[b];
          aRe[b] = t;
          t = aIm[a];
          aIm[a] = aIm[b];
          aIm[b] = t;
        }
        const t = jpvt[p];
        jpvt[p] = jpvt[j];
        jpvt[j] = t;
        vn1[p] = vn1[j];
        vn2[p] = vn2[j];
      }
    }

    // zlarfg on A(j:m-1, j)
    const d = j + j * m;
    const alphRe = aRe[d],
      alphIm = aIm[d];
    const xnorm = colNorm(j, j + 1);
    if (xnorm === 0 && alphIm === 0) {
      tauRe[j] = tauIm[j] = 0;
    } else {
      const nrm = Math.hypot(alphRe, alphIm, xnorm);
      const beta = alphRe >= 0 ? -nrm : nrm;
      tauRe[j] = (beta - alphRe) / beta;
      tauIm[j] = -alphIm / beta;
      // v = x / (alpha - beta)
      const dr = alphRe - beta,
        di = alphIm;
      const dd = dr * dr + di * di;
      for (let i = j + 1; i < m; i++) {
        const xr = aRe[i + j * m],
          xi = aIm[i + j * m];
        aRe[i + j * m] = (xr * dr + xi * di) / dd;
        aIm[i + j * m] = (xi * dr - xr * di) / dd;
      }
      aRe[d] = beta;
      aIm[d] = 0;

      // Apply H_j^H = I - conj(tau) v v^H to A(j:m-1, j+1:n-1)
      const ctr = tauRe[j],
        cti = -tauIm[j];
      for (let c = j + 1; c < n; c++) {
        let wr = aRe[j + c * m],
          wi = aIm[j + c * m];
        for (let i = j + 1; i < m; i++) {
          const vr = aRe[i + j * m],
            vi = aIm[i + j * m];
          const xr = aRe[i + c * m],
            xi = aIm[i + c * m];
          wr += vr * xr + vi * xi;
          wi += vr * xi - vi * xr;
        }
        const sr = ctr * wr - cti * wi,
          si = ctr * wi + cti * wr;
        aRe[j + c * m] -= sr;
        aIm[j + c * m] -= si;
        for (let i = j + 1; i < m; i++) {
          const vr = aRe[i + j * m],
            vi = aIm[i + j * m];
          aRe[i + c * m] -= vr * sr - vi * si;
          aIm[i + c * m] -= vr * si + vi * sr;
        }
      }
    }

    // Downdate partial column norms (zlaqp2)
    if (jpvt && vn1 && vn2) {
      for (let c = j + 1; c < n; c++) {
        if (vn1[c] === 0) continue;
        const r = Math.hypot(aRe[j + c * m], aIm[j + c * m]) / vn1[c];
        const temp = Math.max(0, 1 - r * r);
        const q = vn1[c] / vn2[c];
        if (temp * q * q <= tol3z) {
          vn1[c] = vn2[c] = j + 1 < m ? colNorm(c, j + 1) : 0;
        } else {
          vn1[c] *= Math.sqrt(temp);
        }
      }
    }
  }
}

/** Extract R (rRows×n upper triangle) and optionally form Q (m×qCols). */
function qrOutputs(
  aRe: Float64Array,
  aIm: Float64Array,
  m: number,
  n: number,
  tauRe: Float64Array,
  tauIm: Float64Array,
  econ: boolean,
  wantQ: boolean
) {
  const k = Math.min(m, n);
  const rRows = econ ? k : m;
  const RRe = allocFloat64Array(rRows * n),
    RIm = allocFloat64Array(rRows * n);
  for (let j = 0; j < n; j++)
    for (let i = 0; i <= Math.min(j, rRows - 1); i++) {
      RRe[i + j * rRows] = aRe[i + j * m];
      RIm[i + j * rRows] = aIm[i + j * m];
    }
  if (!wantQ) return { RRe, RIm };

  // zung2r: Q = H_0 H_1 ... H_{k-1} I(:, 1:qCols)
  const qCols = econ ? k : m;
  const QRe = allocFloat64Array(m * qCols),
    QIm = allocFloat64Array(m * qCols);
  for (let i = 0; i < qCols; i++) QRe[i + i * m] = 1;
  for (let j = k - 1; j >= 0; j--) {
    const tr = tauRe[j],
      ti = tauIm[j];
    if (tr === 0 && ti === 0) continue;
    for (let c = j; c < qCols; c++) {
      let wr = QRe[j + c * m],
        wi = QIm[j + c * m];
      for (let i = j + 1; i < m; i++) {
        const vr = aRe[i + j * m],
          vi = aIm[i + j * m];
        const xr = QRe[i + c * m],
          xi = QIm[i + c * m];
        wr += vr * xr + vi * xi;
        wi += vr * xi - vi * xr;
      }
      const sr = tr * wr - ti * wi,
        si = tr * wi + ti * wr;
      QRe[j + c * m] -= sr;
      QIm[j + c * m] -= si;
      for (let i = j + 1; i < m; i++) {
        const vr = aRe[i + j * m],
          vi = aIm[i + j * m];
        QRe[i + c * m] -= vr * sr - vi * si;
        QIm[i + c * m] -= vr * si + vi * sr;
      }
    }
  }
  return { QRe, QIm, RRe, RIm };
}

export function qrComplexTs(
  dataRe: Float64Array,
  dataIm: Float64Array,
  m: number,
  n: number,
  econ: boolean,
  wantQ: boolean
): {
  QRe?: Float64Array;
  QIm?: Float64Array;
  RRe: Float64Array;
  RIm: Float64Array;
} {
  const k = Math.min(m, n);
  const aRe = allocFloat64Array(dataRe),
    aIm = allocFloat64Array(dataIm);
  const tauRe = allocFloat64Array(k),
    tauIm = allocFloat64Array(k);
  zgeqr2(aRe, aIm, m, n, tauRe, tauIm);
  return qrOutputs(aRe, aIm, m, n, tauRe, tauIm, econ, wantQ);
}

export function qrPivotComplexTs(
  dataRe: Float64Array,
  dataIm: Float64Array,
  m: number,
  n: number,
  econ: boolean
): {
  QRe: Float64Array;
  QIm: Float64Array;
  RRe: Float64Array;
  RIm: Float64Array;
  jpvt: Int32Array;
} {
  const k = Math.min(m, n);
  const aRe = allocFloat64Array(dataRe),
    aIm = allocFloat64Array(dataIm);
  const tauRe = allocFloat64Array(k),
    tauIm = allocFloat64Array(k);
  const jpvt = new Int32Array(n);
  zgeqr2(aRe, aIm, m, n, tauRe, tauIm, jpvt);
  const { QRe, QIm, RRe, RIm } = qrOutputs(
    aRe,
    aIm,
    m,
    n,
    tauRe,
    tauIm,
    econ,
    true
  );
  return { QRe: QRe!, QIm: QIm!, RRe, RIm, jpvt };
}

/** zgetf2: LU with partial pivoting, L unit lower and U packed in place. */
export function luComplexTs(
  dataRe: Float64Array,
  dataIm: Float64Array,
  m: number,
  n: number
): { LURe: Float64Array; LUIm: Float64Array; ipiv: Int32Array } {
  const re = allocFloat64Array(dataRe),
    im = allocFloat64Array(dataIm);
  const k = Math.min(m, n);
  const ipiv = new Int32Array(k);
  for (let j = 0; j < k; j++) {
    let p = j,
      best = -1;
    for (let i = j; i < m; i++) {
      const v = Math.abs(re[i + j * m]) + Math.abs(im[i + j * m]);
      if (v > best) {
        best = v;
        p = i;
      }
    }
    ipiv[j] = p + 1;
    const dr0 = re[p + j * m],
      di0 = im[p + j * m];
    if (dr0 === 0 && di0 === 0) continue;
    if (p !== j) {
      for (let c = 0; c < n; c++) {
        const a = j + c * m,
          b = p + c * m;
        let t = re[a];
        re[a] = re[b];
        re[b] = t;
        t = im[a];
        im[a] = im[b];
        im[b] = t;
      }
    }
    const dd = dr0 * dr0 + di0 * di0;
    for (let i = j + 1; i < m; i++) {
      const xr = re[i + j * m],
        xi = im[i + j * m];
      re[i + j * m] = (xr * dr0 + xi * di0) / dd;
      im[i + j * m] = (xi * dr0 - xr * di0) / dd;
    }
    for (let c = j + 1; c < n; c++) {
      const ur = re[j + c * m],
        ui = im[j + c * m];
      if (ur === 0 && ui === 0) continue;
      for (let i = j + 1; i < m; i++) {
        const lr = re[i + j * m],
          li = im[i + j * m];
        re[i + c * m] -= lr * ur - li * ui;
        im[i + c * m] -= lr * ui + li * ur;
      }
    }
  }
  return { LURe: re, LUIm: im, ipiv };
}
