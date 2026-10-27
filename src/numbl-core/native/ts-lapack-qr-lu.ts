/**
 * QR and LU entry points of the ts-lapack bridge that are not plain calls
 * into src/ts-lapack: real column-pivoted QR, and complex QR (zgeqr2/zung2r),
 * column-pivoted QR (zlaqp2) and LU (zgetf2).
 *
 * The complex routines are unblocked translations that follow the LAPACK
 * conventions (zlarfg: real diagonal of R with beta = -sign(Re(alpha))*norm;
 * izamax pivoting by |re|+|im|) so results agree with MATLAB up to rounding.
 */

import { allocFloat64Array } from "../runtime/alloc.js";
import { zlarfg } from "./ts-lapack-complex.js";

/**
 * Real column-pivoted QR, A*P = Q*R (dgeqp3 + dorgqr equivalent): Householder
 * with greedy pivoting on downdated squared column norms. jpvt is 1-based.
 */
export function qrPivotTs(
  data: Float64Array,
  m: number,
  n: number,
  econ: boolean
): { Q: Float64Array; R: Float64Array; jpvt: Int32Array } {
  const k = Math.min(m, n);
  const R_data = allocFloat64Array(data);
  const perm = allocFloat64Array(n);
  for (let i = 0; i < n; i++) perm[i] = i; // 0-based during computation

  // Compute initial column norms
  const colNorms = allocFloat64Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < m; i++) {
      const v = R_data[j * m + i];
      s += v * v;
    }
    colNorms[j] = s;
  }

  const vecs: Float64Array[] = [];
  const taus: number[] = [];

  for (let j = 0; j < k; j++) {
    // Find column with max remaining norm
    let maxNorm = colNorms[j];
    let maxCol = j;
    for (let c = j + 1; c < n; c++) {
      if (colNorms[c] > maxNorm) {
        maxNorm = colNorms[c];
        maxCol = c;
      }
    }
    // Swap columns j and maxCol in R_data, perm, colNorms
    if (maxCol !== j) {
      for (let i = 0; i < m; i++) {
        const ij = j * m + i;
        const ic = maxCol * m + i;
        const tmp = R_data[ij];
        R_data[ij] = R_data[ic];
        R_data[ic] = tmp;
      }
      const tmpP = perm[j];
      perm[j] = perm[maxCol];
      perm[maxCol] = tmpP;
      const tmpN = colNorms[j];
      colNorms[j] = colNorms[maxCol];
      colNorms[maxCol] = tmpN;
    }

    // Householder reflector for column j
    const len = m - j;
    const x = allocFloat64Array(len);
    for (let i = 0; i < len; i++) x[i] = R_data[j * m + j + i];
    let normx = 0;
    for (let i = 0; i < len; i++) normx += x[i] * x[i];
    normx = Math.sqrt(normx);
    if (normx === 0) {
      vecs.push(allocFloat64Array(len));
      taus.push(0);
      // Update remaining column norms
      for (let c = j + 1; c < n; c++) {
        const vj = R_data[c * m + j];
        colNorms[c] -= vj * vj;
        if (colNorms[c] < 0) colNorms[c] = 0;
      }
      continue;
    }
    const sign = x[0] >= 0 ? 1 : -1;
    const alpha = -sign * normx;
    const v = allocFloat64Array(len);
    v[0] = x[0] - alpha;
    for (let i = 1; i < len; i++) v[i] = x[i];
    let vnorm = 0;
    for (let i = 0; i < len; i++) vnorm += v[i] * v[i];
    const tau = vnorm === 0 ? 0 : 2.0 / vnorm;
    vecs.push(v);
    taus.push(tau);
    for (let c = j; c < n; c++) {
      let dot = 0;
      for (let i = 0; i < len; i++) dot += v[i] * R_data[c * m + j + i];
      const scale = tau * dot;
      for (let i = 0; i < len; i++) R_data[c * m + j + i] -= scale * v[i];
    }
    // Update remaining column norms
    for (let c = j + 1; c < n; c++) {
      const vj = R_data[c * m + j];
      colNorms[c] -= vj * vj;
      if (colNorms[c] < 0) colNorms[c] = 0;
    }
  }

  // Build Q
  const qCols = econ ? k : m;
  const Q_data = allocFloat64Array(m * qCols);
  for (let i = 0; i < Math.min(m, qCols); i++) Q_data[i * m + i] = 1;
  for (let j = k - 1; j >= 0; j--) {
    const v = vecs[j],
      tau = taus[j];
    if (tau === 0) continue;
    const len = m - j;
    for (let c = j; c < qCols; c++) {
      let dot = 0;
      for (let i = 0; i < len; i++) dot += v[i] * Q_data[c * m + j + i];
      const scale = tau * dot;
      for (let i = 0; i < len; i++) Q_data[c * m + j + i] -= scale * v[i];
    }
  }

  // Build R
  const rRows = econ ? k : m;
  const R_out = allocFloat64Array(rRows * n);
  for (let r = 0; r < rRows; r++)
    for (let c = r; c < n; c++) R_out[c * rRows + r] = R_data[c * m + r];

  const jpvt = new Int32Array(n);
  for (let i = 0; i < n; i++) jpvt[i] = perm[i] + 1;
  return { Q: Q_data, R: R_out, jpvt };
}

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
    const [tr, ti] = zlarfg(m - j, aRe, aIm, j + j * m, 1);
    tauRe[j] = tr;
    tauIm[j] = ti;
    if (tr !== 0 || ti !== 0) {
      // Apply H_j^H = I - conj(tau) v v^H to A(j:m-1, j+1:n-1)
      const ctr = tr,
        cti = -ti;
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
