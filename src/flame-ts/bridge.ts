// FLAME-TS bridge: assembles a LapackBridge using blocked BLIS/FLAME routines.

import type { LapackBridge } from "../numbl-core/native/lapack-bridge.js";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";
import { matmul } from "./blas/dgemm.js";
import { dgetrf } from "./lapack/dgetrf.js";
import { dpotrf } from "./lapack/dpotrf.js";
import { dgeqrf } from "./lapack/dgeqrf.js";
import { dorgqr_optimized } from "../ts-lapack/src/SRC/dorgqr_optimized.js";
import { dtrsmLUNN, dtrsmLLNUnit } from "./blas/dtrsm.js";

const tsBridge = getTsLapackBridge();

function inv(data: Float64Array, n: number): Float64Array {
  const a = new Float64Array(data);
  const ipiv = new Int32Array(n);
  const info = dgetrf(n, n, a, 0, n, ipiv);
  if (info > 0) throw new Error("inv: matrix is singular (dgetrf)");

  const eye = new Float64Array(n * n);
  for (let i = 0; i < n; i++) eye[i + i * n] = 1;

  for (let i = 0; i < n; i++) {
    const pi = ipiv[i] - 1;
    if (pi !== i) {
      for (let c = 0; c < n; c++) {
        const tmp = eye[i + c * n];
        eye[i + c * n] = eye[pi + c * n];
        eye[pi + c * n] = tmp;
      }
    }
  }

  dtrsmLLNUnit(n, n, 1.0, a, 0, n, eye, 0, n);
  dtrsmLUNN(n, n, 1.0, a, 0, n, eye, 0, n);
  return eye;
}

function lu(
  data: Float64Array,
  m: number,
  n: number
): { LU: Float64Array; ipiv: Int32Array } {
  const a = new Float64Array(data);
  const k = Math.min(m, n);
  const ipiv = new Int32Array(k);
  dgetrf(m, n, a, 0, m, ipiv);
  return { LU: a, ipiv };
}

function qr(
  data: Float64Array,
  m: number,
  n: number,
  econ: boolean,
  wantQ: boolean
): { Q: Float64Array; R: Float64Array } {
  const k = Math.min(m, n);
  const a = new Float64Array(data);
  const tau = new Float64Array(k);

  dgeqrf(m, n, a, 0, m, tau, 0);

  // Extract R from upper triangle
  const rRows = econ ? k : m;
  const R = new Float64Array(rRows * n);
  for (let j = 0; j < n; j++) {
    const ilim = Math.min(j, k - 1);
    for (let i = 0; i <= ilim; i++) {
      R[i + j * rRows] = a[i + j * m];
    }
  }

  // Generate Q
  const qCols = econ ? k : m;
  let Q: Float64Array;

  if (wantQ) {
    const qBuf = new Float64Array(m * qCols);
    const colsToCopy = Math.min(n, qCols);
    for (let j = 0; j < colsToCopy; j++) {
      for (let i = 0; i < m; i++) {
        qBuf[i + j * m] = a[i + j * m];
      }
    }
    dorgqr_optimized(m, qCols, k, qBuf, 0, m, tau, 0);
    Q = qBuf;
  } else {
    Q = new Float64Array(0);
  }

  return { Q, R };
}

function linsolve(
  A: Float64Array,
  m: number,
  n: number,
  B: Float64Array,
  nrhs: number
): Float64Array {
  if (m === n) {
    const a = new Float64Array(A);
    const b = new Float64Array(B);
    const ipiv = new Int32Array(n);
    const info = dgetrf(n, n, a, 0, n, ipiv);
    if (info > 0) throw new Error("linsolve: matrix is singular");

    for (let i = 0; i < n; i++) {
      const pi = ipiv[i] - 1;
      if (pi !== i) {
        for (let c = 0; c < nrhs; c++) {
          const tmp = b[i + c * n];
          b[i + c * n] = b[pi + c * n];
          b[pi + c * n] = tmp;
        }
      }
    }

    dtrsmLLNUnit(n, nrhs, 1.0, a, 0, n, b, 0, n);
    dtrsmLUNN(n, nrhs, 1.0, a, 0, n, b, 0, n);
    return b;
  }

  if (m > n) {
    // Overdetermined: thin QR then solve R * X = Q' * B
    const a = new Float64Array(A);
    const tau = new Float64Array(n);
    dgeqrf(m, n, a, 0, m, tau, 0);

    // Apply Q' to B
    const b = new Float64Array(B);
    for (let j = 0; j < n; j++) {
      const tauJ = tau[j];
      if (tauJ === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        let vdotb = b[j + c * m];
        for (let i = 1; i < m - j; i++) {
          vdotb += a[j + i + j * m] * b[j + i + c * m];
        }
        const scale = tauJ * vdotb;
        b[j + c * m] -= scale;
        for (let i = 1; i < m - j; i++) {
          b[j + i + c * m] -= scale * a[j + i + j * m];
        }
      }
    }

    // Back-substitute R * X = (Q'*B)[0:n, :]
    const x = new Float64Array(n * nrhs);
    for (let c = 0; c < nrhs; c++) {
      for (let i = n - 1; i >= 0; i--) {
        let val = b[i + c * m];
        for (let k = i + 1; k < n; k++) {
          val -= a[i + k * m] * x[k + c * n];
        }
        x[i + c * n] = val / a[i + i * m];
      }
    }
    return x;
  }

  // Underdetermined: fall back to ts-lapack
  return tsBridge.linsolve!(A, m, n, B, nrhs);
}

function chol(
  data: Float64Array,
  n: number,
  upper: boolean
): { R: Float64Array; info: number } {
  if (upper) {
    const a = new Float64Array(n * n);
    for (let j = 0; j < n; j++)
      for (let i = j; i < n; i++) a[i + j * n] = data[j + i * n];

    const info = dpotrf(n, a, 0, n);

    const R = new Float64Array(n * n);
    for (let j = 0; j < n; j++)
      for (let i = 0; i <= j; i++) R[i + j * n] = a[j + i * n];

    return { R, info };
  } else {
    const a = new Float64Array(data);
    const info = dpotrf(n, a, 0, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < j; i++) a[i + j * n] = 0;
    return { R: a, info };
  }
}

// SVD: for tall matrices (m > 2*n), first QR-reduce to square, then use ts-lapack SVD.
// This leverages our fast blocked QR for the expensive reduction step.
function svd(
  data: Float64Array,
  m: number,
  n: number,
  econ: boolean,
  computeUV: boolean
): { U?: Float64Array; S: Float64Array; V?: Float64Array } {
  // QR-first optimization for tall matrices
  if (computeUV && m >= 2 * n) {
    const { Q: Q1, R } = qr(data, m, n, true, true);
    const inner = tsBridge.svd!(R, n, n, econ, true);

    // U = Q1 * inner.U (m×k = m×n * n×k)
    const k = econ ? n : n;
    const U = matmul(Q1, m, n, inner.U!, k);

    return { U, S: inner.S, V: inner.V };
  }

  return tsBridge.svd!(data, m, n, econ, computeUV);
}

// Eig: delegate to ts-lapack. The iterative QR algorithm doesn't benefit from blocking.
function eig(
  data: Float64Array,
  n: number,
  computeVL: boolean,
  computeVR: boolean,
  balance: boolean
): {
  wr: Float64Array;
  wi: Float64Array;
  VL?: Float64Array;
  VR?: Float64Array;
} {
  return tsBridge.eig!(data, n, computeVL, computeVR, balance);
}

const _bridge: LapackBridge = {
  matmul,
  inv,
  lu,
  qr,
  linsolve,
  chol,
  svd,
  eig,
  linsolveComplex: tsBridge.linsolveComplex!.bind(tsBridge),
  cholComplex: tsBridge.cholComplex!.bind(tsBridge),
};

export function getFlameBridge(): LapackBridge {
  return _bridge;
}
