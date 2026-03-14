/* eslint-disable @typescript-eslint/no-unused-vars */
/**
 * Pure-TypeScript LAPACK bridge — implements the LapackBridge interface using
 * the ts-lapack translations of LAPACK Fortran routines.
 *
 * This is always available (no native addon required) and is used as the
 * fallback when the native C++ addon is not loaded (e.g. in the browser).
 */

import { dgetrf } from "../../ts-lapack/src/SRC/dgetrf.js";
import { dgetri } from "../../ts-lapack/src/SRC/dgetri.js";
import { dgeev } from "../../ts-lapack/src/SRC/dgeev.js";
import { dgesvd as _dgesvd } from "../../ts-lapack/src/SRC/dgesvd.js";
import type { LapackBridge } from "./lapack-bridge.js";
import { NOTRANS } from "../../ts-lapack/src/utils/constants.js";
import { dorgqr_optimized } from "../../ts-lapack/src/SRC/dorgqr_optimized.js";
import { dgeqrf_optimized } from "../../ts-lapack/src/SRC/dgeqrf_optimized.js";
import { dgemm_optimized } from "../../ts-lapack/src/BLAS/dgemm_optimized.js";

function inv(data: Float64Array, n: number): Float64Array {
  // Copy — dgetrf/dgetri operate in place
  const a = new Float64Array(data);
  const ipiv = new Int32Array(n);

  const info1 = dgetrf(n, n, a, n, ipiv);
  if (info1 > 0) throw new Error("inv: matrix is singular (dgetrf)");

  const info2 = dgetri(n, a, n, ipiv);
  if (info2 > 0) throw new Error("inv: matrix is singular (dgetri)");

  return a;
}

// C = A * B  (transa='N', transb='N', alpha=1, beta=0)
// A is m×k, B is k×n, C is m×n — all column-major, zero-offset
function matmul(
  A: Float64Array,
  m: number,
  k: number,
  B: Float64Array,
  n: number
): Float64Array {
  const c = new Float64Array(m * n);
  // Even this optimized version is around 50 times slower than the native
  // openblas addon, for multiplying two 1000x1000 matrices. The
  // WebGPU-accelerated version is much faster thank this typescript version,
  // but that only works 32-bit floats. I determined it is better to go for
  // 64-bit floats, and recommend desktop (or remote) usage for heavy linear
  // algebra.
  dgemm_optimized(
    NOTRANS,
    NOTRANS,
    m,
    n,
    k,
    1.0,
    A,
    0,
    m,
    B,
    0,
    k,
    0.0,
    c,
    0,
    m
  );
  return c;
}

/**
 * QR decomposition: A = Q * R
 *
 * Uses LAPACK dgeqrf (QR factorisation) + dorgqr (generate Q).
 * data is a column-major Float64Array of length m*n (not modified).
 * econ=true:  economy/thin QR — Q is m×k, R is k×n  (k = min(m,n))
 * econ=false: full QR         — Q is m×m, R is m×n
 * wantQ=false: skips dorgqr; Q will be absent from the returned object.
 * Returns { Q, R } as Float64Arrays in column-major order.
 */
function qr(
  data: Float64Array,
  m: number,
  n: number,
  econ: boolean,
  wantQ: boolean
): { Q: Float64Array; R: Float64Array } {
  const k = Math.min(m, n);

  // Step 1: QR factorisation (dgeqrf) — operates in place on a copy
  const a = new Float64Array(data);
  const tau = new Float64Array(k);

  // In browser, the _optimized versions are around 15 times faster than the
  // non-optimized versions for 1000x1000 matrix. Unfortunately, this is still
  // around 10 times slower than the native openblas addon. And even that is
  // slower than the MATLAB qr implementation.
  const info1 = dgeqrf_optimized(m, n, a, 0, m, tau, 0);
  if (info1 < 0) throw new Error(`qr: dgeqrf argument error (info=${info1})`);

  // Step 2: Extract R from the upper triangle of the factored matrix
  const rRows = econ ? k : m;
  const R = new Float64Array(rRows * n);
  for (let j = 0; j < n; j++) {
    const ilim = Math.min(j, k - 1); // min(j, k-1) — 0-based
    for (let i = 0; i <= ilim; i++) {
      R[i + j * rRows] = a[i + j * m];
    }
  }

  // Step 3: Generate Q via dorgqr (only if wantQ)
  const qCols = econ ? k : m;
  let Q: Float64Array;

  if (wantQ) {
    // Build buffer of size m×qCols containing first qCols columns of a
    const qBuf = new Float64Array(m * qCols);
    const colsToCopy = Math.min(n, qCols);
    for (let j = 0; j < colsToCopy; j++) {
      for (let i = 0; i < m; i++) {
        qBuf[i + j * m] = a[i + j * m];
      }
    }

    const info2 = dorgqr_optimized(m, qCols, k, qBuf, 0, m, tau, 0);
    if (info2 < 0) throw new Error(`qr: dorgqr argument error (info=${info2})`);

    Q = qBuf;
  } else {
    Q = new Float64Array(0);
  }

  return { Q, R };
}

/**
 * Solve A * X = B.
 *
 * Square (m === n):
 *   LU factorisation via dgetrf, then forward/back substitution.
 *
 * Overdetermined (m > n):
 *   Thin QR via dgeqrf_optimized; apply Q^T to B in place, then
 *   back-substitute through R.  Least-squares solution.
 *
 * Underdetermined (m < n):
 *   Thin QR of A^T (= LQ of A) via dgeqrf_optimized; forward-substitute
 *   through R^T, then apply Q from the left.  Minimum-norm solution.
 *
 * All matrices are column-major Float64Arrays.
 * A and B are not modified; returns a new n×nrhs Float64Array X.
 */
function linsolve(
  A: Float64Array,
  m: number,
  n: number,
  B: Float64Array,
  nrhs: number
): Float64Array {
  if (m === n) {
    // ── Square: LU factorisation + solve ───────────────────────────────────────
    const a = new Float64Array(A);
    const b = new Float64Array(B); // n × nrhs column-major
    const ipiv = new Int32Array(n);

    const info1 = dgetrf(n, n, a, n, ipiv);
    if (info1 > 0) throw new Error("linsolve: matrix is singular (dgetrf)");

    // Apply row permutations to B (ipiv is 1-based)
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

    // Forward substitution: solve L * Y = P*B  (L unit lower triangular)
    for (let c = 0; c < nrhs; c++) {
      for (let i = 1; i < n; i++) {
        for (let k = 0; k < i; k++) {
          b[i + c * n] -= a[i + k * n] * b[k + c * n];
        }
      }
    }

    // Backward substitution: solve U * X = Y  (U upper triangular)
    for (let c = 0; c < nrhs; c++) {
      for (let i = n - 1; i >= 0; i--) {
        for (let k = i + 1; k < n; k++) {
          b[i + c * n] -= a[i + k * n] * b[k + c * n];
        }
        b[i + c * n] /= a[i + i * n];
      }
    }

    return b; // n × nrhs
  } else if (m > n) {
    // ── Overdetermined: thin QR then solve R * X = Q^T * B ────────────────────
    const kk = n; // k = min(m,n) = n  (since m > n)
    const a = new Float64Array(A);
    const tau = new Float64Array(kk);
    dgeqrf_optimized(m, n, a, 0, m, tau, 0);

    // Apply Q^T to B in place: Q^T = H_0 * H_1 * … * H_{k-1}
    const b = new Float64Array(B); // m × nrhs
    for (let j = 0; j < kk; j++) {
      const tauJ = tau[j];
      if (tauJ === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        // v = [1; a[j+1..m-1, j]],  apply H_j = I - tau*v*v^T  to b[j..m-1, c]
        let vdotb = b[j + c * m]; // v[0] = 1
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

    // Back-substitute: solve R * X = (Q^T*B)[0:n, :]
    // R is in the upper triangle of a; a[i + j*m] for i <= j  (lda = m)
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

    return x; // n × nrhs
  } else {
    // ── Underdetermined (m < n): LQ via QR of A^T, minimum-norm solve ─────────
    // Transpose A → At is n×m
    const At = new Float64Array(n * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        At[j + i * n] = A[i + j * m];
      }
    }

    const kk = m; // k = min(n,m) = m  (since n > m)
    const tauT = new Float64Array(kk);
    dgeqrf_optimized(n, m, At, 0, n, tauT, 0);
    // At now holds: R (m×m upper tri at At[i + j*n] for i<=j<m)
    //               Householder vectors for Q of A^T (n×m)

    // Forward substitution: solve R^T * Y = B
    // R^T is lower triangular: R^T[i,j] = R[j,i] = At[j + i*n]  (j <= i)
    const Y = new Float64Array(m * nrhs);
    for (let c = 0; c < nrhs; c++) {
      for (let i = 0; i < m; i++) {
        let val = B[i + c * m];
        for (let k = 0; k < i; k++) {
          val -= At[k + i * n] * Y[k + c * m]; // R^T[i,k] = At[k + i*n]
        }
        Y[i + c * m] = val / At[i + i * n]; // diagonal: R[i,i] = At[i + i*n]
      }
    }

    // Compute X = Q * Y  where Q is n×m stored as Householder reflectors in At
    // Initialise x = [Y; 0]  (n × nrhs, zero-padded below row m)
    const x = new Float64Array(n * nrhs);
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < nrhs; c++) {
        x[i + c * n] = Y[i + c * m];
      }
    }

    // Q = H_0 * H_1 * … * H_{k-1}, so Q*y = H_0*(H_1*(…*(H_{k-1}*y)…))
    // Apply reflectors in reverse order
    for (let j = kk - 1; j >= 0; j--) {
      const tauJ = tauT[j];
      if (tauJ === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        // v = [1; At[j+1..n-1, j]],  apply H_j to x[j..n-1, c]
        let vdotx = x[j + c * n]; // v[0] = 1
        for (let i = 1; i < n - j; i++) {
          vdotx += At[j + i + j * n] * x[j + i + c * n];
        }
        const scale = tauJ * vdotx;
        x[j + c * n] -= scale;
        for (let i = 1; i < n - j; i++) {
          x[j + i + c * n] -= scale * At[j + i + j * n];
        }
      }
    }

    return x; // n × nrhs
  }
}

/**
 * Eigenvalue decomposition: A = V * D * V^{-1}
 *
 * Uses LAPACK dgeev to compute eigenvalues (wr + i*wi) and optionally
 * left/right eigenvectors of an n×n real matrix A.
 */
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
  const a = new Float64Array(data); // dgeev overwrites A
  const wr = new Float64Array(n);
  const wi = new Float64Array(n);
  const ldvl = computeVL ? n : 1;
  const ldvr = computeVR ? n : 1;
  const vl = new Float64Array(ldvl * (computeVL ? n : 0));
  const vr = new Float64Array(ldvr * (computeVR ? n : 0));

  const jobvl = computeVL ? 1 : 0; // 0='N', 1='V'
  const jobvr = computeVR ? 1 : 0;

  // Workspace query
  const workQuery = new Float64Array(1);
  dgeev(
    jobvl,
    jobvr,
    n,
    a,
    0,
    n,
    wr,
    0,
    wi,
    0,
    vl,
    0,
    ldvl,
    vr,
    0,
    ldvr,
    workQuery,
    0,
    -1,
    balance
  );
  const lwork = Math.max(1, Math.floor(workQuery[0]));
  const work = new Float64Array(lwork);

  // Reset a since workspace query may have modified it
  a.set(data);

  const info = dgeev(
    jobvl,
    jobvr,
    n,
    a,
    0,
    n,
    wr,
    0,
    wi,
    0,
    vl,
    0,
    ldvl,
    vr,
    0,
    ldvr,
    work,
    0,
    lwork,
    balance
  );
  if (info > 0)
    throw new Error(`eig: QR algorithm failed to converge (info=${info})`);
  if (info < 0) throw new Error(`eig: illegal argument (info=${info})`);

  return {
    wr,
    wi,
    VL: computeVL ? vl : undefined,
    VR: computeVR ? vr : undefined,
  };
}

function linsolveComplex(
  ARe: Float64Array,
  AIm: Float64Array,
  m: number,
  n: number,
  BRe: Float64Array,
  BIm: Float64Array,
  nrhs: number
): { re: Float64Array; im: Float64Array } {
  if (m === n) {
    // ── Square: complex LU with partial pivoting ────────────────────────────
    const aRe = new Float64Array(ARe);
    const aIm = new Float64Array(AIm);
    const bRe = new Float64Array(BRe);
    const bIm = new Float64Array(BIm);
    const ipiv = new Int32Array(n);

    for (let k = 0; k < n; k++) {
      // Find pivot: max |A(i,k)| for i >= k
      let maxVal =
        aRe[k + k * n] * aRe[k + k * n] + aIm[k + k * n] * aIm[k + k * n];
      let maxIdx = k;
      for (let i = k + 1; i < n; i++) {
        const v =
          aRe[i + k * n] * aRe[i + k * n] + aIm[i + k * n] * aIm[i + k * n];
        if (v > maxVal) {
          maxVal = v;
          maxIdx = i;
        }
      }
      ipiv[k] = maxIdx;

      if (maxIdx !== k) {
        for (let j = 0; j < n; j++) {
          let tmp = aRe[k + j * n];
          aRe[k + j * n] = aRe[maxIdx + j * n];
          aRe[maxIdx + j * n] = tmp;
          tmp = aIm[k + j * n];
          aIm[k + j * n] = aIm[maxIdx + j * n];
          aIm[maxIdx + j * n] = tmp;
        }
      }

      if (maxVal === 0) throw new Error("linsolveComplex: matrix is singular");

      const dRe = aRe[k + k * n];
      const dIm = aIm[k + k * n];
      const dAbs2 = dRe * dRe + dIm * dIm;

      for (let i = k + 1; i < n; i++) {
        const numRe = aRe[i + k * n];
        const numIm = aIm[i + k * n];
        const lRe = (numRe * dRe + numIm * dIm) / dAbs2;
        const lIm = (numIm * dRe - numRe * dIm) / dAbs2;
        aRe[i + k * n] = lRe;
        aIm[i + k * n] = lIm;

        for (let j = k + 1; j < n; j++) {
          aRe[i + j * n] -= lRe * aRe[k + j * n] - lIm * aIm[k + j * n];
          aIm[i + j * n] -= lRe * aIm[k + j * n] + lIm * aRe[k + j * n];
        }
      }
    }

    // Apply row permutations to B
    for (let i = 0; i < n; i++) {
      const pi = ipiv[i];
      if (pi !== i) {
        for (let c = 0; c < nrhs; c++) {
          let tmp = bRe[i + c * n];
          bRe[i + c * n] = bRe[pi + c * n];
          bRe[pi + c * n] = tmp;
          tmp = bIm[i + c * n];
          bIm[i + c * n] = bIm[pi + c * n];
          bIm[pi + c * n] = tmp;
        }
      }
    }

    // Forward substitution: L * Y = P*B
    for (let c = 0; c < nrhs; c++) {
      for (let i = 1; i < n; i++) {
        for (let k = 0; k < i; k++) {
          const lRe = aRe[i + k * n];
          const lIm = aIm[i + k * n];
          bRe[i + c * n] -= lRe * bRe[k + c * n] - lIm * bIm[k + c * n];
          bIm[i + c * n] -= lRe * bIm[k + c * n] + lIm * bRe[k + c * n];
        }
      }
    }

    // Backward substitution: U * X = Y
    for (let c = 0; c < nrhs; c++) {
      for (let i = n - 1; i >= 0; i--) {
        for (let k = i + 1; k < n; k++) {
          const uRe = aRe[i + k * n];
          const uIm = aIm[i + k * n];
          bRe[i + c * n] -= uRe * bRe[k + c * n] - uIm * bIm[k + c * n];
          bIm[i + c * n] -= uRe * bIm[k + c * n] + uIm * bRe[k + c * n];
        }
        const dRe = aRe[i + i * n];
        const dIm = aIm[i + i * n];
        const dAbs2 = dRe * dRe + dIm * dIm;
        const xRe = (bRe[i + c * n] * dRe + bIm[i + c * n] * dIm) / dAbs2;
        const xIm = (bIm[i + c * n] * dRe - bRe[i + c * n] * dIm) / dAbs2;
        bRe[i + c * n] = xRe;
        bIm[i + c * n] = xIm;
      }
    }

    return { re: bRe, im: bIm };
  } else if (m > n) {
    // ── Overdetermined: complex Householder QR, solve R*X = Q^H*B ───────────
    const kk = n;
    const aRe = new Float64Array(ARe);
    const aIm = new Float64Array(AIm);
    const tauRe = new Float64Array(kk);
    const tauIm = new Float64Array(kk);

    // Complex Householder QR factorization
    for (let j = 0; j < kk; j++) {
      // Compute norm of A(j:m-1, j)
      let xnorm2 = 0;
      for (let i = j + 1; i < m; i++) {
        xnorm2 +=
          aRe[i + j * m] * aRe[i + j * m] + aIm[i + j * m] * aIm[i + j * m];
      }
      const alphaRe = aRe[j + j * m];
      const alphaIm = aIm[j + j * m];
      const alphaAbs2 = alphaRe * alphaRe + alphaIm * alphaIm;

      if (xnorm2 === 0 && alphaIm === 0) {
        tauRe[j] = 0;
        tauIm[j] = 0;
        continue;
      }

      const xnorm = Math.sqrt(alphaAbs2 + xnorm2);
      // beta = -sign(Re(alpha)) * xnorm
      const betaRe = alphaRe >= 0 ? -xnorm : xnorm;

      // tau = (beta - alpha) / beta
      const diffRe = betaRe - alphaRe;
      const diffIm = -alphaIm;
      tauRe[j] = diffRe / betaRe;
      tauIm[j] = diffIm / betaRe;

      // v(j+1:m-1) = A(j+1:m-1, j) / (alpha - beta)
      const denomRe = alphaRe - betaRe;
      const denomIm = alphaIm;
      const denomAbs2 = denomRe * denomRe + denomIm * denomIm;
      for (let i = j + 1; i < m; i++) {
        const numRe = aRe[i + j * m];
        const numIm = aIm[i + j * m];
        aRe[i + j * m] = (numRe * denomRe + numIm * denomIm) / denomAbs2;
        aIm[i + j * m] = (numIm * denomRe - numRe * denomIm) / denomAbs2;
      }

      // Set diagonal to beta
      aRe[j + j * m] = betaRe;
      aIm[j + j * m] = 0;

      // Apply H_j to A(j:m-1, j+1:n-1)
      for (let col = j + 1; col < n; col++) {
        // w = v^H * A(:, col), v[0]=1, v[i] stored in aRe/aIm[j+i + j*m]
        let wRe = aRe[j + col * m];
        let wIm = aIm[j + col * m];
        for (let i = 1; i < m - j; i++) {
          // conj(v[i]) * A[j+i, col]
          const vRe = aRe[j + i + j * m];
          const vIm = -aIm[j + i + j * m]; // conjugate
          wRe += vRe * aRe[j + i + col * m] - vIm * aIm[j + i + col * m];
          wIm += vRe * aIm[j + i + col * m] + vIm * aRe[j + i + col * m];
        }
        // A(:, col) -= tau * v * w
        const twRe = tauRe[j] * wRe - tauIm[j] * wIm;
        const twIm = tauRe[j] * wIm + tauIm[j] * wRe;
        aRe[j + col * m] -= twRe;
        aIm[j + col * m] -= twIm;
        for (let i = 1; i < m - j; i++) {
          const vRe = aRe[j + i + j * m];
          const vIm = aIm[j + i + j * m];
          aRe[j + i + col * m] -= vRe * twRe - vIm * twIm;
          aIm[j + i + col * m] -= vRe * twIm + vIm * twRe;
        }
      }
    }

    // Apply Q^H to B: Q^H = H_{k-1} * ... * H_1 * H_0 (each H is Hermitian)
    const bRe = new Float64Array(BRe);
    const bIm = new Float64Array(BIm);
    for (let j = 0; j < kk; j++) {
      if (tauRe[j] === 0 && tauIm[j] === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        // w = v^H * b(:, c)
        let wRe = bRe[j + c * m];
        let wIm = bIm[j + c * m];
        for (let i = 1; i < m - j; i++) {
          const vRe = aRe[j + i + j * m];
          const vIm = -aIm[j + i + j * m]; // conjugate
          wRe += vRe * bRe[j + i + c * m] - vIm * bIm[j + i + c * m];
          wIm += vRe * bIm[j + i + c * m] + vIm * bRe[j + i + c * m];
        }
        // b -= tau * v * w
        const twRe = tauRe[j] * wRe - tauIm[j] * wIm;
        const twIm = tauRe[j] * wIm + tauIm[j] * wRe;
        bRe[j + c * m] -= twRe;
        bIm[j + c * m] -= twIm;
        for (let i = 1; i < m - j; i++) {
          const vRe = aRe[j + i + j * m];
          const vIm = aIm[j + i + j * m];
          bRe[j + i + c * m] -= vRe * twRe - vIm * twIm;
          bIm[j + i + c * m] -= vRe * twIm + vIm * twRe;
        }
      }
    }

    // Back-substitute: R * X = (Q^H*B)[0:n, :]
    const xRe = new Float64Array(n * nrhs);
    const xIm = new Float64Array(n * nrhs);
    for (let c = 0; c < nrhs; c++) {
      for (let i = n - 1; i >= 0; i--) {
        let valRe = bRe[i + c * m];
        let valIm = bIm[i + c * m];
        for (let k = i + 1; k < n; k++) {
          const rRe = aRe[i + k * m];
          const rIm = aIm[i + k * m];
          valRe -= rRe * xRe[k + c * n] - rIm * xIm[k + c * n];
          valIm -= rRe * xIm[k + c * n] + rIm * xRe[k + c * n];
        }
        const dRe = aRe[i + i * m];
        const dIm = aIm[i + i * m];
        const dAbs2 = dRe * dRe + dIm * dIm;
        xRe[i + c * n] = (valRe * dRe + valIm * dIm) / dAbs2;
        xIm[i + c * n] = (valIm * dRe - valRe * dIm) / dAbs2;
      }
    }

    return { re: xRe, im: xIm };
  } else {
    // ── Underdetermined (m < n): QR of A^H, minimum-norm solve ──────────────
    // Conjugate-transpose A → AH is n×m
    const ahRe = new Float64Array(n * m);
    const ahIm = new Float64Array(n * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        ahRe[j + i * n] = ARe[i + j * m];
        ahIm[j + i * n] = -AIm[i + j * m]; // conjugate
      }
    }

    const kk = m;
    const tauRe = new Float64Array(kk);
    const tauIm = new Float64Array(kk);

    // Complex Householder QR of A^H (n×m)
    for (let j = 0; j < kk; j++) {
      let xnorm2 = 0;
      for (let i = j + 1; i < n; i++) {
        xnorm2 +=
          ahRe[i + j * n] * ahRe[i + j * n] + ahIm[i + j * n] * ahIm[i + j * n];
      }
      const alphaRe = ahRe[j + j * n];
      const alphaIm = ahIm[j + j * n];
      const alphaAbs2 = alphaRe * alphaRe + alphaIm * alphaIm;

      if (xnorm2 === 0 && alphaIm === 0) {
        tauRe[j] = 0;
        tauIm[j] = 0;
        continue;
      }

      const xnorm = Math.sqrt(alphaAbs2 + xnorm2);
      const betaRe = alphaRe >= 0 ? -xnorm : xnorm;

      const diffRe = betaRe - alphaRe;
      const diffIm = -alphaIm;
      tauRe[j] = diffRe / betaRe;
      tauIm[j] = diffIm / betaRe;

      const denomRe = alphaRe - betaRe;
      const denomIm = alphaIm;
      const denomAbs2 = denomRe * denomRe + denomIm * denomIm;
      for (let i = j + 1; i < n; i++) {
        const numRe = ahRe[i + j * n];
        const numIm = ahIm[i + j * n];
        ahRe[i + j * n] = (numRe * denomRe + numIm * denomIm) / denomAbs2;
        ahIm[i + j * n] = (numIm * denomRe - numRe * denomIm) / denomAbs2;
      }

      ahRe[j + j * n] = betaRe;
      ahIm[j + j * n] = 0;

      for (let col = j + 1; col < m; col++) {
        let wRe = ahRe[j + col * n];
        let wIm = ahIm[j + col * n];
        for (let i = 1; i < n - j; i++) {
          const vRe = ahRe[j + i + j * n];
          const vIm = -ahIm[j + i + j * n];
          wRe += vRe * ahRe[j + i + col * n] - vIm * ahIm[j + i + col * n];
          wIm += vRe * ahIm[j + i + col * n] + vIm * ahRe[j + i + col * n];
        }
        const twRe = tauRe[j] * wRe - tauIm[j] * wIm;
        const twIm = tauRe[j] * wIm + tauIm[j] * wRe;
        ahRe[j + col * n] -= twRe;
        ahIm[j + col * n] -= twIm;
        for (let i = 1; i < n - j; i++) {
          const vRe = ahRe[j + i + j * n];
          const vIm = ahIm[j + i + j * n];
          ahRe[j + i + col * n] -= vRe * twRe - vIm * twIm;
          ahIm[j + i + col * n] -= vRe * twIm + vIm * twRe;
        }
      }
    }

    // Forward substitution: R^H * Y = B
    // R is upper triangular in ahRe/ahIm; R^H[i,k] = conj(R[k,i]) = conj(ah[k + i*n])
    const yRe = new Float64Array(m * nrhs);
    const yIm = new Float64Array(m * nrhs);
    for (let c = 0; c < nrhs; c++) {
      for (let i = 0; i < m; i++) {
        let valRe = BRe[i + c * m];
        let valIm = BIm[i + c * m];
        for (let k = 0; k < i; k++) {
          // R^H[i,k] = conj(ah[k + i*n])
          const rRe = ahRe[k + i * n];
          const rIm = -ahIm[k + i * n]; // conjugate
          valRe -= rRe * yRe[k + c * m] - rIm * yIm[k + c * m];
          valIm -= rRe * yIm[k + c * m] + rIm * yRe[k + c * m];
        }
        // Diagonal: R^H[i,i] = conj(ah[i + i*n])
        const dRe = ahRe[i + i * n];
        const dIm = -ahIm[i + i * n]; // conjugate
        const dAbs2 = dRe * dRe + dIm * dIm;
        yRe[i + c * m] = (valRe * dRe + valIm * dIm) / dAbs2;
        yIm[i + c * m] = (valIm * dRe - valRe * dIm) / dAbs2;
      }
    }

    // X = Q * Y: apply Householder reflectors in reverse order
    const xRe = new Float64Array(n * nrhs);
    const xIm = new Float64Array(n * nrhs);
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < nrhs; c++) {
        xRe[i + c * n] = yRe[i + c * m];
        xIm[i + c * n] = yIm[i + c * m];
      }
    }

    for (let j = kk - 1; j >= 0; j--) {
      if (tauRe[j] === 0 && tauIm[j] === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        let wRe = xRe[j + c * n];
        let wIm = xIm[j + c * n];
        for (let i = 1; i < n - j; i++) {
          const vRe = ahRe[j + i + j * n];
          const vIm = -ahIm[j + i + j * n]; // conjugate
          wRe += vRe * xRe[j + i + c * n] - vIm * xIm[j + i + c * n];
          wIm += vRe * xIm[j + i + c * n] + vIm * xRe[j + i + c * n];
        }
        const twRe = tauRe[j] * wRe - tauIm[j] * wIm;
        const twIm = tauRe[j] * wIm + tauIm[j] * wRe;
        xRe[j + c * n] -= twRe;
        xIm[j + c * n] -= twIm;
        for (let i = 1; i < n - j; i++) {
          const vRe = ahRe[j + i + j * n];
          const vIm = ahIm[j + i + j * n];
          xRe[j + i + c * n] -= vRe * twRe - vIm * twIm;
          xIm[j + i + c * n] -= vRe * twIm + vIm * twRe;
        }
      }
    }

    return { re: xRe, im: xIm };
  }
}

function lu(
  data: Float64Array,
  m: number,
  n: number
): { LU: Float64Array; ipiv: Int32Array } {
  const a = new Float64Array(data);
  const k = Math.min(m, n);
  const ipiv = new Int32Array(k);

  const info = dgetrf(m, n, a, m, ipiv);
  if (info < 0) throw new Error(`lu: illegal argument (info=${info})`);
  // info > 0 means U(info,info) is zero — factorization completed but U is singular.
  // Still returns a result in this case, so we don't throw.

  return { LU: a, ipiv };
}

/**
 * SVD decomposition: A = U * SIGMA * V'
 *
 * Uses LAPACK dgesvd. Returns singular values S, and optionally U and V.
 * Note: dgesvd returns VT (V transposed); we transpose it to return V.
 */
function svd(
  data: Float64Array,
  m: number,
  n: number,
  econ: boolean,
  computeUV: boolean
): { U?: Float64Array; S: Float64Array; V?: Float64Array } {
  const k = Math.min(m, n);
  const a = new Float64Array(data); // dgesvd overwrites A

  const s = new Float64Array(k);

  // JOBU/JOBVT encoding: 0='A', 1='S', 2='O', 3='N'
  const JOBU_A = 0;
  const JOBU_S = 1;
  const JOBU_N = 3;
  const JOBVT_A = 0;
  const JOBVT_S = 1;
  const JOBVT_N = 3;

  let jobu: number;
  let jobvt: number;
  let uCols: number;
  let vtRows: number;

  if (!computeUV) {
    jobu = JOBU_N;
    jobvt = JOBVT_N;
    uCols = 0;
    vtRows = 0;
  } else if (econ) {
    jobu = JOBU_S;
    jobvt = JOBVT_S;
    uCols = k;
    vtRows = k;
  } else {
    jobu = JOBU_A;
    jobvt = JOBVT_A;
    uCols = m;
    vtRows = n;
  }

  const ldu = computeUV ? Math.max(1, m) : 1;
  const ldvt = computeUV ? Math.max(1, vtRows) : 1;
  const u = new Float64Array(ldu * uCols);
  const vt = new Float64Array(ldvt * n);

  // Workspace query
  const workQuery = new Float64Array(1);
  _dgesvd(
    jobu,
    jobvt,
    m,
    n,
    a,
    0,
    m,
    s,
    0,
    u,
    0,
    ldu,
    vt,
    0,
    ldvt,
    workQuery,
    0,
    -1
  );
  const lwork = Math.max(1, Math.floor(workQuery[0]));
  const work = new Float64Array(lwork);

  // Reset a since workspace query may have modified it
  a.set(data);

  const info = _dgesvd(
    jobu,
    jobvt,
    m,
    n,
    a,
    0,
    m,
    s,
    0,
    u,
    0,
    ldu,
    vt,
    0,
    ldvt,
    work,
    0,
    lwork
  );
  if (info > 0) throw new Error(`svd: DBDSQR did not converge (info=${info})`);
  if (info < 0) throw new Error(`svd: illegal argument (info=${info})`);

  if (!computeUV) {
    return { S: s };
  }

  // Transpose VT to get V (bridge interface returns V, not VT)
  const vCols = econ ? k : n;
  const v = new Float64Array(n * vCols);
  for (let j = 0; j < vCols; j++) {
    for (let i = 0; i < n; i++) {
      v[i + j * n] = vt[j + i * ldvt]; // V(i,j) = VT(j,i)
    }
  }

  return { U: u, S: s, V: v };
}

const _bridge: LapackBridge = {
  inv,
  matmul,
  qr,
  linsolve,
  linsolveComplex,
  eig,
  lu,
  svd,
};

export function getTsLapackBridge(): LapackBridge {
  return _bridge;
}
