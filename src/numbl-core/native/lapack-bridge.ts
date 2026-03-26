/**
 * LAPACK bridge — a module-level singleton that holds a reference to the
 * active acceleration bridge.
 *
 * In Node.js this is usually the native addon. In the browser it can also be a
 * Wasm-backed bridge that implements a subset of the same interface.
 *
 * Usage (CLI startup):
 *   import { setLapackBridge } from './numbl-core/native/lapack-bridge.js';
 *   setLapackBridge(require('<path-to-addon>'));
 *
 * Usage (builtin):
 *   import { getLapackBridge } from '../native/lapack-bridge.js';
 *   const bridge = getLapackBridge();
 *   if (bridge) { ... use bridge.inv(...) ... }
 */

/**
 * Expected native addon version. Bump this whenever the C++ addon API changes
 * (must match ADDON_VERSION in numbl_addon.cpp).
 */
export const NATIVE_ADDON_EXPECTED_VERSION = 1;

export interface LapackBridge {
  /** Human-readable label for logging/debugging. */
  bridgeName?: string;

  /** Returns the native addon's version number. */
  addonVersion?(): number;

  /**
   * Invert an n×n real matrix stored in column-major order (MATLAB/LAPACK convention).
   * @param data  Column-major Float64Array of length n*n (not modified).
   * @param n     Matrix dimension.
   * @returns     Inverted matrix as a new Float64Array in column-major order.
   * @throws      Error if the matrix is singular.
   */
  inv(data: Float64Array, n: number): Float64Array;

  /**
   * Invert an n×n complex matrix stored in column-major order (MATLAB/LAPACK convention).
   * Complex matrices are stored with separate real and imaginary parts.
   * @param dataRe  Real parts in column-major Float64Array of length n*n (not modified).
   * @param dataIm  Imaginary parts in column-major Float64Array of length n*n (not modified).
   * @param n       Matrix dimension.
   * @returns       Object with {re, im} - inverted matrix parts as Float64Arrays in column-major order.
   * @throws        Error if the matrix is singular.
   */
  invComplex?(
    dataRe: Float64Array,
    dataIm: Float64Array,
    n: number
  ): { re: Float64Array; im: Float64Array };

  /**
   * QR decomposition of an m×n real matrix stored in column-major order.
   * Uses LAPACK dgeqrf (QR factorisation) + dorgqr (generate Q).
   * @param data   Column-major Float64Array of length m*n (not modified).
   * @param m      Number of rows.
   * @param n      Number of columns.
   * @param econ   true → economy/thin QR (Q: m×k, R: k×n); false → full QR (Q: m×m, R: m×n). k = min(m,n).
   * @param wantQ  false → skip dorgqr; Q will be absent from the returned object.
   * @returns      Object with R (always present) and Q (present only when wantQ=true).
   */
  qr?(
    data: Float64Array,
    m: number,
    n: number,
    econ: boolean,
    wantQ: boolean
  ): { Q: Float64Array; R: Float64Array };

  /**
   * Column-pivoted QR decomposition: A*P = Q*R via dgeqp3 + dorgqr.
   * @param data   Column-major Float64Array of length m*n (not modified).
   * @param m      Number of rows.
   * @param n      Number of columns.
   * @param econ   true → economy/thin QR; false → full QR.
   * @returns      Object with Q, R, and jpvt (1-based permutation vector as Int32Array).
   */
  qrPivot?(
    data: Float64Array,
    m: number,
    n: number,
    econ: boolean
  ): { Q: Float64Array; R: Float64Array; jpvt: Int32Array };

  /**
   * Column-pivoted QR decomposition of an m×n complex matrix: A*P = Q*R via zgeqp3 + zungqr.
   */
  qrPivotComplex?(
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
  };

  /**
   * QR decomposition of an m×n complex matrix stored in column-major order.
   * Uses LAPACK zgeqrf (QR factorisation) + zungqr (generate Q).
   * @param dataRe  Real parts — column-major Float64Array of length m*n (not modified).
   * @param dataIm  Imaginary parts — column-major Float64Array of length m*n (not modified).
   * @param m       Number of rows.
   * @param n       Number of columns.
   * @param econ    true → economy/thin QR; false → full QR.
   * @param wantQ   false → skip zungqr; Q properties will be absent from the returned object.
   * @returns       Object with RRe, RIm (always present) and QRe, QIm (present only when wantQ=true).
   */
  qrComplex?(
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
  };

  /**
   * SVD (Singular Value Decomposition) of an m×n real matrix stored in column-major order.
   * Uses LAPACK dgesdd (divide-and-conquer SVD).
   * @param data      Column-major Float64Array of length m*n (not modified).
   * @param m         Number of rows.
   * @param n         Number of columns.
   * @param econ      true → economy SVD (U: m×k, S: k, V: n×k); false → full SVD (U: m×m, S: min(m,n), V: n×n). k = min(m,n).
   * @param computeUV true → compute U and V; false → compute S only.
   * @returns         Object with S (always present as vector of singular values) and optionally U and V.
   */
  svd?(
    data: Float64Array,
    m: number,
    n: number,
    econ: boolean,
    computeUV: boolean
  ): { U?: Float64Array; S: Float64Array; V?: Float64Array };

  /**
   * SVD (Singular Value Decomposition) of an m×n complex matrix stored in column-major order.
   * Uses LAPACK zgesdd (complex divide-and-conquer SVD).
   * @param dataRe    Real parts — column-major Float64Array of length m*n (not modified).
   * @param dataIm    Imaginary parts — column-major Float64Array of length m*n (not modified).
   * @param m         Number of rows.
   * @param n         Number of columns.
   * @param econ      true → economy SVD; false → full SVD.
   * @param computeUV true → compute U and V; false → compute S only.
   * @returns         Object with S (real singular values) and optionally URe, UIm, VRe, VIm.
   */
  svdComplex?(
    dataRe: Float64Array,
    dataIm: Float64Array,
    m: number,
    n: number,
    econ: boolean,
    computeUV: boolean
  ): {
    S: Float64Array;
    URe?: Float64Array;
    UIm?: Float64Array;
    VRe?: Float64Array;
    VIm?: Float64Array;
  };

  /**
   * Real matrix-matrix multiplication using BLAS dgemm.
   * Computes C = A * B in column-major order.
   * @param A  Column-major Float64Array of length m*k  (A is m×k, not modified).
   * @param m  Number of rows in A and C.
   * @param k  Number of columns in A and rows in B.
   * @param B  Column-major Float64Array of length k*n  (B is k×n, not modified).
   * @param n  Number of columns in B and C.
   * @returns  C = A*B as a new Float64Array of length m*n in column-major order.
   */
  matmul?(
    A: Float64Array,
    m: number,
    k: number,
    B: Float64Array,
    n: number
  ): Float64Array;

  /**
   * Complex matrix-matrix multiplication using BLAS zgemm.
   * Computes C = A * B in column-major order.
   * @param ARe  Real parts of A, column-major Float64Array of length m*k.
   * @param AIm  Imaginary parts of A, column-major Float64Array of length m*k.
   * @param m    Number of rows in A and C.
   * @param k    Number of columns in A and rows in B.
   * @param BRe  Real parts of B, column-major Float64Array of length k*n.
   * @param BIm  Imaginary parts of B, column-major Float64Array of length k*n.
   * @param n    Number of columns in B and C.
   * @returns    Object with {re, im?} — complex result as Float64Arrays.
   */
  matmulComplex?(
    ARe: Float64Array,
    AIm: Float64Array,
    m: number,
    k: number,
    BRe: Float64Array,
    BIm: Float64Array,
    n: number
  ): { re: Float64Array; im?: Float64Array };

  /**
   * Solve a linear system A * X = B.
   * If A is square (m === n), uses LU factorization (dgesv / dgetrf+solve).
   * If A is non-square, uses QR / LQ factorization (dgels / dgeqrf+solve):
   *   overdetermined (m > n): least-squares solution minimising ||A*X - B||₂
   *   underdetermined (m < n): minimum-norm solution minimising ||X||₂
   * @param A     Column-major Float64Array of length m*n  (A is m×n, not modified).
   * @param m     Number of rows in A and B.
   * @param n     Number of columns in A; also the number of rows in the result X.
   * @param B     Column-major Float64Array of length m*nrhs  (B is m×nrhs, not modified).
   * @param nrhs  Number of right-hand sides (columns of B and X).
   * @returns     X as a new Float64Array of length n*nrhs in column-major order.
   */
  linsolve?(
    A: Float64Array,
    m: number,
    n: number,
    B: Float64Array,
    nrhs: number
  ): Float64Array;

  /**
   * Solve a complex linear system A * X = B.
   * Input/output are column-major Float64Arrays for real and imaginary parts.
   * If A is square (m === n), uses LU factorization (zgesv).
   * If A is non-square, uses QR / LQ factorization (zgels):
   *   overdetermined (m > n): least-squares solution minimising ||A*X - B||₂
   *   underdetermined (m < n): minimum-norm solution minimising ||X||₂
   * @param ARe   Real parts of A — column-major Float64Array of length m*n (not modified).
   * @param AIm   Imaginary parts of A — column-major Float64Array of length m*n (not modified).
   * @param m     Number of rows in A and B.
   * @param n     Number of columns in A; also the number of rows in the result X.
   * @param BRe   Real parts of B — column-major Float64Array of length m*nrhs (not modified).
   * @param BIm   Imaginary parts of B — column-major Float64Array of length m*nrhs (not modified).
   * @param nrhs  Number of right-hand sides (columns of B and X).
   * @returns     Object with {re, im} — X as Float64Arrays of length n*nrhs in column-major order.
   */
  linsolveComplex?(
    ARe: Float64Array,
    AIm: Float64Array,
    m: number,
    n: number,
    BRe: Float64Array,
    BIm: Float64Array,
    nrhs: number
  ): { re: Float64Array; im: Float64Array };

  /**
   * Eigenvalue decomposition of an n×n real matrix stored in column-major order.
   * Uses LAPACK dgeev.
   * @param data       Column-major Float64Array of length n*n (not modified).
   * @param n          Matrix dimension.
   * @param computeVL  true → compute left eigenvectors.
   * @param computeVR  true → compute right eigenvectors.
   * @param balance    true → balance matrix before computing (default); false → no balancing.
   * @returns          Object with wr/wi (real/imag parts of eigenvalues) and optionally VL/VR.
   */
  eig?(
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
  };

  /**
   * Eigenvalue decomposition of an n×n complex matrix stored in column-major order.
   * Uses LAPACK zgeev.
   * @param dataRe     Real parts — column-major Float64Array of length n*n (not modified).
   * @param dataIm     Imaginary parts — column-major Float64Array of length n*n (not modified).
   * @param n          Matrix dimension.
   * @param computeVL  true → compute left eigenvectors.
   * @param computeVR  true → compute right eigenvectors.
   * @returns          Object with wRe/wIm (real/imag parts of eigenvalues) and optionally VLRe/VLIm/VRRe/VRIm.
   */
  eigComplex?(
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
  };

  /**
   * LU factorization of an m×n real matrix with partial pivoting (dgetrf).
   * @param data   Column-major Float64Array of length m*n (not modified).
   * @param m      Number of rows.
   * @param n      Number of columns.
   * @returns      Object with:
   *   - LU: m×n Float64Array with L (unit lower) and U (upper) packed together.
   *   - ipiv: Int32Array of length min(m,n), 1-based pivot indices from dgetrf.
   */
  lu?(
    data: Float64Array,
    m: number,
    n: number
  ): { LU: Float64Array; ipiv: Int32Array };

  /**
   * LU factorization of an m×n complex matrix with partial pivoting (zgetrf).
   * @param dataRe  Real parts — column-major Float64Array of length m*n (not modified).
   * @param dataIm  Imaginary parts — column-major Float64Array of length m*n (not modified).
   * @param m       Number of rows.
   * @param n       Number of columns.
   * @returns       Object with:
   *   - LURe, LUIm: m×n Float64Arrays with L (unit lower) and U (upper) packed together.
   *   - ipiv: Int32Array of length min(m,n), 1-based pivot indices from zgetrf.
   */
  luComplex?(
    dataRe: Float64Array,
    dataIm: Float64Array,
    m: number,
    n: number
  ): { LURe: Float64Array; LUIm: Float64Array; ipiv: Int32Array };

  /**
   * Cholesky factorization of an n×n real symmetric positive definite matrix.
   * Uses LAPACK dpotrf.
   * @param data   Column-major Float64Array of length n*n (not modified).
   * @param n      Matrix dimension.
   * @param upper  true → compute upper triangular R (A = R'*R);
   *               false → compute lower triangular L (A = L*L').
   * @returns      Object with:
   *   - R: n×n Float64Array with the triangular factor (opposite triangle zeroed).
   *   - info: 0 if successful, >0 if not positive definite (index of failing pivot).
   */
  chol?(
    data: Float64Array,
    n: number,
    upper: boolean
  ): { R: Float64Array; info: number };

  /**
   * Cholesky factorization of an n×n complex Hermitian positive definite matrix.
   * Uses LAPACK zpotrf.
   * @param dataRe  Real parts — column-major Float64Array of length n*n (not modified).
   * @param dataIm  Imaginary parts — column-major Float64Array of length n*n (not modified).
   * @param n       Matrix dimension.
   * @param upper   true → compute upper triangular R (A = R'*R);
   *                false → compute lower triangular L (A = L*L').
   * @returns       Object with:
   *   - RRe, RIm: n×n Float64Arrays with the triangular factor.
   *   - info: 0 if successful, >0 if not positive definite.
   */
  cholComplex?(
    dataRe: Float64Array,
    dataIm: Float64Array,
    n: number,
    upper: boolean
  ): { RRe: Float64Array; RIm: Float64Array; info: number };

  /**
   * QZ factorization (generalized Schur decomposition) of an n×n real matrix pair.
   * Uses LAPACK dgges (and dtgevc for eigenvectors).
   * @param dataA           Column-major Float64Array of length n*n (not modified).
   * @param dataB           Column-major Float64Array of length n*n (not modified).
   * @param n               Matrix dimension.
   * @param computeEigvecs  true → also compute generalized eigenvectors via dtgevc.
   * @returns  Object with:
   *   - AA: n×n Float64Array, upper quasi-triangular Schur form of A.
   *   - BB: n×n Float64Array, upper triangular Schur form of B.
   *   - Q: n×n Float64Array, left orthogonal factor (Q*A*Z = AA).
   *   - Z: n×n Float64Array, right orthogonal factor.
   *   - alphar, alphai, beta: Float64Array(n), generalized eigenvalue components.
   *   - V?: n×n Float64Array, right generalized eigenvectors (packed, like dgeev format).
   *   - W?: n×n Float64Array, left generalized eigenvectors (packed, like dgeev format).
   */
  qz?(
    dataA: Float64Array,
    dataB: Float64Array,
    n: number,
    computeEigvecs: boolean
  ): {
    AA: Float64Array;
    BB: Float64Array;
    Q: Float64Array;
    Z: Float64Array;
    alphar: Float64Array;
    alphai: Float64Array;
    beta: Float64Array;
    V?: Float64Array;
    W?: Float64Array;
  };

  /**
   * Complex QZ factorization (generalized Schur decomposition) of an n×n complex matrix pair.
   * Uses LAPACK zgges (and ztgevc for eigenvectors).
   */
  /**
   * Forward/inverse 1D FFT on real input via FFTW.
   * Does NOT normalize for inverse — caller handles 1/n scaling.
   * @param re      Real input data as Float64Array of length n.
   * @param n       Transform length.
   * @param inverse true for inverse FFT, false for forward.
   * @returns       { re, im } Float64Arrays of length n.
   */
  fft1d?(
    re: Float64Array,
    n: number,
    inverse: boolean
  ): { re: Float64Array; im: Float64Array };

  /**
   * Forward/inverse 1D FFT on complex input via FFTW.
   * Does NOT normalize for inverse — caller handles 1/n scaling.
   * @param re      Real part of input as Float64Array of length n.
   * @param im      Imaginary part of input as Float64Array of length n.
   * @param n       Transform length.
   * @param inverse true for inverse FFT, false for forward.
   * @returns       { re, im } Float64Arrays of length n.
   */
  fft1dComplex?(
    re: Float64Array,
    im: Float64Array,
    n: number,
    inverse: boolean
  ): { re: Float64Array; im: Float64Array };

  /**
   * Batch FFT along a single dimension of a column-major tensor via FFTW.
   * Transforms ALL fibers in one FFTW call using the guru split interface.
   * Does NOT normalize for inverse — caller handles 1/n scaling.
   * @param re      Real part of input tensor (column-major flat array).
   * @param im      Imaginary part, or null for real input.
   * @param shape   Tensor dimensions as a JS array of numbers.
   * @param dim     0-based dimension to transform along.
   * @param n       FFT length (may differ from shape[dim] for pad/truncate).
   * @param inverse true for inverse FFT, false for forward.
   * @returns       { re, im } Float64Arrays for the output tensor.
   */
  fftAlongDim?(
    re: Float64Array,
    im: Float64Array | null,
    shape: number[],
    dim: number,
    n: number,
    inverse: boolean
  ): { re: Float64Array; im: Float64Array };

  qzComplex?(
    dataARe: Float64Array,
    dataAIm: Float64Array,
    dataBRe: Float64Array,
    dataBIm: Float64Array,
    n: number,
    computeEigvecs: boolean
  ): {
    AARe: Float64Array;
    AAIm: Float64Array;
    BBRe: Float64Array;
    BBIm: Float64Array;
    QRe: Float64Array;
    QIm: Float64Array;
    ZRe: Float64Array;
    ZIm: Float64Array;
    alphaRe: Float64Array;
    alphaIm: Float64Array;
    betaRe: Float64Array;
    betaIm: Float64Array;
    VRe?: Float64Array;
    VIm?: Float64Array;
    WRe?: Float64Array;
    WIm?: Float64Array;
  };

  /** Element-wise binary op on real Float64Arrays. op: 0=add, 1=sub, 2=mul, 3=div */
  elemwise?(a: Float64Array, b: Float64Array, op: number): Float64Array;

  /** Element-wise binary op on complex Float64Arrays. op: 0=add, 1=sub, 2=mul, 3=div */
  elemwiseComplex?(
    aRe: Float64Array,
    aIm: Float64Array | null,
    bRe: Float64Array,
    bIm: Float64Array | null,
    op: number
  ): { re: Float64Array; im?: Float64Array };
}

let _bridge: LapackBridge | null = null;

export function setLapackBridge(bridge: LapackBridge | null): void {
  _bridge = bridge;
}

export function getLapackBridge(): LapackBridge | null {
  return _bridge;
}
