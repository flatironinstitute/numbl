// Blocked Cholesky factorization: A = L * L' (lower triangular)
// Uses dgemm for the bulk update (dsyrk pattern) and dtrsm for panel solve.

import { FLAME_CONFIG } from "../config.js";
import { dgemm } from "../blas/dgemm.js";
import { dtrsmRLTN } from "../blas/dtrsm.js";

const { NB } = FLAME_CONFIG;

// dsyrk-like: C -= A * A' where A is m×k, C is m×m (lower triangle updated)
// We explicitly transpose A into a temp buffer and call dgemm.
let _syrktmp = new Float64Array(0);
function dsyrkLower(
  m: number,
  k: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  // Transpose A (m×k col-major) → AT (k×m col-major)
  const needed = k * m;
  if (_syrktmp.length < needed) _syrktmp = new Float64Array(needed);
  for (let j = 0; j < k; j++)
    for (let i = 0; i < m; i++) _syrktmp[j + i * k] = a[aOff + i + j * lda]; // AT(j,i) = A(i,j)
  dgemm(m, m, k, -1.0, a, aOff, lda, _syrktmp, 0, k, 1.0, c, cOff, ldc);
}

// gemm for panel update: C -= A * B' where A is m×k, B is n×k
function dgemmNT(
  m: number,
  n: number,
  k: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const needed = k * n;
  if (_syrktmp.length < needed) _syrktmp = new Float64Array(needed);
  for (let j = 0; j < k; j++)
    for (let i = 0; i < n; i++) _syrktmp[j + i * k] = b[bOff + i + j * ldb];
  dgemm(m, n, k, -1.0, a, aOff, lda, _syrktmp, 0, k, 1.0, c, cOff, ldc);
}

// Unblocked Cholesky for small diagonal blocks
function dpotf2Lower(
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  for (let j = 0; j < n; j++) {
    let s = a[aOff + j + j * lda];
    for (let k = 0; k < j; k++) {
      s -= a[aOff + j + k * lda] * a[aOff + j + k * lda];
    }
    if (s <= 0 || Number.isNaN(s)) return j + 1;
    const ljj = Math.sqrt(s);
    a[aOff + j + j * lda] = ljj;
    for (let i = j + 1; i < n; i++) {
      let v = a[aOff + i + j * lda];
      for (let k = 0; k < j; k++) {
        v -= a[aOff + i + k * lda] * a[aOff + j + k * lda];
      }
      a[aOff + i + j * lda] = v / ljj;
    }
  }
  return 0;
}

export function dpotrf(
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  for (let j = 0; j < n; j += NB) {
    const jb = Math.min(NB, n - j);
    const jOff = aOff + j + j * lda;

    // 1. Update diagonal block: A[j:j+jb, j:j+jb] -= A[j:j+jb, 0:j] * A[j:j+jb, 0:j]'
    if (j > 0) {
      dsyrkLower(jb, j, a, aOff + j, lda, a, jOff, lda);
    }

    // 2. Factor diagonal block (unblocked)
    const info = dpotf2Lower(jb, a, jOff, lda);
    if (info !== 0) return j + info;

    // 3. Update below-diagonal panel: A[j+jb:n, j:j+jb]
    if (j + jb < n) {
      const rows = n - j - jb;
      if (j > 0) {
        // A[j+jb:n, j:j+jb] -= A[j+jb:n, 0:j] * A[j:j+jb, 0:j]'
        dgemmNT(
          rows,
          jb,
          j,
          a,
          aOff + (j + jb),
          lda,
          a,
          aOff + j,
          lda,
          a,
          aOff + (j + jb) + j * lda,
          lda
        );
      }
      // Solve: X * L[j:j+jb, j:j+jb]' = A[j+jb:n, j:j+jb]
      dtrsmRLTN(rows, jb, 1.0, a, jOff, lda, a, aOff + (j + jb) + j * lda, lda);
    }
  }
  return 0;
}
