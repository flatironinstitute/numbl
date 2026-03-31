// Blocked LU factorization with partial pivoting (right-looking).
// A = P * L * U, returns ipiv (1-based pivot indices).

import { FLAME_CONFIG } from "../config.js";
import { dgemm } from "../blas/dgemm.js";
import { dtrsmLLNUnit } from "../blas/dtrsm.js";

const { NB } = FLAME_CONFIG;

// Swap rows i and j in matrix a, columns c0..c0+nc-1
function dswapRows(
  a: Float64Array,
  aOff: number,
  lda: number,
  i: number,
  j: number,
  c0: number,
  nc: number
): void {
  if (i === j) return;
  for (let c = c0; c < c0 + nc; c++) {
    const ci = aOff + i + c * lda;
    const cj = aOff + j + c * lda;
    const tmp = a[ci];
    a[ci] = a[cj];
    a[cj] = tmp;
  }
}

// Unblocked LU with partial pivoting for a panel m×n (m >= n)
function dgetf2(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  ipiv: Int32Array,
  ipivOff: number
): number {
  const mn = Math.min(m, n);
  for (let j = 0; j < mn; j++) {
    // Find pivot
    let maxVal = Math.abs(a[aOff + j + j * lda]);
    let maxIdx = j;
    for (let i = j + 1; i < m; i++) {
      const v = Math.abs(a[aOff + i + j * lda]);
      if (v > maxVal) {
        maxVal = v;
        maxIdx = i;
      }
    }
    ipiv[ipivOff + j] = maxIdx + 1; // 1-based

    if (a[aOff + maxIdx + j * lda] === 0) continue;

    // Swap rows
    if (maxIdx !== j) dswapRows(a, aOff, lda, j, maxIdx, 0, n);

    // Scale below-diagonal
    const diag = a[aOff + j + j * lda];
    for (let i = j + 1; i < m; i++) {
      a[aOff + i + j * lda] /= diag;
    }

    // Rank-1 update on trailing submatrix
    for (let c = j + 1; c < n; c++) {
      const ac = aOff + c * lda;
      for (let i = j + 1; i < m; i++) {
        a[ac + i] -= a[aOff + i + j * lda] * a[aOff + j + c * lda];
      }
    }
  }
  return 0;
}

export function dgetrf(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  ipiv: Int32Array
): number {
  const mn = Math.min(m, n);

  // Small matrix: use unblocked
  if (mn <= NB) return dgetf2(m, n, a, aOff, lda, ipiv, 0);

  for (let j = 0; j < mn; j += NB) {
    const jb = Math.min(NB, mn - j);

    // 1. Factor panel A[j:m, j:j+jb] (unblocked)
    dgetf2(m - j, jb, a, aOff + j + j * lda, lda, ipiv, j);

    // Adjust ipiv to global indices
    for (let i = j; i < j + jb; i++) ipiv[i] += j;

    // 2. Apply pivots to columns 0..j-1 (left of panel)
    for (let i = j; i < j + jb; i++) {
      const pi = ipiv[i] - 1;
      if (pi !== i) dswapRows(a, aOff, lda, i, pi, 0, j);
    }

    // 3. Apply pivots to columns j+jb..n-1 (right of panel)
    if (j + jb < n) {
      for (let i = j; i < j + jb; i++) {
        const pi = ipiv[i] - 1;
        if (pi !== i) dswapRows(a, aOff, lda, i, pi, j + jb, n - j - jb);
      }

      // 4. Solve L[j:j+jb, j:j+jb] * U = A[j:j+jb, j+jb:n] (unit diagonal L)
      dtrsmLLNUnit(
        jb,
        n - j - jb,
        1.0,
        a,
        aOff + j + j * lda,
        lda,
        a,
        aOff + j + (j + jb) * lda,
        lda
      );

      // 5. Update trailing: A[j+jb:m, j+jb:n] -= A[j+jb:m, j:j+jb] * U[j:j+jb, j+jb:n]
      if (j + jb < m) {
        dgemm(
          m - j - jb,
          n - j - jb,
          jb,
          -1.0,
          a,
          aOff + (j + jb) + j * lda,
          lda,
          a,
          aOff + j + (j + jb) * lda,
          lda,
          1.0,
          a,
          aOff + (j + jb) + (j + jb) * lda,
          lda
        );
      }
    }
  }
  return 0;
}
