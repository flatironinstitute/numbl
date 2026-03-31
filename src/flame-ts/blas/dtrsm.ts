// Triangular solve: op(A) * X = alpha * B  or  X * op(A) = alpha * B
// A is upper or lower triangular, X overwrites B.
// Column-major layout throughout.

// Solve L * X = B  (lower triangular, no transpose, left side)
// B is m×n, L is m×m lower triangular. Overwrites B with X.
export function dtrsmLLNN(
  m: number,
  n: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number
): void {
  if (m === 0 || n === 0) return;
  if (alpha !== 1) {
    for (let j = 0; j < n; j++) {
      const base = bOff + j * ldb;
      for (let i = 0; i < m; i++) b[base + i] *= alpha;
    }
  }
  for (let j = 0; j < n; j++) {
    const bCol = bOff + j * ldb;
    for (let k = 0; k < m; k++) {
      if (b[bCol + k] !== 0) {
        b[bCol + k] /= a[aOff + k + k * lda];
        const bk = b[bCol + k];
        for (let i = k + 1; i < m; i++) {
          b[bCol + i] -= bk * a[aOff + i + k * lda];
        }
      }
    }
  }
}

// Solve L * X = B where L has unit diagonal (implicit 1s on diagonal)
// Used after dgetrf where L is stored with unit diagonal.
export function dtrsmLLNUnit(
  m: number,
  n: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number
): void {
  if (m === 0 || n === 0) return;
  if (alpha !== 1) {
    for (let j = 0; j < n; j++) {
      const base = bOff + j * ldb;
      for (let i = 0; i < m; i++) b[base + i] *= alpha;
    }
  }
  for (let j = 0; j < n; j++) {
    const bCol = bOff + j * ldb;
    for (let k = 0; k < m; k++) {
      if (b[bCol + k] !== 0) {
        const bk = b[bCol + k];
        for (let i = k + 1; i < m; i++) {
          b[bCol + i] -= bk * a[aOff + i + k * lda];
        }
      }
    }
  }
}

// Solve U * X = B  (upper triangular, no transpose, left side)
export function dtrsmLUNN(
  m: number,
  n: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number
): void {
  if (m === 0 || n === 0) return;
  if (alpha !== 1) {
    for (let j = 0; j < n; j++) {
      const base = bOff + j * ldb;
      for (let i = 0; i < m; i++) b[base + i] *= alpha;
    }
  }
  for (let j = 0; j < n; j++) {
    const bCol = bOff + j * ldb;
    for (let k = m - 1; k >= 0; k--) {
      if (b[bCol + k] !== 0) {
        b[bCol + k] /= a[aOff + k + k * lda];
        const bk = b[bCol + k];
        for (let i = 0; i < k; i++) {
          b[bCol + i] -= bk * a[aOff + i + k * lda];
        }
      }
    }
  }
}

// Solve X * L' = B  (lower triangular, transpose, right side)
// Used by blocked Cholesky. B is m×n, L is n×n.
export function dtrsmRLTN(
  m: number,
  n: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number
): void {
  if (m === 0 || n === 0) return;
  if (alpha !== 1) {
    for (let j = 0; j < n; j++) {
      const base = bOff + j * ldb;
      for (let i = 0; i < m; i++) b[base + i] *= alpha;
    }
  }
  // X * L' = B => columns of X: solve L' * x_col = b_col from left
  // But right-side: iterate over columns of result
  for (let j = 0; j < n; j++) {
    const bCol = bOff + j * ldb;
    for (let k = 0; k < j; k++) {
      const lkj = a[aOff + j + k * lda]; // L'(k,j) = L(j,k)
      const bkCol = bOff + k * ldb;
      for (let i = 0; i < m; i++) b[bCol + i] -= lkj * b[bkCol + i];
    }
    const ljj = a[aOff + j + j * lda];
    for (let i = 0; i < m; i++) b[bCol + i] /= ljj;
  }
}
