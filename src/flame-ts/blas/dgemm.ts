// BLIS-style blocked dgemm: C := alpha * op(A) * op(B) + beta * C
// Implements the Goto/BLIS 5-loop structure with packing and micro-kernel.

import { FLAME_CONFIG } from "../config.js";

const { MC, KC, NC, MR, NR } = FLAME_CONFIG;

// Pre-allocated packing buffers (reused across calls)
let packA = new Float64Array(MC * KC);
let packB = new Float64Array(KC * NC);

function ensurePackBuffers(mc: number, kc: number, nc: number): void {
  if (packA.length < mc * kc) packA = new Float64Array(mc * kc);
  if (packB.length < kc * nc) packB = new Float64Array(kc * nc);
}

// Pack a mc×kc block of A into contiguous mr-wide column panels
function packBlockA(
  a: Float64Array,
  aOff: number,
  lda: number,
  mc: number,
  kc: number,
  buf: Float64Array
): void {
  let p = 0;
  for (let i = 0; i < mc; i += MR) {
    const mr = Math.min(MR, mc - i);
    for (let l = 0; l < kc; l++) {
      const col = aOff + l * lda + i;
      for (let ii = 0; ii < mr; ii++) buf[p++] = a[col + ii];
      // Pad to MR if needed
      for (let ii = mr; ii < MR; ii++) buf[p++] = 0;
    }
  }
}

// Pack a kc×nc block of B into contiguous nr-wide row panels
function packBlockB(
  b: Float64Array,
  bOff: number,
  ldb: number,
  kc: number,
  nc: number,
  buf: Float64Array
): void {
  let p = 0;
  for (let j = 0; j < nc; j += NR) {
    const nr = Math.min(NR, nc - j);
    for (let l = 0; l < kc; l++) {
      const row = bOff + l + j * ldb;
      for (let jj = 0; jj < nr; jj++) buf[p++] = b[row + jj * ldb];
      for (let jj = nr; jj < NR; jj++) buf[p++] = 0;
    }
  }
}

// 4×4 micro-kernel: accumulates kc rank-1 updates into a 4×4 tile of C
// Ã is packed row-panel (MR elements per k-step), B̃ is packed col-panel (NR per k-step)
function microKernel4x4(
  kc: number,
  pA: Float64Array,
  aPos: number,
  pB: Float64Array,
  bPos: number,
  c: Float64Array,
  cOff: number,
  ldc: number,
  mr: number,
  nr: number,
  alpha: number,
  beta: number
): void {
  // Accumulator registers
  let c00 = 0,
    c10 = 0,
    c20 = 0,
    c30 = 0;
  let c01 = 0,
    c11 = 0,
    c21 = 0,
    c31 = 0;
  let c02 = 0,
    c12 = 0,
    c22 = 0,
    c32 = 0;
  let c03 = 0,
    c13 = 0,
    c23 = 0,
    c33 = 0;

  // Rank-1 updates: C_tile += A_col * B_row
  let ap = aPos;
  let bp = bPos;
  for (let p = 0; p < kc; p++) {
    const a0 = pA[ap],
      a1 = pA[ap + 1],
      a2 = pA[ap + 2],
      a3 = pA[ap + 3];
    const b0 = pB[bp],
      b1 = pB[bp + 1],
      b2 = pB[bp + 2],
      b3 = pB[bp + 3];
    c00 += a0 * b0;
    c10 += a1 * b0;
    c20 += a2 * b0;
    c30 += a3 * b0;
    c01 += a0 * b1;
    c11 += a1 * b1;
    c21 += a2 * b1;
    c31 += a3 * b1;
    c02 += a0 * b2;
    c12 += a1 * b2;
    c22 += a2 * b2;
    c32 += a3 * b2;
    c03 += a0 * b3;
    c13 += a1 * b3;
    c23 += a2 * b3;
    c33 += a3 * b3;
    ap += MR;
    bp += NR;
  }

  // Store back to C with alpha/beta scaling, respecting actual dimensions
  const c0 = cOff,
    c1 = cOff + ldc,
    c2 = cOff + 2 * ldc,
    c3 = cOff + 3 * ldc;
  if (beta === 0) {
    if (mr >= 1 && nr >= 1) c[c0] = alpha * c00;
    if (mr >= 2 && nr >= 1) c[c0 + 1] = alpha * c10;
    if (mr >= 3 && nr >= 1) c[c0 + 2] = alpha * c20;
    if (mr >= 4 && nr >= 1) c[c0 + 3] = alpha * c30;
    if (mr >= 1 && nr >= 2) c[c1] = alpha * c01;
    if (mr >= 2 && nr >= 2) c[c1 + 1] = alpha * c11;
    if (mr >= 3 && nr >= 2) c[c1 + 2] = alpha * c21;
    if (mr >= 4 && nr >= 2) c[c1 + 3] = alpha * c31;
    if (mr >= 1 && nr >= 3) c[c2] = alpha * c02;
    if (mr >= 2 && nr >= 3) c[c2 + 1] = alpha * c12;
    if (mr >= 3 && nr >= 3) c[c2 + 2] = alpha * c22;
    if (mr >= 4 && nr >= 3) c[c2 + 3] = alpha * c32;
    if (mr >= 1 && nr >= 4) c[c3] = alpha * c03;
    if (mr >= 2 && nr >= 4) c[c3 + 1] = alpha * c13;
    if (mr >= 3 && nr >= 4) c[c3 + 2] = alpha * c23;
    if (mr >= 4 && nr >= 4) c[c3 + 3] = alpha * c33;
  } else {
    if (mr >= 1 && nr >= 1) c[c0] = alpha * c00 + beta * c[c0];
    if (mr >= 2 && nr >= 1) c[c0 + 1] = alpha * c10 + beta * c[c0 + 1];
    if (mr >= 3 && nr >= 1) c[c0 + 2] = alpha * c20 + beta * c[c0 + 2];
    if (mr >= 4 && nr >= 1) c[c0 + 3] = alpha * c30 + beta * c[c0 + 3];
    if (mr >= 1 && nr >= 2) c[c1] = alpha * c01 + beta * c[c1];
    if (mr >= 2 && nr >= 2) c[c1 + 1] = alpha * c11 + beta * c[c1 + 1];
    if (mr >= 3 && nr >= 2) c[c1 + 2] = alpha * c21 + beta * c[c1 + 2];
    if (mr >= 4 && nr >= 2) c[c1 + 3] = alpha * c31 + beta * c[c1 + 3];
    if (mr >= 1 && nr >= 3) c[c2] = alpha * c02 + beta * c[c2];
    if (mr >= 2 && nr >= 3) c[c2 + 1] = alpha * c12 + beta * c[c2 + 1];
    if (mr >= 3 && nr >= 3) c[c2 + 2] = alpha * c22 + beta * c[c2 + 2];
    if (mr >= 4 && nr >= 3) c[c2 + 3] = alpha * c32 + beta * c[c2 + 3];
    if (mr >= 1 && nr >= 4) c[c3] = alpha * c03 + beta * c[c3];
    if (mr >= 2 && nr >= 4) c[c3 + 1] = alpha * c13 + beta * c[c3 + 1];
    if (mr >= 3 && nr >= 4) c[c3 + 2] = alpha * c23 + beta * c[c3 + 2];
    if (mr >= 4 && nr >= 4) c[c3 + 3] = alpha * c33 + beta * c[c3 + 3];
  }
}

// Public API: C = alpha * A * B + beta * C
// A is m×k, B is k×n, C is m×n — all column-major
export function dgemm(
  m: number,
  n: number,
  k: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number,
  beta: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  if (m === 0 || n === 0) return;
  if (alpha === 0 || k === 0) {
    // C := beta * C
    for (let j = 0; j < n; j++) {
      const base = cOff + j * ldc;
      if (beta === 0) c.fill(0, base, base + m);
      else if (beta !== 1) for (let i = 0; i < m; i++) c[base + i] *= beta;
    }
    return;
  }

  ensurePackBuffers(MC, KC, NC);

  // 5-loop BLIS structure
  for (let jc = 0; jc < n; jc += NC) {
    const nc = Math.min(NC, n - jc);
    for (let pc = 0; pc < k; pc += KC) {
      const kc = Math.min(KC, k - pc);

      // Pack B panel: B[pc:pc+kc, jc:jc+nc]
      packBlockB(b, bOff + pc + jc * ldb, ldb, kc, nc, packB);

      for (let ic = 0; ic < m; ic += MC) {
        const mc = Math.min(MC, m - ic);

        // Pack A block: A[ic:ic+mc, pc:pc+kc]
        packBlockA(a, aOff + ic + pc * lda, lda, mc, kc, packA);

        // Beta scaling: only on first k-block (pc === 0)
        const useBeta = pc === 0 ? beta : 1.0;

        // Macro-kernel: iterate over micro-tiles
        for (let jr = 0; jr < nc; jr += NR) {
          const nr = Math.min(NR, nc - jr);
          for (let ir = 0; ir < mc; ir += MR) {
            const mr = Math.min(MR, mc - ir);
            microKernel4x4(
              kc,
              packA,
              ((ir / MR) | 0) * MR * kc,
              packB,
              ((jr / NR) | 0) * NR * kc,
              c,
              cOff + (ic + ir) + (jc + jr) * ldc,
              ldc,
              mr,
              nr,
              alpha,
              useBeta
            );
          }
        }
      }
    }
  }
}

// Convenience wrapper matching LapackBridge.matmul signature
export function matmul(
  A: Float64Array,
  m: number,
  k: number,
  B: Float64Array,
  n: number
): Float64Array {
  const C = new Float64Array(m * n);
  dgemm(m, n, k, 1.0, A, 0, m, B, 0, k, 0.0, C, 0, m);
  return C;
}
