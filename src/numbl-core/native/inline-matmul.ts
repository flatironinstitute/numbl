/**
 * The reference matrix-product loop, used for small products at every
 * `--opt` level: the interpreter reaches it through the stack's `inline`
 * provider (lapack-bridge.ts), and the JIT's matmul helpers (JS and C) run
 * the same loop inline rather than calling the host.
 */

/**
 * Products with at most this many multiply-adds (`m*k*n`) run on the
 * reference loop. At these sizes the loop beats a BLAS call, whose fixed
 * cost is about 1 us, and keeping one algorithm per size range keeps the
 * levels bit-identical (BLAS kernels use FMA and blocking, so they differ
 * from the loop in the last bits).
 */
export const MATMUL_INLINE_MAX_WORK = 1024;

/** True when `A (m×k) * B (k×n)` runs on the reference loop. */
export function matmulIsInline(m: number, k: number, n: number): boolean {
  return m * k * n <= MATMUL_INLINE_MAX_WORK;
}

/** C = A*B, column-major; each element sums its products in order of `p`. */
export function matmulLoop(
  a: Float64Array,
  m: number,
  k: number,
  b: Float64Array,
  n: number
): Float64Array {
  const r = new Float64Array(m * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += a[i + p * m] * b[p + j * k];
      r[i + j * m] = s;
    }
  }
  return r;
}

/** Complex reference loop; `im` is omitted when the result is exactly real
 *  (as in the TypeScript `matmulComplex`). */
export function matmulComplexLoop(
  aRe: Float64Array,
  aIm: Float64Array,
  m: number,
  k: number,
  bRe: Float64Array,
  bIm: Float64Array,
  n: number
): { re: Float64Array; im?: Float64Array } {
  const re = new Float64Array(m * n);
  const im = new Float64Array(m * n);
  let anyIm = false;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      let sr = 0;
      let si = 0;
      for (let p = 0; p < k; p++) {
        const ar = aRe[i + p * m];
        const ai = aIm[i + p * m];
        const br = bRe[p + j * k];
        const bi = bIm[p + j * k];
        sr += ar * br - ai * bi;
        si += ar * bi + ai * br;
      }
      re[i + j * m] = sr;
      im[i + j * m] = si;
      if (si !== 0) anyIm = true;
    }
  }
  return anyIm ? { re, im } : { re };
}
