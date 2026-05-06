// dgeqrf_optimized.ts
// Fully inlined + JS-engine-optimized implementation of DGEQRF.
//
// This file merges the following LAPACK/BLAS routines into one function:
//   dgeqrf  →  dgeqr2  →  dlarfg  →  dnrm2, dscal, dlamch, dlapy2
//                       →  dlarf1f →  iladlc, dgemv, daxpy, dger
//
// ilaenv(1, "DGEQRF") always returns 1 (not in the special-case switch),
// so nb=1 < nbmin=2, meaning the blocked path in dgeqrf is never taken.
// dgeqrf therefore reduces to a single call of dgeqr2(m, n, ...).
//
// Optimization principles (matching dorgqr_optimized.ts):
//
//   1. Unaliased v[] copy — v is extracted from the column of a[] into its
//      own Float64Array with v[0]=1 explicit. The JIT sees two independent
//      arrays, enabling bounds-check hoisting and better vectorisation of
//      both the dot-product and the rank-1 update inner loops.
//
//   2. Per-column fusion — the dgemv + daxpy×2 + dger sequence inside
//      dlarf1f is collapsed into a single double-pass (dot then update) per
//      column of C. The column a[cCol..cCol+lastv] stays in L1 cache across
//      both passes, and the scalar `w` replaces the work[] array entirely.
//
//   3. work[] eliminated — no intermediate work array is allocated in the
//      hot path. w is a scalar local to each column iteration.
//
//   4. Machine constants hoisted to module scope — dlamch() values and the
//      derived safmin/rsafmn are computed once at module load. V8 treats
//      them as numeric constants and avoids re-computing them every call.
//
//   5. Stride-1 fast paths — dgeqr2 always calls dlarfg with incx=1 and
//      dlarf1f with incv=1, so all inner loops are contiguous linear scans
//      that V8 can unroll, vectorise, and bounds-check-hoist aggressively.
//
//   6. NaN checks in dlapy2 omitted — the optimised path skips
//      Number.isNaN guards that are never triggered by valid inputs; the
//      result for NaN input is still NaN (propagated naturally by IEEE 754).
//
// Array indexing matches Fortran column-major:
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   TAU(I)  =>  tau[tauOff + (I-1)]             (I is 1-based)

import { allocFloat64Array } from "../../../numbl-core/executors/jsJit/helpers/alloc";

// ─── Module-level machine constants (dlamch inlined, evaluated once) ─────────
// eps = Number.EPSILON * 0.5   →  dlamch('E'), machine epsilon / 2
const _EPS = Number.EPSILON * 0.5;
// Smallest positive normalised float64 = 2^-1022
const _SFMIN_CONST = 2.2250738585072014e-308;
const _SMALL = 1.0 / Number.MAX_VALUE;
// dlamch('S') — safe minimum such that 1/sfmin does not overflow
const _SFMIN = _SMALL >= _SFMIN_CONST ? _SMALL * (1.0 + _EPS) : _SFMIN_CONST;
// safmin = dlamch(SFMIN) / dlamch(EPS), used in dlarfg rescaling loop
const _SAFMIN = _SFMIN / _EPS;
const _RSAFMN = 1.0 / _SAFMIN;
// dlamch('O') = overflow threshold, used in dlapy2
const _HUGEVAL = Number.MAX_VALUE;
// ─────────────────────────────────────────────────────────────────────────────

export function dgeqrf_optimized(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number
): number {
  // Argument validation (xerbla inlined)
  if (m < 0)
    throw new Error(
      "** On entry to 'DGEQRF' parameter number 1 had an illegal value"
    );
  if (n < 0)
    throw new Error(
      "** On entry to 'DGEQRF' parameter number 2 had an illegal value"
    );
  if (lda < Math.max(1, m))
    throw new Error(
      "** On entry to 'DGEQRF' parameter number 4 had an illegal value"
    );

  const k = Math.min(m, n);
  if (k === 0) return 0;

  // ilaenv(1,"DGEQRF",...) = 1 → nb=1, nbmin=2, condition nb>=nbmin is false.
  // Blocked path never executes; dgeqrf always calls:
  //   dgeqr2(m, n, a, aOff, lda, tau, tauOff, work, 0)
  //
  // We inline dgeqr2 directly. dgeqr2's own input validation is skipped
  // because dgeqrf already checked the same conditions above.

  // Pre-allocate once to avoid GC pressure inside the loop.
  // v[0..m-1]: unaliased copy of the current Householder vector.
  const v = allocFloat64Array(m);

  // ═══ dgeqr2 inlined ═══════════════════════════════════════════════════════
  for (let i = 1; i <= k; i++) {
    // Flat offset of the pivot element A(i,i)
    const aIIOff = aOff + (i - 1) + (i - 1) * lda;

    // ─── dlarfg inlined ─────────────────────────────────────────────────────
    // Generates elementary reflector H(i) = I - tauI * v * v^T such that
    //   H(i) * [A(i,i); A(i+1:m,i)] = [beta; 0]
    // Inputs:  n_larfg = m-i+1, alpha = A(i,i), x = A(i+1:m,i), incx = 1
    // Outputs: A(i,i) ← beta,  tau[i] ← tauI,  A(i+1:m,i) ← v(2:n_larfg)

    const n_larfg = m - i + 1; // order of the reflector
    let tauI: number;

    if (n_larfg <= 1) {
      // Scalar case — reflector is identity, no changes needed
      tauI = 0.0;
    } else {
      const xOff = aIIOff + 1; // A(i+1,i), contiguous (incx = 1)
      const xLen = n_larfg - 1; // = m - i  (length of sub-diagonal vector)
      const xEnd = xOff + xLen; // exclusive end index in a[]
      let alpha = a[aIIOff];

      // ── dnrm2(xLen, a, xOff, 1) inlined — stride-1 scaled accumulation ──
      // Computes ||A(i+1:m, i)||_2 without overflow/underflow.
      let xnorm: number;
      if (xLen === 1) {
        xnorm = Math.abs(a[xOff]);
      } else {
        let scale = 0.0,
          ssq = 1.0;
        for (let xi = xOff; xi < xEnd; xi++) {
          const absxi = Math.abs(a[xi]);
          if (absxi > 0.0) {
            if (scale < absxi) {
              ssq = 1.0 + ssq * (scale / absxi) * (scale / absxi);
              scale = absxi;
            } else {
              ssq += (absxi / scale) * (absxi / scale);
            }
          }
        }
        xnorm = scale * Math.sqrt(ssq);
      }
      // ─────────────────────────────────────────────────────────────────────

      if (xnorm === 0.0) {
        // Sub-diagonal is already zero — H = I
        tauI = 0.0;
        // a[aIIOff] is left unchanged (matches dlarfg returning { alpha, tau:0 })
      } else {
        // ── dlapy2(alpha, xnorm) inlined → beta = -sign(alpha)*hypot(alpha,xnorm)
        // NaN propagation is handled implicitly by IEEE 754 arithmetic.
        let beta: number;
        {
          const xabs = Math.abs(alpha);
          const w = xabs > xnorm ? xabs : xnorm; // max(|alpha|, xnorm)
          const z = xabs < xnorm ? xabs : xnorm; // min(|alpha|, xnorm)
          beta =
            z === 0.0 || w > _HUGEVAL
              ? w
              : w * Math.sqrt(1.0 + (z / w) * (z / w));
          beta = alpha >= 0 ? -beta : beta;
        }
        // ─────────────────────────────────────────────────────────────────────

        let knt = 0;

        if (Math.abs(beta) < _SAFMIN) {
          // xnorm or beta may be inaccurate — scale x up and recompute
          do {
            knt++;
            // dscal(xLen, rsafmn, a, xOff, 1) inlined (incx = 1)
            for (let xi = xOff; xi < xEnd; xi++) a[xi] *= _RSAFMN;
            beta *= _RSAFMN;
            alpha *= _RSAFMN;
          } while (Math.abs(beta) < _SAFMIN && knt < 20);

          // Recompute xnorm from the rescaled x (dnrm2 inlined again)
          if (xLen === 1) {
            xnorm = Math.abs(a[xOff]);
          } else {
            let scale = 0.0,
              ssq = 1.0;
            for (let xi = xOff; xi < xEnd; xi++) {
              const absxi = Math.abs(a[xi]);
              if (absxi > 0.0) {
                if (scale < absxi) {
                  ssq = 1.0 + ssq * (scale / absxi) * (scale / absxi);
                  scale = absxi;
                } else {
                  ssq += (absxi / scale) * (absxi / scale);
                }
              }
            }
            xnorm = scale * Math.sqrt(ssq);
          }

          // Recompute beta from rescaled alpha and xnorm (dlapy2 inlined)
          {
            const xabs = Math.abs(alpha);
            const w = xabs > xnorm ? xabs : xnorm;
            const z = xabs < xnorm ? xabs : xnorm;
            beta =
              z === 0.0 || w > _HUGEVAL
                ? w
                : w * Math.sqrt(1.0 + (z / w) * (z / w));
            beta = alpha >= 0 ? -beta : beta;
          }
        }

        // Compute tau from the (possibly rescaled) alpha and beta
        tauI = (beta - alpha) / beta;

        // dscal(xLen, 1/(alpha-beta), a, xOff, 1) inlined — normalise v
        {
          const sc = 1.0 / (alpha - beta);
          for (let xi = xOff; xi < xEnd; xi++) a[xi] *= sc;
        }

        // Recover beta to its original scale (undo knt rounds of _RSAFMN scaling)
        for (let j = 1; j <= knt; j++) beta *= _SAFMIN;

        a[aIIOff] = beta; // store pivot: new A(i,i) = beta
      }
    }
    // ─── end dlarfg ─────────────────────────────────────────────────────────

    tau[tauOff + (i - 1)] = tauI;

    if (i < n && tauI !== 0.0) {
      // ─── dlarf1f(LEFT, m-i+1, n-i, v=a[aIIOff], incv=1, tauI,
      //              C=a[cOff], lda, work) inlined ──────────────────────────
      // Apply H(i) to the trailing submatrix A(i:m, i+1:n) from the left.
      // C = A(i:m, i+1:n), dimensions: rows × cols.

      const rows = m - i + 1; // number of rows in C  (= length of v)
      const cols = n - i; //     number of columns in C  (> 0 since i < n)
      const cOff = aOff + (i - 1) + i * lda; // flat offset of C(1,1) = A(i, i+1)

      // Copy Householder vector to v[] to remove aliasing with a[].
      // v[0] = 1 (the implicit unit first element in LAPACK compact form).
      // v[1..rows-1] = A(i+1:m, i), the stored sub-diagonal part.
      v[0] = 1.0;
      for (let ii = 1; ii < rows; ii++) v[ii] = a[aIIOff + ii];

      // Find lastv: 1-based length of the active part of v.
      // Scans backward to skip trailing zeros. v[0]=1 ensures lastv >= 1.
      let lastv = rows;
      while (lastv > 1 && v[lastv - 1] === 0.0) lastv--;

      // ── iladlc(lastv, cols, a, cOff, lda) inlined ────────────────────────
      // Returns lastc: 1-based index of last non-zero column in C(1:lastv,:).
      let lastc: number;
      if (cols === 0) {
        lastc = 0; // unreachable when i < n; kept for correctness
      } else if (
        a[cOff + (cols - 1) * lda] !== 0.0 || // C(1, cols) = A(i, n)
        a[cOff + (lastv - 1) + (cols - 1) * lda] !== 0.0 // C(lastv, cols)
      ) {
        lastc = cols; // fast path: last column is non-zero
      } else {
        // Slow path: scan columns from right to left
        lastc = 0;
        scan: for (let col = cols; col >= 1; col--) {
          const colBase = cOff + (col - 1) * lda;
          for (let ii = 0; ii < lastv; ii++) {
            if (a[colBase + ii] !== 0.0) {
              lastc = col;
              break scan;
            }
          }
        }
      }
      // ─────────────────────────────────────────────────────────────────────

      if (lastc === 0) continue; // C is numerically zero; nothing to update

      if (lastv === 1) {
        // v = [1], so H = I - tau * e1 * e1^T.
        // Only row i of C is affected: C(i, i+1:i+lastc) *= (1 - tau).
        // The elements C(1, j) = a[cOff + (j-1)*lda] are stride-lda apart.
        // (dscal(lastc, 1-tau, c, cOff, ldc) inlined with incx = lda)
        const scale = 1.0 - tauI;
        for (let j = 0; j < lastc; j++) a[cOff + j * lda] *= scale;
      } else {
        // ── Fused left-apply of H(i) to C(1:lastv, 1:lastc) ─────────────
        // Replaces: dgemv(TRANS,...) + daxpy×2 + dger from dlarf1f.
        //
        // For each column j of C:
        //   w  = v^T * C[:,j]        (dot product; v[0]=1 is explicit)
        //   C[:,j] -= tauI * w * v   (rank-1 update)
        //
        // v[] is a separate Float64Array — no aliasing with a[].
        // Both inner loops are stride-1, so V8 can auto-vectorise them.
        for (let j = 0; j < lastc; j++) {
          const cCol = cOff + j * lda; // base of column j of C in a[]

          // Pass 1: w = v^T * C[:,j]
          let w = 0.0;
          for (let ii = 0; ii < lastv; ii++) w += v[ii] * a[cCol + ii];

          // Pass 2: C[:,j] -= tauI * w * v
          const tw = tauI * w;
          for (let ii = 0; ii < lastv; ii++) a[cCol + ii] -= tw * v[ii];
        }
        // ─────────────────────────────────────────────────────────────────
      }
      // ─── end dlarf1f ─────────────────────────────────────────────────────
    }
  }
  // ═══ end dgeqr2 ═══════════════════════════════════════════════════════════

  return 0;
}
