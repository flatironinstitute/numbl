// dorgqr_optimized.ts
// Fully inlined + JS-engine-optimized implementation of DORGQR.
//
// Core design principles (vs. LAPACK-faithful inlining):
//
//   1. Unaliased v[] copy — v is extracted into its own Float64Array with
//      v[0]=1 explicit. The JIT sees two independent arrays (v and a) with
//      no aliasing; it can bounds-check-hoist and vectorize both inner loops.
//
//   2. Per-column fusion — dot product and rank-1 update are done inside the
//      same j-iteration. The column a[cCol..cCol+rows] is loaded for the dot
//      and still in L1 cache when the update writes it. This replaces the
//      two-pass (all-columns-dot then all-columns-update) pattern.
//
//   3. work[] eliminated — w is a scalar local to each j-iteration. Removes
//      an array allocation and eliminates the second pass over work[].
//
//   4. No branches in inner loops — the if(wj!==0) guard is removed.
//      For dense matrices branch misprediction costs more than a zero-multiply.
//
//   5. Minimal pointer arithmetic in hot loops — cCol = cBase + j*lda is
//      computed once per j; inner loops only add ii (stride-1 linear scan).
//      Similarly diagOff+ii is the only ii-dependent term in v reads.
//      But v[ii] (stride-1 into a separate array) beats a[diagOff+ii]
//      (stride-1 but aliased with the write target a[]).
//
//   6. a.fill() for all zero-init — V8 lowers TypedArray.fill(0) to memset.
//
//   7. v pre-allocated outside outer loop — one Float64Array(m) allocation
//      total, reused every iteration, avoiding GC pressure inside the loop.
//
// Array indexing matches Fortran column-major:
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   TAU(I)  =>  tau[tauOff + (I-1)]             (I is 1-based)

import { allocFloat64Array } from "../../../numbl-core/runtime/alloc";

export function dorgqr_optimized(
  m: number,
  n: number,
  k: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number
): number {
  if (m < 0)
    throw new Error(
      "** On entry to 'DORGQR' parameter number 1 had an illegal value"
    );
  if (n < 0 || n > m)
    throw new Error(
      "** On entry to 'DORGQR' parameter number 2 had an illegal value"
    );
  if (k < 0 || k > n)
    throw new Error(
      "** On entry to 'DORGQR' parameter number 3 had an illegal value"
    );
  if (lda < Math.max(1, m))
    throw new Error(
      "** On entry to 'DORGQR' parameter number 5 had an illegal value"
    );

  if (n <= 0) return 0;

  // ilaenv("DORGQR") = 1 => nb=1 < nbmin=2 => kk=0 => unblocked path only.

  // Pre-allocate the Householder vector buffer once.
  // Reused each outer iteration; holds at most m elements.
  const v = allocFloat64Array(m);

  // Initialize columns k+1..n (0-based k..n-1) to identity columns.
  for (let j = k; j < n; j++) {
    const colBase = aOff + j * lda;
    a.fill(0.0, colBase, colBase + m);
    a[colBase + j] = 1.0;
  }

  // Apply H(k), H(k-1), ..., H(1) in reverse order (dorg2r logic).
  for (let i = k; i >= 1; i--) {
    const col = i - 1; //            0-based pivot column
    const colOff = aOff + col * lda; // base of column i
    const diagOff = colOff + col; //   A(i,i)
    const tauI = tau[tauOff + col];

    if (i < n && tauI !== 0.0) {
      const rows = m - col; // = m - i + 1  (length of v, rows of C)
      const cols = n - i; //               (number of columns of C)
      const cBase = diagOff + lda; //       A(i, i+1) — first element of C

      // Copy Householder vector into unaliased v[].
      // v[0] = 1 (implicit in LAPACK's compact form, explicit here).
      // v[1..rows-1] = A(i+1:m, i).
      v[0] = 1.0;
      for (let ii = 1; ii < rows; ii++) {
        v[ii] = a[diagOff + ii];
      }

      // Apply H(i) to C = A(i:m, i+1:n), one column at a time.
      // The column a[cCol..cCol+rows] fits in L1; both passes hit it while hot.
      for (let j = 0; j < cols; j++) {
        const cCol = cBase + j * lda; // base of column j of C

        // Pass 1: w = v^T * C[:,j]
        let w = 0.0;
        for (let ii = 0; ii < rows; ii++) {
          w += v[ii] * a[cCol + ii];
        }

        // Pass 2: C[:,j] -= tauI * w * v
        const tw = tauI * w;
        for (let ii = 0; ii < rows; ii++) {
          a[cCol + ii] -= tw * v[ii];
        }
      }
    }

    // Scale A(i+1:m, i) by -tauI  (contiguous slice, stride-1).
    if (i < m) {
      const scalFactor = -tauI;
      if (scalFactor !== 1.0) {
        const subEnd = colOff + m;
        for (let si = diagOff + 1; si < subEnd; si++) {
          a[si] *= scalFactor;
        }
      }
    }

    a[diagOff] = 1.0 - tauI; //       A(i,i) = 1 - tauI
    a.fill(0.0, colOff, diagOff); //   A(1:i-1, i) = 0  (memset via fill)
  }

  return 0;
}
