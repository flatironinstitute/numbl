// Tests for dgetrf and dgetri
// Run: npx tsc && node --input-type=module < dist/test.js
//
// Strategy:
//   dgetrf: factor A = P*L*U, reconstruct P^-1*L*U and compare to original A
//   dgetri: factor then invert A, verify A_inv * A ≈ I

import { allocFloat64Array } from "../../numbl-core/executors/jsJit/helpers/alloc.js";
import { dgetrf } from "./SRC/dgetrf.js";
import { dgetri } from "./SRC/dgetri.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Element A(i,j) in column-major storage, 1-based */
function idx(i: number, j: number, lda: number): number {
  return i - 1 + (j - 1) * lda;
}

/** Matrix-matrix multiply C = A*B, all n×n, column-major */
function matmul(a: Float64Array, b: Float64Array, n: number): Float64Array {
  const c = allocFloat64Array(n * n);
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= n; j++) {
      let s = 0.0;
      for (let k = 1; k <= n; k++) s += a[idx(i, k, n)] * b[idx(k, j, n)];
      c[idx(i, j, n)] = s;
    }
  }
  return c;
}

/** Max absolute element of a Float64Array */
function maxabs(v: Float64Array): number {
  let m = 0;
  for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]));
  return m;
}

/** Subtract two Float64Arrays element-wise */
function sub(a: Float64Array, b: Float64Array): Float64Array {
  const r = allocFloat64Array(a.length);
  for (let i = 0; i < a.length; i++) r[i] = a[i] - b[i];
  return r;
}

/** Identity matrix n×n, column-major */
function eye(n: number): Float64Array {
  const I = allocFloat64Array(n * n);
  for (let i = 1; i <= n; i++) I[idx(i, i, n)] = 1.0;
  return I;
}

/** Seeded pseudo-random matrix (deterministic) */
function randMatrix(n: number, seed: number): Float64Array {
  const a = allocFloat64Array(n * n);
  let s = seed;
  for (let i = 0; i < n * n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s / 0x100000000) * 2.0 - 1.0;
  }
  return a;
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

let passed = 0,
  failed = 0;

function check(name: string, err: number, tol: number): void {
  if (err < tol) {
    console.log(`  PASS  ${name}  (err=${err.toExponential(3)})`);
    passed++;
  } else {
    console.log(
      `  FAIL  ${name}  (err=${err.toExponential(3)}, tol=${tol.toExponential(3)})`
    );
    failed++;
  }
}

// ---------------------------------------------------------------------------
// DGETRF tests
// ---------------------------------------------------------------------------

console.log("\n=== dgetrf tests ===");

function testDgetrf(n: number, label: string): void {
  const A = randMatrix(n, 42 + n);
  const origA = A.slice();
  const ipiv = new Int32Array(n);

  const info = dgetrf(n, n, A, n, ipiv);

  if (info !== 0) {
    console.log(`  SKIP  ${label}  (singular matrix, info=${info})`);
    return;
  }

  // Extract L and U from the packed LU
  const L = allocFloat64Array(n * n);
  const U = allocFloat64Array(n * n);
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= n; j++) {
      if (i > j) L[idx(i, j, n)] = A[idx(i, j, n)];
      else if (i === j) {
        L[idx(i, j, n)] = 1.0;
        U[idx(i, j, n)] = A[idx(i, j, n)];
      } else U[idx(i, j, n)] = A[idx(i, j, n)];
    }
  }

  // LU product
  const LU = matmul(L, U, n);

  // Undo row swaps (apply pivots in reverse) to recover original A
  const PA = LU.slice();
  for (let i = n; i >= 1; i--) {
    const ip = ipiv[i - 1];
    if (ip !== i) {
      for (let j = 1; j <= n; j++) {
        const tmp = PA[idx(i, j, n)];
        PA[idx(i, j, n)] = PA[idx(ip, j, n)];
        PA[idx(ip, j, n)] = tmp;
      }
    }
  }

  const err = maxabs(sub(PA, origA)) / (n * maxabs(origA));
  check(`dgetrf ${label}  P^-1*L*U == A`, err, 1e-12);
}

testDgetrf(1, "1×1");
testDgetrf(2, "2×2");
testDgetrf(3, "3×3");
testDgetrf(10, "10×10");
testDgetrf(50, "50×50");
testDgetrf(100, "100×100");
testDgetrf(200, "200×200"); // exercises blocked path (nb=64)

// ---------------------------------------------------------------------------
// DGETRI tests
// ---------------------------------------------------------------------------

console.log("\n=== dgetri tests ===");

function testDgetri(n: number, label: string): void {
  const A = randMatrix(n, 99 + n);
  const origA = A.slice();
  const ipiv = new Int32Array(n);

  // Factorize
  const infoF = dgetrf(n, n, A, n, ipiv);
  if (infoF !== 0) {
    console.log(`  SKIP  ${label}  (singular at factor step, info=${infoF})`);
    return;
  }

  // Invert
  const infoI = dgetri(n, A, n, ipiv);
  if (infoI !== 0) {
    console.log(`  SKIP  ${label}  (singular at invert step, info=${infoI})`);
    return;
  }
  const Ainv = A; // now holds the inverse

  // Verify A_inv * origA ≈ I  (and origA * A_inv ≈ I)
  const AinvA = matmul(Ainv, origA, n);
  const I = eye(n);
  const scale = maxabs(origA);

  const err = maxabs(sub(AinvA, I)) / (n * scale);
  check(`dgetri ${label}  inv(A)*A == I`, err, 1e-10);
}

testDgetri(1, "1×1");
testDgetri(2, "2×2");
testDgetri(3, "3×3");
testDgetri(10, "10×10");
testDgetri(50, "50×50");
testDgetri(100, "100×100");
testDgetri(200, "200×200"); // exercises blocked dgetri path (nb=64)

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed`);
if (failed > 0) throw new Error(`${failed} test(s) failed`);
