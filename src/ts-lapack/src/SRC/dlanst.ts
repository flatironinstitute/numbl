// Translated from SRC/dlanst.f
// DLANST returns the value of the 1-norm, Frobenius norm, infinity-norm, or
// the element of largest absolute value of a real symmetric tridiagonal
// matrix A with diagonal D and off-diagonal E.
//
// norm codes (matching dlange):
//   0 = 'M' — max(abs(A(i,j)))
//   1 = 'O' or '1' — one-norm
//   2 = 'I' — infinity-norm (same as the one-norm, A is symmetric)
//   3 = 'F' or 'E' — Frobenius norm
//
// Array indexing convention:
//   D(I) => d[dOff + (I-1)], E(I) => e[eOff + (I-1)]   (I is 1-based)

import { dlassq } from "./dlassq.js";

// norm constants
const NORM_MAX = 0; // 'M'
const NORM_ONE = 1; // 'O' or '1'
const NORM_INF = 2; // 'I'
const NORM_FRO = 3; // 'F' or 'E'

export function dlanst(
  norm: number,
  n: number,
  d: Float64Array,
  dOff: number,
  e: Float64Array,
  eOff: number
): number {
  let anorm = 0.0;

  if (n <= 0) {
    anorm = 0.0;
  } else if (norm === NORM_MAX) {
    // Find max(abs(A(i,j)))
    anorm = Math.abs(d[dOff + (n - 1)]);
    for (let i = 1; i <= n - 1; i++) {
      let sum = Math.abs(d[dOff + (i - 1)]);
      if (anorm < sum || Number.isNaN(sum)) anorm = sum;
      sum = Math.abs(e[eOff + (i - 1)]);
      if (anorm < sum || Number.isNaN(sum)) anorm = sum;
    }
  } else if (norm === NORM_ONE || norm === NORM_INF) {
    // Find norm1(A)
    if (n === 1) {
      anorm = Math.abs(d[dOff]);
    } else {
      anorm = Math.abs(d[dOff]) + Math.abs(e[eOff]);
      let sum = Math.abs(e[eOff + (n - 2)]) + Math.abs(d[dOff + (n - 1)]);
      if (anorm < sum || Number.isNaN(sum)) anorm = sum;
      for (let i = 2; i <= n - 1; i++) {
        sum =
          Math.abs(d[dOff + (i - 1)]) +
          Math.abs(e[eOff + (i - 1)]) +
          Math.abs(e[eOff + (i - 2)]);
        if (anorm < sum || Number.isNaN(sum)) anorm = sum;
      }
    }
  } else if (norm === NORM_FRO) {
    // Find normF(A)
    const scale = { val: 0.0 };
    const sum = { val: 1.0 };
    if (n > 1) {
      dlassq(n - 1, e, eOff, 1, scale, sum);
      sum.val = 2 * sum.val;
    }
    dlassq(n, d, dOff, 1, scale, sum);
    anorm = scale.val * Math.sqrt(sum.val);
  }

  return anorm;
}
