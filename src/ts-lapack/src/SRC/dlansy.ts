// Translated from SRC/dlansy.f
// DLANSY returns the value of the 1-norm, Frobenius norm, infinity-norm, or
// the element of largest absolute value of a real symmetric matrix A, of which
// only the triangle selected by `uplo` is referenced.
//
// norm codes (matching dlange):
//   0 = 'M' — max(abs(A(i,j)))
//   1 = 'O' or '1' — one-norm
//   2 = 'I' — infinity-norm (same as the one-norm, A is symmetric)
//   3 = 'F' or 'E' — Frobenius norm
//
// Array indexing convention (column-major):
//   A(I,J)  => a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   WORK(I) => work[workOff + (I-1)]
//
// WORK is only referenced when norm = 1 or 2, and must have length >= N.

import { UPPER } from "../utils/constants.js";
import { dlassq } from "./dlassq.js";

// norm constants
const NORM_MAX = 0; // 'M'
const NORM_ONE = 1; // 'O' or '1'
const NORM_INF = 2; // 'I'
const NORM_FRO = 3; // 'F' or 'E'

export function dlansy(
  norm: number,
  uplo: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  work: Float64Array,
  workOff: number
): number {
  let value = 0.0;

  if (n === 0) {
    value = 0.0;
  } else if (norm === NORM_MAX) {
    // Find max(abs(A(i,j)))
    value = 0.0;
    if (uplo === UPPER) {
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= j; i++) {
          const sum = Math.abs(a[aOff + (i - 1) + (j - 1) * lda]);
          if (value < sum || Number.isNaN(sum)) value = sum;
        }
      }
    } else {
      for (let j = 1; j <= n; j++) {
        for (let i = j; i <= n; i++) {
          const sum = Math.abs(a[aOff + (i - 1) + (j - 1) * lda]);
          if (value < sum || Number.isNaN(sum)) value = sum;
        }
      }
    }
  } else if (norm === NORM_ONE || norm === NORM_INF) {
    // Find normI(A) ( = norm1(A), since A is symmetric)
    value = 0.0;
    if (uplo === UPPER) {
      for (let j = 1; j <= n; j++) {
        let sum = 0.0;
        for (let i = 1; i <= j - 1; i++) {
          const absa = Math.abs(a[aOff + (i - 1) + (j - 1) * lda]);
          sum += absa;
          work[workOff + (i - 1)] += absa;
        }
        work[workOff + (j - 1)] =
          sum + Math.abs(a[aOff + (j - 1) + (j - 1) * lda]);
      }
      for (let i = 1; i <= n; i++) {
        const sum = work[workOff + (i - 1)];
        if (value < sum || Number.isNaN(sum)) value = sum;
      }
    } else {
      for (let i = 1; i <= n; i++) work[workOff + (i - 1)] = 0.0;
      for (let j = 1; j <= n; j++) {
        let sum =
          work[workOff + (j - 1)] + Math.abs(a[aOff + (j - 1) + (j - 1) * lda]);
        for (let i = j + 1; i <= n; i++) {
          const absa = Math.abs(a[aOff + (i - 1) + (j - 1) * lda]);
          sum += absa;
          work[workOff + (i - 1)] += absa;
        }
        if (value < sum || Number.isNaN(sum)) value = sum;
      }
    }
  } else if (norm === NORM_FRO) {
    // Find normF(A)
    const scale = { val: 0.0 };
    const sum = { val: 1.0 };
    if (uplo === UPPER) {
      for (let j = 2; j <= n; j++) {
        dlassq(j - 1, a, aOff + (j - 1) * lda, 1, scale, sum);
      }
    } else {
      for (let j = 1; j <= n - 1; j++) {
        dlassq(n - j, a, aOff + j + (j - 1) * lda, 1, scale, sum);
      }
    }
    sum.val = 2 * sum.val;
    dlassq(n, a, aOff, lda + 1, scale, sum);
    value = scale.val * Math.sqrt(sum.val);
  }

  return value;
}
