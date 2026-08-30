// Translated from BLAS/SRC/dsymv.f
// DSYMV performs the matrix-vector operation
//   y := alpha*A*x + beta*y
// where alpha and beta are scalars, x and y are n element vectors and A is an
// n by n symmetric matrix.  Only the triangle selected by `uplo` is referenced.
//
// Array indexing convention (column-major):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   X(I)   => x[xOff + (I-1)*incx]          (incx > 0; see kx for incx < 0)

import { UPPER, LOWER } from "../utils/constants.js";
import { xerbla } from "../utils/xerbla.js";

export function dsymv(
  uplo: number,
  n: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  x: Float64Array,
  xOff: number,
  incx: number,
  beta: number,
  y: Float64Array,
  yOff: number,
  incy: number
): void {
  const ZERO = 0.0;
  const ONE = 1.0;

  // Test the input parameters
  let info = 0;
  if (uplo !== UPPER && uplo !== LOWER) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (lda < Math.max(1, n)) {
    info = 5;
  } else if (incx === 0) {
    info = 7;
  } else if (incy === 0) {
    info = 10;
  }
  if (info !== 0) {
    xerbla("DSYMV ", info);
    return;
  }

  // Quick return if possible
  if (n === 0 || (alpha === ZERO && beta === ONE)) return;

  // Set up the start points in X and Y (0-based)
  const kx = incx > 0 ? 0 : -(n - 1) * incx;
  const ky = incy > 0 ? 0 : -(n - 1) * incy;

  // First form y := beta*y
  if (beta !== ONE) {
    let iy = ky;
    if (beta === ZERO) {
      for (let i = 0; i < n; i++) {
        y[yOff + iy] = ZERO;
        iy += incy;
      }
    } else {
      for (let i = 0; i < n; i++) {
        y[yOff + iy] *= beta;
        iy += incy;
      }
    }
  }
  if (alpha === ZERO) return;

  if (uplo === UPPER) {
    // Form y when A is stored in the upper triangle
    let jx = kx;
    let jy = ky;
    for (let j = 1; j <= n; j++) {
      const temp1 = alpha * x[xOff + jx];
      let temp2 = ZERO;
      const aCol = aOff + (j - 1) * lda;
      let ix = kx;
      let iy = ky;
      for (let i = 1; i <= j - 1; i++) {
        const aij = a[aCol + (i - 1)];
        y[yOff + iy] += temp1 * aij;
        temp2 += aij * x[xOff + ix];
        ix += incx;
        iy += incy;
      }
      y[yOff + jy] += temp1 * a[aCol + (j - 1)] + alpha * temp2;
      jx += incx;
      jy += incy;
    }
  } else {
    // Form y when A is stored in the lower triangle
    let jx = kx;
    let jy = ky;
    for (let j = 1; j <= n; j++) {
      const temp1 = alpha * x[xOff + jx];
      let temp2 = ZERO;
      const aCol = aOff + (j - 1) * lda;
      y[yOff + jy] += temp1 * a[aCol + (j - 1)];
      let ix = jx;
      let iy = jy;
      for (let i = j + 1; i <= n; i++) {
        ix += incx;
        iy += incy;
        const aij = a[aCol + (i - 1)];
        y[yOff + iy] += temp1 * aij;
        temp2 += aij * x[xOff + ix];
      }
      y[yOff + jy] += alpha * temp2;
      jx += incx;
      jy += incy;
    }
  }
}
