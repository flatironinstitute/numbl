// Translated from BLAS/SRC/dsyr2.f
// DSYR2 performs the symmetric rank 2 operation
//   A := alpha*x*y**T + alpha*y*x**T + A
// where alpha is a scalar, x and y are n element vectors and A is an n by n
// symmetric matrix.  Only the triangle selected by `uplo` is updated.
//
// Array indexing convention (column-major):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)

import { UPPER, LOWER } from "../utils/constants.js";
import { xerbla } from "../utils/xerbla.js";

export function dsyr2(
  uplo: number,
  n: number,
  alpha: number,
  x: Float64Array,
  xOff: number,
  incx: number,
  y: Float64Array,
  yOff: number,
  incy: number,
  a: Float64Array,
  aOff: number,
  lda: number
): void {
  const ZERO = 0.0;

  // Test the input parameters
  let info = 0;
  if (uplo !== UPPER && uplo !== LOWER) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (incx === 0) {
    info = 5;
  } else if (incy === 0) {
    info = 7;
  } else if (lda < Math.max(1, n)) {
    info = 9;
  }
  if (info !== 0) {
    xerbla("DSYR2 ", info);
    return;
  }

  // Quick return if possible
  if (n === 0 || alpha === ZERO) return;

  // Set up the start points in X and Y (0-based)
  const kx = incx > 0 ? 0 : -(n - 1) * incx;
  const ky = incy > 0 ? 0 : -(n - 1) * incy;

  let jx = kx;
  let jy = ky;

  if (uplo === UPPER) {
    // Form A when A is stored in the upper triangle
    for (let j = 1; j <= n; j++) {
      const xj = x[xOff + jx];
      const yj = y[yOff + jy];
      if (xj !== ZERO || yj !== ZERO) {
        const temp1 = alpha * yj;
        const temp2 = alpha * xj;
        const aCol = aOff + (j - 1) * lda;
        let ix = kx;
        let iy = ky;
        for (let i = 1; i <= j; i++) {
          a[aCol + (i - 1)] += x[xOff + ix] * temp1 + y[yOff + iy] * temp2;
          ix += incx;
          iy += incy;
        }
      }
      jx += incx;
      jy += incy;
    }
  } else {
    // Form A when A is stored in the lower triangle
    for (let j = 1; j <= n; j++) {
      const xj = x[xOff + jx];
      const yj = y[yOff + jy];
      if (xj !== ZERO || yj !== ZERO) {
        const temp1 = alpha * yj;
        const temp2 = alpha * xj;
        const aCol = aOff + (j - 1) * lda;
        let ix = jx;
        let iy = jy;
        for (let i = j; i <= n; i++) {
          a[aCol + (i - 1)] += x[xOff + ix] * temp1 + y[yOff + iy] * temp2;
          ix += incx;
          iy += incy;
        }
      }
      jx += incx;
      jy += incy;
    }
  }
}
