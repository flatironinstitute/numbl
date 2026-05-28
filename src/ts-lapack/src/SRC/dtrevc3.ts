// Translated from SRC/dtrevc3.f
// DTREVC3 computes some or all of the right and/or left eigenvectors of
// a real upper quasi-triangular matrix T.
//
// Matrices of this type are produced by the Schur factorization of
// a real general matrix:  A = Q*T*Q**T, as computed by DHSEQR.
//
// The right eigenvector x and the left eigenvector y of T corresponding
// to an eigenvalue w are defined by:
//
//    T*x = w*x,     (y**T)*T = w*(y**T)
//
// This uses a Level 3 BLAS version of the back transformation.
//
// Array indexing convention (column-major, matching Fortran):
//   T(I,J)  => t[tOff + (I-1) + (J-1)*ldt]    (I,J are 1-based)
//   VL(I,J) => vl[vlOff + (I-1) + (J-1)*ldvl]  (I,J are 1-based)
//   VR(I,J) => vr[vrOff + (I-1) + (J-1)*ldvr]  (I,J are 1-based)
//   WORK is treated as a 1-D array with 1-based Fortran indexing:
//     WORK(K + IV*N) => work[workOff + (K-1) + IV*n]

import { LEFT, RIGHT, NOTRANS } from "../utils/constants.js";
import { MACH_SFMIN, MACH_PREC } from "../utils/constants.js";
import { dlamch } from "./dlamch.js";
import { dlaln2 } from "./dlaln2.js";
import { dlacpy } from "./dlacpy.js";
import { dlaset } from "./dlaset.js";
import { dscal } from "../BLAS/dscal.js";
import { daxpy } from "../BLAS/daxpy.js";
import { dcopy } from "../BLAS/dcopy.js";
import { ddot } from "../BLAS/ddot.js";
import { dgemm } from "../BLAS/dgemm.js";
import { dgemv } from "../BLAS/dgemv.js";
import { idamax } from "../BLAS/idamax.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

// SIDE constants
const BOTH = 2; // 'B' — compute both left and right eigenvectors

// HOWMNY constants
const HOWMNY_ALL = 0; // 'A' — compute all eigenvectors
const HOWMNY_BACKTRANSFORM = 1; // 'B' — backtransform
const HOWMNY_SELECTED = 2; // 'S' — selected eigenvectors only

// Block size parameters
const NBMIN = 8;
const NBMAX = 128;

/**
 * DTREVC3 computes some or all of the right and/or left eigenvectors of
 * a real upper quasi-triangular matrix T.
 *
 * @param side - LEFT (0), RIGHT (1), or BOTH (2)
 * @param howmny - 0='A' all, 1='B' backtransform, 2='S' selected
 * @param select - logical array, only used when howmny=2
 * @param n - order of matrix T
 * @param t - upper quasi-triangular matrix T, dimension (ldt, n)
 * @param tOff - offset into t
 * @param ldt - leading dimension of t
 * @param vl - left eigenvector matrix, dimension (ldvl, mm)
 * @param vlOff - offset into vl
 * @param ldvl - leading dimension of vl
 * @param vr - right eigenvector matrix, dimension (ldvr, mm)
 * @param vrOff - offset into vr
 * @param ldvr - leading dimension of vr
 * @param mm - number of columns in VL/VR
 * @param work - workspace array, dimension (max(1, lwork))
 * @param workOff - offset into work
 * @param lwork - dimension of work array
 * @returns { m: number of eigenvectors computed, info: 0 on success }
 */
export function dtrevc3(
  side: number,
  howmny: number,
  select: boolean[],
  n: number,
  t: Float64Array,
  tOff: number,
  ldt: number,
  vl: Float64Array,
  vlOff: number,
  ldvl: number,
  vr: Float64Array,
  vrOff: number,
  ldvr: number,
  mm: number,
  work: Float64Array,
  workOff: number,
  lwork: number
): { m: number; info: number } {
  const ZERO = 0.0;
  const ONE = 1.0;

  // Decode and test the input parameters
  const bothv = side === BOTH;
  const rightv = side === RIGHT || bothv;
  const leftv = side === LEFT || bothv;

  const allv = howmny === HOWMNY_ALL;
  const over = howmny === HOWMNY_BACKTRANSFORM;
  const somev = howmny === HOWMNY_SELECTED;

  let info = 0;

  // Use a fixed NB for workspace query; ILAENV is not available,
  // so we use NBMAX as an upper bound for the optimal block size.
  let nb = NBMAX;
  const maxwrk = Math.max(1, n + 2 * n * nb);
  work[workOff] = maxwrk;
  const lquery = lwork === -1;

  if (!rightv && !leftv) {
    info = -1;
  } else if (!allv && !over && !somev) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (ldt < Math.max(1, n)) {
    info = -6;
  } else if (ldvl < 1 || (leftv && ldvl < n)) {
    info = -8;
  } else if (ldvr < 1 || (rightv && ldvr < n)) {
    info = -10;
  } else if (lwork < Math.max(1, 3 * n) && !lquery) {
    info = -14;
  }

  let m = 0;

  if (info === 0) {
    // Set M to the number of columns required to store the selected
    // eigenvectors, standardize the array SELECT if necessary, and
    // test MM.
    if (somev) {
      m = 0;
      let pair = false;
      for (let j = 1; j <= n; j++) {
        if (pair) {
          pair = false;
          select[j - 1] = false;
        } else {
          if (j < n) {
            // T(J+1,J)
            if (t[tOff + j + (j - 1) * ldt] === ZERO) {
              if (select[j - 1]) {
                m = m + 1;
              }
            } else {
              pair = true;
              if (select[j - 1] || select[j]) {
                select[j - 1] = true;
                m = m + 2;
              }
            }
          } else {
            if (select[n - 1]) {
              m = m + 1;
            }
          }
        }
      }
    } else {
      m = n;
    }

    if (mm < m) {
      info = -11;
    }
  }

  if (info !== 0) {
    return { m: 0, info };
  } else if (lquery) {
    return { m, info: 0 };
  }

  // Quick return if possible.
  if (n === 0) {
    return { m, info: 0 };
  }

  // Use blocked version of back-transformation if sufficient workspace.
  // Zero-out the workspace to avoid potential NaN propagation.
  if (over && lwork >= n + 2 * n * NBMIN) {
    nb = Math.floor((lwork - n) / (2 * n));
    nb = Math.min(nb, NBMAX);
    // DLASET('F', N, 1+2*NB, ZERO, ZERO, WORK, N)
    // The work array is treated as an N x (1+2*NB) column-major matrix with leading dim N.
    // WORK(I + J*N) for I=1..N, J=0..2*NB
    dlaset(
      -1, // 'F' = full, neither UPPER nor LOWER
      n,
      1 + 2 * nb,
      ZERO,
      ZERO,
      work,
      workOff,
      n
    );
  } else {
    nb = 1;
  }

  // Set the constants to control overflow.
  const unfl = dlamch(MACH_SFMIN);
  // const ovfl = ONE / unfl;
  const ulp = dlamch(MACH_PREC);
  const smlnum = unfl * (n / ulp);
  const bignum = (ONE - ulp) / smlnum;

  // Compute 1-norm of each column of strictly upper triangular
  // part of T to control overflow in triangular solver.
  // WORK(1) = ZERO
  work[workOff] = ZERO;
  for (let j = 2; j <= n; j++) {
    // WORK(J) = ZERO
    work[workOff + (j - 1)] = ZERO;
    for (let i = 1; i <= j - 1; i++) {
      // WORK(J) = WORK(J) + ABS(T(I,J))
      work[workOff + (j - 1)] += Math.abs(t[tOff + (i - 1) + (j - 1) * ldt]);
    }
  }

  // Local 2x2 array X(2,2), stored column-major as flat array of length 4
  const x = allocFloat64Array(4);

  // ISCOMPLEX array for blocked back-transform (0-based, size NBMAX)
  const iscomplex = new Int32Array(NBMAX);

  // Index IP is used to specify the real or complex eigenvalue:
  //   IP = 0, real eigenvalue,
  //        1, first  of conjugate complex pair: (wr,wi)
  //       -1, second of conjugate complex pair: (wr,wi)
  // ISCOMPLEX array stores IP for each column in current block.

  if (rightv) {
    // ============================================================
    // Compute right eigenvectors.
    //
    // IV is index of column in current block (1-based).
    // For complex right vector, uses IV-1 for real part and IV for complex part.
    // Non-blocked version always uses IV=2;
    // blocked     version starts with IV=NB, goes down to 1 or 2.
    // (Note the "0-th" column is used for 1-norms computed above.)
    let iv = 2;
    if (nb > 2) {
      iv = nb;
    }

    let ip = 0;
    let is_ = m; // 'is' is reserved in JS strict mode
    for (let ki = n; ki >= 1; ki--) {
      if (ip === -1) {
        // previous iteration (ki+1) was second of conjugate pair,
        // so this ki is first of conjugate pair; skip to end of loop
        ip = 1;
        continue;
      } else if (ki === 1) {
        // last column, so this ki must be real eigenvalue
        ip = 0;
      } else if (t[tOff + (ki - 1) + (ki - 2) * ldt] === ZERO) {
        // zero on sub-diagonal, so this ki is real eigenvalue
        // T(KI, KI-1) = t[tOff + (ki-1) + (ki-2)*ldt]
        ip = 0;
      } else {
        // non-zero on sub-diagonal, so this ki is second of conjugate pair
        ip = -1;
      }

      if (somev) {
        if (ip === 0) {
          if (!select[ki - 1]) continue;
        } else {
          if (!select[ki - 2]) continue;
        }
      }

      // Compute the KI-th eigenvalue (WR,WI).
      // WR = T(KI,KI)
      const wr = t[tOff + (ki - 1) + (ki - 1) * ldt];
      let wi = ZERO;
      if (ip !== 0) {
        // WI = SQRT(ABS(T(KI,KI-1))) * SQRT(ABS(T(KI-1,KI)))
        wi =
          Math.sqrt(Math.abs(t[tOff + (ki - 1) + (ki - 2) * ldt])) *
          Math.sqrt(Math.abs(t[tOff + (ki - 2) + (ki - 1) * ldt]));
      }
      const smin = Math.max(ulp * (Math.abs(wr) + Math.abs(wi)), smlnum);

      if (ip === 0) {
        // --------------------------------------------------------
        // Real right eigenvector

        // WORK(KI + IV*N) = ONE
        work[workOff + (ki - 1) + iv * n] = ONE;

        // Form right-hand side.
        for (let k = 1; k <= ki - 1; k++) {
          // WORK(K + IV*N) = -T(K,KI)
          work[workOff + (k - 1) + iv * n] =
            -t[tOff + (k - 1) + (ki - 1) * ldt];
        }

        // Solve upper quasi-triangular system:
        // [T(1:KI-1,1:KI-1) - WR]*X = SCALE*WORK.
        let jnxt = ki - 1;
        for (let j = ki - 1; j >= 1; j--) {
          if (j > jnxt) continue;
          let j1 = j;
          const j2 = j;
          jnxt = j - 1;
          if (j > 1) {
            // T(J,J-1)
            if (t[tOff + (j - 1) + (j - 2) * ldt] !== ZERO) {
              j1 = j - 1;
              jnxt = j - 2;
            }
          }

          if (j1 === j2) {
            // 1-by-1 diagonal block
            // DLALN2(.FALSE., 1, 1, SMIN, ONE, T(J,J), LDT, ONE, ONE,
            //        WORK(J+IV*N), N, WR, ZERO, X, 2, SCALE, XNORM, IERR)
            const result = dlaln2(
              false,
              1,
              1,
              smin,
              ONE,
              t,
              tOff + (j - 1) + (j - 1) * ldt,
              ldt,
              ONE,
              ONE,
              work,
              workOff + (j - 1) + iv * n,
              n,
              wr,
              ZERO,
              x,
              0,
              2
            );
            let scale = result.scale;
            const xnorm = result.xnorm;

            // Scale X(1,1) to avoid overflow when updating
            // the right-hand side.
            if (xnorm > ONE) {
              if (work[workOff + (j - 1)] > bignum / xnorm) {
                x[0] = x[0] / xnorm; // X(1,1)
                scale = scale / xnorm;
              }
            }

            // Scale if necessary
            if (scale !== ONE) {
              dscal(ki, scale, work, workOff + iv * n, 1);
            }
            // WORK(J+IV*N) = X(1,1)
            work[workOff + (j - 1) + iv * n] = x[0];

            // Update right-hand side
            // DAXPY(J-1, -X(1,1), T(1,J), 1, WORK(1+IV*N), 1)
            daxpy(
              j - 1,
              -x[0],
              t,
              tOff + (j - 1) * ldt,
              1,
              work,
              workOff + iv * n,
              1
            );
          } else {
            // 2-by-2 diagonal block
            // DLALN2(.FALSE., 2, 1, SMIN, ONE, T(J-1,J-1), LDT, ONE, ONE,
            //        WORK(J-1+IV*N), N, WR, ZERO, X, 2, SCALE, XNORM, IERR)
            const result = dlaln2(
              false,
              2,
              1,
              smin,
              ONE,
              t,
              tOff + (j - 2) + (j - 2) * ldt,
              ldt,
              ONE,
              ONE,
              work,
              workOff + (j - 2) + iv * n,
              n,
              wr,
              ZERO,
              x,
              0,
              2
            );
            let scale = result.scale;
            const xnorm = result.xnorm;

            // Scale X(1,1) and X(2,1) to avoid overflow when
            // updating the right-hand side.
            if (xnorm > ONE) {
              const beta = Math.max(
                work[workOff + (j - 2)],
                work[workOff + (j - 1)]
              );
              if (beta > bignum / xnorm) {
                x[0] = x[0] / xnorm; // X(1,1)
                x[1] = x[1] / xnorm; // X(2,1)
                scale = scale / xnorm;
              }
            }

            // Scale if necessary
            if (scale !== ONE) {
              dscal(ki, scale, work, workOff + iv * n, 1);
            }
            // WORK(J-1+IV*N) = X(1,1)
            work[workOff + (j - 2) + iv * n] = x[0];
            // WORK(J+IV*N) = X(2,1)
            work[workOff + (j - 1) + iv * n] = x[1];

            // Update right-hand side
            // DAXPY(J-2, -X(1,1), T(1,J-1), 1, WORK(1+IV*N), 1)
            daxpy(
              j - 2,
              -x[0],
              t,
              tOff + (j - 2) * ldt,
              1,
              work,
              workOff + iv * n,
              1
            );
            // DAXPY(J-2, -X(2,1), T(1,J), 1, WORK(1+IV*N), 1)
            daxpy(
              j - 2,
              -x[1],
              t,
              tOff + (j - 1) * ldt,
              1,
              work,
              workOff + iv * n,
              1
            );
          }
        }

        // Copy the vector x or Q*x to VR and normalize.
        if (!over) {
          // no back-transform: copy x to VR and normalize.
          // DCOPY(KI, WORK(1+IV*N), 1, VR(1,IS), 1)
          dcopy(ki, work, workOff + iv * n, 1, vr, vrOff + (is_ - 1) * ldvr, 1);

          // II = IDAMAX(KI, VR(1,IS), 1)
          const ii = idamax(ki, vr, vrOff + (is_ - 1) * ldvr, 1);
          const remax = ONE / Math.abs(vr[vrOff + (ii - 1) + (is_ - 1) * ldvr]);
          dscal(ki, remax, vr, vrOff + (is_ - 1) * ldvr, 1);

          for (let k = ki + 1; k <= n; k++) {
            // VR(K,IS) = ZERO
            vr[vrOff + (k - 1) + (is_ - 1) * ldvr] = ZERO;
          }
        } else if (nb === 1) {
          // version 1: back-transform each vector with GEMV, Q*x.
          if (ki > 1) {
            // DGEMV('N', N, KI-1, ONE, VR, LDVR, WORK(1+IV*N), 1,
            //        WORK(KI+IV*N), VR(1,KI), 1)
            dgemv(
              NOTRANS,
              n,
              ki - 1,
              ONE,
              vr,
              vrOff,
              ldvr,
              work,
              workOff + iv * n,
              1,
              work[workOff + (ki - 1) + iv * n],
              vr,
              vrOff + (ki - 1) * ldvr,
              1
            );
          }

          // II = IDAMAX(N, VR(1,KI), 1)
          const ii = idamax(n, vr, vrOff + (ki - 1) * ldvr, 1);
          const remax = ONE / Math.abs(vr[vrOff + (ii - 1) + (ki - 1) * ldvr]);
          dscal(n, remax, vr, vrOff + (ki - 1) * ldvr, 1);
        } else {
          // version 2: back-transform block of vectors with GEMM
          // zero out below vector
          for (let k = ki + 1; k <= n; k++) {
            work[workOff + (k - 1) + iv * n] = ZERO;
          }
          iscomplex[iv - 1] = ip;
          // back-transform and normalization is done below
        }
      } else {
        // --------------------------------------------------------
        // Complex right eigenvector.
        //
        // Initial solve
        // [ (T(KI-1,KI-1) T(KI-1,KI)) - (WR + I*WI) ]*X = 0.
        // [ (T(KI,  KI-1) T(KI,  KI))               ]

        // T(KI-1,KI) = t[tOff + (ki-2) + (ki-1)*ldt]
        // T(KI,KI-1) = t[tOff + (ki-1) + (ki-2)*ldt]
        if (
          Math.abs(t[tOff + (ki - 2) + (ki - 1) * ldt]) >=
          Math.abs(t[tOff + (ki - 1) + (ki - 2) * ldt])
        ) {
          // WORK(KI-1 + (IV-1)*N) = ONE
          work[workOff + (ki - 2) + (iv - 1) * n] = ONE;
          // WORK(KI + IV*N) = WI / T(KI-1,KI)
          work[workOff + (ki - 1) + iv * n] =
            wi / t[tOff + (ki - 2) + (ki - 1) * ldt];
        } else {
          // WORK(KI-1 + (IV-1)*N) = -WI / T(KI,KI-1)
          work[workOff + (ki - 2) + (iv - 1) * n] =
            -wi / t[tOff + (ki - 1) + (ki - 2) * ldt];
          // WORK(KI + IV*N) = ONE
          work[workOff + (ki - 1) + iv * n] = ONE;
        }
        // WORK(KI + (IV-1)*N) = ZERO
        work[workOff + (ki - 1) + (iv - 1) * n] = ZERO;
        // WORK(KI-1 + IV*N) = ZERO
        work[workOff + (ki - 2) + iv * n] = ZERO;

        // Form right-hand side.
        for (let k = 1; k <= ki - 2; k++) {
          // WORK(K+(IV-1)*N) = -WORK(KI-1+(IV-1)*N)*T(K,KI-1)
          work[workOff + (k - 1) + (iv - 1) * n] =
            -work[workOff + (ki - 2) + (iv - 1) * n] *
            t[tOff + (k - 1) + (ki - 2) * ldt];
          // WORK(K+IV*N) = -WORK(KI+IV*N)*T(K,KI)
          work[workOff + (k - 1) + iv * n] =
            -work[workOff + (ki - 1) + iv * n] *
            t[tOff + (k - 1) + (ki - 1) * ldt];
        }

        // Solve upper quasi-triangular system:
        // [T(1:KI-2,1:KI-2) - (WR+i*WI)]*X = SCALE*(WORK+i*WORK2)
        let jnxt = ki - 2;
        for (let j = ki - 2; j >= 1; j--) {
          if (j > jnxt) continue;
          let j1 = j;
          const j2 = j;
          jnxt = j - 1;
          if (j > 1) {
            if (t[tOff + (j - 1) + (j - 2) * ldt] !== ZERO) {
              j1 = j - 1;
              jnxt = j - 2;
            }
          }

          if (j1 === j2) {
            // 1-by-1 diagonal block
            const result = dlaln2(
              false,
              1,
              2,
              smin,
              ONE,
              t,
              tOff + (j - 1) + (j - 1) * ldt,
              ldt,
              ONE,
              ONE,
              work,
              workOff + (j - 1) + (iv - 1) * n,
              n,
              wr,
              wi,
              x,
              0,
              2
            );
            let scale = result.scale;
            const xnorm = result.xnorm;

            // Scale X(1,1) and X(1,2) to avoid overflow when
            // updating the right-hand side.
            if (xnorm > ONE) {
              if (work[workOff + (j - 1)] > bignum / xnorm) {
                x[0] = x[0] / xnorm; // X(1,1)
                x[2] = x[2] / xnorm; // X(1,2)
                scale = scale / xnorm;
              }
            }

            // Scale if necessary
            if (scale !== ONE) {
              dscal(ki, scale, work, workOff + (iv - 1) * n, 1);
              dscal(ki, scale, work, workOff + iv * n, 1);
            }
            // WORK(J+(IV-1)*N) = X(1,1)
            work[workOff + (j - 1) + (iv - 1) * n] = x[0];
            // WORK(J+IV*N) = X(1,2)
            work[workOff + (j - 1) + iv * n] = x[2];

            // Update the right-hand side
            // DAXPY(J-1, -X(1,1), T(1,J), 1, WORK(1+(IV-1)*N), 1)
            daxpy(
              j - 1,
              -x[0],
              t,
              tOff + (j - 1) * ldt,
              1,
              work,
              workOff + (iv - 1) * n,
              1
            );
            // DAXPY(J-1, -X(1,2), T(1,J), 1, WORK(1+IV*N), 1)
            daxpy(
              j - 1,
              -x[2],
              t,
              tOff + (j - 1) * ldt,
              1,
              work,
              workOff + iv * n,
              1
            );
          } else {
            // 2-by-2 diagonal block
            const result = dlaln2(
              false,
              2,
              2,
              smin,
              ONE,
              t,
              tOff + (j - 2) + (j - 2) * ldt,
              ldt,
              ONE,
              ONE,
              work,
              workOff + (j - 2) + (iv - 1) * n,
              n,
              wr,
              wi,
              x,
              0,
              2
            );
            let scale = result.scale;
            const xnorm = result.xnorm;

            // Scale X to avoid overflow when updating
            // the right-hand side.
            if (xnorm > ONE) {
              const beta = Math.max(
                work[workOff + (j - 2)],
                work[workOff + (j - 1)]
              );
              if (beta > bignum / xnorm) {
                const rec = ONE / xnorm;
                x[0] = x[0] * rec; // X(1,1)
                x[2] = x[2] * rec; // X(1,2)
                x[1] = x[1] * rec; // X(2,1)
                x[3] = x[3] * rec; // X(2,2)
                scale = scale * rec;
              }
            }

            // Scale if necessary
            if (scale !== ONE) {
              dscal(ki, scale, work, workOff + (iv - 1) * n, 1);
              dscal(ki, scale, work, workOff + iv * n, 1);
            }
            // WORK(J-1+(IV-1)*N) = X(1,1)
            work[workOff + (j - 2) + (iv - 1) * n] = x[0];
            // WORK(J+(IV-1)*N) = X(2,1)
            work[workOff + (j - 1) + (iv - 1) * n] = x[1];
            // WORK(J-1+IV*N) = X(1,2)
            work[workOff + (j - 2) + iv * n] = x[2];
            // WORK(J+IV*N) = X(2,2)
            work[workOff + (j - 1) + iv * n] = x[3];

            // Update the right-hand side
            // DAXPY(J-2, -X(1,1), T(1,J-1), 1, WORK(1+(IV-1)*N), 1)
            daxpy(
              j - 2,
              -x[0],
              t,
              tOff + (j - 2) * ldt,
              1,
              work,
              workOff + (iv - 1) * n,
              1
            );
            // DAXPY(J-2, -X(2,1), T(1,J), 1, WORK(1+(IV-1)*N), 1)
            daxpy(
              j - 2,
              -x[1],
              t,
              tOff + (j - 1) * ldt,
              1,
              work,
              workOff + (iv - 1) * n,
              1
            );
            // DAXPY(J-2, -X(1,2), T(1,J-1), 1, WORK(1+IV*N), 1)
            daxpy(
              j - 2,
              -x[2],
              t,
              tOff + (j - 2) * ldt,
              1,
              work,
              workOff + iv * n,
              1
            );
            // DAXPY(J-2, -X(2,2), T(1,J), 1, WORK(1+IV*N), 1)
            daxpy(
              j - 2,
              -x[3],
              t,
              tOff + (j - 1) * ldt,
              1,
              work,
              workOff + iv * n,
              1
            );
          }
        }

        // Copy the vector x or Q*x to VR and normalize.
        if (!over) {
          // no back-transform: copy x to VR and normalize.
          // DCOPY(KI, WORK(1+(IV-1)*N), 1, VR(1,IS-1), 1)
          dcopy(
            ki,
            work,
            workOff + (iv - 1) * n,
            1,
            vr,
            vrOff + (is_ - 2) * ldvr,
            1
          );
          // DCOPY(KI, WORK(1+IV*N), 1, VR(1,IS), 1)
          dcopy(ki, work, workOff + iv * n, 1, vr, vrOff + (is_ - 1) * ldvr, 1);

          let emax = ZERO;
          for (let k = 1; k <= ki; k++) {
            emax = Math.max(
              emax,
              Math.abs(vr[vrOff + (k - 1) + (is_ - 2) * ldvr]) +
                Math.abs(vr[vrOff + (k - 1) + (is_ - 1) * ldvr])
            );
          }
          const remax = ONE / emax;
          dscal(ki, remax, vr, vrOff + (is_ - 2) * ldvr, 1);
          dscal(ki, remax, vr, vrOff + (is_ - 1) * ldvr, 1);

          for (let k = ki + 1; k <= n; k++) {
            vr[vrOff + (k - 1) + (is_ - 2) * ldvr] = ZERO;
            vr[vrOff + (k - 1) + (is_ - 1) * ldvr] = ZERO;
          }
        } else if (nb === 1) {
          // version 1: back-transform each vector with GEMV, Q*x.
          if (ki > 2) {
            // DGEMV('N', N, KI-2, ONE, VR, LDVR, WORK(1+(IV-1)*N), 1,
            //        WORK(KI-1+(IV-1)*N), VR(1,KI-1), 1)
            dgemv(
              NOTRANS,
              n,
              ki - 2,
              ONE,
              vr,
              vrOff,
              ldvr,
              work,
              workOff + (iv - 1) * n,
              1,
              work[workOff + (ki - 2) + (iv - 1) * n],
              vr,
              vrOff + (ki - 2) * ldvr,
              1
            );
            // DGEMV('N', N, KI-2, ONE, VR, LDVR, WORK(1+IV*N), 1,
            //        WORK(KI+IV*N), VR(1,KI), 1)
            dgemv(
              NOTRANS,
              n,
              ki - 2,
              ONE,
              vr,
              vrOff,
              ldvr,
              work,
              workOff + iv * n,
              1,
              work[workOff + (ki - 1) + iv * n],
              vr,
              vrOff + (ki - 1) * ldvr,
              1
            );
          } else {
            // DSCAL(N, WORK(KI-1+(IV-1)*N), VR(1,KI-1), 1)
            dscal(
              n,
              work[workOff + (ki - 2) + (iv - 1) * n],
              vr,
              vrOff + (ki - 2) * ldvr,
              1
            );
            // DSCAL(N, WORK(KI+IV*N), VR(1,KI), 1)
            dscal(
              n,
              work[workOff + (ki - 1) + iv * n],
              vr,
              vrOff + (ki - 1) * ldvr,
              1
            );
          }

          let emax = ZERO;
          for (let k = 1; k <= n; k++) {
            emax = Math.max(
              emax,
              Math.abs(vr[vrOff + (k - 1) + (ki - 2) * ldvr]) +
                Math.abs(vr[vrOff + (k - 1) + (ki - 1) * ldvr])
            );
          }
          const remax = ONE / emax;
          dscal(n, remax, vr, vrOff + (ki - 2) * ldvr, 1);
          dscal(n, remax, vr, vrOff + (ki - 1) * ldvr, 1);
        } else {
          // version 2: back-transform block of vectors with GEMM
          // zero out below vector
          for (let k = ki + 1; k <= n; k++) {
            work[workOff + (k - 1) + (iv - 1) * n] = ZERO;
            work[workOff + (k - 1) + iv * n] = ZERO;
          }
          iscomplex[iv - 2] = -ip; // ISCOMPLEX(IV-1) in 1-based
          iscomplex[iv - 1] = ip; // ISCOMPLEX(IV) in 1-based
          iv = iv - 1;
          // back-transform and normalization is done below
        }
      }

      if (nb > 1) {
        // --------------------------------------------------------
        // Blocked version of back-transform
        // For complex case, KI2 includes both vectors (KI-1 and KI)
        let ki2: number;
        if (ip === 0) {
          ki2 = ki;
        } else {
          ki2 = ki - 1;
        }

        // Columns IV:NB of work are valid vectors.
        // When the number of vectors stored reaches NB-1 or NB,
        // or if this was last vector, do the GEMM
        if (iv <= 2 || ki2 === 1) {
          // DGEMM('N','N', N, NB-IV+1, KI2+NB-IV, ONE,
          //       VR, LDVR, WORK(1+IV*N), N, ZERO, WORK(1+(NB+IV)*N), N)
          dgemm(
            NOTRANS,
            NOTRANS,
            n,
            nb - iv + 1,
            ki2 + nb - iv,
            ONE,
            vr,
            vrOff,
            ldvr,
            work,
            workOff + iv * n,
            n,
            ZERO,
            work,
            workOff + (nb + iv) * n,
            n
          );
          // normalize vectors
          let remax = ONE;
          for (let k = iv; k <= nb; k++) {
            if (iscomplex[k - 1] === 0) {
              // real eigenvector
              const ii = idamax(n, work, workOff + (nb + k) * n, 1);
              remax = ONE / Math.abs(work[workOff + (ii - 1) + (nb + k) * n]);
            } else if (iscomplex[k - 1] === 1) {
              // first eigenvector of conjugate pair
              let emax = ZERO;
              for (let ii = 1; ii <= n; ii++) {
                emax = Math.max(
                  emax,
                  Math.abs(work[workOff + (ii - 1) + (nb + k) * n]) +
                    Math.abs(work[workOff + (ii - 1) + (nb + k + 1) * n])
                );
              }
              remax = ONE / emax;
            }
            // else if iscomplex[k-1] === -1:
            //   second eigenvector of conjugate pair
            //   reuse same REMAX as previous k
            dscal(n, remax, work, workOff + (nb + k) * n, 1);
          }
          // DLACPY('F', N, NB-IV+1, WORK(1+(NB+IV)*N), N, VR(1,KI2), LDVR)
          dlacpy(
            -1, // 'F' = full
            n,
            nb - iv + 1,
            work,
            workOff + (nb + iv) * n,
            n,
            vr,
            vrOff + (ki2 - 1) * ldvr,
            ldvr
          );
          iv = nb;
        } else {
          iv = iv - 1;
        }
      } // blocked back-transform

      is_ = is_ - 1;
      if (ip !== 0) {
        is_ = is_ - 1;
      }
    }
  }

  if (leftv) {
    // ============================================================
    // Compute left eigenvectors.
    //
    // IV is index of column in current block (1-based).
    // For complex left vector, uses IV for real part and IV+1 for complex part.
    // Non-blocked version always uses IV=1;
    // blocked     version starts with IV=1, goes up to NB-1 or NB.
    // (Note the "0-th" column is used for 1-norms computed above.)
    let iv = 1;
    let ip = 0;
    let is_ = 1;
    for (let ki = 1; ki <= n; ki++) {
      if (ip === 1) {
        // previous iteration (ki-1) was first of conjugate pair,
        // so this ki is second of conjugate pair; skip to end of loop
        ip = -1;
        continue;
      } else if (ki === n) {
        // last column, so this ki must be real eigenvalue
        ip = 0;
      } else if (t[tOff + ki + (ki - 1) * ldt] === ZERO) {
        // T(KI+1,KI) = t[tOff + ki + (ki-1)*ldt]
        // zero on sub-diagonal, so this ki is real eigenvalue
        ip = 0;
      } else {
        // non-zero on sub-diagonal, so this ki is first of conjugate pair
        ip = 1;
      }

      if (somev) {
        if (!select[ki - 1]) continue;
      }

      // Compute the KI-th eigenvalue (WR,WI).
      // WR = T(KI,KI)
      const wr = t[tOff + (ki - 1) + (ki - 1) * ldt];
      let wi = ZERO;
      if (ip !== 0) {
        // WI = SQRT(ABS(T(KI,KI+1))) * SQRT(ABS(T(KI+1,KI)))
        wi =
          Math.sqrt(Math.abs(t[tOff + (ki - 1) + ki * ldt])) *
          Math.sqrt(Math.abs(t[tOff + ki + (ki - 1) * ldt]));
      }
      const smin = Math.max(ulp * (Math.abs(wr) + Math.abs(wi)), smlnum);

      if (ip === 0) {
        // --------------------------------------------------------
        // Real left eigenvector

        // WORK(KI + IV*N) = ONE
        work[workOff + (ki - 1) + iv * n] = ONE;

        // Form right-hand side.
        for (let k = ki + 1; k <= n; k++) {
          // WORK(K + IV*N) = -T(KI, K)
          work[workOff + (k - 1) + iv * n] =
            -t[tOff + (ki - 1) + (k - 1) * ldt];
        }

        // Solve transposed quasi-triangular system:
        // [T(KI+1:N,KI+1:N) - WR]**T * X = SCALE*WORK
        let vmax = ONE;
        let vcrit = bignum;

        let jnxt = ki + 1;
        for (let j = ki + 1; j <= n; j++) {
          if (j < jnxt) continue;
          const j1 = j;
          let j2 = j;
          jnxt = j + 1;
          if (j < n) {
            // T(J+1,J)
            if (t[tOff + j + (j - 1) * ldt] !== ZERO) {
              j2 = j + 1;
              jnxt = j + 2;
            }
          }

          if (j1 === j2) {
            // 1-by-1 diagonal block
            // Scale if necessary to avoid overflow when forming
            // the right-hand side.
            if (work[workOff + (j - 1)] > vcrit) {
              const rec = ONE / vmax;
              dscal(n - ki + 1, rec, work, workOff + (ki - 1) + iv * n, 1);
              vmax = ONE;
              vcrit = bignum;
            }

            // WORK(J+IV*N) = WORK(J+IV*N) -
            //   DDOT(J-KI-1, T(KI+1,J), 1, WORK(KI+1+IV*N), 1)
            work[workOff + (j - 1) + iv * n] -= ddot(
              j - ki - 1,
              t,
              tOff + ki + (j - 1) * ldt,
              1,
              work,
              workOff + ki + iv * n,
              1
            );

            // Solve [T(J,J) - WR]**T * X = WORK
            const result = dlaln2(
              false,
              1,
              1,
              smin,
              ONE,
              t,
              tOff + (j - 1) + (j - 1) * ldt,
              ldt,
              ONE,
              ONE,
              work,
              workOff + (j - 1) + iv * n,
              n,
              wr,
              ZERO,
              x,
              0,
              2
            );
            const scale = result.scale;

            // Scale if necessary
            if (scale !== ONE) {
              dscal(n - ki + 1, scale, work, workOff + (ki - 1) + iv * n, 1);
            }
            // WORK(J+IV*N) = X(1,1)
            work[workOff + (j - 1) + iv * n] = x[0];
            vmax = Math.max(Math.abs(work[workOff + (j - 1) + iv * n]), vmax);
            vcrit = bignum / vmax;
          } else {
            // 2-by-2 diagonal block
            // Scale if necessary to avoid overflow when forming
            // the right-hand side.
            const beta = Math.max(work[workOff + (j - 1)], work[workOff + j]);
            if (beta > vcrit) {
              const rec = ONE / vmax;
              dscal(n - ki + 1, rec, work, workOff + (ki - 1) + iv * n, 1);
              vmax = ONE;
              vcrit = bignum;
            }

            // WORK(J+IV*N) = WORK(J+IV*N) -
            //   DDOT(J-KI-1, T(KI+1,J), 1, WORK(KI+1+IV*N), 1)
            work[workOff + (j - 1) + iv * n] -= ddot(
              j - ki - 1,
              t,
              tOff + ki + (j - 1) * ldt,
              1,
              work,
              workOff + ki + iv * n,
              1
            );

            // WORK(J+1+IV*N) = WORK(J+1+IV*N) -
            //   DDOT(J-KI-1, T(KI+1,J+1), 1, WORK(KI+1+IV*N), 1)
            work[workOff + j + iv * n] -= ddot(
              j - ki - 1,
              t,
              tOff + ki + j * ldt,
              1,
              work,
              workOff + ki + iv * n,
              1
            );

            // Solve
            // [T(J,J)-WR   T(J,J+1)     ]**T * X = SCALE*( WORK1 )
            // [T(J+1,J)    T(J+1,J+1)-WR]                ( WORK2 )
            const result = dlaln2(
              true,
              2,
              1,
              smin,
              ONE,
              t,
              tOff + (j - 1) + (j - 1) * ldt,
              ldt,
              ONE,
              ONE,
              work,
              workOff + (j - 1) + iv * n,
              n,
              wr,
              ZERO,
              x,
              0,
              2
            );
            const scale = result.scale;

            // Scale if necessary
            if (scale !== ONE) {
              dscal(n - ki + 1, scale, work, workOff + (ki - 1) + iv * n, 1);
            }
            // WORK(J+IV*N) = X(1,1)
            work[workOff + (j - 1) + iv * n] = x[0];
            // WORK(J+1+IV*N) = X(2,1)
            work[workOff + j + iv * n] = x[1];

            vmax = Math.max(
              Math.abs(work[workOff + (j - 1) + iv * n]),
              Math.abs(work[workOff + j + iv * n]),
              vmax
            );
            vcrit = bignum / vmax;
          }
        }

        // Copy the vector x or Q*x to VL and normalize.
        if (!over) {
          // no back-transform: copy x to VL and normalize.
          // DCOPY(N-KI+1, WORK(KI+IV*N), 1, VL(KI,IS), 1)
          dcopy(
            n - ki + 1,
            work,
            workOff + (ki - 1) + iv * n,
            1,
            vl,
            vlOff + (ki - 1) + (is_ - 1) * ldvl,
            1
          );

          // II = IDAMAX(N-KI+1, VL(KI,IS), 1) + KI - 1
          const ii =
            idamax(n - ki + 1, vl, vlOff + (ki - 1) + (is_ - 1) * ldvl, 1) +
            ki -
            1;
          const remax = ONE / Math.abs(vl[vlOff + (ii - 1) + (is_ - 1) * ldvl]);
          dscal(n - ki + 1, remax, vl, vlOff + (ki - 1) + (is_ - 1) * ldvl, 1);

          for (let k = 1; k <= ki - 1; k++) {
            // VL(K,IS) = ZERO
            vl[vlOff + (k - 1) + (is_ - 1) * ldvl] = ZERO;
          }
        } else if (nb === 1) {
          // version 1: back-transform each vector with GEMV, Q*x.
          if (ki < n) {
            // DGEMV('N', N, N-KI, ONE, VL(1,KI+1), LDVL,
            //        WORK(KI+1+IV*N), 1, WORK(KI+IV*N), VL(1,KI), 1)
            dgemv(
              NOTRANS,
              n,
              n - ki,
              ONE,
              vl,
              vlOff + ki * ldvl,
              ldvl,
              work,
              workOff + ki + iv * n,
              1,
              work[workOff + (ki - 1) + iv * n],
              vl,
              vlOff + (ki - 1) * ldvl,
              1
            );
          }

          // II = IDAMAX(N, VL(1,KI), 1)
          const ii = idamax(n, vl, vlOff + (ki - 1) * ldvl, 1);
          const remax = ONE / Math.abs(vl[vlOff + (ii - 1) + (ki - 1) * ldvl]);
          dscal(n, remax, vl, vlOff + (ki - 1) * ldvl, 1);
        } else {
          // version 2: back-transform block of vectors with GEMM
          // zero out above vector
          for (let k = 1; k <= ki - 1; k++) {
            work[workOff + (k - 1) + iv * n] = ZERO;
          }
          iscomplex[iv - 1] = ip;
          // back-transform and normalization is done below
        }
      } else {
        // --------------------------------------------------------
        // Complex left eigenvector.
        //
        // Initial solve:
        // [ (T(KI,KI)    T(KI,KI+1) )**T - (WR - I*WI) ]*X = 0.
        // [ (T(KI+1,KI) T(KI+1,KI+1))                   ]

        // T(KI,KI+1) = t[tOff + (ki-1) + ki*ldt]
        // T(KI+1,KI) = t[tOff + ki + (ki-1)*ldt]
        if (
          Math.abs(t[tOff + (ki - 1) + ki * ldt]) >=
          Math.abs(t[tOff + ki + (ki - 1) * ldt])
        ) {
          // WORK(KI + IV*N) = WI / T(KI,KI+1)
          work[workOff + (ki - 1) + iv * n] =
            wi / t[tOff + (ki - 1) + ki * ldt];
          // WORK(KI+1 + (IV+1)*N) = ONE
          work[workOff + ki + (iv + 1) * n] = ONE;
        } else {
          // WORK(KI + IV*N) = ONE
          work[workOff + (ki - 1) + iv * n] = ONE;
          // WORK(KI+1 + (IV+1)*N) = -WI / T(KI+1,KI)
          work[workOff + ki + (iv + 1) * n] =
            -wi / t[tOff + ki + (ki - 1) * ldt];
        }
        // WORK(KI+1 + IV*N) = ZERO
        work[workOff + ki + iv * n] = ZERO;
        // WORK(KI + (IV+1)*N) = ZERO
        work[workOff + (ki - 1) + (iv + 1) * n] = ZERO;

        // Form right-hand side.
        for (let k = ki + 2; k <= n; k++) {
          // WORK(K+(IV)*N) = -WORK(KI+(IV)*N)*T(KI,K)
          work[workOff + (k - 1) + iv * n] =
            -work[workOff + (ki - 1) + iv * n] *
            t[tOff + (ki - 1) + (k - 1) * ldt];
          // WORK(K+(IV+1)*N) = -WORK(KI+1+(IV+1)*N)*T(KI+1,K)
          work[workOff + (k - 1) + (iv + 1) * n] =
            -work[workOff + ki + (iv + 1) * n] * t[tOff + ki + (k - 1) * ldt];
        }

        // Solve transposed quasi-triangular system:
        // [T(KI+2:N,KI+2:N)**T - (WR-i*WI)]*X = WORK1+i*WORK2
        let vmax = ONE;
        let vcrit = bignum;

        let jnxt = ki + 2;
        for (let j = ki + 2; j <= n; j++) {
          if (j < jnxt) continue;
          const j1 = j;
          let j2 = j;
          jnxt = j + 1;
          if (j < n) {
            if (t[tOff + j + (j - 1) * ldt] !== ZERO) {
              j2 = j + 1;
              jnxt = j + 2;
            }
          }

          if (j1 === j2) {
            // 1-by-1 diagonal block
            // Scale if necessary to avoid overflow when
            // forming the right-hand side elements.
            if (work[workOff + (j - 1)] > vcrit) {
              const rec = ONE / vmax;
              dscal(n - ki + 1, rec, work, workOff + (ki - 1) + iv * n, 1);
              dscal(
                n - ki + 1,
                rec,
                work,
                workOff + (ki - 1) + (iv + 1) * n,
                1
              );
              vmax = ONE;
              vcrit = bignum;
            }

            // WORK(J+IV*N) = WORK(J+IV*N) -
            //   DDOT(J-KI-2, T(KI+2,J), 1, WORK(KI+2+IV*N), 1)
            work[workOff + (j - 1) + iv * n] -= ddot(
              j - ki - 2,
              t,
              tOff + (ki + 1) + (j - 1) * ldt,
              1,
              work,
              workOff + (ki + 1) + iv * n,
              1
            );
            // WORK(J+(IV+1)*N) = WORK(J+(IV+1)*N) -
            //   DDOT(J-KI-2, T(KI+2,J), 1, WORK(KI+2+(IV+1)*N), 1)
            work[workOff + (j - 1) + (iv + 1) * n] -= ddot(
              j - ki - 2,
              t,
              tOff + (ki + 1) + (j - 1) * ldt,
              1,
              work,
              workOff + (ki + 1) + (iv + 1) * n,
              1
            );

            // Solve [T(J,J)-(WR-i*WI)]*(X11+i*X12)= WK+I*WK2
            const result = dlaln2(
              false,
              1,
              2,
              smin,
              ONE,
              t,
              tOff + (j - 1) + (j - 1) * ldt,
              ldt,
              ONE,
              ONE,
              work,
              workOff + (j - 1) + iv * n,
              n,
              wr,
              -wi,
              x,
              0,
              2
            );
            const scale = result.scale;

            // Scale if necessary
            if (scale !== ONE) {
              dscal(n - ki + 1, scale, work, workOff + (ki - 1) + iv * n, 1);
              dscal(
                n - ki + 1,
                scale,
                work,
                workOff + (ki - 1) + (iv + 1) * n,
                1
              );
            }
            // WORK(J+IV*N) = X(1,1)
            work[workOff + (j - 1) + iv * n] = x[0];
            // WORK(J+(IV+1)*N) = X(1,2)
            work[workOff + (j - 1) + (iv + 1) * n] = x[2];
            vmax = Math.max(
              Math.abs(work[workOff + (j - 1) + iv * n]),
              Math.abs(work[workOff + (j - 1) + (iv + 1) * n]),
              vmax
            );
            vcrit = bignum / vmax;
          } else {
            // 2-by-2 diagonal block
            // Scale if necessary to avoid overflow when forming
            // the right-hand side elements.
            const beta = Math.max(work[workOff + (j - 1)], work[workOff + j]);
            if (beta > vcrit) {
              const rec = ONE / vmax;
              dscal(n - ki + 1, rec, work, workOff + (ki - 1) + iv * n, 1);
              dscal(
                n - ki + 1,
                rec,
                work,
                workOff + (ki - 1) + (iv + 1) * n,
                1
              );
              vmax = ONE;
              vcrit = bignum;
            }

            // WORK(J+(IV)*N) = WORK(J+(IV)*N) -
            //   DDOT(J-KI-2, T(KI+2,J), 1, WORK(KI+2+(IV)*N), 1)
            work[workOff + (j - 1) + iv * n] -= ddot(
              j - ki - 2,
              t,
              tOff + (ki + 1) + (j - 1) * ldt,
              1,
              work,
              workOff + (ki + 1) + iv * n,
              1
            );

            // WORK(J+(IV+1)*N) = WORK(J+(IV+1)*N) -
            //   DDOT(J-KI-2, T(KI+2,J), 1, WORK(KI+2+(IV+1)*N), 1)
            work[workOff + (j - 1) + (iv + 1) * n] -= ddot(
              j - ki - 2,
              t,
              tOff + (ki + 1) + (j - 1) * ldt,
              1,
              work,
              workOff + (ki + 1) + (iv + 1) * n,
              1
            );

            // WORK(J+1+(IV)*N) = WORK(J+1+(IV)*N) -
            //   DDOT(J-KI-2, T(KI+2,J+1), 1, WORK(KI+2+(IV)*N), 1)
            work[workOff + j + iv * n] -= ddot(
              j - ki - 2,
              t,
              tOff + (ki + 1) + j * ldt,
              1,
              work,
              workOff + (ki + 1) + iv * n,
              1
            );

            // WORK(J+1+(IV+1)*N) = WORK(J+1+(IV+1)*N) -
            //   DDOT(J-KI-2, T(KI+2,J+1), 1, WORK(KI+2+(IV+1)*N), 1)
            work[workOff + j + (iv + 1) * n] -= ddot(
              j - ki - 2,
              t,
              tOff + (ki + 1) + j * ldt,
              1,
              work,
              workOff + (ki + 1) + (iv + 1) * n,
              1
            );

            // Solve 2-by-2 complex linear equation
            // [ (T(j,j)   T(j,j+1)  )**T - (wr-i*wi)*I ]*X = SCALE*B
            // [ (T(j+1,j) T(j+1,j+1))                  ]
            const result = dlaln2(
              true,
              2,
              2,
              smin,
              ONE,
              t,
              tOff + (j - 1) + (j - 1) * ldt,
              ldt,
              ONE,
              ONE,
              work,
              workOff + (j - 1) + iv * n,
              n,
              wr,
              -wi,
              x,
              0,
              2
            );
            const scale = result.scale;

            // Scale if necessary
            if (scale !== ONE) {
              dscal(n - ki + 1, scale, work, workOff + (ki - 1) + iv * n, 1);
              dscal(
                n - ki + 1,
                scale,
                work,
                workOff + (ki - 1) + (iv + 1) * n,
                1
              );
            }
            // WORK(J+(IV)*N) = X(1,1)
            work[workOff + (j - 1) + iv * n] = x[0];
            // WORK(J+(IV+1)*N) = X(1,2)
            work[workOff + (j - 1) + (iv + 1) * n] = x[2];
            // WORK(J+1+(IV)*N) = X(2,1)
            work[workOff + j + iv * n] = x[1];
            // WORK(J+1+(IV+1)*N) = X(2,2)
            work[workOff + j + (iv + 1) * n] = x[3];
            vmax = Math.max(
              Math.abs(x[0]),
              Math.abs(x[2]),
              Math.abs(x[1]),
              Math.abs(x[3]),
              vmax
            );
            vcrit = bignum / vmax;
          }
        }

        // Copy the vector x or Q*x to VL and normalize.
        if (!over) {
          // no back-transform: copy x to VL and normalize.
          // DCOPY(N-KI+1, WORK(KI+(IV)*N), 1, VL(KI,IS), 1)
          dcopy(
            n - ki + 1,
            work,
            workOff + (ki - 1) + iv * n,
            1,
            vl,
            vlOff + (ki - 1) + (is_ - 1) * ldvl,
            1
          );
          // DCOPY(N-KI+1, WORK(KI+(IV+1)*N), 1, VL(KI,IS+1), 1)
          dcopy(
            n - ki + 1,
            work,
            workOff + (ki - 1) + (iv + 1) * n,
            1,
            vl,
            vlOff + (ki - 1) + is_ * ldvl,
            1
          );

          let emax = ZERO;
          for (let k = ki; k <= n; k++) {
            emax = Math.max(
              emax,
              Math.abs(vl[vlOff + (k - 1) + (is_ - 1) * ldvl]) +
                Math.abs(vl[vlOff + (k - 1) + is_ * ldvl])
            );
          }
          const remax = ONE / emax;
          dscal(n - ki + 1, remax, vl, vlOff + (ki - 1) + (is_ - 1) * ldvl, 1);
          dscal(n - ki + 1, remax, vl, vlOff + (ki - 1) + is_ * ldvl, 1);

          for (let k = 1; k <= ki - 1; k++) {
            vl[vlOff + (k - 1) + (is_ - 1) * ldvl] = ZERO;
            vl[vlOff + (k - 1) + is_ * ldvl] = ZERO;
          }
        } else if (nb === 1) {
          // version 1: back-transform each vector with GEMV, Q*x.
          if (ki < n - 1) {
            // DGEMV('N', N, N-KI-1, ONE, VL(1,KI+2), LDVL,
            //        WORK(KI+2+(IV)*N), 1, WORK(KI+(IV)*N), VL(1,KI), 1)
            dgemv(
              NOTRANS,
              n,
              n - ki - 1,
              ONE,
              vl,
              vlOff + (ki + 1) * ldvl,
              ldvl,
              work,
              workOff + (ki + 1) + iv * n,
              1,
              work[workOff + (ki - 1) + iv * n],
              vl,
              vlOff + (ki - 1) * ldvl,
              1
            );
            // DGEMV('N', N, N-KI-1, ONE, VL(1,KI+2), LDVL,
            //        WORK(KI+2+(IV+1)*N), 1, WORK(KI+1+(IV+1)*N), VL(1,KI+1), 1)
            dgemv(
              NOTRANS,
              n,
              n - ki - 1,
              ONE,
              vl,
              vlOff + (ki + 1) * ldvl,
              ldvl,
              work,
              workOff + (ki + 1) + (iv + 1) * n,
              1,
              work[workOff + ki + (iv + 1) * n],
              vl,
              vlOff + ki * ldvl,
              1
            );
          } else {
            // DSCAL(N, WORK(KI+(IV)*N), VL(1,KI), 1)
            dscal(
              n,
              work[workOff + (ki - 1) + iv * n],
              vl,
              vlOff + (ki - 1) * ldvl,
              1
            );
            // DSCAL(N, WORK(KI+1+(IV+1)*N), VL(1,KI+1), 1)
            dscal(
              n,
              work[workOff + ki + (iv + 1) * n],
              vl,
              vlOff + ki * ldvl,
              1
            );
          }

          let emax = ZERO;
          for (let k = 1; k <= n; k++) {
            emax = Math.max(
              emax,
              Math.abs(vl[vlOff + (k - 1) + (ki - 1) * ldvl]) +
                Math.abs(vl[vlOff + (k - 1) + ki * ldvl])
            );
          }
          const remax = ONE / emax;
          dscal(n, remax, vl, vlOff + (ki - 1) * ldvl, 1);
          dscal(n, remax, vl, vlOff + ki * ldvl, 1);
        } else {
          // version 2: back-transform block of vectors with GEMM
          // zero out above vector
          for (let k = 1; k <= ki - 1; k++) {
            work[workOff + (k - 1) + iv * n] = ZERO;
            work[workOff + (k - 1) + (iv + 1) * n] = ZERO;
          }
          iscomplex[iv - 1] = ip; // ISCOMPLEX(IV) = IP
          iscomplex[iv] = -ip; // ISCOMPLEX(IV+1) = -IP
          iv = iv + 1;
          // back-transform and normalization is done below
        }
      }

      if (nb > 1) {
        // --------------------------------------------------------
        // Blocked version of back-transform
        // For complex case, KI2 includes both vectors (KI and KI+1)
        let ki2: number;
        if (ip === 0) {
          ki2 = ki;
        } else {
          ki2 = ki + 1;
        }

        // Columns 1:IV of work are valid vectors.
        // When the number of vectors stored reaches NB-1 or NB,
        // or if this was last vector, do the GEMM
        if (iv >= nb - 1 || ki2 === n) {
          // DGEMM('N','N', N, IV, N-KI2+IV, ONE,
          //       VL(1,KI2-IV+1), LDVL, WORK(KI2-IV+1+(1)*N), N,
          //       ZERO, WORK(1+(NB+1)*N), N)
          dgemm(
            NOTRANS,
            NOTRANS,
            n,
            iv,
            n - ki2 + iv,
            ONE,
            vl,
            vlOff + (ki2 - iv) * ldvl,
            ldvl,
            work,
            workOff + (ki2 - iv) + 1 * n,
            n,
            ZERO,
            work,
            workOff + (nb + 1) * n,
            n
          );
          // normalize vectors
          let remax = ONE;
          for (let k = 1; k <= iv; k++) {
            if (iscomplex[k - 1] === 0) {
              // real eigenvector
              const ii = idamax(n, work, workOff + (nb + k) * n, 1);
              remax = ONE / Math.abs(work[workOff + (ii - 1) + (nb + k) * n]);
            } else if (iscomplex[k - 1] === 1) {
              // first eigenvector of conjugate pair
              let emax = ZERO;
              for (let ii = 1; ii <= n; ii++) {
                emax = Math.max(
                  emax,
                  Math.abs(work[workOff + (ii - 1) + (nb + k) * n]) +
                    Math.abs(work[workOff + (ii - 1) + (nb + k + 1) * n])
                );
              }
              remax = ONE / emax;
            }
            // else if iscomplex[k-1] === -1:
            //   second eigenvector of conjugate pair
            //   reuse same REMAX as previous k
            dscal(n, remax, work, workOff + (nb + k) * n, 1);
          }
          // DLACPY('F', N, IV, WORK(1+(NB+1)*N), N, VL(1,KI2-IV+1), LDVL)
          dlacpy(
            -1, // 'F' = full
            n,
            iv,
            work,
            workOff + (nb + 1) * n,
            n,
            vl,
            vlOff + (ki2 - iv) * ldvl,
            ldvl
          );
          iv = 1;
        } else {
          iv = iv + 1;
        }
      } // blocked back-transform

      is_ = is_ + 1;
      if (ip !== 0) {
        is_ = is_ + 1;
      }
    }
  }

  return { m, info: 0 };
}
