// Translated from SRC/dlaexc.f
// DLAEXC swaps adjacent diagonal blocks T11 and T22 of order 1 or 2 in
// an upper quasi-triangular matrix T by an orthogonal similarity
// transformation.
//
// T must be in Schur canonical form, that is, block upper triangular
// with 1-by-1 and 2-by-2 diagonal blocks; each 2-by-2 diagonal block
// has its diagonal elements equal and its off-diagonal elements of
// opposite sign.
//
// Array indexing convention (column-major, matching Fortran):
//   T(I,J) => t[tOff + (I-1) + (J-1)*ldt]   (I,J are 1-based)
//   Q(I,J) => q[qOff + (I-1) + (J-1)*ldq]   (I,J are 1-based)
//   WORK(I) => work[workOff + (I-1)]          (I is 1-based)

import { dlamch } from "./dlamch.js";
import { dlasy2 } from "./dlasy2.js";
import { dlanv2 } from "./dlanv2.js";
import { dlacpy } from "./dlacpy.js";
import { dlarfg } from "./dlarfg.js";
import { dlarfx } from "./dlarfx.js";
import { dlange } from "./dlange.js";
import { dlartg } from "./dlartg.js";
import { drot } from "../BLAS/drot.js";
import { MACH_PREC, MACH_SFMIN, LEFT, RIGHT } from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

// "ALL" uplo constant for dlacpy (anything other than UPPER/LOWER)
const ALL = -1;
// norm code for max norm in dlange
const NORM_MAX = 0;

/**
 * DLAEXC swaps adjacent diagonal blocks T11 and T22 of order 1 or 2 in
 * an upper quasi-triangular matrix T by an orthogonal similarity
 * transformation.
 *
 * @param wantq - true: accumulate transformation in Q; false: do not
 * @param n - order of the matrix T (>= 0)
 * @param t - upper quasi-triangular matrix, dimension (ldt, n)
 * @param tOff - offset into t
 * @param ldt - leading dimension of t
 * @param q - orthogonal matrix, dimension (ldq, n), updated if wantq
 * @param qOff - offset into q
 * @param ldq - leading dimension of q
 * @param j1 - index of the first row of the first block T11 (1-based)
 * @param n1 - order of the first block T11 (0, 1, or 2)
 * @param n2 - order of the second block T22 (0, 1, or 2)
 * @param work - workspace, dimension (n)
 * @param workOff - offset into work
 * @returns INFO: 0 = success, 1 = swap rejected (ill-conditioned)
 */
export function dlaexc(
  wantq: boolean,
  n: number,
  t: Float64Array,
  tOff: number,
  ldt: number,
  q: Float64Array,
  qOff: number,
  ldq: number,
  j1: number,
  n1: number,
  n2: number,
  work: Float64Array,
  workOff: number
): number {
  const ZERO = 0.0;
  const ONE = 1.0;
  const TEN = 10.0;
  const LDD = 4;
  const LDX = 2;

  // Helpers for column-major indexing (1-based I,J)
  const T_ = (i: number, j: number) => tOff + (i - 1) + (j - 1) * ldt;
  const Q_ = (i: number, j: number) => qOff + (i - 1) + (j - 1) * ldq;
  // Local arrays D(LDD,4) and X(LDX,2) in column-major
  const D = (i: number, j: number) => i - 1 + (j - 1) * LDD;
  const X = (i: number, j: number) => i - 1 + (j - 1) * LDX;

  const info = 0;

  // Quick return if possible
  if (n === 0 || n1 === 0 || n2 === 0) return 0;
  if (j1 + n1 > n) return 0;

  const j2 = j1 + 1;
  const j3 = j1 + 2;
  const j4 = j1 + 3;

  if (n1 === 1 && n2 === 1) {
    // Swap two 1-by-1 blocks.
    const t11 = t[T_(j1, j1)];
    const t22 = t[T_(j2, j2)];

    // Determine the transformation to perform the interchange.
    const res = dlartg(t[T_(j1, j2)], t22 - t11);
    const cs = res.cs;
    const sn = res.sn;

    // Apply transformation to the matrix T.
    if (j3 <= n) {
      drot(n - j1 - 1, t, T_(j1, j3), ldt, t, T_(j2, j3), ldt, cs, sn);
    }
    drot(j1 - 1, t, T_(1, j1), 1, t, T_(1, j2), 1, cs, sn);

    t[T_(j1, j1)] = t22;
    t[T_(j2, j2)] = t11;

    if (wantq) {
      // Accumulate transformation in the matrix Q.
      drot(n, q, Q_(1, j1), 1, q, Q_(1, j2), 1, cs, sn);
    }
  } else {
    // Swapping involves at least one 2-by-2 block.

    // Copy the diagonal block of order N1+N2 to the local array D
    // and compute its norm.
    const nd = n1 + n2;
    const d = allocFloat64Array(LDD * 4);
    dlacpy(ALL, nd, nd, t, T_(j1, j1), ldt, d, 0, LDD);
    const dnorm = dlange(NORM_MAX, nd, nd, d, 0, LDD, work, workOff);

    // Compute machine-dependent threshold for test for accepting swap.
    const eps = dlamch(MACH_PREC);
    const smlnum = dlamch(MACH_SFMIN) / eps;
    const thresh = Math.max(TEN * eps * dnorm, smlnum);

    // Solve T11*X - X*T22 = scale*T12 for X.
    const x = allocFloat64Array(LDX * 2);
    const solveRes = dlasy2(
      false,
      false,
      -1,
      n1,
      n2,
      d,
      0,
      LDD,
      d,
      n1 + n1 * LDD,
      LDD,
      d,
      n1 * LDD,
      LDD,
      x,
      0,
      LDX
    );
    const scale = solveRes.scale;
    // const xnorm = solveRes.xnorm;

    // Swap the adjacent diagonal blocks.
    const k = n1 + n1 + n2 - 3;

    if (k === 1) {
      // N1 = 1, N2 = 2: generate elementary reflector H so that:
      //   ( scale, X11, X12 ) H = ( 0, 0, * )
      const uArr = allocFloat64Array(3);
      uArr[0] = scale;
      uArr[1] = x[X(1, 1)];
      uArr[2] = x[X(1, 2)];
      const rfg = dlarfg(3, uArr[2], uArr, 0, 1);
      uArr[2] = ONE;
      const tau = rfg.tau;
      const t11 = t[T_(j1, j1)];

      // Perform swap provisionally on diagonal block in D.
      dlarfx(LEFT, 3, 3, uArr, 0, tau, d, 0, LDD, work, workOff);
      dlarfx(RIGHT, 3, 3, uArr, 0, tau, d, 0, LDD, work, workOff);

      // Test whether to reject swap.
      if (
        Math.max(
          Math.abs(d[D(3, 1)]),
          Math.abs(d[D(3, 2)]),
          Math.abs(d[D(3, 3)] - t11)
        ) > thresh
      ) {
        // Swap rejected.
        return 1;
      }

      // Accept swap: apply transformation to the entire matrix T.
      dlarfx(
        LEFT,
        3,
        n - j1 + 1,
        uArr,
        0,
        tau,
        t,
        T_(j1, j1),
        ldt,
        work,
        workOff
      );
      dlarfx(RIGHT, j2, 3, uArr, 0, tau, t, T_(1, j1), ldt, work, workOff);

      t[T_(j3, j1)] = ZERO;
      t[T_(j3, j2)] = ZERO;
      t[T_(j3, j3)] = t11;

      if (wantq) {
        // Accumulate transformation in the matrix Q.
        dlarfx(RIGHT, n, 3, uArr, 0, tau, q, Q_(1, j1), ldq, work, workOff);
      }
    } else if (k === 2) {
      // N1 = 2, N2 = 1: generate elementary reflector H so that:
      //   H ( -X11 ) = ( * )
      //     ( -X21 )   ( 0 )
      //     ( scale)   ( 0 )
      const uArr = allocFloat64Array(3);
      uArr[0] = -x[X(1, 1)];
      uArr[1] = -x[X(2, 1)];
      uArr[2] = scale;
      const rfg = dlarfg(3, uArr[0], uArr, 1, 1);
      uArr[0] = ONE;
      const tau = rfg.tau;
      const t33 = t[T_(j3, j3)];

      // Perform swap provisionally on diagonal block in D.
      dlarfx(LEFT, 3, 3, uArr, 0, tau, d, 0, LDD, work, workOff);
      dlarfx(RIGHT, 3, 3, uArr, 0, tau, d, 0, LDD, work, workOff);

      // Test whether to reject swap.
      if (
        Math.max(
          Math.abs(d[D(2, 1)]),
          Math.abs(d[D(3, 1)]),
          Math.abs(d[D(1, 1)] - t33)
        ) > thresh
      ) {
        // Swap rejected.
        return 1;
      }

      // Accept swap: apply transformation to the entire matrix T.
      dlarfx(RIGHT, j3, 3, uArr, 0, tau, t, T_(1, j1), ldt, work, workOff);
      dlarfx(LEFT, 3, n - j1, uArr, 0, tau, t, T_(j1, j2), ldt, work, workOff);

      t[T_(j1, j1)] = t33;
      t[T_(j2, j1)] = ZERO;
      t[T_(j3, j1)] = ZERO;

      if (wantq) {
        // Accumulate transformation in the matrix Q.
        dlarfx(RIGHT, n, 3, uArr, 0, tau, q, Q_(1, j1), ldq, work, workOff);
      }
    } else if (k === 3) {
      // N1 = 2, N2 = 2: generate elementary reflectors H(1) and H(2)
      const u1 = allocFloat64Array(3);
      u1[0] = -x[X(1, 1)];
      u1[1] = -x[X(2, 1)];
      u1[2] = scale;
      const rfg1 = dlarfg(3, u1[0], u1, 1, 1);
      u1[0] = ONE;
      const tau1 = rfg1.tau;

      const temp = -tau1 * (x[X(1, 2)] + u1[1] * x[X(2, 2)]);
      const u2 = allocFloat64Array(3);
      u2[0] = -temp * u1[1] - x[X(2, 2)];
      u2[1] = -temp * u1[2];
      u2[2] = scale;
      const rfg2 = dlarfg(3, u2[0], u2, 1, 1);
      u2[0] = ONE;
      const tau2 = rfg2.tau;

      // Perform swap provisionally on diagonal block in D.
      dlarfx(LEFT, 3, 4, u1, 0, tau1, d, 0, LDD, work, workOff);
      dlarfx(RIGHT, 4, 3, u1, 0, tau1, d, 0, LDD, work, workOff);
      dlarfx(LEFT, 3, 4, u2, 0, tau2, d, D(2, 1), LDD, work, workOff);
      dlarfx(RIGHT, 4, 3, u2, 0, tau2, d, D(1, 2), LDD, work, workOff);

      // Test whether to reject swap.
      if (
        Math.max(
          Math.abs(d[D(3, 1)]),
          Math.abs(d[D(3, 2)]),
          Math.abs(d[D(4, 1)]),
          Math.abs(d[D(4, 2)])
        ) > thresh
      ) {
        // Swap rejected.
        return 1;
      }

      // Accept swap: apply transformation to the entire matrix T.
      dlarfx(
        LEFT,
        3,
        n - j1 + 1,
        u1,
        0,
        tau1,
        t,
        T_(j1, j1),
        ldt,
        work,
        workOff
      );
      dlarfx(RIGHT, j4, 3, u1, 0, tau1, t, T_(1, j1), ldt, work, workOff);
      dlarfx(
        LEFT,
        3,
        n - j1 + 1,
        u2,
        0,
        tau2,
        t,
        T_(j2, j1),
        ldt,
        work,
        workOff
      );
      dlarfx(RIGHT, j4, 3, u2, 0, tau2, t, T_(1, j2), ldt, work, workOff);

      t[T_(j3, j1)] = ZERO;
      t[T_(j3, j2)] = ZERO;
      t[T_(j4, j1)] = ZERO;
      t[T_(j4, j2)] = ZERO;

      if (wantq) {
        // Accumulate transformation in the matrix Q.
        dlarfx(RIGHT, n, 3, u1, 0, tau1, q, Q_(1, j1), ldq, work, workOff);
        dlarfx(RIGHT, n, 3, u2, 0, tau2, q, Q_(1, j2), ldq, work, workOff);
      }
    }

    // label 40 — Standardize new 2-by-2 blocks after swap
    if (n2 === 2) {
      // Standardize new 2-by-2 block T11
      const res = dlanv2(
        t[T_(j1, j1)],
        t[T_(j1, j2)],
        t[T_(j2, j1)],
        t[T_(j2, j2)]
      );
      t[T_(j1, j1)] = res.a;
      t[T_(j1, j2)] = res.b;
      t[T_(j2, j1)] = res.c;
      t[T_(j2, j2)] = res.d;
      const cs = res.cs;
      const sn = res.sn;
      drot(n - j1 - 1, t, T_(j1, j1 + 2), ldt, t, T_(j2, j1 + 2), ldt, cs, sn);
      drot(j1 - 1, t, T_(1, j1), 1, t, T_(1, j2), 1, cs, sn);
      if (wantq) {
        drot(n, q, Q_(1, j1), 1, q, Q_(1, j2), 1, cs, sn);
      }
    }

    if (n1 === 2) {
      // Standardize new 2-by-2 block T22
      const j3b = j1 + n2;
      const j4b = j3b + 1;
      const res = dlanv2(
        t[T_(j3b, j3b)],
        t[T_(j3b, j4b)],
        t[T_(j4b, j3b)],
        t[T_(j4b, j4b)]
      );
      t[T_(j3b, j3b)] = res.a;
      t[T_(j3b, j4b)] = res.b;
      t[T_(j4b, j3b)] = res.c;
      t[T_(j4b, j4b)] = res.d;
      const cs = res.cs;
      const sn = res.sn;
      if (j3b + 2 <= n) {
        drot(
          n - j3b - 1,
          t,
          T_(j3b, j3b + 2),
          ldt,
          t,
          T_(j4b, j3b + 2),
          ldt,
          cs,
          sn
        );
      }
      drot(j3b - 1, t, T_(1, j3b), 1, t, T_(1, j4b), 1, cs, sn);
      if (wantq) {
        drot(n, q, Q_(1, j3b), 1, q, Q_(1, j4b), 1, cs, sn);
      }
    }
  }
  return info;
}
