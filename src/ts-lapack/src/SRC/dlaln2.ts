// Translated from SRC/dlaln2.f
// DLALN2 solves a system of the form  (ca A - w D) X = s B
// or (ca A**T - w D) X = s B   with possible scaling ("s") and
// perturbation of A.  (A**T means A-transpose.)
//
// A is an NA x NA real matrix, ca is a real scalar, D is an NA x NA
// real diagonal matrix, w is a real or complex value, and X and B are
// NA x 1 matrices -- real if w is real, complex if w is complex.
// NA may be 1 or 2.

import { dlamch } from "./dlamch.js";
import { dladiv } from "./dladiv.js";
import { MACH_SFMIN } from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

// ZSWAP: indexed 1..4 (Fortran DATA), stored as 0..3 here
const ZSWAP = [false, false, true, true];
// RSWAP: indexed 1..4
const RSWAP = [false, true, false, true];
// IPIVOT(4,4): column-major in Fortran.
// Fortran DATA: IPIVOT / 1,2,3,4, 2,1,4,3, 3,4,1,2, 4,3,2,1 /
// IPIVOT(i,j) with i=1..4, j=1..4 stored column-major:
//   col1: 1,2,3,4  col2: 2,1,4,3  col3: 3,4,1,2  col4: 4,3,2,1
// We store as ipivot[j-1][i-1] for access as ipivot[icmax-1][row-1]
const IPIVOT = [
  [1, 2, 3, 4], // column 1: IPIVOT(*,1)
  [2, 1, 4, 3], // column 2: IPIVOT(*,2)
  [3, 4, 1, 2], // column 3: IPIVOT(*,3)
  [4, 3, 2, 1], // column 4: IPIVOT(*,4)
];

export function dlaln2(
  ltrans: boolean,
  na: number,
  nw: number,
  smin: number,
  ca: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  d1: number,
  d2: number,
  b: Float64Array,
  bOff: number,
  ldb: number,
  wr: number,
  wi: number,
  x: Float64Array,
  xOff: number,
  ldx: number
): { scale: number; xnorm: number; info: number } {
  const ZERO = 0.0;
  const ONE = 1.0;
  const TWO = 2.0;

  // Helper to access column-major A(i,j) where i,j are 1-based
  const A = (i: number, j: number) => a[aOff + (i - 1) + (j - 1) * lda];
  const B = (i: number, j: number) => b[bOff + (i - 1) + (j - 1) * ldb];
  const setX = (i: number, j: number, val: number) => {
    x[xOff + (i - 1) + (j - 1) * ldx] = val;
  };
  const getX = (i: number, j: number) => x[xOff + (i - 1) + (j - 1) * ldx];

  // Compute BIGNUM
  const smlnum = TWO * dlamch(MACH_SFMIN);
  const bignum = ONE / smlnum;
  const smini = Math.max(smin, smlnum);

  // Standard Initializations
  let info = 0;
  let scale = ONE;
  let xnorm: number;

  if (na === 1) {
    // 1 x 1 (i.e., scalar) system   C X = B

    if (nw === 1) {
      // Real 1x1 system.
      // C = ca A - w D
      let csr = ca * A(1, 1) - wr * d1;
      let cnorm = Math.abs(csr);

      // If | C | < SMINI, use C = SMINI
      if (cnorm < smini) {
        csr = smini;
        cnorm = smini;
        info = 1;
      }

      // Check scaling for  X = B / C
      const bnorm = Math.abs(B(1, 1));
      if (cnorm < ONE && bnorm > ONE) {
        if (bnorm > bignum * cnorm) {
          scale = ONE / bnorm;
        }
      }

      // Compute X
      setX(1, 1, (B(1, 1) * scale) / csr);
      xnorm = Math.abs(getX(1, 1));
    } else {
      // Complex 1x1 system (w is complex)
      // C = ca A - w D
      let csr = ca * A(1, 1) - wr * d1;
      let csi = -wi * d1;
      let cnorm = Math.abs(csr) + Math.abs(csi);

      // If | C | < SMINI, use C = SMINI
      if (cnorm < smini) {
        csr = smini;
        csi = ZERO;
        cnorm = smini;
        info = 1;
      }

      // Check scaling for  X = B / C
      const bnorm = Math.abs(B(1, 1)) + Math.abs(B(1, 2));
      if (cnorm < ONE && bnorm > ONE) {
        if (bnorm > bignum * cnorm) {
          scale = ONE / bnorm;
        }
      }

      // Compute X
      const div = dladiv(scale * B(1, 1), scale * B(1, 2), csr, csi);
      setX(1, 1, div.p);
      setX(1, 2, div.q);
      xnorm = Math.abs(getX(1, 1)) + Math.abs(getX(1, 2));
    }
  } else {
    // 2x2 System

    // Compute the real part of  C = ca A - w D  (or  ca A**T - w D)
    // CR and CI are 2x2 matrices stored as flat arrays (column-major):
    //   crv[0] = CR(1,1), crv[1] = CR(2,1), crv[2] = CR(1,2), crv[3] = CR(2,2)
    const crv = allocFloat64Array(4);
    const civ = allocFloat64Array(4);

    // CR(1,1) = ca*A(1,1) - wr*D1
    crv[0] = ca * A(1, 1) - wr * d1;
    // CR(2,2) = ca*A(2,2) - wr*D2
    crv[3] = ca * A(2, 2) - wr * d2;
    if (ltrans) {
      // CR(1,2) = ca*A(2,1)
      crv[2] = ca * A(2, 1);
      // CR(2,1) = ca*A(1,2)
      crv[1] = ca * A(1, 2);
    } else {
      // CR(2,1) = ca*A(2,1)
      crv[1] = ca * A(2, 1);
      // CR(1,2) = ca*A(1,2)
      crv[2] = ca * A(1, 2);
    }

    if (nw === 1) {
      // Real 2x2 system (w is real)

      // Find the largest element in C
      let cmax = ZERO;
      let icmax = 0;

      for (let j = 0; j < 4; j++) {
        if (Math.abs(crv[j]) > cmax) {
          cmax = Math.abs(crv[j]);
          icmax = j + 1; // 1-based like Fortran
        }
      }

      // If norm(C) < SMINI, use SMINI*identity.
      if (cmax < smini) {
        const bnorm = Math.max(Math.abs(B(1, 1)), Math.abs(B(2, 1)));
        if (smini < ONE && bnorm > ONE) {
          if (bnorm > bignum * smini) {
            scale = ONE / bnorm;
          }
        }
        const temp = scale / smini;
        setX(1, 1, temp * B(1, 1));
        setX(2, 1, temp * B(2, 1));
        xnorm = temp * bnorm;
        info = 1;
        return { scale, xnorm, info };
      }

      // Gaussian elimination with complete pivoting.
      // icmax is 1-based; IPIVOT is accessed as IPIVOT(row, icmax)
      const ur11 = crv[icmax - 1];
      const cr21 = crv[IPIVOT[icmax - 1][1] - 1]; // IPIVOT(2, ICMAX)
      const ur12 = crv[IPIVOT[icmax - 1][2] - 1]; // IPIVOT(3, ICMAX)
      const cr22 = crv[IPIVOT[icmax - 1][3] - 1]; // IPIVOT(4, ICMAX)
      const ur11r = ONE / ur11;
      const lr21 = ur11r * cr21;
      let ur22 = cr22 - ur12 * lr21;

      // If smaller pivot < SMINI, use SMINI
      if (Math.abs(ur22) < smini) {
        ur22 = smini;
        info = 1;
      }

      let br1: number;
      let br2: number;
      if (RSWAP[icmax - 1]) {
        br1 = B(2, 1);
        br2 = B(1, 1);
      } else {
        br1 = B(1, 1);
        br2 = B(2, 1);
      }
      br2 = br2 - lr21 * br1;

      const bbnd = Math.max(Math.abs(br1 * (ur22 * ur11r)), Math.abs(br2));
      if (bbnd > ONE && Math.abs(ur22) < ONE) {
        if (bbnd >= bignum * Math.abs(ur22)) {
          scale = ONE / bbnd;
        }
      }

      const xr2 = (br2 * scale) / ur22;
      const xr1 = scale * br1 * ur11r - xr2 * (ur11r * ur12);
      if (ZSWAP[icmax - 1]) {
        setX(1, 1, xr2);
        setX(2, 1, xr1);
      } else {
        setX(1, 1, xr1);
        setX(2, 1, xr2);
      }
      xnorm = Math.max(Math.abs(xr1), Math.abs(xr2));

      // Further scaling if  norm(A) norm(X) > overflow
      if (xnorm > ONE && cmax > ONE) {
        if (xnorm > bignum / cmax) {
          const temp = cmax / bignum;
          setX(1, 1, temp * getX(1, 1));
          setX(2, 1, temp * getX(2, 1));
          xnorm = temp * xnorm;
          scale = temp * scale;
        }
      }
    } else {
      // Complex 2x2 system (w is complex)

      // CI(1,1) = -WI*D1, CI(2,1) = 0, CI(1,2) = 0, CI(2,2) = -WI*D2
      civ[0] = -wi * d1;
      civ[1] = ZERO;
      civ[2] = ZERO;
      civ[3] = -wi * d2;

      // Find the largest element in C
      let cmax = ZERO;
      let icmax = 0;

      for (let j = 0; j < 4; j++) {
        if (Math.abs(crv[j]) + Math.abs(civ[j]) > cmax) {
          cmax = Math.abs(crv[j]) + Math.abs(civ[j]);
          icmax = j + 1; // 1-based
        }
      }

      // If norm(C) < SMINI, use SMINI*identity.
      if (cmax < smini) {
        const bnorm = Math.max(
          Math.abs(B(1, 1)) + Math.abs(B(1, 2)),
          Math.abs(B(2, 1)) + Math.abs(B(2, 2))
        );
        if (smini < ONE && bnorm > ONE) {
          if (bnorm > bignum * smini) {
            scale = ONE / bnorm;
          }
        }
        const temp = scale / smini;
        setX(1, 1, temp * B(1, 1));
        setX(2, 1, temp * B(2, 1));
        setX(1, 2, temp * B(1, 2));
        setX(2, 2, temp * B(2, 2));
        xnorm = temp * bnorm;
        info = 1;
        return { scale, xnorm, info };
      }

      // Gaussian elimination with complete pivoting.
      const ur11 = crv[icmax - 1];
      const ui11 = civ[icmax - 1];
      const cr21 = crv[IPIVOT[icmax - 1][1] - 1]; // IPIVOT(2, ICMAX)
      const ci21 = civ[IPIVOT[icmax - 1][1] - 1];
      const ur12 = crv[IPIVOT[icmax - 1][2] - 1]; // IPIVOT(3, ICMAX)
      const ui12 = civ[IPIVOT[icmax - 1][2] - 1];
      const cr22 = crv[IPIVOT[icmax - 1][3] - 1]; // IPIVOT(4, ICMAX)
      const ci22 = civ[IPIVOT[icmax - 1][3] - 1];

      let ur11r: number;
      let ui11r: number;
      let lr21: number;
      let li21: number;
      let ur12s: number;
      let ui12s: number;
      let ur22: number;
      let ui22: number;

      if (icmax === 1 || icmax === 4) {
        // Code when off-diagonals of pivoted C are real
        if (Math.abs(ur11) > Math.abs(ui11)) {
          const temp = ui11 / ur11;
          ur11r = ONE / (ur11 * (ONE + temp * temp));
          ui11r = -temp * ur11r;
        } else {
          const temp = ur11 / ui11;
          ui11r = -ONE / (ui11 * (ONE + temp * temp));
          ur11r = -temp * ui11r;
        }
        lr21 = cr21 * ur11r;
        li21 = cr21 * ui11r;
        ur12s = ur12 * ur11r;
        ui12s = ur12 * ui11r;
        ur22 = cr22 - ur12 * lr21;
        ui22 = ci22 - ur12 * li21;
      } else {
        // Code when diagonals of pivoted C are real
        ur11r = ONE / ur11;
        ui11r = ZERO;
        lr21 = cr21 * ur11r;
        li21 = ci21 * ur11r;
        ur12s = ur12 * ur11r;
        ui12s = ui12 * ur11r;
        ur22 = cr22 - ur12 * lr21 + ui12 * li21;
        ui22 = -ur12 * li21 - ui12 * lr21;
      }

      let u22abs = Math.abs(ur22) + Math.abs(ui22);

      // If smaller pivot < SMINI, use SMINI
      if (u22abs < smini) {
        ur22 = smini;
        ui22 = ZERO;
        info = 1;
      }

      let br1: number;
      let br2: number;
      let bi1: number;
      let bi2: number;
      if (RSWAP[icmax - 1]) {
        br2 = B(1, 1);
        br1 = B(2, 1);
        bi2 = B(1, 2);
        bi1 = B(2, 2);
      } else {
        br1 = B(1, 1);
        br2 = B(2, 1);
        bi1 = B(1, 2);
        bi2 = B(2, 2);
      }
      br2 = br2 - lr21 * br1 + li21 * bi1;
      bi2 = bi2 - li21 * br1 - lr21 * bi1;

      // Recompute u22abs after possible SMINI substitution
      u22abs = Math.abs(ur22) + Math.abs(ui22);

      const bbnd = Math.max(
        (Math.abs(br1) + Math.abs(bi1)) *
          (u22abs * (Math.abs(ur11r) + Math.abs(ui11r))),
        Math.abs(br2) + Math.abs(bi2)
      );
      if (bbnd > ONE && u22abs < ONE) {
        if (bbnd >= bignum * u22abs) {
          scale = ONE / bbnd;
          br1 = scale * br1;
          bi1 = scale * bi1;
          br2 = scale * br2;
          bi2 = scale * bi2;
        }
      }

      const div = dladiv(br2, bi2, ur22, ui22);
      const xr2 = div.p;
      const xi2 = div.q;
      const xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
      const xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
      if (ZSWAP[icmax - 1]) {
        setX(1, 1, xr2);
        setX(2, 1, xr1);
        setX(1, 2, xi2);
        setX(2, 2, xi1);
      } else {
        setX(1, 1, xr1);
        setX(2, 1, xr2);
        setX(1, 2, xi1);
        setX(2, 2, xi2);
      }
      xnorm = Math.max(
        Math.abs(xr1) + Math.abs(xi1),
        Math.abs(xr2) + Math.abs(xi2)
      );

      // Further scaling if  norm(A) norm(X) > overflow
      if (xnorm > ONE && cmax > ONE) {
        if (xnorm > bignum / cmax) {
          const temp = cmax / bignum;
          setX(1, 1, temp * getX(1, 1));
          setX(2, 1, temp * getX(2, 1));
          setX(1, 2, temp * getX(1, 2));
          setX(2, 2, temp * getX(2, 2));
          xnorm = temp * xnorm;
          scale = temp * scale;
        }
      }
    }
  }

  return { scale, xnorm: xnorm!, info };
}
