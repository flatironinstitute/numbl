// Translated from SRC/dlasy2.f
// DLASY2 solves for the N1 by N2 matrix X, 1 <= N1,N2 <= 2, in
//
//   op(TL)*X + ISGN*X*op(TR) = SCALE*B,
//
// where TL is N1 by N1, TR is N2 by N2, B is N1 by N2, and ISGN = 1 or -1.
// op(T) = T or T**T.
//
// Indexing convention (matching Fortran column-major):
//   TL(I,J)  => tl[tlOff + (I-1) + (J-1)*ldtl]
//   TR(I,J)  => tr[trOff + (I-1) + (J-1)*ldtr]
//   B(I,J)   => b[bOff   + (I-1) + (J-1)*ldb]
//   X(I,J)   => x[xOff   + (I-1) + (J-1)*ldx]

import { dlamch } from "./dlamch.js";
import { MACH_PREC, MACH_SFMIN } from "../utils/constants.js";
import { allocFloat64Array } from "../../../numbl-core/runtime/alloc.js";

// Lookup tables (matching Fortran DATA statements, 0-indexed)
// Fortran: LOCU12 / 3, 4, 1, 2 /
const LOCU12 = [2, 3, 0, 1]; // 0-based
// Fortran: LOCL21 / 2, 1, 4, 3 /
const LOCL21 = [1, 0, 3, 2];
// Fortran: LOCU22 / 4, 3, 2, 1 /
const LOCU22 = [3, 2, 1, 0];
// Fortran: XSWPIV / .FALSE., .FALSE., .TRUE., .TRUE. /
const XSWPIV = [false, false, true, true];
// Fortran: BSWPIV / .FALSE., .TRUE., .FALSE., .TRUE. /
const BSWPIV = [false, true, false, true];

export function dlasy2(
  ltranl: boolean,
  ltranr: boolean,
  isgn: number,
  n1: number,
  n2: number,
  tl: Float64Array,
  tlOff: number,
  ldtl: number,
  tr: Float64Array,
  trOff: number,
  ldtr: number,
  b: Float64Array,
  bOff: number,
  ldb: number,
  x: Float64Array,
  xOff: number,
  ldx: number
): { scale: number; xnorm: number; info: number } {
  let info = 0;
  let scale = 1.0;
  let xnorm = 0.0;

  // Quick return if possible
  if (n1 === 0 || n2 === 0) {
    return { scale, xnorm, info };
  }

  // Set constants to control overflow
  const eps = dlamch(MACH_PREC);
  const smlnum = dlamch(MACH_SFMIN) / eps;
  const sgn = isgn;

  const k = n1 + n1 + n2 - 2;
  // k=1 => 1x1, k=2 => 1x2, k=3 => 2x1, k=4 => 2x2

  // Helper macros for column-major access
  // TL(i,j) => tl[tlOff + (i-1) + (j-1)*ldtl]  (1-based i,j)
  // TR(i,j) => tr[trOff + (i-1) + (j-1)*ldtr]
  // B(i,j)  => b[bOff + (i-1) + (j-1)*ldb]
  // X(i,j)  => x[xOff + (i-1) + (j-1)*ldx]

  if (k === 1) {
    // 1 by 1: TL11*X + SGN*X*TR11 = B11
    let tau1 = tl[tlOff] + sgn * tr[trOff];
    let bet = Math.abs(tau1);
    if (bet <= smlnum) {
      tau1 = smlnum;
      bet = smlnum;
      info = 1;
    }

    scale = 1.0;
    const gam = Math.abs(b[bOff]);
    if (smlnum * gam > bet) {
      scale = 1.0 / gam;
    }

    x[xOff] = (b[bOff] * scale) / tau1;
    xnorm = Math.abs(x[xOff]);
    return { scale, xnorm, info };
  }

  if (k === 2) {
    // 1 by 2:
    // TL11*[X11 X12] + ISGN*[X11 X12]*op[TR11 TR12] = [B11 B12]
    //                                    [TR21 TR22]
    const smin = Math.max(
      eps *
        Math.max(
          Math.abs(tl[tlOff]),
          Math.abs(tr[trOff]),
          Math.abs(tr[trOff + ldtr]),
          Math.abs(tr[trOff + 1]),
          Math.abs(tr[trOff + 1 + ldtr])
        ),
      smlnum
    );
    const tmp = allocFloat64Array(4);
    tmp[0] = tl[tlOff] + sgn * tr[trOff];
    tmp[3] = tl[tlOff] + sgn * tr[trOff + 1 + ldtr];
    if (ltranr) {
      tmp[1] = sgn * tr[trOff + 1]; // SGN*TR(2,1)
      tmp[2] = sgn * tr[trOff + ldtr]; // SGN*TR(1,2)
    } else {
      tmp[1] = sgn * tr[trOff + ldtr]; // SGN*TR(1,2)
      tmp[2] = sgn * tr[trOff + 1]; // SGN*TR(2,1)
    }
    const btmp = allocFloat64Array(4);
    btmp[0] = b[bOff];
    btmp[1] = b[bOff + ldb];

    return solve2x2(tmp, btmp, smin, smlnum, n1, x, xOff, ldx);
  }

  if (k === 3) {
    // 2 by 1:
    // op[TL11 TL12]*[X11] + ISGN* [X11]*TR11 = [B11]
    //   [TL21 TL22] [X21]         [X21]         [B21]
    const smin = Math.max(
      eps *
        Math.max(
          Math.abs(tr[trOff]),
          Math.abs(tl[tlOff]),
          Math.abs(tl[tlOff + ldtl]),
          Math.abs(tl[tlOff + 1]),
          Math.abs(tl[tlOff + 1 + ldtl])
        ),
      smlnum
    );
    const tmp = allocFloat64Array(4);
    tmp[0] = tl[tlOff] + sgn * tr[trOff];
    tmp[3] = tl[tlOff + 1 + ldtl] + sgn * tr[trOff];
    if (ltranl) {
      tmp[1] = tl[tlOff + ldtl]; // TL(1,2)
      tmp[2] = tl[tlOff + 1]; // TL(2,1)
    } else {
      tmp[1] = tl[tlOff + 1]; // TL(2,1)
      tmp[2] = tl[tlOff + ldtl]; // TL(1,2)
    }
    const btmp = allocFloat64Array(4);
    btmp[0] = b[bOff];
    btmp[1] = b[bOff + 1];

    return solve2x2(tmp, btmp, smin, smlnum, n1, x, xOff, ldx);
  }

  // k === 4: 2 by 2
  // op[TL11 TL12]*[X11 X12] +ISGN* [X11 X12]*op[TR11 TR12] = [B11 B12]
  //   [TL21 TL22] [X21 X22]        [X21 X22]   [TR21 TR22]   [B21 B22]
  //
  // Solve equivalent 4 by 4 system using complete pivoting.

  let smin = Math.max(
    Math.abs(tr[trOff]),
    Math.abs(tr[trOff + ldtr]),
    Math.abs(tr[trOff + 1]),
    Math.abs(tr[trOff + 1 + ldtr])
  );
  smin = Math.max(
    smin,
    Math.abs(tl[tlOff]),
    Math.abs(tl[tlOff + ldtl]),
    Math.abs(tl[tlOff + 1]),
    Math.abs(tl[tlOff + 1 + ldtl])
  );
  smin = Math.max(eps * smin, smlnum);

  // T16 is 4x4, stored column-major: T16(i,j) => t16[(i-1) + (j-1)*4], 1-based
  const t16 = allocFloat64Array(16); // initialized to zero

  t16[0] = tl[tlOff] + sgn * tr[trOff]; // T16(1,1)
  t16[1 + 1 * 4] = tl[tlOff + 1 + ldtl] + sgn * tr[trOff]; // T16(2,2)
  t16[2 + 2 * 4] = tl[tlOff] + sgn * tr[trOff + 1 + ldtr]; // T16(3,3)
  t16[3 + 3 * 4] = tl[tlOff + 1 + ldtl] + sgn * tr[trOff + 1 + ldtr]; // T16(4,4)

  if (ltranl) {
    t16[0 + 1 * 4] = tl[tlOff + 1]; // T16(1,2) = TL(2,1)
    t16[1 + 0 * 4] = tl[tlOff + ldtl]; // T16(2,1) = TL(1,2)
    t16[2 + 3 * 4] = tl[tlOff + 1]; // T16(3,4) = TL(2,1)
    t16[3 + 2 * 4] = tl[tlOff + ldtl]; // T16(4,3) = TL(1,2)
  } else {
    t16[0 + 1 * 4] = tl[tlOff + ldtl]; // T16(1,2) = TL(1,2)
    t16[1 + 0 * 4] = tl[tlOff + 1]; // T16(2,1) = TL(2,1)
    t16[2 + 3 * 4] = tl[tlOff + ldtl]; // T16(3,4) = TL(1,2)
    t16[3 + 2 * 4] = tl[tlOff + 1]; // T16(4,3) = TL(2,1)
  }

  if (ltranr) {
    t16[0 + 2 * 4] = sgn * tr[trOff + ldtr]; // T16(1,3) = SGN*TR(1,2)
    t16[1 + 3 * 4] = sgn * tr[trOff + ldtr]; // T16(2,4) = SGN*TR(1,2)
    t16[2 + 0 * 4] = sgn * tr[trOff + 1]; // T16(3,1) = SGN*TR(2,1)
    t16[3 + 1 * 4] = sgn * tr[trOff + 1]; // T16(4,2) = SGN*TR(2,1)
  } else {
    t16[0 + 2 * 4] = sgn * tr[trOff + 1]; // T16(1,3) = SGN*TR(2,1)
    t16[1 + 3 * 4] = sgn * tr[trOff + 1]; // T16(2,4) = SGN*TR(2,1)
    t16[2 + 0 * 4] = sgn * tr[trOff + ldtr]; // T16(3,1) = SGN*TR(1,2)
    t16[3 + 1 * 4] = sgn * tr[trOff + ldtr]; // T16(4,2) = SGN*TR(1,2)
  }

  const btmp = allocFloat64Array(4);
  btmp[0] = b[bOff]; // B(1,1)
  btmp[1] = b[bOff + 1]; // B(2,1)
  btmp[2] = b[bOff + ldb]; // B(1,2)
  btmp[3] = b[bOff + 1 + ldb]; // B(2,2)

  // Perform elimination with complete pivoting
  const jpiv = new Int32Array(4);

  for (let i = 0; i < 3; i++) {
    let xmax = 0.0;
    let ipsv = i;
    let jpsv = i;
    for (let ip = i; ip < 4; ip++) {
      for (let jp = i; jp < 4; jp++) {
        if (Math.abs(t16[ip + jp * 4]) >= xmax) {
          xmax = Math.abs(t16[ip + jp * 4]);
          ipsv = ip;
          jpsv = jp;
        }
      }
    }

    // Swap rows ipsv and i in T16 and btmp
    if (ipsv !== i) {
      // DSWAP(4, T16(IPSV,1), 4, T16(I,1), 4) — swap rows with stride 4
      for (let jj = 0; jj < 4; jj++) {
        const tmp = t16[ipsv + jj * 4];
        t16[ipsv + jj * 4] = t16[i + jj * 4];
        t16[i + jj * 4] = tmp;
      }
      const tmp = btmp[i];
      btmp[i] = btmp[ipsv];
      btmp[ipsv] = tmp;
    }

    // Swap columns jpsv and i in T16
    if (jpsv !== i) {
      // DSWAP(4, T16(1,JPSV), 1, T16(1,I), 1) — swap columns with stride 1
      for (let ii = 0; ii < 4; ii++) {
        const tmp = t16[ii + jpsv * 4];
        t16[ii + jpsv * 4] = t16[ii + i * 4];
        t16[ii + i * 4] = tmp;
      }
    }
    jpiv[i] = jpsv;

    if (Math.abs(t16[i + i * 4]) < smin) {
      info = 1;
      t16[i + i * 4] = smin;
    }

    for (let j = i + 1; j < 4; j++) {
      t16[j + i * 4] = t16[j + i * 4] / t16[i + i * 4];
      btmp[j] = btmp[j] - t16[j + i * 4] * btmp[i];
      for (let kk = i + 1; kk < 4; kk++) {
        t16[j + kk * 4] = t16[j + kk * 4] - t16[j + i * 4] * t16[i + kk * 4];
      }
    }
  }

  if (Math.abs(t16[3 + 3 * 4]) < smin) {
    info = 1;
    t16[3 + 3 * 4] = smin;
  }

  scale = 1.0;
  if (
    8.0 * smlnum * Math.abs(btmp[0]) > Math.abs(t16[0]) ||
    8.0 * smlnum * Math.abs(btmp[1]) > Math.abs(t16[1 + 1 * 4]) ||
    8.0 * smlnum * Math.abs(btmp[2]) > Math.abs(t16[2 + 2 * 4]) ||
    8.0 * smlnum * Math.abs(btmp[3]) > Math.abs(t16[3 + 3 * 4])
  ) {
    scale =
      0.125 /
      Math.max(
        Math.abs(btmp[0]),
        Math.abs(btmp[1]),
        Math.abs(btmp[2]),
        Math.abs(btmp[3])
      );
    btmp[0] *= scale;
    btmp[1] *= scale;
    btmp[2] *= scale;
    btmp[3] *= scale;
  }

  // Back-substitution
  const tmp = allocFloat64Array(4);
  for (let i = 0; i < 4; i++) {
    const kk = 3 - i; // k = 4,3,2,1 in Fortran => 3,2,1,0 in 0-based
    const temp = 1.0 / t16[kk + kk * 4];
    tmp[kk] = btmp[kk] * temp;
    for (let j = kk + 1; j < 4; j++) {
      tmp[kk] = tmp[kk] - temp * t16[kk + j * 4] * tmp[j];
    }
  }

  // Undo column permutations
  for (let i = 0; i < 3; i++) {
    // Fortran: DO I=1,3; IF(JPIV(4-I).NE.4-I) swap TMP(4-I) and TMP(JPIV(4-I))
    // In 0-based: swap tmp[2-i] and tmp[jpiv[2-i]]
    const idx = 2 - i;
    if (jpiv[idx] !== idx) {
      const temp = tmp[idx];
      tmp[idx] = tmp[jpiv[idx]];
      tmp[jpiv[idx]] = temp;
    }
  }

  x[xOff] = tmp[0]; // X(1,1)
  x[xOff + 1] = tmp[1]; // X(2,1)
  x[xOff + ldx] = tmp[2]; // X(1,2)
  x[xOff + 1 + ldx] = tmp[3]; // X(2,2)

  xnorm = Math.max(
    Math.abs(tmp[0]) + Math.abs(tmp[2]),
    Math.abs(tmp[1]) + Math.abs(tmp[3])
  );

  return { scale, xnorm, info };
}

// Shared helper for the 2x2 solve (used by both 1x2 and 2x1 cases)
function solve2x2(
  tmp: Float64Array,
  btmp: Float64Array,
  smin: number,
  smlnum: number,
  n1: number,
  x: Float64Array,
  xOff: number,
  ldx: number
): { scale: number; xnorm: number; info: number } {
  let info = 0;
  let scale: number;
  let xnorm: number;

  // Find pivot (1-based IDAMAX equivalent, but we do it inline for a 4-element array)
  let ipiv = 0;
  let maxVal = Math.abs(tmp[0]);
  for (let i = 1; i < 4; i++) {
    if (Math.abs(tmp[i]) > maxVal) {
      maxVal = Math.abs(tmp[i]);
      ipiv = i;
    }
  }

  let u11 = tmp[ipiv];
  if (Math.abs(u11) <= smin) {
    info = 1;
    u11 = smin;
  }

  const u12 = tmp[LOCU12[ipiv]];
  const l21 = tmp[LOCL21[ipiv]] / u11;
  let u22 = tmp[LOCU22[ipiv]] - u12 * l21;
  const xswap = XSWPIV[ipiv];
  const bswap = BSWPIV[ipiv];

  if (Math.abs(u22) <= smin) {
    info = 1;
    u22 = smin;
  }

  if (bswap) {
    const temp = btmp[1];
    btmp[1] = btmp[0] - l21 * temp;
    btmp[0] = temp;
  } else {
    btmp[1] = btmp[1] - l21 * btmp[0];
  }

  scale = 1.0;
  if (
    2.0 * smlnum * Math.abs(btmp[1]) > Math.abs(u22) ||
    2.0 * smlnum * Math.abs(btmp[0]) > Math.abs(u11)
  ) {
    scale = 0.5 / Math.max(Math.abs(btmp[0]), Math.abs(btmp[1]));
    btmp[0] *= scale;
    btmp[1] *= scale;
  }

  let x2_1 = btmp[1] / u22;
  let x2_0 = btmp[0] / u11 - (u12 / u11) * x2_1;

  if (xswap) {
    const temp = x2_1;
    x2_1 = x2_0;
    x2_0 = temp;
  }

  x[xOff] = x2_0; // X(1,1)
  if (n1 === 1) {
    x[xOff + ldx] = x2_1; // X(1,2)
    xnorm = Math.abs(x[xOff]) + Math.abs(x[xOff + ldx]);
  } else {
    x[xOff + 1] = x2_1; // X(2,1)
    xnorm = Math.max(Math.abs(x[xOff]), Math.abs(x[xOff + 1]));
  }

  return { scale, xnorm, info };
}
