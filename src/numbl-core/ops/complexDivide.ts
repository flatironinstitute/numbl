/**
 * Complex division `a / b`, the one formula numbl uses for it: the scalar
 * and element-wise paths here, the native kernel
 * (native/ops/complex_binary_elemwise.c) and the JIT runtime's `jit_cdiv`
 * mirror it, so every path gives the same bits.
 *
 * Smith's algorithm avoids spurious overflow and underflow in |b|^2. A
 * zero divisor gives NaN for 0/0 and otherwise a signed Inf per nonzero
 * numerator part (NaN stays NaN). An infinite divisor with a finite numerator gives 0
 * where Smith's ratio is Inf/Inf (the C99 Annex G recovery), as in MATLAB:
 * (1+2i)/(Inf+Inf*i) is 0.
 */

/** The result of the last `complexDivide` call (reused). */
export const cdivResult = { re: 0, im: 0 };

/** x / 0 for one part of the numerator: a NaN part stays NaN. */
function signedInf(x: number): number {
  return x > 0 ? Infinity : x < 0 ? -Infinity : x === 0 ? 0 : NaN;
}

function isInf(x: number): boolean {
  return x === Infinity || x === -Infinity;
}

/** `a / b`; the result is in `cdivResult`. */
export function complexDivide(
  aRe: number,
  aIm: number,
  bRe: number,
  bIm: number
): void {
  let re: number;
  let im: number;
  if (bRe === 0 && bIm === 0) {
    if (aRe === 0 && aIm === 0) {
      re = NaN;
      im = 0;
    } else {
      re = signedInf(aRe);
      im = signedInf(aIm);
    }
  } else if (Math.abs(bRe) >= Math.abs(bIm)) {
    const r = bIm / bRe;
    const d = bRe + bIm * r;
    re = (aRe + aIm * r) / d;
    im = (aIm - aRe * r) / d;
  } else {
    const r = bRe / bIm;
    const d = bIm + bRe * r;
    re = (aRe * r + aIm) / d;
    im = (aIm * r - aRe) / d;
  }
  if (
    re !== re &&
    im !== im &&
    isInf(bRe) &&
    isInf(bIm) &&
    Number.isFinite(aRe) &&
    Number.isFinite(aIm)
  ) {
    re = 0;
    im = 0;
  }
  cdivResult.re = re;
  cdivResult.im = im;
}
