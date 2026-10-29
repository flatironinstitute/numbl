// JS sibling of `cdiv.h`. Scalar complex division matching numbl's
// signed-Inf-on-zero-divisor behavior via Smith's algorithm. Mirrors
// the C side so output stays byte-for-byte.

export function jit_cdiv(a, b) {
  const ar = a.re,
    ai = a.im;
  const br = b.re,
    bi = b.im;
  // Zero divisor: match the interpreter (helpers/arithmetic.ts) — 0/0 is
  // NaN, a nonzero numerator yields a signed Inf per component. Smith's
  // algorithm below would otherwise produce NaN+NaNi here.
  if (br === 0 && bi === 0) {
    if (ar === 0 && ai === 0) return { re: NaN, im: 0 };
    // x / 0 per component: a NaN part stays NaN (MATLAB: (NaN + 1i) / 0
    // is NaN + Inf i).
    const signedInf = x =>
      x > 0 ? Infinity : x < 0 ? -Infinity : x === 0 ? 0 : NaN;
    return { re: signedInf(ar), im: signedInf(ai) };
  }
  let re, im;
  if (Math.abs(br) >= Math.abs(bi)) {
    const r = bi / br;
    const den = br + r * bi;
    re = (ar + ai * r) / den;
    im = (ai - ar * r) / den;
  } else {
    const r = br / bi;
    const den = bi + r * br;
    re = (ar * r + ai) / den;
    im = (ai * r - ar) / den;
  }
  // An infinite divisor with a finite numerator: 0 where Smith's ratio is
  // Inf/Inf (as the interpreter's ops/complexDivide.ts).
  if (
    re !== re &&
    im !== im &&
    Math.abs(br) === Infinity &&
    Math.abs(bi) === Infinity &&
    Number.isFinite(ar) &&
    Number.isFinite(ai)
  ) {
    re = 0;
    im = 0;
  }
  return { re, im };
}

// Scalar `/` with MATLAB's mixed real/complex rule (see `cdiv.h`).
export function jit_cdiv_mixed(a, b) {
  if (b.im === 0) {
    if (a.im === 0) return { re: a.re / b.re, im: 0 };
    return { re: a.re === 0 ? 0 : a.re / b.re, im: a.im / b.re };
  }
  return jit_cdiv(a.im === 0 ? { re: a.re, im: 0 } : a, b);
}
