// JS sibling of `cdiv.h`. Scalar complex division matching numbl's
// signed-Inf-on-zero-divisor behavior via Smith's algorithm. Mirrors
// the C side so cross-runner output stays byte-for-byte.

export function mtoc2_cdiv(a, b) {
  const ar = a.re,
    ai = a.im;
  const br = b.re,
    bi = b.im;
  // Zero divisor: match the interpreter (helpers/arithmetic.ts) — 0/0 is
  // NaN, a nonzero numerator yields a signed Inf per component. Smith's
  // algorithm below would otherwise produce NaN+NaNi here.
  if (br === 0 && bi === 0) {
    if (ar === 0 && ai === 0) return { re: NaN, im: 0 };
    const signedInf = x => (x > 0 ? Infinity : x < 0 ? -Infinity : 0);
    return { re: signedInf(ar), im: signedInf(ai) };
  }
  if (Math.abs(br) >= Math.abs(bi)) {
    const r = bi / br;
    const den = br + r * bi;
    return { re: (ar + ai * r) / den, im: (ai - ar * r) / den };
  }
  const r = br / bi;
  const den = bi + r * br;
  return { re: (ar * r + ai) / den, im: (ai * r - ar) / den };
}
