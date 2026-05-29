// JS sibling of `cdiv.h`. Scalar complex division matching numbl's
// signed-Inf-on-zero-divisor behavior via Smith's algorithm. Mirrors
// the C side so cross-runner output stays byte-for-byte.

export function mtoc2_cdiv(a, b) {
  const ar = a.re,
    ai = a.im;
  const br = b.re,
    bi = b.im;
  if (Math.abs(br) >= Math.abs(bi)) {
    const r = bi / br;
    const den = br + r * bi;
    return { re: (ar + ai * r) / den, im: (ai - ar * r) / den };
  }
  const r = br / bi;
  const den = bi + r * br;
  return { re: (ar * r + ai) / den, im: (ai * r - ar) / den };
}
