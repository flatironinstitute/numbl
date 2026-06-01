/**
 * JS-side scalar-complex fold functions for unary math builtins.
 *
 * Mirrors the cscalar.h `mtoc2_c*` helpers — both the C side and
 * c2js's JS-runtime side must agree with these formulas. Used by
 * `_unary_real.ts`'s factory at type-system-fold time when an input
 * has an exact `{re, im}` carrier.
 *
 * Real input is projected to `{re: x, im: 0}` upstream of these
 * helpers via `exactScalarAsComplex`; that case is rare for unary
 * math since the same builtin's real-path fold already catches it.
 */

export type C = { re: number; im: number };

export function cSqrt(z: C): C {
  // Smith's stable formula — matches `mtoc2_csqrt` in cscalar.js
  // exactly. The polar form (`mag * cos/sin(phase/2)`) carries a
  // ~6e-17 real residue for pure-real negative input because
  // `Math.cos(pi/2)` isn't exactly 0; the Smith form returns clean
  // `{re: 0, im: sqrt(-re)}` for that case.
  const r = Math.hypot(z.re, z.im);
  if (r === 0) return { re: 0, im: 0 };
  const re = Math.sqrt((r + z.re) / 2);
  const im = (z.im >= 0 ? 1 : -1) * Math.sqrt((r - z.re) / 2);
  return { re, im };
}

export function cExp(z: C): C {
  const m = Math.exp(z.re);
  return { re: m * Math.cos(z.im), im: m * Math.sin(z.im) };
}

export function cLog(z: C): C {
  return {
    re: 0.5 * Math.log(z.re * z.re + z.im * z.im),
    im: Math.atan2(z.im, z.re),
  };
}

export function cLog2(z: C): C {
  const w = cLog(z);
  const k = Math.log(2);
  return { re: w.re / k, im: w.im / k };
}

export function cLog10(z: C): C {
  const w = cLog(z);
  const k = Math.log(10);
  return { re: w.re / k, im: w.im / k };
}

export function cSin(z: C): C {
  return {
    re: Math.sin(z.re) * Math.cosh(z.im),
    im: Math.cos(z.re) * Math.sinh(z.im),
  };
}

export function cCos(z: C): C {
  return {
    re: Math.cos(z.re) * Math.cosh(z.im),
    im: -Math.sin(z.re) * Math.sinh(z.im),
  };
}

function cdiv(a: C, b: C): C {
  // Smith's algorithm (matches mtoc2_cdiv).
  if (Math.abs(b.re) >= Math.abs(b.im)) {
    const r = b.im / b.re;
    const den = b.re + r * b.im;
    return { re: (a.re + a.im * r) / den, im: (a.im - a.re * r) / den };
  }
  const r = b.re / b.im;
  const den = b.im + r * b.re;
  return { re: (a.re * r + a.im) / den, im: (a.im * r - a.re) / den };
}

export function cTan(z: C): C {
  return cdiv(cSin(z), cCos(z));
}

export function cSinh(z: C): C {
  return {
    re: Math.sinh(z.re) * Math.cos(z.im),
    im: Math.cosh(z.re) * Math.sin(z.im),
  };
}

export function cCosh(z: C): C {
  return {
    re: Math.cosh(z.re) * Math.cos(z.im),
    im: Math.sinh(z.re) * Math.sin(z.im),
  };
}

export function cTanh(z: C): C {
  // Double-angle form, matching `mtoc2_ctanh` (cscalar.js) and the
  // interpreter — avoids overflow that sinh(z)/cosh(z) would hit for
  // large real parts.
  const denom = Math.cosh(2 * z.re) + Math.cos(2 * z.im);
  return { re: Math.sinh(2 * z.re) / denom, im: Math.sin(2 * z.im) / denom };
}

export function cAtan(z: C): C {
  // atan(z) = (i/2) * log((1 - i*z) / (1 + i*z))
  const iz: C = { re: -z.im, im: z.re };
  const num: C = { re: 1 - iz.re, im: -iz.im };
  const den: C = { re: 1 + iz.re, im: iz.im };
  const q = cdiv(num, den);
  const l = cLog(q);
  return { re: -l.im / 2, im: l.re / 2 };
}

export function cFloor(z: C): C {
  return { re: Math.floor(z.re), im: Math.floor(z.im) };
}

export function cCeil(z: C): C {
  return { re: Math.ceil(z.re), im: Math.ceil(z.im) };
}

/** Half-away-from-zero on each component (matches C99 `round`). */
export function cRound(z: C): C {
  const r = (x: number): number =>
    x >= 0 ? Math.floor(x + 0.5) : -Math.floor(-x + 0.5);
  return { re: r(z.re), im: r(z.im) };
}

export function cFix(z: C): C {
  return { re: Math.trunc(z.re), im: Math.trunc(z.im) };
}

export function cSign(z: C): C {
  if (z.re === 0 && z.im === 0) return { re: 0, im: 0 };
  const m = Math.hypot(z.re, z.im);
  return { re: z.re / m, im: z.im / m };
}
