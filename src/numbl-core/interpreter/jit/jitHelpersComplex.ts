/**
 * Complex number arithmetic helpers for JIT-compiled code.
 * These are pure functions used by the $h helpers object.
 */

import type { RuntimeComplexNumber } from "../../runtime/types.js";

export function re(v: unknown): number {
  if (typeof v === "number") return v;
  return (v as RuntimeComplexNumber).re;
}

export function im(v: unknown): number {
  if (typeof v === "number") return 0;
  return (v as RuntimeComplexNumber).im;
}

export function mkc(r: number, i: number): number | RuntimeComplexNumber {
  if (i === 0) return r;
  return { kind: "complex_number", re: r, im: i };
}

export function cAdd(a: unknown, b: unknown): number | RuntimeComplexNumber {
  return mkc(re(a) + re(b), im(a) + im(b));
}

export function cSub(a: unknown, b: unknown): number | RuntimeComplexNumber {
  return mkc(re(a) - re(b), im(a) - im(b));
}

export function cMul(a: unknown, b: unknown): number | RuntimeComplexNumber {
  const ar = re(a),
    ai = im(a),
    br = re(b),
    bi = im(b);
  return mkc(ar * br - ai * bi, ar * bi + ai * br);
}

/**
 * Complex division using Smith's method to avoid overflow/underflow.
 * Handles division by zero consistently with MATLAB (produces Inf, not NaN).
 */
export function cDiv(a: unknown, b: unknown): number | RuntimeComplexNumber {
  const ar = re(a),
    ai = im(a),
    br = re(b),
    bi = im(b);
  if (Math.abs(br) >= Math.abs(bi)) {
    if (br === 0 && bi === 0) {
      return mkc(ar / 0, ai / 0);
    }
    const r = bi / br;
    const d = br + bi * r;
    return mkc((ar + ai * r) / d, (ai - ar * r) / d);
  } else {
    const r = br / bi;
    const d = bi + br * r;
    return mkc((ar * r + ai) / d, (ai * r - ar) / d);
  }
}

export function cNeg(a: unknown): number | RuntimeComplexNumber {
  return mkc(-re(a), -im(a));
}

export function cConj(a: unknown): number | RuntimeComplexNumber {
  return mkc(re(a), -im(a));
}

export function cAngle(a: unknown): number {
  return Math.atan2(im(a), re(a));
}

/** Complex truthiness: nonzero if either real or imag part is nonzero. */
export function cTruthy(v: unknown): boolean {
  if (typeof v === "number") return v !== 0;
  const c = v as RuntimeComplexNumber;
  return c.re !== 0 || c.im !== 0;
}
