/**
 * Pure-TS implementation of complex unary element-wise ops, split storage.
 * Mirrors native/ops/complex_unary_elemwise.c.
 *
 * ABS is intentionally unsupported (use tsComplexAbs — real-valued output).
 */

import { OpUnary } from "./opCodes.js";

const INV_LN2 = 1 / Math.log(2);
const INV_LN10 = 1 / Math.log(10);

// ── Building blocks (principal branches, matching C99 <complex.h>) ──────

function cexp(re: number, im: number): [number, number] {
  const e = Math.exp(re);
  return [e * Math.cos(im), e * Math.sin(im)];
}

function clog(re: number, im: number): [number, number] {
  return [Math.log(Math.hypot(re, im)), Math.atan2(im, re)];
}

function csqrt(re: number, im: number): [number, number] {
  if (re === 0 && im === 0) return [0, 0];
  const m = Math.hypot(re, im);
  const outRe = Math.sqrt((m + re) / 2);
  const outIm = Math.sign(im) * Math.sqrt((m - re) / 2);
  // When im == 0, sign(0)=0; handle negative real axis.
  if (im === 0 && re < 0) return [0, Math.sqrt(-re)];
  return [outRe, outIm];
}

function csin(re: number, im: number): [number, number] {
  return [Math.sin(re) * Math.cosh(im), Math.cos(re) * Math.sinh(im)];
}

function ccos(re: number, im: number): [number, number] {
  return [Math.cos(re) * Math.cosh(im), -Math.sin(re) * Math.sinh(im)];
}

function ctan(re: number, im: number): [number, number] {
  const [sr, si] = csin(re, im);
  const [cr, ci] = ccos(re, im);
  const denom = cr * cr + ci * ci;
  return [(sr * cr + si * ci) / denom, (si * cr - sr * ci) / denom];
}

function csinh(re: number, im: number): [number, number] {
  return [Math.sinh(re) * Math.cos(im), Math.cosh(re) * Math.sin(im)];
}

function ccosh(re: number, im: number): [number, number] {
  return [Math.cosh(re) * Math.cos(im), Math.sinh(re) * Math.sin(im)];
}

function ctanh(re: number, im: number): [number, number] {
  const [sr, si] = csinh(re, im);
  const [cr, ci] = ccosh(re, im);
  const denom = cr * cr + ci * ci;
  return [(sr * cr + si * ci) / denom, (si * cr - sr * ci) / denom];
}

// asin(z) = -i * log(i*z + sqrt(1 - z^2))
function casin(re: number, im: number): [number, number] {
  // z^2 = (re + i*im)^2 = re^2 - im^2 + 2i*re*im
  const z2re = re * re - im * im;
  const z2im = 2 * re * im;
  // 1 - z^2
  const ar = 1 - z2re;
  const ai = -z2im;
  // sqrt(1 - z^2)
  const [sr, si] = csqrt(ar, ai);
  // i*z + sqrt(1 - z^2) = (sr - im) + i*(si + re)
  const [lr, li] = clog(sr - im, si + re);
  // -i * log(...) = -i * (lr + i*li) = li - i*lr
  return [li, -lr];
}

// acos(z) = -i * log(z + i * sqrt(1 - z^2))
function cacos(re: number, im: number): [number, number] {
  const z2re = re * re - im * im;
  const z2im = 2 * re * im;
  const ar = 1 - z2re;
  const ai = -z2im;
  const [sr, si] = csqrt(ar, ai);
  // i * sqrt(1 - z^2) = -si + i*sr; then + z = (re - si) + i*(im + sr)
  const [lr, li] = clog(re - si, im + sr);
  return [li, -lr];
}

// atan(z) = (i/2) * (log(1 - i*z) - log(1 + i*z))
function catan(re: number, im: number): [number, number] {
  // i*z = -im + i*re
  // 1 - i*z = (1 + im) + i*(-re)
  // 1 + i*z = (1 - im) + i*(re)
  const [l1r, l1i] = clog(1 + im, -re);
  const [l2r, l2i] = clog(1 - im, re);
  // log(1 - i*z) - log(1 + i*z)
  const dr = l1r - l2r;
  const di = l1i - l2i;
  // multiply by (i/2): (i/2) * (dr + i*di) = -di/2 + i*dr/2
  return [-di / 2, dr / 2];
}

function csign(re: number, im: number): [number, number] {
  const m = Math.hypot(re, im);
  if (m === 0) return [0, 0];
  return [re / m, im / m];
}

// ── Main dispatcher ────────────────────────────────────────────────────

function roundHaf(x: number): number {
  return x >= 0 ? Math.round(x) : -Math.round(-x);
}

export function tsComplexUnaryElemwise(
  op: number,
  n: number,
  aRe: Float64Array,
  aIm: Float64Array | null,
  outRe: Float64Array,
  outIm: Float64Array
): void {
  let fn: ((re: number, im: number) => [number, number]) | null = null;
  switch (op) {
    case OpUnary.EXP:
      fn = cexp;
      break;
    case OpUnary.LOG:
      fn = clog;
      break;
    case OpUnary.LOG2:
      fn = (r, i) => {
        const [lr, li] = clog(r, i);
        return [lr * INV_LN2, li * INV_LN2];
      };
      break;
    case OpUnary.LOG10:
      fn = (r, i) => {
        const [lr, li] = clog(r, i);
        return [lr * INV_LN10, li * INV_LN10];
      };
      break;
    case OpUnary.SQRT:
      fn = csqrt;
      break;
    case OpUnary.ABS:
      throw new Error(
        "tsComplexUnaryElemwise: ABS is not supported (use tsComplexAbs)"
      );
    case OpUnary.FLOOR:
      for (let i = 0; i < n; i++) {
        outRe[i] = Math.floor(aRe[i]);
        outIm[i] = aIm ? Math.floor(aIm[i]) : 0;
      }
      return;
    case OpUnary.CEIL:
      for (let i = 0; i < n; i++) {
        outRe[i] = Math.ceil(aRe[i]);
        outIm[i] = aIm ? Math.ceil(aIm[i]) : 0;
      }
      return;
    case OpUnary.ROUND:
      for (let i = 0; i < n; i++) {
        outRe[i] = roundHaf(aRe[i]);
        outIm[i] = aIm ? roundHaf(aIm[i]) : 0;
      }
      return;
    case OpUnary.TRUNC:
      for (let i = 0; i < n; i++) {
        outRe[i] = Math.trunc(aRe[i]);
        outIm[i] = aIm ? Math.trunc(aIm[i]) : 0;
      }
      return;
    case OpUnary.SIN:
      fn = csin;
      break;
    case OpUnary.COS:
      fn = ccos;
      break;
    case OpUnary.TAN:
      fn = ctan;
      break;
    case OpUnary.ASIN:
      fn = casin;
      break;
    case OpUnary.ACOS:
      fn = cacos;
      break;
    case OpUnary.ATAN:
      fn = catan;
      break;
    case OpUnary.SINH:
      fn = csinh;
      break;
    case OpUnary.COSH:
      fn = ccosh;
      break;
    case OpUnary.TANH:
      fn = ctanh;
      break;
    case OpUnary.SIGN:
      fn = csign;
      break;
    default:
      throw new Error(`tsComplexUnaryElemwise: unknown op ${op}`);
  }
  const f = fn!;
  for (let i = 0; i < n; i++) {
    const [r, im] = f(aRe[i], aIm ? aIm[i] : 0);
    outRe[i] = r;
    outIm[i] = im;
  }
}

export function tsComplexAbs(
  n: number,
  aRe: Float64Array,
  aIm: Float64Array | null,
  out: Float64Array
): void {
  if (aIm) {
    for (let i = 0; i < n; i++) out[i] = Math.hypot(aRe[i], aIm[i]);
  } else {
    for (let i = 0; i < n; i++) out[i] = Math.abs(aRe[i]);
  }
}
