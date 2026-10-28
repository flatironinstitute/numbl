// JS sibling of `tensor_elemwise_complex.h`. Per-op tensor-tensor /
// tensor-scalar / scalar-tensor / broadcast helpers for complex
// operands. Mirrors the real elemwise scaffold but each element is a
// `{re, im}` pair carried by the `data` + `imag` lanes.

import { jit_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";
import { jit_cadd, jit_csub, jit_cmul, jit_cneg } from "../system/cscalar.js";
import { jit_cdiv } from "../system/cdiv.js";

function cLaneOf(t, idx) {
  return { re: t.data[idx], im: t.imag != null ? t.imag[idx] : 0 };
}

function cTtKernel(a, b, fn) {
  const r = jit_tensor_alloc_nd_complex(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    const v = fn(cLaneOf(a, i), cLaneOf(b, i));
    r.data[i] = v.re;
    r.imag[i] = v.im;
  }
  return r;
}

function cTsKernel(a, s, fn) {
  const r = jit_tensor_alloc_nd_complex(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    const v = fn(cLaneOf(a, i), s);
    r.data[i] = v.re;
    r.imag[i] = v.im;
  }
  return r;
}

function cStKernel(s, a, fn) {
  const r = jit_tensor_alloc_nd_complex(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    const v = fn(s, cLaneOf(a, i));
    r.data[i] = v.re;
    r.imag[i] = v.im;
  }
  return r;
}

// Broadcast helper — same shape walk as the real elemwise broadcast
// path but each cell carries two lanes.
function cBcastKernel(a, b, fn) {
  const ndim = Math.max(a.shape.length, b.shape.length);
  const ashape = a.shape.slice();
  const bshape = b.shape.slice();
  while (ashape.length < ndim) ashape.push(1);
  while (bshape.length < ndim) bshape.push(1);
  const outShape = new Array(ndim);
  for (let i = 0; i < ndim; i++) {
    // Singleton axis expands to the other; a 0-size axis vs a 1-size axis
    // broadcasts to 0, not Math.max's 1 (mirrors the C (adim==1)?bdim:adim).
    outShape[i] = ashape[i] === 1 ? bshape[i] : ashape[i];
  }
  const r = jit_tensor_alloc_nd_complex(ndim, outShape);
  const aStrides = new Array(ndim);
  const bStrides = new Array(ndim);
  let as = 1;
  let bs = 1;
  for (let i = 0; i < ndim; i++) {
    aStrides[i] = ashape[i] === 1 ? 0 : as;
    bStrides[i] = bshape[i] === 1 ? 0 : bs;
    as *= ashape[i];
    bs *= bshape[i];
  }
  const idx = new Array(ndim).fill(0);
  for (let k = 0; k < r.data.length; k++) {
    let aOff = 0;
    let bOff = 0;
    for (let i = 0; i < ndim; i++) {
      aOff += idx[i] * aStrides[i];
      bOff += idx[i] * bStrides[i];
    }
    const v = fn(cLaneOf(a, aOff), cLaneOf(b, bOff));
    r.data[k] = v.re;
    r.imag[k] = v.im;
    for (let i = 0; i < ndim; i++) {
      idx[i]++;
      if (idx[i] < outShape[i]) break;
      idx[i] = 0;
    }
  }
  return r;
}

// Mixed real/complex times / rdivide, as `tensor_elemwise_complex.h`
// (and the interpreter's `mixedRealComplex`): an operand with no imag
// lane, or an all-zero one, acts as real, and a real operand scales each
// part of a complex one without promotion, so a result part stays 0
// where the complex operand's part is 0 (no NaN*0 = NaN).
function cActsReal(t) {
  if (t.imag == null) return true;
  for (let i = 0; i < t.imag.length; i++) if (t.imag[i] !== 0) return false;
  return true;
}

function cTimesMixed(a, ar, b, br) {
  if (ar && br) return { re: a.re * b.re, im: 0 };
  if (br)
    return {
      re: a.re === 0 ? 0 : a.re * b.re,
      im: a.im === 0 ? 0 : a.im * b.re,
    };
  if (ar)
    return {
      re: b.re === 0 ? 0 : b.re * a.re,
      im: b.im === 0 ? 0 : b.im * a.re,
    };
  return jit_cmul(a, b);
}

function cRdivideMixed(a, ar, b, br) {
  if (ar && br) return { re: a.re / b.re, im: 0 };
  if (br)
    return {
      re: a.re === 0 ? 0 : a.re / b.re,
      im: a.im === 0 ? 0 : a.im / b.re,
    };
  if (ar && b.im === 0) return { re: a.re / b.re, im: 0 };
  return jit_cdiv(ar ? { re: a.re, im: 0 } : a, b);
}

// ── plus ──────────────────────────────────────────────────────────────────
export function jit_tensor_plus_complex_tt(a, b) {
  return cTtKernel(a, b, jit_cadd);
}
export function jit_tensor_plus_complex_ts(a, s) {
  return cTsKernel(a, s, jit_cadd);
}
export function jit_tensor_plus_complex_bcast_tt(a, b) {
  return cBcastKernel(a, b, jit_cadd);
}

// ── minus ─────────────────────────────────────────────────────────────────
export function jit_tensor_minus_complex_tt(a, b) {
  return cTtKernel(a, b, jit_csub);
}
export function jit_tensor_minus_complex_ts(a, s) {
  return cTsKernel(a, s, jit_csub);
}
export function jit_tensor_minus_complex_st(s, a) {
  return cStKernel(s, a, jit_csub);
}
export function jit_tensor_minus_complex_bcast_tt(a, b) {
  return cBcastKernel(a, b, jit_csub);
}

// ── times ──────────────────────────────────────────────────────────────────
export function jit_tensor_times_complex_tt(a, b) {
  const ar = cActsReal(a),
    br = cActsReal(b);
  return cTtKernel(a, b, (x, y) => cTimesMixed(x, ar, y, br));
}
export function jit_tensor_times_complex_ts(a, s) {
  const ar = cActsReal(a),
    sr = s.im === 0;
  return cTsKernel(a, s, (x, y) => cTimesMixed(x, ar, y, sr));
}
export function jit_tensor_times_complex_bcast_tt(a, b) {
  const ar = cActsReal(a),
    br = cActsReal(b);
  return cBcastKernel(a, b, (x, y) => cTimesMixed(x, ar, y, br));
}

// ── rdivide ───────────────────────────────────────────────────────────────
export function jit_tensor_rdivide_complex_tt(a, b) {
  const ar = cActsReal(a),
    br = cActsReal(b);
  return cTtKernel(a, b, (x, y) => cRdivideMixed(x, ar, y, br));
}
export function jit_tensor_rdivide_complex_ts(a, s) {
  const ar = cActsReal(a),
    sr = s.im === 0;
  return cTsKernel(a, s, (x, y) => cRdivideMixed(x, ar, y, sr));
}
export function jit_tensor_rdivide_complex_st(s, a) {
  const sr = s.im === 0,
    ar = cActsReal(a);
  return cStKernel(s, a, (x, y) => cRdivideMixed(x, sr, y, ar));
}
export function jit_tensor_rdivide_complex_bcast_tt(a, b) {
  const ar = cActsReal(a),
    br = cActsReal(b);
  return cBcastKernel(a, b, (x, y) => cRdivideMixed(x, ar, y, br));
}

// ── uminus ────────────────────────────────────────────────────────────────
export function jit_tensor_uminus_complex(a) {
  const r = jit_tensor_alloc_nd_complex(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    const v = jit_cneg(cLaneOf(a, i));
    r.data[i] = v.re;
    r.imag[i] = v.im;
  }
  return r;
}
