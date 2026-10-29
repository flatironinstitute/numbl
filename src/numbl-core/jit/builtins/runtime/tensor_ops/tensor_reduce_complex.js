// JS sibling of `tensor_reduce_complex.h`. Complex-tensor reductions.
// Mirrors the real reducer shape (sum/prod/mean → complex; min/max
// → complex via magnitude+atan2 tiebreak; any/all → real).

import { jit_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { jit_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";
import { jit_raise_bad_dim } from "../system/raise.js";

function cSqueezeTrailing(dims) {
  while (dims.length > 2 && dims[dims.length - 1] === 1) dims.pop();
  return dims;
}

function cReduceLaneIm(t, i) {
  return t.imag !== undefined ? t.imag[i] : 0;
}

// True when the tensor carries no imaginary content (no lane, or every
// element zero). Such a tensor is real in value — min/max must order by
// value, not magnitude, to match the interpreter and MATLAB on real data.
function cReduceAllImagZero(t) {
  if (t.imag === undefined) return true;
  for (let i = 0; i < t.imag.length; i++) {
    if (t.imag[i] !== 0) return false;
  }
  return true;
}

// Numeric (sum/prod/mean) — complex accumulator { re, im }.
function complexAccumAll(t, init, accum, finalize) {
  let acc = { ...init };
  for (let i = 0; i < t.data.length; i++) {
    acc = accum(acc, { re: t.data[i], im: cReduceLaneIm(t, i) });
  }
  return finalize(acc, t.data.length);
}

function complexAccumDim(t, dim, init, accum, finalize) {
  if (dim < 1) jit_raise_bad_dim();
  if (dim > t.shape.length) {
    // No-op axis: return a fresh complex copy.
    const out = jit_tensor_alloc_nd_complex(t.shape.length, t.shape.slice());
    out.data.set(t.data);
    if (t.imag !== undefined) out.imag.set(t.imag);
    return out;
  }
  const dimIdx = dim - 1;
  const axis = t.shape[dimIdx];
  let before = 1;
  for (let i = 0; i < dimIdx; i++) before *= t.shape[i];
  let after = 1;
  for (let i = dimIdx + 1; i < t.shape.length; i++) after *= t.shape[i];
  const outDims = t.shape.slice();
  outDims[dimIdx] = 1;
  cSqueezeTrailing(outDims);
  const out = jit_tensor_alloc_nd_complex(outDims.length, outDims);
  for (let aft = 0; aft < after; aft++) {
    for (let bef = 0; bef < before; bef++) {
      const base = aft * before * axis + bef;
      let acc = { ...init };
      for (let k = 0; k < axis; k++) {
        const off = base + k * before;
        acc = accum(acc, { re: t.data[off], im: cReduceLaneIm(t, off) });
      }
      const fin = finalize(acc, axis);
      const dst = aft * before + bef;
      out.data[dst] = fin.re;
      out.imag[dst] = fin.im;
    }
  }
  return out;
}

const cSumInit = { re: 0, im: 0 };
const cProdInit = { re: 1, im: 0 };
const cSumAccum = (a, x) => ({ re: a.re + x.re, im: a.im + x.im });
const cProdAccum = (a, x) => ({
  re: a.re * x.re - a.im * x.im,
  im: a.re * x.im + a.im * x.re,
});
const cIdFinalize = a => a;
const cMeanFinalize = (a, n) =>
  n === 0 ? { re: NaN, im: NaN } : { re: a.re / n, im: a.im / n };

export const jit_sum_complex_all = t =>
  complexAccumAll(t, cSumInit, cSumAccum, cIdFinalize);
export const jit_sum_complex_dim = (t, d) =>
  complexAccumDim(t, d, cSumInit, cSumAccum, cIdFinalize);
export const jit_prod_complex_all = t =>
  complexAccumAll(t, cProdInit, cProdAccum, cIdFinalize);
export const jit_prod_complex_dim = (t, d) =>
  complexAccumDim(t, d, cProdInit, cProdAccum, cIdFinalize);
export const jit_mean_complex_all = t =>
  complexAccumAll(t, cSumInit, cSumAccum, cMeanFinalize);
export const jit_mean_complex_dim = (t, d) =>
  complexAccumDim(t, d, cSumInit, cSumAccum, cMeanFinalize);

// Min / max — magnitude compare with atan2 tiebreak (numbl's
// complexIsBetter). Skip NaN-lane elements; result is complex.
function complexMinmaxAll(t, cmp) {
  const realMode = cReduceAllImagZero(t);
  let found = false;
  let mRe = NaN;
  let mIm = 0;
  for (let i = 0; i < t.data.length; i++) {
    const xr = t.data[i];
    const xi = cReduceLaneIm(t, i);
    if (xr !== xr || xi !== xi) continue;
    const better = realMode
      ? cmp === "<"
        ? xr < mRe
        : xr > mRe
      : complexBetter(xr, xi, mRe, mIm, cmp);
    if (!found || better) {
      mRe = xr;
      mIm = xi;
      found = true;
    }
  }
  return { re: mRe, im: mIm };
}

function complexBetter(aRe, aIm, bRe, bIm, cmp) {
  const absA = Math.hypot(aRe, aIm);
  const absB = Math.hypot(bRe, bIm);
  if (absA !== absB) return cmp === "<" ? absA < absB : absA > absB;
  return cmp === "<"
    ? Math.atan2(aIm, aRe) < Math.atan2(bIm, bRe)
    : Math.atan2(aIm, aRe) > Math.atan2(bIm, bRe);
}

function complexMinmaxDim(t, dim, cmp) {
  if (dim < 1) jit_raise_bad_dim();
  if (dim > t.shape.length) {
    const out = jit_tensor_alloc_nd_complex(t.shape.length, t.shape.slice());
    out.data.set(t.data);
    if (t.imag !== undefined) out.imag.set(t.imag);
    return out;
  }
  const dimIdx = dim - 1;
  const realMode = cReduceAllImagZero(t);
  const axis = t.shape[dimIdx];
  let before = 1;
  for (let i = 0; i < dimIdx; i++) before *= t.shape[i];
  let after = 1;
  for (let i = dimIdx + 1; i < t.shape.length; i++) after *= t.shape[i];
  const outDims = t.shape.slice();
  outDims[dimIdx] = 1;
  cSqueezeTrailing(outDims);
  const out = jit_tensor_alloc_nd_complex(outDims.length, outDims);
  for (let aft = 0; aft < after; aft++) {
    for (let bef = 0; bef < before; bef++) {
      const base = aft * before * axis + bef;
      let found = false;
      let mRe = NaN;
      let mIm = 0;
      for (let k = 0; k < axis; k++) {
        const off = base + k * before;
        const xr = t.data[off];
        const xi = cReduceLaneIm(t, off);
        if (xr !== xr || xi !== xi) continue;
        const better = realMode
          ? cmp === "<"
            ? xr < mRe
            : xr > mRe
          : complexBetter(xr, xi, mRe, mIm, cmp);
        if (!found || better) {
          mRe = xr;
          mIm = xi;
          found = true;
        }
      }
      const dst = aft * before + bef;
      out.data[dst] = mRe;
      out.imag[dst] = mIm;
    }
  }
  return out;
}

export const jit_min_complex_all = t => complexMinmaxAll(t, "<");
export const jit_min_complex_dim = (t, d) => complexMinmaxDim(t, d, "<");
export const jit_max_complex_all = t => complexMinmaxAll(t, ">");
export const jit_max_complex_dim = (t, d) => complexMinmaxDim(t, d, ">");

// any / all — real result; toBool per element (either lane nonzero).
function complexLogicalAll(t, emptyResult, shortPredicate, toBool) {
  if (t.data.length === 0) return emptyResult;
  for (let i = 0; i < t.data.length; i++) {
    const x = toBool(t.data[i], cReduceLaneIm(t, i));
    if (shortPredicate(x)) return emptyResult === 1 ? 0 : 1;
  }
  return emptyResult;
}

function complexLogicalDim(t, dim, emptyResult, shortPredicate, toBool) {
  if (dim < 1) jit_raise_bad_dim();
  if (dim > t.shape.length) {
    const out = jit_tensor_alloc_nd(t.shape.length, t.shape.slice());
    for (let i = 0; i < t.data.length; i++) {
      out.data[i] = toBool(t.data[i], cReduceLaneIm(t, i)) ? 1 : 0;
    }
    return out;
  }
  const dimIdx = dim - 1;
  const axis = t.shape[dimIdx];
  let before = 1;
  for (let i = 0; i < dimIdx; i++) before *= t.shape[i];
  let after = 1;
  for (let i = dimIdx + 1; i < t.shape.length; i++) after *= t.shape[i];
  const outDims = t.shape.slice();
  outDims[dimIdx] = 1;
  cSqueezeTrailing(outDims);
  const out = jit_tensor_alloc_nd(outDims.length, outDims);
  for (let aft = 0; aft < after; aft++) {
    for (let bef = 0; bef < before; bef++) {
      const base = aft * before * axis + bef;
      let res = emptyResult;
      for (let k = 0; k < axis; k++) {
        const off = base + k * before;
        const x = toBool(t.data[off], cReduceLaneIm(t, off));
        if (shortPredicate(x)) {
          res = emptyResult === 1 ? 0 : 1;
          break;
        }
      }
      out.data[aft * before + bef] = res;
    }
  }
  return out;
}

const cAnyShort = x => x;
const cAllShort = x => !x;
// `any` ignores NaN elements (either part NaN), as for real input; `all`
// counts them as nonzero.
const cAnyBool = (xr, xi) =>
  (xr !== 0 || xi !== 0) && !Number.isNaN(xr) && !Number.isNaN(xi);
const cAllBool = (xr, xi) => xr !== 0 || xi !== 0;
export const jit_any_complex_all = t =>
  complexLogicalAll(t, 0, cAnyShort, cAnyBool);
export const jit_any_complex_dim = (t, d) =>
  complexLogicalDim(t, d, 0, cAnyShort, cAnyBool);
export const jit_all_complex_all = t =>
  complexLogicalAll(t, 1, cAllShort, cAllBool);
export const jit_all_complex_dim = (t, d) =>
  complexLogicalDim(t, d, 1, cAllShort, cAllBool);
