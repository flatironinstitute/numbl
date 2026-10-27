// JS sibling of `tensor_elemwise_real_fn.h`. Per-op tensor kernels
// for the function-call binary family (`atan2`, `hypot`, `mod`,
// `rem`, `power`, two-argument `max`/`min`). Mirrors
// `tensor_elemwise_real.js`: each op has its own loop with the scalar
// function called directly (a shared kernel taking it as a callback
// goes megamorphic in V8); only the broadcast path is shared.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

function bcast(a, b, fn) {
  const ndim = Math.max(a.shape.length, b.shape.length);
  const ashape = a.shape.slice();
  const bshape = b.shape.slice();
  while (ashape.length < ndim) ashape.push(1);
  while (bshape.length < ndim) bshape.push(1);
  const outShape = new Array(ndim);
  // Singleton axis expands to the other; a 0-size axis vs a 1-size axis
  // broadcasts to 0, not Math.max's 1 (mirrors the C (adim==1)?bdim:adim).
  for (let i = 0; i < ndim; i++)
    outShape[i] = ashape[i] === 1 ? bshape[i] : ashape[i];
  const r = mtoc2_tensor_alloc_nd(ndim, outShape);
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
    r.data[k] = fn(a.data[aOff], b.data[bOff]);
    for (let i = 0; i < ndim; i++) {
      idx[i]++;
      if (idx[i] < outShape[i]) break;
      idx[i] = 0;
    }
  }
  return r;
}

// MATLAB-style mod (sign tracks divisor). Mirrors `mtoc2_mod_real`
// in the .h. Special-case: `mod(a, 0) = a`.
function mtoc2_mod_real_js(a, b) {
  if (b === 0) return a;
  let r = a % b;
  if (r !== 0 && r < 0 !== b < 0) r += b;
  return r;
}

// rem: JS `%` matches C `fmod` semantics (sign tracks a).
const remFn = (a, b) => a % b;

// Two-argument max/min: the non-NaN operand wins, as in MATLAB and C99
// fmax/fmin (JS Math.max/min propagate NaN).
const maxFn = (a, b) => (a !== a ? b : b !== b ? a : Math.max(a, b));
const minFn = (a, b) => (a !== a ? b : b !== b ? a : Math.min(a, b));

export function mtoc2_tensor_atan2_tt(a, b) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = Math.atan2(x[i], y[i]);
  return r;
}
export function mtoc2_tensor_atan2_ts(a, s) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = Math.atan2(x[i], s);
  return r;
}
export function mtoc2_tensor_atan2_st(s, a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = Math.atan2(s, x[i]);
  return r;
}
export const mtoc2_tensor_atan2_bcast_tt = (a, b) => bcast(a, b, Math.atan2);

// hypot is commutative — no `_st` in the C side.
export function mtoc2_tensor_hypot_tt(a, b) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = Math.hypot(x[i], y[i]);
  return r;
}
export function mtoc2_tensor_hypot_ts(a, s) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = Math.hypot(x[i], s);
  return r;
}
export const mtoc2_tensor_hypot_bcast_tt = (a, b) => bcast(a, b, Math.hypot);

export function mtoc2_tensor_power_tt(a, b) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = Math.pow(x[i], y[i]);
  return r;
}
export function mtoc2_tensor_power_ts(a, s) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = Math.pow(x[i], s);
  return r;
}
export function mtoc2_tensor_power_st(s, a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = Math.pow(s, x[i]);
  return r;
}
export const mtoc2_tensor_power_bcast_tt = (a, b) => bcast(a, b, Math.pow);

export function mtoc2_tensor_rem_tt(a, b) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = remFn(x[i], y[i]);
  return r;
}
export function mtoc2_tensor_rem_ts(a, s) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = remFn(x[i], s);
  return r;
}
export function mtoc2_tensor_rem_st(s, a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = remFn(s, x[i]);
  return r;
}
export const mtoc2_tensor_rem_bcast_tt = (a, b) => bcast(a, b, remFn);

export function mtoc2_tensor_mod_tt(a, b) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = mtoc2_mod_real_js(x[i], y[i]);
  return r;
}
export function mtoc2_tensor_mod_ts(a, s) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = mtoc2_mod_real_js(x[i], s);
  return r;
}
export function mtoc2_tensor_mod_st(s, a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = mtoc2_mod_real_js(s, x[i]);
  return r;
}
export const mtoc2_tensor_mod_bcast_tt = (a, b) =>
  bcast(a, b, mtoc2_mod_real_js);

export function mtoc2_tensor_max_tt(a, b) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = maxFn(x[i], y[i]);
  return r;
}
export function mtoc2_tensor_max_ts(a, s) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = maxFn(x[i], s);
  return r;
}
export const mtoc2_tensor_max_bcast_tt = (a, b) => bcast(a, b, maxFn);

export function mtoc2_tensor_min_tt(a, b) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = minFn(x[i], y[i]);
  return r;
}
export function mtoc2_tensor_min_ts(a, s) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = minFn(x[i], s);
  return r;
}
export const mtoc2_tensor_min_bcast_tt = (a, b) => bcast(a, b, minFn);
