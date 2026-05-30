// JS sibling of `tensor_elemwise_real_fn.h`. Per-op tensor kernels
// for the function-call binary family (`atan2`, `hypot`, `mod`,
// `rem`, `power`). Mirrors `tensor_elemwise_real.js`'s structure;
// only the scalar fn differs.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

function tt(a, b, fn) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = fn(a.data[i], b.data[i]);
  return r;
}
function ts(a, s, fn) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = fn(a.data[i], s);
  return r;
}
function st(s, a, fn) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = fn(s, a.data[i]);
  return r;
}
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

// atan2 / hypot / power
export const mtoc2_tensor_atan2_tt = (a, b) => tt(a, b, Math.atan2);
export const mtoc2_tensor_atan2_ts = (a, s) => ts(a, s, Math.atan2);
export const mtoc2_tensor_atan2_st = (s, a) => st(s, a, Math.atan2);
export const mtoc2_tensor_atan2_bcast_tt = (a, b) => bcast(a, b, Math.atan2);

export const mtoc2_tensor_hypot_tt = (a, b) => tt(a, b, Math.hypot);
export const mtoc2_tensor_hypot_ts = (a, s) => ts(a, s, Math.hypot);
// hypot is commutative — no `_st` in the C side; included for symmetry.
export const mtoc2_tensor_hypot_bcast_tt = (a, b) => bcast(a, b, Math.hypot);

export const mtoc2_tensor_power_tt = (a, b) => tt(a, b, Math.pow);
export const mtoc2_tensor_power_ts = (a, s) => ts(a, s, Math.pow);
export const mtoc2_tensor_power_st = (s, a) => st(s, a, Math.pow);
export const mtoc2_tensor_power_bcast_tt = (a, b) => bcast(a, b, Math.pow);

// rem: JS `%` matches C `fmod` semantics (sign tracks a).
const remFn = (a, b) => a % b;
export const mtoc2_tensor_rem_tt = (a, b) => tt(a, b, remFn);
export const mtoc2_tensor_rem_ts = (a, s) => ts(a, s, remFn);
export const mtoc2_tensor_rem_st = (s, a) => st(s, a, remFn);
export const mtoc2_tensor_rem_bcast_tt = (a, b) => bcast(a, b, remFn);

// mod: MATLAB convention (sign tracks b).
export const mtoc2_tensor_mod_tt = (a, b) => tt(a, b, mtoc2_mod_real_js);
export const mtoc2_tensor_mod_ts = (a, s) => ts(a, s, mtoc2_mod_real_js);
export const mtoc2_tensor_mod_st = (s, a) => st(s, a, mtoc2_mod_real_js);
export const mtoc2_tensor_mod_bcast_tt = (a, b) =>
  bcast(a, b, mtoc2_mod_real_js);
