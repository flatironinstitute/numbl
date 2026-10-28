// JS sibling of `tensor_elemwise_real.h`. Per-op tensor-tensor /
// tensor-scalar / scalar-tensor / broadcast helpers.
//
// Each op has its own loop with the operator written inline, like the
// C macros expand to. A shared kernel taking the op as a callback would
// see a different callback per op, go megamorphic in V8 and stop
// inlining it, which made large element-wise work several times slower
// than the interpreter. Only the (rare) broadcast path is shared.
//
// Tensors of at least EW_HOST_MIN elements go through the host's kernels
// (`$elemwise`, the interpreter's native-or-TS kernels) when installed, so
// large work runs as SIMD C and matches --opt 0. Below it the per-call
// overhead outweighs the JS loop.

import { jit_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

const EW_HOST_MIN = 1024;

function ew_host(n) {
  return n >= EW_HOST_MIN && typeof $elemwise === "object" && $elemwise;
}

function ew_alloc_like(a) {
  return jit_tensor_alloc_nd(a.shape.length, a.shape);
}

// uminus on a tensor — separate signature (unary). Kept here because
// the C side groups it with the elemwise file.
export function jit_tensor_uminus(a) {
  const r = ew_alloc_like(a);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = -x[i];
  return r;
}

function shapes_equal(x, y) {
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}

// Broadcast helper. Pads the shorter shape with trailing 1s on the
// right; treats any axis whose size is 1 as a stride-0 axis (so the
// scalar slot replicates).
function bcast_kernel(a, b, fn) {
  const ndim = Math.max(a.shape.length, b.shape.length);
  const ashape = a.shape.slice();
  const bshape = b.shape.slice();
  while (ashape.length < ndim) ashape.push(1);
  while (bshape.length < ndim) bshape.push(1);
  const outShape = new Array(ndim);
  for (let i = 0; i < ndim; i++) {
    // MATLAB broadcast rule: axis sizes must be equal or one must be 1.
    if (ashape[i] !== bshape[i] && ashape[i] !== 1 && bshape[i] !== 1) {
      throw new Error("Matrix dimensions must agree");
    }
    // A singleton axis expands to the other; otherwise they're equal. Must
    // NOT use Math.max: a 0-size axis vs a 1-size axis broadcasts to 0, not 1
    // (mirrors the C twin's (adim==1)?bdim:adim).
    outShape[i] = ashape[i] === 1 ? bshape[i] : ashape[i];
  }
  const r = jit_tensor_alloc_nd(ndim, outShape);
  // Column-major strides for each operand. A singleton axis gets a
  // stride of 0 so its index "doesn't move".
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
  // Walk the output index space column-major. Decompose linear
  // output index `k` into per-axis subscripts, then accumulate the
  // operand offsets via the per-axis strides.
  const idx = new Array(ndim).fill(0);
  for (let k = 0; k < r.data.length; k++) {
    let aOff = 0;
    let bOff = 0;
    for (let i = 0; i < ndim; i++) {
      aOff += idx[i] * aStrides[i];
      bOff += idx[i] * bStrides[i];
    }
    r.data[k] = fn(a.data[aOff], b.data[bOff]);
    // Increment subscripts (column-major: axis 0 varies fastest).
    for (let i = 0; i < ndim; i++) {
      idx[i]++;
      if (idx[i] < outShape[i]) break;
      idx[i] = 0;
    }
  }
  return r;
}

const opPlus = (x, y) => x + y;
const opMinus = (x, y) => x - y;
const opTimes = (x, y) => x * y;
const opRdivide = (x, y) => x / y;

// `_tt`: the emitter picks it whenever it can't statically prove the
// shapes match (unknown dims), so the shapes are re-checked at runtime: a
// mismatch defers to the broadcast kernel, which broadcasts compatible
// shapes and errors on incompatible ones (instead of reading past `b`).
export function jit_tensor_plus_tt(a, b) {
  if (!shapes_equal(a.shape, b.shape)) return bcast_kernel(a, b, opPlus);
  const r = ew_alloc_like(a);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.binary("plus", x, y, o);
  else for (let i = 0; i < o.length; i++) o[i] = x[i] + y[i];
  return r;
}
export function jit_tensor_minus_tt(a, b) {
  if (!shapes_equal(a.shape, b.shape)) return bcast_kernel(a, b, opMinus);
  const r = ew_alloc_like(a);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.binary("minus", x, y, o);
  else for (let i = 0; i < o.length; i++) o[i] = x[i] - y[i];
  return r;
}
export function jit_tensor_times_tt(a, b) {
  if (!shapes_equal(a.shape, b.shape)) return bcast_kernel(a, b, opTimes);
  const r = ew_alloc_like(a);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.binary("times", x, y, o);
  else for (let i = 0; i < o.length; i++) o[i] = x[i] * y[i];
  return r;
}
export function jit_tensor_rdivide_tt(a, b) {
  if (!shapes_equal(a.shape, b.shape)) return bcast_kernel(a, b, opRdivide);
  const r = ew_alloc_like(a);
  const x = a.data;
  const y = b.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.binary("rdivide", x, y, o);
  else for (let i = 0; i < o.length; i++) o[i] = x[i] / y[i];
  return r;
}

export function jit_tensor_plus_ts(a, s) {
  const r = ew_alloc_like(a);
  const x = a.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.scalarBinary("plus", s, x, false, o);
  else for (let i = 0; i < o.length; i++) o[i] = x[i] + s;
  return r;
}
export function jit_tensor_minus_ts(a, s) {
  const r = ew_alloc_like(a);
  const x = a.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.scalarBinary("minus", s, x, false, o);
  else for (let i = 0; i < o.length; i++) o[i] = x[i] - s;
  return r;
}
export function jit_tensor_times_ts(a, s) {
  const r = ew_alloc_like(a);
  const x = a.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.scalarBinary("times", s, x, false, o);
  else for (let i = 0; i < o.length; i++) o[i] = x[i] * s;
  return r;
}
export function jit_tensor_rdivide_ts(a, s) {
  const r = ew_alloc_like(a);
  const x = a.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.scalarBinary("rdivide", s, x, false, o);
  else for (let i = 0; i < o.length; i++) o[i] = x[i] / s;
  return r;
}

// Non-commutative ops also need scalar-OP-tensor.
export function jit_tensor_minus_st(s, a) {
  const r = ew_alloc_like(a);
  const x = a.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.scalarBinary("minus", s, x, true, o);
  else for (let i = 0; i < o.length; i++) o[i] = s - x[i];
  return r;
}
export function jit_tensor_rdivide_st(s, a) {
  const r = ew_alloc_like(a);
  const x = a.data;
  const o = r.data;
  const h = ew_host(o.length);
  if (h) h.scalarBinary("rdivide", s, x, true, o);
  else for (let i = 0; i < o.length; i++) o[i] = s / x[i];
  return r;
}

export function jit_tensor_plus_bcast_tt(a, b) {
  return bcast_kernel(a, b, opPlus);
}
export function jit_tensor_minus_bcast_tt(a, b) {
  return bcast_kernel(a, b, opMinus);
}
export function jit_tensor_times_bcast_tt(a, b) {
  return bcast_kernel(a, b, opTimes);
}
export function jit_tensor_rdivide_bcast_tt(a, b) {
  return bcast_kernel(a, b, opRdivide);
}
