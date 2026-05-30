// JS sibling of `tensor_elemwise_real.h`. Per-op tensor-tensor /
// tensor-scalar / scalar-tensor / broadcast helpers. C macros
// generate these; JS uses a single generic kernel + thin wrappers
// to preserve the call-site shape the emitter produces.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

// uminus on a tensor — separate signature (unary). Kept here because
// the C side groups it with the elemwise file.
function uminus_kernel(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = -a.data[i];
  return r;
}

function shapes_equal(x, y) {
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}

function tt_kernel(a, b, fn) {
  // Same-shape fast path. The emitter picks `_tt` whenever it can't
  // statically prove broadcasting is needed (unknown dims), so the
  // shapes must be checked at runtime: equal → elementwise; otherwise
  // defer to the broadcast kernel, which broadcasts compatible shapes
  // and errors on incompatible ones. Without this, a runtime mismatch
  // read past `b` and silently produced NaN.
  if (!shapes_equal(a.shape, b.shape)) return bcast_kernel(a, b, fn);
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = fn(a.data[i], b.data[i]);
  return r;
}

function ts_kernel(a, s, fn) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = fn(a.data[i], s);
  return r;
}

function st_kernel(s, a, fn) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = fn(s, a.data[i]);
  return r;
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
  const r = mtoc2_tensor_alloc_nd(ndim, outShape);
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

export function mtoc2_tensor_uminus(a) {
  return uminus_kernel(a);
}

const opPlus = (x, y) => x + y;
const opMinus = (x, y) => x - y;
const opTimes = (x, y) => x * y;
const opRdivide = (x, y) => x / y;

export function mtoc2_tensor_plus_tt(a, b) {
  return tt_kernel(a, b, opPlus);
}
export function mtoc2_tensor_minus_tt(a, b) {
  return tt_kernel(a, b, opMinus);
}
export function mtoc2_tensor_times_tt(a, b) {
  return tt_kernel(a, b, opTimes);
}
export function mtoc2_tensor_rdivide_tt(a, b) {
  return tt_kernel(a, b, opRdivide);
}

export function mtoc2_tensor_plus_ts(a, s) {
  return ts_kernel(a, s, opPlus);
}
export function mtoc2_tensor_minus_ts(a, s) {
  return ts_kernel(a, s, opMinus);
}
export function mtoc2_tensor_times_ts(a, s) {
  return ts_kernel(a, s, opTimes);
}
export function mtoc2_tensor_rdivide_ts(a, s) {
  return ts_kernel(a, s, opRdivide);
}

// Non-commutative ops also need scalar-OP-tensor.
export function mtoc2_tensor_minus_st(s, a) {
  return st_kernel(s, a, opMinus);
}
export function mtoc2_tensor_rdivide_st(s, a) {
  return st_kernel(s, a, opRdivide);
}

export function mtoc2_tensor_plus_bcast_tt(a, b) {
  return bcast_kernel(a, b, opPlus);
}
export function mtoc2_tensor_minus_bcast_tt(a, b) {
  return bcast_kernel(a, b, opMinus);
}
export function mtoc2_tensor_times_bcast_tt(a, b) {
  return bcast_kernel(a, b, opTimes);
}
export function mtoc2_tensor_rdivide_bcast_tt(a, b) {
  return bcast_kernel(a, b, opRdivide);
}
