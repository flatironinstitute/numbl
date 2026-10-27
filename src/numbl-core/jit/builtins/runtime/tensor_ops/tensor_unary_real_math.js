// JS sibling of `tensor_unary_real_math.h`. Per-op unary math
// kernels for tensors (`mtoc2_tensor_sin`, `_cos`, etc.). Matches
// the per-op function names the codegen emits. Each op has its own
// loop (not a shared kernel taking the op as a callback, which goes
// megamorphic in V8 and stops inlining it). Large tensors go through
// the host's kernels when installed; see `tensor_elemwise_real.js`.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

const EW_UNARY_HOST_MIN = 1024;

function ew_host_unary(op, x, o) {
  return (
    o.length >= EW_UNARY_HOST_MIN &&
    typeof $elemwise === "object" &&
    !!$elemwise &&
    $elemwise.unary(op, x, o)
  );
}

function matlabRound(x) {
  return Math.sign(x) * Math.round(Math.abs(x));
}

// realsqrt(x): MATLAB raises an error rather than returning a complex
// result for x < 0. NaN passes through as NaN.
export function mtoc2_realsqrt(x) {
  if (x < 0) throw new Error("Realsqrt produced complex result.");
  return Math.sqrt(x);
}

export function mtoc2_tensor_cos(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("cos", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.cos(x[i]);
  }
  return r;
}
export function mtoc2_tensor_sin(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("sin", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.sin(x[i]);
  }
  return r;
}
export function mtoc2_tensor_tan(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("tan", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.tan(x[i]);
  }
  return r;
}
export function mtoc2_tensor_atan(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("atan", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.atan(x[i]);
  }
  return r;
}
export function mtoc2_tensor_sinh(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("sinh", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.sinh(x[i]);
  }
  return r;
}
export function mtoc2_tensor_cosh(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("cosh", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.cosh(x[i]);
  }
  return r;
}
export function mtoc2_tensor_tanh(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("tanh", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.tanh(x[i]);
  }
  return r;
}
export function mtoc2_tensor_asinh(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("asinh", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.asinh(x[i]);
  }
  return r;
}
export function mtoc2_tensor_exp(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("exp", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.exp(x[i]);
  }
  return r;
}
export function mtoc2_tensor_log(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("log", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.log(x[i]);
  }
  return r;
}
export function mtoc2_tensor_log2(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("log2", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.log2(x[i]);
  }
  return r;
}
export function mtoc2_tensor_log10(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("log10", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.log10(x[i]);
  }
  return r;
}
export function mtoc2_tensor_sqrt(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("sqrt", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.sqrt(x[i]);
  }
  return r;
}
export function mtoc2_tensor_realsqrt(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  for (let i = 0; i < o.length; i++) o[i] = mtoc2_realsqrt(x[i]);
  return r;
}
export function mtoc2_tensor_abs(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("abs", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.abs(x[i]);
  }
  return r;
}
export function mtoc2_tensor_floor(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("floor", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.floor(x[i]);
  }
  return r;
}
export function mtoc2_tensor_ceil(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("ceil", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.ceil(x[i]);
  }
  return r;
}
export function mtoc2_tensor_fix(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("fix", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.trunc(x[i]);
  }
  return r;
}
export function mtoc2_tensor_round(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("round", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = matlabRound(x[i]);
  }
  return r;
}
export function mtoc2_tensor_sign(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const x = a.data;
  const o = r.data;
  if (!ew_host_unary("sign", x, o)) {
    for (let i = 0; i < o.length; i++) o[i] = Math.sign(x[i]);
  }
  return r;
}
