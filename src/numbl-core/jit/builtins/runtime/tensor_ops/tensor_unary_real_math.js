// JS sibling of `tensor_unary_real_math.h`. Per-op unary math
// kernels for tensors (`mtoc2_tensor_sin`, `_cos`, etc.). Matches
// the per-op function names the codegen emits.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

function unary_kernel(a, fn) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = fn(a.data[i]);
  return r;
}

function matlabRound(x) {
  return Math.sign(x) * Math.round(Math.abs(x));
}

export function mtoc2_tensor_cos(a) {
  return unary_kernel(a, Math.cos);
}
export function mtoc2_tensor_sin(a) {
  return unary_kernel(a, Math.sin);
}
export function mtoc2_tensor_tan(a) {
  return unary_kernel(a, Math.tan);
}
export function mtoc2_tensor_atan(a) {
  return unary_kernel(a, Math.atan);
}
export function mtoc2_tensor_sinh(a) {
  return unary_kernel(a, Math.sinh);
}
export function mtoc2_tensor_cosh(a) {
  return unary_kernel(a, Math.cosh);
}
export function mtoc2_tensor_tanh(a) {
  return unary_kernel(a, Math.tanh);
}
export function mtoc2_tensor_asinh(a) {
  return unary_kernel(a, Math.asinh);
}
export function mtoc2_tensor_exp(a) {
  return unary_kernel(a, Math.exp);
}
export function mtoc2_tensor_log(a) {
  return unary_kernel(a, Math.log);
}
export function mtoc2_tensor_log2(a) {
  return unary_kernel(a, Math.log2);
}
export function mtoc2_tensor_log10(a) {
  return unary_kernel(a, Math.log10);
}
export function mtoc2_tensor_sqrt(a) {
  return unary_kernel(a, Math.sqrt);
}
export function mtoc2_tensor_abs(a) {
  return unary_kernel(a, Math.abs);
}
export function mtoc2_tensor_floor(a) {
  return unary_kernel(a, Math.floor);
}
export function mtoc2_tensor_ceil(a) {
  return unary_kernel(a, Math.ceil);
}
export function mtoc2_tensor_fix(a) {
  return unary_kernel(a, Math.trunc);
}
export function mtoc2_tensor_round(a) {
  return unary_kernel(a, matlabRound);
}
export function mtoc2_tensor_sign(a) {
  return unary_kernel(a, Math.sign);
}
