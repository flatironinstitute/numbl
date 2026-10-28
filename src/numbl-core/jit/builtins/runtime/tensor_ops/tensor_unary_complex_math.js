// JS sibling of `tensor_unary_complex_math.h`. Elementwise unary
// math on complex tensors. Each helper allocates a fresh tensor
// (real or complex, depending on the op) and walks every element
// through the per-op `jit_c*` scalar helper.

import { jit_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";
import { jit_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import {
  jit_csqrt,
  jit_cexp,
  jit_clog,
  jit_clog2,
  jit_clog10,
  jit_csin,
  jit_ccos,
  jit_ctan,
  jit_catan,
  jit_csinh,
  jit_ccosh,
  jit_ctanh,
  jit_cfloor,
  jit_cceil,
  jit_cround,
  jit_cfix,
  jit_csign,
  jit_cconj,
} from "../system/cscalar.js";

function cUnaryLaneIm(t, i) {
  return t.imag !== undefined ? t.imag[i] : 0;
}

function cUnary(a, fn) {
  const r = jit_tensor_alloc_nd_complex(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    const v = fn({ re: a.data[i], im: cUnaryLaneIm(a, i) });
    r.data[i] = v.re;
    r.imag[i] = v.im;
  }
  return r;
}

export function jit_tensor_sqrt_complex(a) {
  return cUnary(a, jit_csqrt);
}
export function jit_tensor_exp_complex(a) {
  return cUnary(a, jit_cexp);
}
export function jit_tensor_log_complex(a) {
  return cUnary(a, jit_clog);
}
export function jit_tensor_log2_complex(a) {
  return cUnary(a, jit_clog2);
}
export function jit_tensor_log10_complex(a) {
  return cUnary(a, jit_clog10);
}
export function jit_tensor_sin_complex(a) {
  return cUnary(a, jit_csin);
}
export function jit_tensor_cos_complex(a) {
  return cUnary(a, jit_ccos);
}
export function jit_tensor_tan_complex(a) {
  return cUnary(a, jit_ctan);
}
export function jit_tensor_atan_complex(a) {
  return cUnary(a, jit_catan);
}
export function jit_tensor_sinh_complex(a) {
  return cUnary(a, jit_csinh);
}
export function jit_tensor_cosh_complex(a) {
  return cUnary(a, jit_ccosh);
}
export function jit_tensor_tanh_complex(a) {
  return cUnary(a, jit_ctanh);
}
export function jit_tensor_floor_complex(a) {
  return cUnary(a, jit_cfloor);
}
export function jit_tensor_ceil_complex(a) {
  return cUnary(a, jit_cceil);
}
export function jit_tensor_round_complex(a) {
  return cUnary(a, jit_cround);
}
export function jit_tensor_fix_complex(a) {
  return cUnary(a, jit_cfix);
}
export function jit_tensor_sign_complex(a) {
  return cUnary(a, jit_csign);
}
export function jit_tensor_conj_complex(a) {
  return cUnary(a, jit_cconj);
}

// `imag`, `real`, `angle`, `abs` on a complex tensor all return a
// REAL tensor (the imag lane is NOT allocated).
export function jit_tensor_imag_complex(a) {
  const r = jit_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = cUnaryLaneIm(a, i);
  return r;
}
export function jit_tensor_real_complex(a) {
  const r = jit_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = a.data[i];
  return r;
}
export function jit_tensor_angle_complex(a) {
  const r = jit_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    r.data[i] = Math.atan2(cUnaryLaneIm(a, i), a.data[i]);
  }
  return r;
}
export function jit_tensor_abs_complex(a) {
  const r = jit_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    r.data[i] = Math.hypot(a.data[i], cUnaryLaneIm(a, i));
  }
  return r;
}
