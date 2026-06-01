// JS sibling of `tensor_unary_complex_math.h`. Elementwise unary
// math on complex tensors. Each helper allocates a fresh tensor
// (real or complex, depending on the op) and walks every element
// through the per-op `mtoc2_c*` scalar helper.

import { mtoc2_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";
import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import {
  mtoc2_csqrt,
  mtoc2_cexp,
  mtoc2_clog,
  mtoc2_clog2,
  mtoc2_clog10,
  mtoc2_csin,
  mtoc2_ccos,
  mtoc2_ctan,
  mtoc2_catan,
  mtoc2_csinh,
  mtoc2_ccosh,
  mtoc2_ctanh,
  mtoc2_cfloor,
  mtoc2_cceil,
  mtoc2_cround,
  mtoc2_cfix,
  mtoc2_csign,
  mtoc2_cconj,
} from "../system/cscalar.js";

function cUnaryLaneIm(t, i) {
  return t.imag !== undefined ? t.imag[i] : 0;
}

function cUnary(a, fn) {
  const r = mtoc2_tensor_alloc_nd_complex(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    const v = fn({ re: a.data[i], im: cUnaryLaneIm(a, i) });
    r.data[i] = v.re;
    r.imag[i] = v.im;
  }
  return r;
}

export function mtoc2_tensor_sqrt_complex(a) {
  return cUnary(a, mtoc2_csqrt);
}
export function mtoc2_tensor_exp_complex(a) {
  return cUnary(a, mtoc2_cexp);
}
export function mtoc2_tensor_log_complex(a) {
  return cUnary(a, mtoc2_clog);
}
export function mtoc2_tensor_log2_complex(a) {
  return cUnary(a, mtoc2_clog2);
}
export function mtoc2_tensor_log10_complex(a) {
  return cUnary(a, mtoc2_clog10);
}
export function mtoc2_tensor_sin_complex(a) {
  return cUnary(a, mtoc2_csin);
}
export function mtoc2_tensor_cos_complex(a) {
  return cUnary(a, mtoc2_ccos);
}
export function mtoc2_tensor_tan_complex(a) {
  return cUnary(a, mtoc2_ctan);
}
export function mtoc2_tensor_atan_complex(a) {
  return cUnary(a, mtoc2_catan);
}
export function mtoc2_tensor_sinh_complex(a) {
  return cUnary(a, mtoc2_csinh);
}
export function mtoc2_tensor_cosh_complex(a) {
  return cUnary(a, mtoc2_ccosh);
}
export function mtoc2_tensor_tanh_complex(a) {
  return cUnary(a, mtoc2_ctanh);
}
export function mtoc2_tensor_floor_complex(a) {
  return cUnary(a, mtoc2_cfloor);
}
export function mtoc2_tensor_ceil_complex(a) {
  return cUnary(a, mtoc2_cceil);
}
export function mtoc2_tensor_round_complex(a) {
  return cUnary(a, mtoc2_cround);
}
export function mtoc2_tensor_fix_complex(a) {
  return cUnary(a, mtoc2_cfix);
}
export function mtoc2_tensor_sign_complex(a) {
  return cUnary(a, mtoc2_csign);
}
export function mtoc2_tensor_conj_complex(a) {
  return cUnary(a, mtoc2_cconj);
}

// `imag`, `real`, `angle`, `abs` on a complex tensor all return a
// REAL tensor (the imag lane is NOT allocated).
export function mtoc2_tensor_imag_complex(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = cUnaryLaneIm(a, i);
  return r;
}
export function mtoc2_tensor_real_complex(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) r.data[i] = a.data[i];
  return r;
}
export function mtoc2_tensor_angle_complex(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    r.data[i] = Math.atan2(cUnaryLaneIm(a, i), a.data[i]);
  }
  return r;
}
export function mtoc2_tensor_abs_complex(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    r.data[i] = Math.hypot(a.data[i], cUnaryLaneIm(a, i));
  }
  return r;
}
