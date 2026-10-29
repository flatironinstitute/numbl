// JS sibling of `tensor_predicate.h`. Real-tensor → logical-tensor
// predicate kernels for the JS-JIT, plus their `_complex`
// siblings (each reads `imag[i]` when defined, treats it as 0
// otherwise). Result carries `isLogical: true` so downstream index-
// slot resolution treats it as a mask.

import { jit_raise_error } from "../system/raise.js";

function pred_kernel(a, fn) {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = fn(a.data[i]) ? 1 : 0;
  return {
    jitTag: "tensor",
    shape: a.shape.slice(),
    data: out,
    isLogical: true,
  };
}

function pred_kernel_complex(a, fn) {
  const out = new Float64Array(a.data.length);
  const im = a.imag;
  for (let i = 0; i < a.data.length; i++) {
    out[i] = fn(a.data[i], im !== undefined ? im[i] : 0) ? 1 : 0;
  }
  return {
    jitTag: "tensor",
    shape: a.shape.slice(),
    data: out,
    isLogical: true,
  };
}

export function jit_tensor_isnan(a) {
  return pred_kernel(a, Number.isNaN);
}

// `logical(x)` of one element: MATLAB raises on NaN.
export function jit_logical_elem(x) {
  if (Number.isNaN(x))
    jit_raise_error(
      "MATLAB:nologicalnan",
      "NaN values cannot be converted to logicals."
    );
  return x !== 0;
}

export function jit_tensor_logical(a) {
  return pred_kernel(a, jit_logical_elem);
}

export function jit_tensor_isinf(a) {
  return pred_kernel(a, x => x === Infinity || x === -Infinity);
}

export function jit_tensor_isfinite(a) {
  return pred_kernel(a, Number.isFinite);
}

export function jit_tensor_isnan_complex(a) {
  return pred_kernel_complex(
    a,
    (re, im) => Number.isNaN(re) || Number.isNaN(im)
  );
}

export function jit_tensor_isinf_complex(a) {
  const isInf = x => x === Infinity || x === -Infinity;
  return pred_kernel_complex(a, (re, im) => isInf(re) || isInf(im));
}

export function jit_tensor_isfinite_complex(a) {
  return pred_kernel_complex(
    a,
    (re, im) => Number.isFinite(re) && Number.isFinite(im)
  );
}
