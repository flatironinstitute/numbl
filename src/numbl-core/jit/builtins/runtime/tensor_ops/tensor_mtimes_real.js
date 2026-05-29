// JS sibling of `tensor_mtimes_real.h`. Real 2-D matrix
// multiplication A * B. Column-major in, column-major out.
// `mtoc2_tensor_mtimes_real_scalar` mirrors the C 1×k * k×1
// inner-product path that returns a bare number instead of a
// freshly-allocated 1×1 tensor.

import { mtoc2_tensor_alloc } from "../tensor/tensor_alloc.js";

export function mtoc2_tensor_mtimes_real(a, b) {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error("mtoc2_tensor_mtimes_real: inputs must be 2-D");
  }
  const m = a.shape[0];
  const k = a.shape[1];
  const k2 = b.shape[0];
  const n = b.shape[1];
  if (k !== k2) {
    throw new Error(
      `mtoc2_tensor_mtimes_real: inner dimensions disagree (${k} vs ${k2})`
    );
  }
  const r = mtoc2_tensor_alloc(m, n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let p = 0; p < k; p++) {
        s += a.data[i + p * m] * b.data[p + j * k];
      }
      r.data[i + j * m] = s;
    }
  }
  return r;
}

export function mtoc2_tensor_mtimes_real_scalar(a, b) {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error("mtoc2_tensor_mtimes_real_scalar: inputs must be 2-D");
  }
  // The translate-time type system has already proven a is 1×k and b
  // is k×1; this helper trusts that and just runs the inner product.
  const k = a.shape[1];
  let s = 0;
  for (let p = 0; p < k; p++) {
    s += a.data[p * a.shape[0]] * b.data[p];
  }
  return s;
}
