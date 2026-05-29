// JS sibling of tensor_dot.h — dot product on real and complex
// tensors. Complex variant uses `sum(conj(a) .* b)` matching numbl /
// MATLAB. A missing `imag` lane is treated as zero.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { mtoc2_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";

export function mtoc2_dot_real(a, b) {
  if (a.data.length !== b.data.length) {
    throw new Error("dot: vectors must be same length");
  }
  let acc = 0.0;
  for (let i = 0; i < a.data.length; i++) acc += a.data[i] * b.data[i];
  return acc;
}

export function mtoc2_dot_real_matrix(a, b) {
  const rows = a.shape[0];
  const cols = a.shape[1];
  const r = mtoc2_tensor_alloc_nd(2, [1, cols]);
  for (let j = 0; j < cols; j++) {
    let acc = 0.0;
    for (let i = 0; i < rows; i++) {
      const off = j * rows + i;
      acc += a.data[off] * b.data[off];
    }
    r.data[j] = acc;
  }
  return r;
}

export function mtoc2_dot_complex(a, b) {
  if (a.data.length !== b.data.length) {
    throw new Error("dot_complex: vectors must be same length");
  }
  const aIm = a.imag;
  const bIm = b.imag;
  let accRe = 0.0;
  let accIm = 0.0;
  for (let i = 0; i < a.data.length; i++) {
    const aRe = a.data[i];
    const aXi = aIm !== undefined ? aIm[i] : 0;
    const bRe = b.data[i];
    const bXi = bIm !== undefined ? bIm[i] : 0;
    accRe += aRe * bRe + aXi * bXi;
    accIm += aRe * bXi - aXi * bRe;
  }
  return { re: accRe, im: accIm };
}

export function mtoc2_dot_complex_matrix(a, b) {
  const rows = a.shape[0];
  const cols = a.shape[1];
  const r = mtoc2_tensor_alloc_nd_complex(2, [1, cols]);
  const aIm = a.imag;
  const bIm = b.imag;
  for (let j = 0; j < cols; j++) {
    let accRe = 0.0;
    let accIm = 0.0;
    for (let i = 0; i < rows; i++) {
      const off = j * rows + i;
      const aRe = a.data[off];
      const aXi = aIm !== undefined ? aIm[off] : 0;
      const bRe = b.data[off];
      const bXi = bIm !== undefined ? bIm[off] : 0;
      accRe += aRe * bRe + aXi * bXi;
      accIm += aRe * bXi - aXi * bRe;
    }
    r.data[j] = accRe;
    r.imag[j] = accIm;
  }
  return r;
}
