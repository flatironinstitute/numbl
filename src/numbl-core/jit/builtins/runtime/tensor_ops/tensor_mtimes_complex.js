// JS sibling of `tensor_mtimes_complex.h`. 2-D matrix multiplication
// producing a complex result. Handles all three mixed cases
// (complex × complex, complex × real, real × complex) by treating an
// undefined imag lane as implicit zeros.

import { mtoc2_tensor_alloc_complex } from "../tensor/tensor_alloc_complex.js";

export function mtoc2_tensor_mtimes_complex_scalar(a, b) {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error("mtoc2_tensor_mtimes_complex_scalar: inputs must be 2-D");
  }
  if (a.shape[0] !== 1 || b.shape[1] !== 1) {
    throw new Error("mtoc2_tensor_mtimes_complex_scalar: requires 1×k * k×1");
  }
  if (a.shape[1] !== b.shape[0]) {
    throw new Error("mtoc2_tensor_mtimes_complex_scalar: inner-dim mismatch");
  }
  const k = a.shape[1];
  const aim = a.imag;
  const bim = b.imag;
  let accR = 0;
  let accI = 0;
  for (let p = 0; p < k; p++) {
    const ar = a.data[p];
    const ai = aim !== undefined ? aim[p] : 0;
    const br = b.data[p];
    const bi = bim !== undefined ? bim[p] : 0;
    accR += ar * br - ai * bi;
    accI += ar * bi + ai * br;
  }
  return { re: accR, im: accI };
}

export function mtoc2_tensor_mtimes_complex(a, b) {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error("mtoc2_tensor_mtimes_complex: inputs must be 2-D");
  }
  const m = a.shape[0];
  const k = a.shape[1];
  const k2 = b.shape[0];
  const n = b.shape[1];
  if (k !== k2) {
    throw new Error(
      `mtoc2_tensor_mtimes_complex: inner-dim mismatch (${m}×${k} * ${k2}×${n})`
    );
  }
  const out = mtoc2_tensor_alloc_complex(m, n);
  const aim = a.imag;
  const bim = b.imag;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      let accR = 0;
      let accI = 0;
      for (let p = 0; p < k; p++) {
        const ar = a.data[i + p * m];
        const ai = aim !== undefined ? aim[i + p * m] : 0;
        const br = b.data[p + j * k];
        const bi = bim !== undefined ? bim[p + j * k] : 0;
        accR += ar * br - ai * bi;
        accI += ar * bi + ai * br;
      }
      out.data[i + j * m] = accR;
      out.imag[i + j * m] = accI;
    }
  }
  return out;
}
