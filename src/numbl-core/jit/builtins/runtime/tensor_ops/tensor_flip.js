// JS sibling of `tensor_flip.h`. Two helpers:
//   - `mtoc2_tensor_flip(t, dimIdx)` — real-input variant.
//   - `mtoc2_tensor_flip_complex(t, dimIdx)` — walks both lanes;
//     if `imag` is undefined the output imag stays zero.
// Out-of-range axis acts as a deep-copy no-op in both.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { mtoc2_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";

export function mtoc2_tensor_flip(a, dimIdx) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const axisSize = dimIdx >= 0 && dimIdx < a.shape.length ? a.shape[dimIdx] : 1;
  if (axisSize <= 1) {
    r.data.set(a.data);
    return r;
  }
  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= a.shape[d];
  const slabSize = strideDim * axisSize;
  const total = r.data.length;
  const numOuter = total / slabSize;
  for (let outer = 0; outer < numOuter; outer++) {
    const base = outer * slabSize;
    for (let k = 0; k < axisSize; k++) {
      const srcOff = base + k * strideDim;
      const dstOff = base + (axisSize - 1 - k) * strideDim;
      for (let i = 0; i < strideDim; i++) {
        r.data[dstOff + i] = a.data[srcOff + i];
      }
    }
  }
  return r;
}

export function mtoc2_tensor_flip_complex(a, dimIdx) {
  const r = mtoc2_tensor_alloc_nd_complex(a.shape.length, a.shape);
  const im = a.imag;
  const axisSize = dimIdx >= 0 && dimIdx < a.shape.length ? a.shape[dimIdx] : 1;
  if (axisSize <= 1) {
    r.data.set(a.data);
    if (im !== undefined) r.imag.set(im);
    return r;
  }
  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= a.shape[d];
  const slabSize = strideDim * axisSize;
  const total = r.data.length;
  const numOuter = total / slabSize;
  for (let outer = 0; outer < numOuter; outer++) {
    const base = outer * slabSize;
    for (let k = 0; k < axisSize; k++) {
      const srcOff = base + k * strideDim;
      const dstOff = base + (axisSize - 1 - k) * strideDim;
      for (let i = 0; i < strideDim; i++) {
        r.data[dstOff + i] = a.data[srcOff + i];
        if (im !== undefined) r.imag[dstOff + i] = im[srcOff + i];
      }
    }
  }
  return r;
}
