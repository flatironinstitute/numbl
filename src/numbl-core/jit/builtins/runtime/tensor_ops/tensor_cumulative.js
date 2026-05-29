// JS sibling of `tensor_cumulative.h`. Prefix-scan helpers
// (`cumsum`, `cumprod`) on real and complex tensors. `_dim` returns
// a freshly-allocated tensor of the SAME shape as the input, scanned
// along the 1-based `dim` axis. Mirrors numbl's `cumOp`
// (helpers/reduction/cumulative.ts) with column-major
// (before × axis × after) fiber traversal.
//
// NaN propagates: once acc becomes NaN, every later output along
// that fiber is NaN. Matches the C side and numbl.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { mtoc2_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";

function cumScan(t, dim, init, op) {
  if (dim < 1) {
    throw new Error(`cumulative _dim: dim must be >= 1 (got ${dim})`);
  }
  const shape = t.shape;
  const out = mtoc2_tensor_alloc_nd(shape.length, shape.slice());
  if (dim > shape.length) {
    out.data.set(t.data);
    return out;
  }
  const dimIdx = dim - 1;
  const axis = shape[dimIdx];
  let before = 1;
  for (let i = 0; i < dimIdx; i++) before *= shape[i];
  let after = 1;
  for (let i = dimIdx + 1; i < shape.length; i++) after *= shape[i];
  for (let outer = 0; outer < after; outer++) {
    const slabBase = outer * before * axis;
    for (let inner = 0; inner < before; inner++) {
      let acc = init;
      for (let k = 0; k < axis; k++) {
        const idx = slabBase + inner + k * before;
        acc = op(acc, t.data[idx]);
        out.data[idx] = acc;
      }
    }
  }
  return out;
}

function cumScanComplex(t, dim, initRe, initIm, op) {
  if (dim < 1) {
    throw new Error(`cumulative_complex _dim: dim must be >= 1 (got ${dim})`);
  }
  const shape = t.shape;
  const out = mtoc2_tensor_alloc_nd_complex(shape.length, shape.slice());
  const im = t.imag;
  if (dim > shape.length) {
    out.data.set(t.data);
    if (im !== undefined) out.imag.set(im);
    return out;
  }
  const dimIdx = dim - 1;
  const axis = shape[dimIdx];
  let before = 1;
  for (let i = 0; i < dimIdx; i++) before *= shape[i];
  let after = 1;
  for (let i = dimIdx + 1; i < shape.length; i++) after *= shape[i];
  for (let outer = 0; outer < after; outer++) {
    const slabBase = outer * before * axis;
    for (let inner = 0; inner < before; inner++) {
      let aRe = initRe;
      let aIm = initIm;
      for (let k = 0; k < axis; k++) {
        const idx = slabBase + inner + k * before;
        const xRe = t.data[idx];
        const xIm = im !== undefined ? im[idx] : 0;
        const next = op(aRe, aIm, xRe, xIm);
        aRe = next[0];
        aIm = next[1];
        out.data[idx] = aRe;
        out.imag[idx] = aIm;
      }
    }
  }
  return out;
}

export const mtoc2_tensor_cumsum_dim = (t, dim) =>
  cumScan(t, dim, 0, (a, x) => a + x);
export const mtoc2_tensor_cumprod_dim = (t, dim) =>
  cumScan(t, dim, 1, (a, x) => a * x);

export const mtoc2_tensor_cumsum_complex_dim = (t, dim) =>
  cumScanComplex(t, dim, 0, 0, (aRe, aIm, xRe, xIm) => [aRe + xRe, aIm + xIm]);

export const mtoc2_tensor_cumprod_complex_dim = (t, dim) =>
  cumScanComplex(t, dim, 1, 0, (aRe, aIm, xRe, xIm) => [
    aRe * xRe - aIm * xIm,
    aRe * xIm + aIm * xRe,
  ]);
