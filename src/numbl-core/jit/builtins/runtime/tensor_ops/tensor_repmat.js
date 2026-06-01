// JS sibling of `tensor_repmat.h`. Two helpers:
//   - `mtoc2_tensor_repmat(in, nreps, reps)` — tile a real tensor.
//   - `mtoc2_tensor_repmat_complex(in, nreps, reps)` — tile both
//     lanes of a complex tensor (zero imag when input is real).
// Negative reps clamp to 0; input shape and reps are right-padded
// with 1s to a common rank.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { mtoc2_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";

export function mtoc2_tensor_repmat(input, nreps, repsIn) {
  const reps = [];
  for (let i = 0; i < nreps; i++) {
    const r = repsIn[i] < 0 ? 0 : repsIn[i];
    reps.push(r);
  }
  const inShape = input.shape;
  const inNdim = inShape.length;
  const outNdim = Math.max(nreps, inNdim);
  const padShape = [];
  const padReps = [];
  const outDims = [];
  for (let i = 0; i < outNdim; i++) {
    padShape.push(i < inNdim ? inShape[i] : 1);
    padReps.push(i < nreps ? reps[i] : 1);
    outDims.push(padShape[i] * padReps[i]);
  }
  const out = mtoc2_tensor_alloc_nd(outNdim, outDims);
  let outTotal = 1;
  for (const d of outDims) outTotal *= d;
  if (outTotal === 0) return out;
  let inTotal = 1;
  for (const d of inShape) inTotal *= d;
  if (inTotal === 0) return out;

  // Initial copy: trailing-1 padding doesn't change column-major layout.
  out.data.set(input.data.subarray(0, inTotal), 0);

  const curShape = padShape.slice();
  let curTotal = inTotal;

  for (let d = 0; d < outNdim; d++) {
    const rep = padReps[d];
    if (rep === 1) continue;
    let blockSize = 1;
    for (let i = 0; i <= d; i++) blockSize *= curShape[i];
    if (rep === 0 || blockSize === 0) {
      // outTotal will be 0; the alloc above already produced an empty
      // tensor. Done.
      return out;
    }
    const numBlocks = curTotal / blockSize;
    // Walk blocks in reverse to avoid overwriting source data.
    for (let b = numBlocks - 1; b >= 0; b--) {
      const srcOff = b * blockSize;
      const dstBase = b * blockSize * rep;
      if (dstBase !== srcOff) {
        // copyWithin handles overlapping moves correctly.
        out.data.copyWithin(dstBase, srcOff, srcOff + blockSize);
      }
      for (let r = 1; r < rep; r++) {
        out.data.copyWithin(
          dstBase + r * blockSize,
          dstBase,
          dstBase + blockSize
        );
      }
    }
    curShape[d] *= rep;
    curTotal *= rep;
  }
  return out;
}

export function mtoc2_tensor_repmat_complex(input, nreps, repsIn) {
  const reps = [];
  for (let i = 0; i < nreps; i++) {
    const r = repsIn[i] < 0 ? 0 : repsIn[i];
    reps.push(r);
  }
  const inShape = input.shape;
  const inNdim = inShape.length;
  const outNdim = Math.max(nreps, inNdim);
  const padShape = [];
  const padReps = [];
  const outDims = [];
  for (let i = 0; i < outNdim; i++) {
    padShape.push(i < inNdim ? inShape[i] : 1);
    padReps.push(i < nreps ? reps[i] : 1);
    outDims.push(padShape[i] * padReps[i]);
  }
  const out = mtoc2_tensor_alloc_nd_complex(outNdim, outDims);
  let outTotal = 1;
  for (const d of outDims) outTotal *= d;
  if (outTotal === 0) return out;
  let inTotal = 1;
  for (const d of inShape) inTotal *= d;
  if (inTotal === 0) return out;

  const im = input.imag;
  out.data.set(input.data.subarray(0, inTotal), 0);
  if (im !== undefined) out.imag.set(im.subarray(0, inTotal), 0);

  const curShape = padShape.slice();
  let curTotal = inTotal;

  for (let d = 0; d < outNdim; d++) {
    const rep = padReps[d];
    if (rep === 1) continue;
    let blockSize = 1;
    for (let i = 0; i <= d; i++) blockSize *= curShape[i];
    if (rep === 0 || blockSize === 0) return out;
    const numBlocks = curTotal / blockSize;
    for (let b = numBlocks - 1; b >= 0; b--) {
      const srcOff = b * blockSize;
      const dstBase = b * blockSize * rep;
      if (dstBase !== srcOff) {
        out.data.copyWithin(dstBase, srcOff, srcOff + blockSize);
        out.imag.copyWithin(dstBase, srcOff, srcOff + blockSize);
      }
      for (let r = 1; r < rep; r++) {
        out.data.copyWithin(
          dstBase + r * blockSize,
          dstBase,
          dstBase + blockSize
        );
        out.imag.copyWithin(
          dstBase + r * blockSize,
          dstBase,
          dstBase + blockSize
        );
      }
    }
    curShape[d] *= rep;
    curTotal *= rep;
  }
  return out;
}
