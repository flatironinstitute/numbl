// JS sibling of `tensor_cat.h`. `jit_tensor_cat(dim, nin, xs)`
// concatenates the inputs along axis `dim` (1-based, like MATLAB /
// numbl). Inputs may be tensor objects (per the `jit_tensor_make`
// shape) or plain numbers — scalars are treated as 1×1 tensors.
//
// `jit_tensor_cat_complex(dim, nin, xs)` is the complex sibling:
// inputs may be `{re, im}` scalars, real numbers, real tensors, or
// complex tensors; result is always a complex tensor.
//
// Mirrors numbl's `catAlongDim`: empty inputs are dropped (and
// asymmetrically kept only when their non-cat dims match the reference).

import { jit_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { jit_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";
import { jit_raise_bail, jit_raise_error } from "../system/raise.js";

export function jit_tensor_cat(dim, nin, xs) {
  if (dim < 1) {
    jit_raise_bail(`JIT: cat dim ${dim} must be >= 1`);
  }
  const dimIdx = dim - 1;

  // Normalize each input: scalar → 1×1; tensor → its shape/data.
  let maxIn = 2;
  if (dim > maxIn) maxIn = dim;
  const parts = [];
  for (let i = 0; i < nin; i++) {
    const x = xs[i];
    if (typeof x === "number") {
      parts.push({ shape: [1, 1], data: new Float64Array([x]) });
    } else {
      const s = x.shape;
      if (s.length > maxIn) maxIn = s.length;
      parts.push({ shape: s.slice(), data: x.data });
    }
  }
  const ndim = maxIn;
  // Pad shapes to ndim with 1s.
  for (const p of parts) {
    p.zero = p.shape.every(d => d === 0);
    while (p.shape.length < ndim) p.shape.push(1);
  }

  // Each input's flat size (under padded shape, which equals its raw
  // numel since trailing 1s preserve column-major layout).
  for (const p of parts) {
    let n = 1;
    for (const d of p.shape) n *= d;
    p.total = n;
  }

  // Step 2: find first non-empty input as the reference.
  let refIdx = -1;
  for (let i = 0; i < parts.length; i++) {
    if (parts[i].total > 0) {
      refIdx = i;
      break;
    }
  }
  // Every input is empty: as numbl's catAlongDim, drop the 0x0 ones
  // (every dim 0) and concatenate the rest, whose non-cat dims must agree
  // (cat(2, zeros(0, 3), zeros(0, 2)) is 0x5).
  const allEmpty = refIdx === -1;
  if (allEmpty) refIdx = parts.findIndex(p => !p.zero);
  if (refIdx === -1) {
    // All inputs empty (or none given): canonical [0, 0] empty.
    return jit_tensor_alloc_nd(2, [0, 0]);
  }

  const refShape = parts[refIdx].shape;

  // Step 3: keep[] decides which inputs participate.
  const keep = parts.map((p, i) => {
    if (p.total > 0 || (allEmpty && !p.zero)) {
      if (i === refIdx) return true;
      for (let d = 0; d < ndim; d++) {
        if (d === dimIdx) continue;
        if (p.shape[d] !== refShape[d]) {
          jit_raise_error(
            "",
            "Dimensions of arrays being concatenated are not consistent"
          );
        }
      }
      return true;
    }
    // Empty input — keep iff non-cat dims match ref.
    for (let d = 0; d < ndim; d++) {
      if (d === dimIdx) continue;
      if (p.shape[d] !== refShape[d]) return false;
    }
    return true;
  });

  // Step 4: result shape.
  const resultShape = refShape.slice();
  let catSum = 0;
  for (let i = 0; i < parts.length; i++) {
    if (keep[i]) catSum += parts[i].shape[dimIdx];
  }
  resultShape[dimIdx] = catSum;

  const out = jit_tensor_alloc_nd(ndim, resultShape);
  let resultTotal = 1;
  for (const d of resultShape) resultTotal *= d;
  if (resultTotal === 0) return out;

  // Step 5: column-major slab copies.
  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= resultShape[d];
  let numOuter = 1;
  for (let d = dimIdx + 1; d < ndim; d++) numOuter *= resultShape[d];

  for (let outer = 0; outer < numOuter; outer++) {
    let dstOff = outer * strideDim * resultShape[dimIdx];
    for (let i = 0; i < parts.length; i++) {
      if (!keep[i]) continue;
      const srcDimSize = parts[i].shape[dimIdx];
      const blockSize = strideDim * srcDimSize;
      const srcOff = outer * blockSize;
      if (blockSize > 0) {
        out.data.set(
          parts[i].data.subarray(srcOff, srcOff + blockSize),
          dstOff
        );
      }
      dstOff += blockSize;
    }
  }
  return out;
}

function inputLanesComplex(x) {
  if (typeof x === "number") {
    return { shape: [1, 1], data: new Float64Array([x]), imag: undefined };
  }
  if (typeof x === "object" && x !== null && "re" in x && "im" in x) {
    return {
      shape: [1, 1],
      data: new Float64Array([x.re]),
      imag: new Float64Array([x.im]),
    };
  }
  return { shape: x.shape.slice(), data: x.data, imag: x.imag };
}

export function jit_tensor_cat_complex(dim, nin, xs) {
  if (dim < 1) {
    jit_raise_bail(`JIT: cat_complex dim ${dim} must be >= 1`);
  }
  const dimIdx = dim - 1;

  let maxIn = 2;
  if (dim > maxIn) maxIn = dim;
  const parts = [];
  for (let i = 0; i < nin; i++) {
    const p = inputLanesComplex(xs[i]);
    if (p.shape.length > maxIn) maxIn = p.shape.length;
    parts.push(p);
  }
  const ndim = maxIn;
  for (const p of parts) {
    p.zero = p.shape.every(d => d === 0);
    while (p.shape.length < ndim) p.shape.push(1);
  }
  for (const p of parts) {
    let n = 1;
    for (const d of p.shape) n *= d;
    p.total = n;
  }

  let refIdx = -1;
  for (let i = 0; i < parts.length; i++) {
    if (parts[i].total > 0) {
      refIdx = i;
      break;
    }
  }
  // Every input is empty: as numbl's catAlongDim, drop the 0x0 ones
  // (every dim 0) and concatenate the rest, whose non-cat dims must agree
  // (cat(2, zeros(0, 3), zeros(0, 2)) is 0x5).
  const allEmpty = refIdx === -1;
  if (allEmpty) refIdx = parts.findIndex(p => !p.zero);
  if (refIdx === -1) {
    return jit_tensor_alloc_nd_complex(2, [0, 0]);
  }
  const refShape = parts[refIdx].shape;

  const keep = parts.map((p, i) => {
    if (p.total > 0 || (allEmpty && !p.zero)) {
      if (i === refIdx) return true;
      for (let d = 0; d < ndim; d++) {
        if (d === dimIdx) continue;
        if (p.shape[d] !== refShape[d]) {
          jit_raise_error(
            "",
            "Dimensions of arrays being concatenated are not consistent"
          );
        }
      }
      return true;
    }
    for (let d = 0; d < ndim; d++) {
      if (d === dimIdx) continue;
      if (p.shape[d] !== refShape[d]) return false;
    }
    return true;
  });

  const resultShape = refShape.slice();
  let catSum = 0;
  for (let i = 0; i < parts.length; i++) {
    if (keep[i]) catSum += parts[i].shape[dimIdx];
  }
  resultShape[dimIdx] = catSum;

  const out = jit_tensor_alloc_nd_complex(ndim, resultShape);
  let resultTotal = 1;
  for (const d of resultShape) resultTotal *= d;
  if (resultTotal === 0) return out;

  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= resultShape[d];
  let numOuter = 1;
  for (let d = dimIdx + 1; d < ndim; d++) numOuter *= resultShape[d];

  for (let outer = 0; outer < numOuter; outer++) {
    let dstOff = outer * strideDim * resultShape[dimIdx];
    for (let i = 0; i < parts.length; i++) {
      if (!keep[i]) continue;
      const srcDimSize = parts[i].shape[dimIdx];
      const blockSize = strideDim * srcDimSize;
      const srcOff = outer * blockSize;
      if (blockSize > 0) {
        out.data.set(
          parts[i].data.subarray(srcOff, srcOff + blockSize),
          dstOff
        );
        if (parts[i].imag !== undefined) {
          out.imag.set(
            parts[i].imag.subarray(srcOff, srcOff + blockSize),
            dstOff
          );
        }
      }
      dstOff += blockSize;
    }
  }
  return out;
}
