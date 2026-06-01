// JS sibling of `tensor_diff.h`. First-order difference along the
// operating axis, byte-for-byte with numbl's `diffOnce`
// (helpers/reduction/cumulative.ts). `dim === 0` selects the default
// axis (row vector → dim 2, else first dim); `dim > 0` is explicit.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

export function mtoc2_tensor_diff(a, dim) {
  const shape = a.shape;
  let opDim;
  if (dim > 0) {
    opDim = dim - 1;
  } else if (shape.length <= 1 || (shape.length === 2 && shape[0] === 1)) {
    opDim = shape.length === 2 && shape[0] === 1 ? 1 : 0;
  } else {
    opDim = 0;
  }
  const dimSize = opDim < shape.length ? shape[opDim] : 1;

  const newDims = shape.slice();
  if (dimSize <= 1) {
    if (opDim < newDims.length) newDims[opDim] = 0;
    while (newDims.length > 2 && newDims[newDims.length - 1] === 1)
      newDims.pop();
    return mtoc2_tensor_alloc_nd(newDims.length, newDims);
  }

  newDims[opDim] = dimSize - 1;
  while (newDims.length > 2 && newDims[newDims.length - 1] === 1) newDims.pop();
  const out = mtoc2_tensor_alloc_nd(newDims.length, newDims);

  let innerCount = 1;
  for (let d = 0; d < opDim; d++) innerCount *= shape[d];
  let outerCount = 1;
  for (let d = opDim + 1; d < shape.length; d++) outerCount *= shape[d];

  let outIdx = 0;
  for (let outer = 0; outer < outerCount; outer++) {
    for (let k = 0; k < dimSize - 1; k++) {
      for (let inner = 0; inner < innerCount; inner++) {
        const base = outer * (dimSize * innerCount) + k * innerCount + inner;
        out.data[outIdx++] = a.data[base + innerCount] - a.data[base];
      }
    }
  }
  return out;
}
