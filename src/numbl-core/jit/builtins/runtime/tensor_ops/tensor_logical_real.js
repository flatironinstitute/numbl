// JS sibling of `tensor_logical_real.h`. Elementwise logical NOT on
// real and complex tensors. Real input: `out[i] = (in[i] == 0) ? 1 : 0`.
// Complex input: fires "true" iff both lanes are exactly zero.
// Result is logical-tagged in both cases.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

export function mtoc2_tensor_not(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  for (let i = 0; i < r.data.length; i++) {
    r.data[i] = a.data[i] === 0 ? 1 : 0;
  }
  // Tag as logical so `a(mask)` / `M(:, mask)` etc. take the mask
  // path in the interpreter (and js-aot, when wired). The tensor
  // alloc helpers return plain numeric tensors; we mutate the field
  // here rather than threading a parameter through every allocator.
  r.isLogical = true;
  return r;
}

export function mtoc2_tensor_not_complex(a) {
  const r = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const im = a.imag;
  for (let i = 0; i < r.data.length; i++) {
    const re = a.data[i];
    const v = im !== undefined ? im[i] : 0;
    r.data[i] = re === 0 && v === 0 ? 1 : 0;
  }
  r.isLogical = true;
  return r;
}
