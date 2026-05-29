// JS sibling of `tensor_size.h`. Build a 1×ndim row tensor whose
// elements are the input's dim sizes.

import { mtoc2_tensor_alloc } from "../tensor/tensor_alloc.js";

export function mtoc2_tensor_size_row(a) {
  const n = a.shape.length;
  const r = mtoc2_tensor_alloc(1, n);
  for (let i = 0; i < n; i++) r.data[i] = a.shape[i];
  return r;
}
