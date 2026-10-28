// JS sibling of `tensor_size.h`. Build a 1×ndim row tensor whose
// elements are the input's dim sizes.

import { jit_tensor_alloc } from "../tensor/tensor_alloc.js";

export function jit_tensor_size_row(a) {
  const n = a.shape.length;
  const r = jit_tensor_alloc(1, n);
  for (let i = 0; i < n; i++) r.data[i] = a.shape[i];
  return r;
}
