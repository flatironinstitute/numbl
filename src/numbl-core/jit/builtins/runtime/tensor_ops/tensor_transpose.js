// JS sibling of `tensor_transpose.h`. Real 2-D non-conjugate
// transpose. Mirrors `transposeCore` semantics: column-major in,
// column-major out.

import { jit_tensor_alloc } from "../tensor/tensor_alloc.js";

export function jit_tensor_transpose(a) {
  const m = a.shape[0];
  const n = a.shape[1];
  const r = jit_tensor_alloc(n, m);
  for (let sc = 0; sc < n; sc++) {
    for (let sr = 0; sr < m; sr++) {
      r.data[sc + sr * n] = a.data[sr + sc * m];
    }
  }
  return r;
}
