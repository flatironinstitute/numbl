// JS sibling of `tensor_transpose_complex.h`. Real 2-D non-conjugate
// transpose for a complex tensor — both lanes get the same index
// permutation. `'` (conjugate transpose) lowers to
// `transpose(conj(z))` upstream, so this helper isn't responsible
// for negating the imag lane.

import { mtoc2_tensor_alloc_complex } from "../tensor/tensor_alloc_complex.js";

export function mtoc2_tensor_transpose_complex(a) {
  const m = a.shape[0];
  const n = a.shape[1];
  const r = mtoc2_tensor_alloc_complex(n, m);
  const aim = a.imag;
  for (let sc = 0; sc < n; sc++) {
    for (let sr = 0; sr < m; sr++) {
      r.data[sc + sr * n] = a.data[sr + sc * m];
      r.imag[sc + sr * n] = aim !== undefined ? aim[sr + sc * m] : 0;
    }
  }
  return r;
}
