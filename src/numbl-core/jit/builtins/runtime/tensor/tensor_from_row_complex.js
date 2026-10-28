// JS sibling of `tensor_from_row_complex.h`. Build a 1×N complex
// tensor from parallel `re` and `im` JS arrays (matching the C side's
// `double[]` sources). The codegen path passes plain JS array
// literals.

import { jit_tensor_alloc_complex } from "./tensor_alloc_complex.js";

export function jit_tensor_from_row_complex(re, im, n) {
  const t = jit_tensor_alloc_complex(1, n);
  for (let i = 0; i < n; i++) {
    t.data[i] = re[i];
    t.imag[i] = im[i];
  }
  return t;
}
