// JS sibling of `tensor_from_matrix_complex.h`. Build a rows×cols
// complex tensor from parallel column-major `re` and `im` arrays.

import { mtoc2_tensor_alloc_complex } from "./tensor_alloc_complex.js";

export function mtoc2_tensor_from_matrix_complex(re, im, rows, cols) {
  const t = mtoc2_tensor_alloc_complex(rows, cols);
  const n = rows * cols;
  for (let i = 0; i < n; i++) {
    t.data[i] = re[i];
    t.imag[i] = im[i];
  }
  return t;
}
