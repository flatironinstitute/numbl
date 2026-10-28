// JS sibling of `tensor_from_matrix.h`. Build an R×C tensor from a
// flat column-major array. The codegen path emits:
//   jit_tensor_from_matrix([col0_row0, col0_row1, ..., col1_row0,
//                             col1_row1, ...], rows, cols)

import { jit_tensor_alloc } from "./tensor_alloc.js";

export function jit_tensor_from_matrix(values, rows, cols) {
  const t = jit_tensor_alloc(rows, cols);
  for (let i = 0; i < rows * cols; i++) t.data[i] = values[i];
  return t;
}
