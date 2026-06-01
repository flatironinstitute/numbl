// JS sibling of `tensor_from_matrix.h`. Build an R×C tensor from a
// flat column-major array. The codegen path emits:
//   mtoc2_tensor_from_matrix([col0_row0, col0_row1, ..., col1_row0,
//                             col1_row1, ...], rows, cols)

import { mtoc2_tensor_alloc } from "./tensor_alloc.js";

export function mtoc2_tensor_from_matrix(values, rows, cols) {
  const t = mtoc2_tensor_alloc(rows, cols);
  for (let i = 0; i < rows * cols; i++) t.data[i] = values[i];
  return t;
}
