// JS sibling of `tensor_eye.h`. Build an m×n identity matrix.

import { mtoc2_tensor_alloc } from "../tensor/tensor_alloc.js";

export function mtoc2_eye_rect(rows, cols) {
  const out = mtoc2_tensor_alloc(rows, cols);
  const m = Math.min(rows, cols);
  for (let i = 0; i < m; i++) out.data[i + i * rows] = 1;
  return out;
}

export function mtoc2_eye_square(n) {
  return mtoc2_eye_rect(n, n);
}
