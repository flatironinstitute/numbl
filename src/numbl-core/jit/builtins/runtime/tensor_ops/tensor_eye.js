// JS sibling of `tensor_eye.h`. Build an m×n identity matrix.

import { jit_tensor_alloc } from "../tensor/tensor_alloc.js";

export function jit_eye_rect(rows, cols) {
  const out = jit_tensor_alloc(rows, cols);
  const m = Math.min(rows, cols);
  for (let i = 0; i < m; i++) out.data[i + i * rows] = 1;
  return out;
}

export function jit_eye_square(n) {
  return jit_eye_rect(n, n);
}
