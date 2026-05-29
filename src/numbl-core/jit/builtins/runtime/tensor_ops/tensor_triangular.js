// JS sibling of `tensor_triangular.h`. Four helpers: `triu` / `tril`
// keep entries where `j - i >= k` / `i - j >= -k`; their `_complex`
// siblings walk both lanes. Mirrors `triPart` in numbl's
// `interpreter/builtins/array-extras.ts`.

import { mtoc2_tensor_alloc } from "../tensor/tensor_alloc.js";
import { mtoc2_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";

export function mtoc2_tensor_triu(a, k) {
  const rows = a.shape[0];
  const cols = a.shape[1];
  const out = mtoc2_tensor_alloc(rows, cols);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      if (j - i >= k) {
        const idx = i + j * rows;
        out.data[idx] = a.data[idx];
      }
    }
  }
  return out;
}

export function mtoc2_tensor_tril(a, k) {
  const rows = a.shape[0];
  const cols = a.shape[1];
  const out = mtoc2_tensor_alloc(rows, cols);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      if (i - j >= -k) {
        const idx = i + j * rows;
        out.data[idx] = a.data[idx];
      }
    }
  }
  return out;
}

export function mtoc2_tensor_triu_complex(a, k) {
  const rows = a.shape[0];
  const cols = a.shape[1];
  const out = mtoc2_tensor_alloc_nd_complex(2, [rows, cols]);
  const im = a.imag;
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      if (j - i >= k) {
        const idx = i + j * rows;
        out.data[idx] = a.data[idx];
        if (im !== undefined) out.imag[idx] = im[idx];
      }
    }
  }
  return out;
}

export function mtoc2_tensor_tril_complex(a, k) {
  const rows = a.shape[0];
  const cols = a.shape[1];
  const out = mtoc2_tensor_alloc_nd_complex(2, [rows, cols]);
  const im = a.imag;
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      if (i - j >= -k) {
        const idx = i + j * rows;
        out.data[idx] = a.data[idx];
        if (im !== undefined) out.imag[idx] = im[idx];
      }
    }
  }
  return out;
}
