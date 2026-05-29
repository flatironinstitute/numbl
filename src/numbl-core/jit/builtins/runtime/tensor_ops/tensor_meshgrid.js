// JS sibling of `tensor_meshgrid.h`. MATLAB-style coordinate grid.

import { mtoc2_tensor_alloc } from "../tensor/tensor_alloc.js";

function numel(t) {
  let n = 1;
  for (const s of t.shape) n *= s;
  return n;
}

export function mtoc2_meshgrid_x(x, y) {
  const M = numel(x);
  const N = numel(y);
  const X = mtoc2_tensor_alloc(N, M);
  for (let j = 0; j < M; j++) {
    const xj = x.data[j];
    for (let i = 0; i < N; i++) X.data[i + j * N] = xj;
  }
  return X;
}

export function mtoc2_meshgrid(x, y) {
  const M = numel(x);
  const N = numel(y);
  const X = mtoc2_tensor_alloc(N, M);
  const Y = mtoc2_tensor_alloc(N, M);
  for (let j = 0; j < M; j++) {
    const xj = x.data[j];
    for (let i = 0; i < N; i++) {
      X.data[i + j * N] = xj;
      Y.data[i + j * N] = y.data[i];
    }
  }
  return { X, Y };
}

export function mtoc2_meshgrid_1arg(x) {
  return mtoc2_meshgrid(x, x);
}
