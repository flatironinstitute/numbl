// JS sibling of `tensor_sort_real.h`. Stable sort on real and
// complex tensors. The descending flag flips the comparator while
// keeping the tie-break on ascending original index. Complex sort
// orders by magnitude then phase (matches numbl).

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";
import { mtoc2_tensor_alloc_nd_complex } from "../tensor/tensor_alloc_nd_complex.js";

function pair_sort_indices(a, descending) {
  const n = a.data.length;
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((p, q) => {
    const av = a.data[p];
    const bv = a.data[q];
    if (av < bv) return descending ? 1 : -1;
    if (av > bv) return descending ? -1 : 1;
    return p - q;
  });
  return idx;
}

function complex_sort_indices(a, descending) {
  const n = a.data.length;
  const im = a.imag;
  const idx = new Array(n);
  const mag = new Float64Array(n);
  const ph = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const re = a.data[i];
    const xi = im !== undefined ? im[i] : 0;
    mag[i] = Math.hypot(re, xi);
    ph[i] = Math.atan2(xi, re);
    idx[i] = i;
  }
  idx.sort((p, q) => {
    if (mag[p] < mag[q]) return descending ? 1 : -1;
    if (mag[p] > mag[q]) return descending ? -1 : 1;
    if (ph[p] < ph[q]) return descending ? 1 : -1;
    if (ph[p] > ph[q]) return descending ? -1 : 1;
    return p - q;
  });
  return idx;
}

export function mtoc2_sort_real(a, descending) {
  const v = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  if (a.data.length === 0) return v;
  const sorted = pair_sort_indices(a, descending);
  for (let i = 0; i < sorted.length; i++) v.data[i] = a.data[sorted[i]];
  return v;
}

export function mtoc2_sort_real_2(a, descending) {
  const v = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  const ix = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  if (a.data.length === 0) return { v, ix };
  const sorted = pair_sort_indices(a, descending);
  for (let i = 0; i < sorted.length; i++) {
    v.data[i] = a.data[sorted[i]];
    ix.data[i] = sorted[i] + 1;
  }
  return { v, ix };
}

export function mtoc2_sort_complex(a, descending) {
  const v = mtoc2_tensor_alloc_nd_complex(a.shape.length, a.shape);
  if (a.data.length === 0) return v;
  const sorted = complex_sort_indices(a, descending);
  const im = a.imag;
  for (let i = 0; i < sorted.length; i++) {
    v.data[i] = a.data[sorted[i]];
    if (im !== undefined) v.imag[i] = im[sorted[i]];
  }
  return v;
}

export function mtoc2_sort_complex_2(a, descending) {
  const v = mtoc2_tensor_alloc_nd_complex(a.shape.length, a.shape);
  const ix = mtoc2_tensor_alloc_nd(a.shape.length, a.shape);
  if (a.data.length === 0) return { v, ix };
  const sorted = complex_sort_indices(a, descending);
  const im = a.imag;
  for (let i = 0; i < sorted.length; i++) {
    v.data[i] = a.data[sorted[i]];
    if (im !== undefined) v.imag[i] = im[sorted[i]];
    ix.data[i] = sorted[i] + 1;
  }
  return { v, ix };
}
