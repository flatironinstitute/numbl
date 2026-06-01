// JS sibling of tensor_uniquetol.h. Same naive pairwise-against-
// running-list scan, preserving first-occurrence order and the
// transitive-chaining behaviour that follows from it.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

export function mtoc2_uniquetol_real(a, tol, rowOut) {
  const n = a.data.length;
  const uniques = [];
  for (let i = 0; i < n; i++) {
    const x = a.data[i];
    let found = false;
    for (let k = 0; k < uniques.length; k++) {
      const u = uniques[k];
      if (Number.isNaN(u) || Number.isNaN(x)) continue;
      if (Math.abs(x - u) <= tol) {
        found = true;
        break;
      }
    }
    if (!found) uniques.push(x);
  }
  const shape = rowOut ? [1, uniques.length] : [uniques.length, 1];
  const r = mtoc2_tensor_alloc_nd(2, shape);
  for (let i = 0; i < uniques.length; i++) r.data[i] = uniques[i];
  return r;
}
