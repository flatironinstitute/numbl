// JS sibling of `tensor_alloc_nd.h`. Allocate a real tensor with
// the given dimensions (column-major). `data` is initialised to all
// zeros by Float64Array's default semantics — the same as the C side
// (which routes through `mtoc2_alloc` + a NULL check; here we trust
// the engine to keep this allocation tiny enough to not need a
// guarded path).

import { mtoc2_tensor_make } from "./tensor.js";

export function mtoc2_tensor_alloc_nd(ndim, dims) {
  const shape = [];
  for (let i = 0; i < ndim; i++) {
    // MATLAB / numbl clamp negative dim values to 0 (empty tensor)
    // rather than aborting. Mirrors the C helper.
    const d = dims[i] < 0 ? 0 : dims[i];
    shape.push(d);
  }
  let total = 1;
  for (const s of shape) total *= s;
  return mtoc2_tensor_make(shape, new Float64Array(total));
}
