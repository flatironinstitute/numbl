// JS sibling of `tensor_var.h`. Variance / standard-deviation
// reductions, byte-for-byte with numbl's `varianceOf`
// (interpreter/builtins/reductions.ts) — including the `** 2`
// squared-deviation so opt0 (interpreter) and opt1 (this kernel)
// agree exactly. Helper names are prefixed to avoid colliding with
// `tensor_reduce_real.js`'s `squeeze_trailing` when both snippets are
// inlined into one generated module.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

function var_squeeze_trailing(dims) {
  while (dims.length > 2 && dims[dims.length - 1] === 1) dims.pop();
  return dims;
}

function variance_fiber(data, base, n, stride, w) {
  if (n === 0) return NaN;
  if (n <= 1 && w === 0) return 0;
  let s = 0;
  for (let k = 0; k < n; k++) s += data[base + k * stride];
  const m = s / n;
  let ss = 0;
  for (let k = 0; k < n; k++) ss += (data[base + k * stride] - m) ** 2;
  return ss / (w === 1 ? n : n - 1);
}

function var_all_impl(t, w, transform) {
  return transform(variance_fiber(t.data, 0, t.data.length, 1, w));
}

function var_dim_impl(t, dim, w, transform) {
  if (dim < 1) throw new Error(`var/std _dim: dim must be >= 1 (got ${dim})`);
  if (dim > t.shape.length) {
    // No-op axis: numbl's reduceDim copies the input verbatim when
    // dim > ndim (forEachSlice returns null), regardless of the
    // reducer — so var/std return the input unchanged, not zeros.
    const out = mtoc2_tensor_alloc_nd(t.shape.length, t.shape.slice());
    out.data.set(t.data);
    return out;
  }
  const dimIdx = dim - 1;
  const axis = t.shape[dimIdx];
  let before = 1;
  for (let i = 0; i < dimIdx; i++) before *= t.shape[i];
  let after = 1;
  for (let i = dimIdx + 1; i < t.shape.length; i++) after *= t.shape[i];
  const outDims = t.shape.slice();
  outDims[dimIdx] = 1;
  var_squeeze_trailing(outDims);
  const out = mtoc2_tensor_alloc_nd(outDims.length, outDims);
  for (let aft = 0; aft < after; aft++) {
    for (let bef = 0; bef < before; bef++) {
      const base = aft * before * axis + bef;
      out.data[aft * before + bef] = transform(
        variance_fiber(t.data, base, axis, before, w)
      );
    }
  }
  return out;
}

export function mtoc2_var_all(t, w) {
  return var_all_impl(t, w, x => x);
}
export function mtoc2_var_dim(t, dim, w) {
  return var_dim_impl(t, dim, w, x => x);
}
export function mtoc2_std_all(t, w) {
  return var_all_impl(t, w, Math.sqrt);
}
export function mtoc2_std_dim(t, dim, w) {
  return var_dim_impl(t, dim, w, Math.sqrt);
}
