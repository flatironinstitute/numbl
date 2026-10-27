// JS sibling of `tensor_reduce_real.h`. Real-tensor reductions —
// `_all` returns a scalar; `_dim` returns a freshly-allocated tensor
// reduced along the 1-based `dim` axis. Mirrors numbl's
// `forEachSlice` semantics with column-major (before × axis × after)
// traversal.
//
// Output shape rule for `_dim`: input dims with `dims[dim-1] = 1`,
// then trailing singletons stripped subject to a 2-axis floor.

import { mtoc2_tensor_alloc_nd } from "../tensor/tensor_alloc_nd.js";

function squeeze_trailing(dims) {
  while (dims.length > 2 && dims[dims.length - 1] === 1) dims.pop();
  return dims;
}

// Accumulator-based reducers (`sum`, `prod`, `mean`). The loops add or
// multiply inline (sequentially, like the C kernel) rather than through
// an accumulator callback, which went megamorphic in V8.
function sum_all_seq(x) {
  let acc = 0;
  for (let i = 0; i < x.length; i++) acc += x[i];
  return acc;
}

function prod_all_seq(x) {
  let acc = 1;
  for (let i = 0; i < x.length; i++) acc *= x[i];
  return acc;
}

function accum_dim(t, dim, isProd, isMean) {
  if (dim < 1) {
    throw new Error(`reducer _dim: dim must be >= 1 (got ${dim})`);
  }
  if (dim > t.shape.length) {
    // Reducing along a trailing singleton axis is the identity: every
    // fiber has length 1, so sum/prod/mean each yield that single
    // element unchanged. Return a same-shape COPY of the data (the C
    // kernel memcpy's here too); allocating without copying left a
    // zero-filled tensor — the original opt1 bug.
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
  const outDims = squeeze_trailing(t.shape.slice());
  outDims[dimIdx] = 1;
  // Re-squeeze after the in-place axis update (the original `out_dims
  // = a.shape.slice(); out_dims[dimIdx] = 1` then squeeze pattern).
  squeeze_trailing(outDims);
  const out = mtoc2_tensor_alloc_nd(outDims.length, outDims);
  const x = t.data;
  for (let aft = 0; aft < after; aft++) {
    for (let bef = 0; bef < before; bef++) {
      const base = aft * before * axis + bef;
      let acc;
      if (isProd) {
        acc = 1;
        for (let k = 0; k < axis; k++) acc *= x[base + k * before];
      } else {
        acc = 0;
        for (let k = 0; k < axis; k++) acc += x[base + k * before];
      }
      out.data[aft * before + bef] = isMean
        ? axis === 0
          ? NaN
          : acc / axis
        : acc;
    }
  }
  return out;
}

// Min/max reducer. Ignores NaN like numbl/MATLAB: NaN entries are
// skipped, and the result is NaN only when every element is NaN.
// Mirrors the interpreter's `minMaxScan` (helpers/reduction/min-max.ts)
// and the C kernel — seeding with data[0] would let a *leading* NaN
// poison the result (`x < NaN` / `x > NaN` are always false).
function minmax_all(t, op /* "min" | "max" */) {
  const x = t.data;
  if (x.length === 0) return op === "min" ? Infinity : -Infinity;
  // Large inputs use the host's kernel (same as the interpreter) when
  // installed; see `tensor_elemwise_real.js`.
  if (x.length >= 1024 && typeof $elemwise === "object" && $elemwise) {
    return $elemwise.minmax(op, x);
  }
  // Seed with the first non-NaN element, then a plain compare loop per op
  // (a NaN never wins a comparison, so it is skipped).
  let i = 0;
  while (i < x.length && x[i] !== x[i]) i++;
  if (i === x.length) return NaN;
  let best = x[i];
  if (op === "min") {
    for (i++; i < x.length; i++) if (x[i] < best) best = x[i];
  } else {
    for (i++; i < x.length; i++) if (x[i] > best) best = x[i];
  }
  return best;
}

function minmax_dim(t, dim, op) {
  if (dim < 1) throw new Error(`reducer _dim: dim must be >= 1 (got ${dim})`);
  if (dim > t.shape.length) {
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
  squeeze_trailing(outDims);
  const out = mtoc2_tensor_alloc_nd(outDims.length, outDims);
  for (let aft = 0; aft < after; aft++) {
    for (let bef = 0; bef < before; bef++) {
      const base = aft * before * axis + bef;
      let best = NaN;
      let found = false;
      for (let k = 0; k < axis; k++) {
        const x = t.data[base + k * before];
        if (x !== x) continue; // skip NaN
        if (!found || (op === "min" ? x < best : x > best)) {
          best = x;
          found = true;
        }
      }
      out.data[aft * before + bef] = found ? best : NaN;
    }
  }
  return out;
}

// Logical reducer (`any`, `all`). `emptyResult` is the value for a
// 0-element reduction; `short` is the early-exit predicate.
function logical_all(t, emptyResult, shortPredicate) {
  if (t.data.length === 0) return emptyResult;
  for (let i = 0; i < t.data.length; i++) {
    if (shortPredicate(t.data[i])) return emptyResult === 1 ? 0 : 1;
  }
  return emptyResult;
}

function logical_dim(t, dim, emptyResult, shortPredicate) {
  if (dim < 1) throw new Error(`reducer _dim: dim must be >= 1 (got ${dim})`);
  if (dim > t.shape.length) {
    // No-op axis: emit a logical cast of the input (each element →
    // 1 if nonzero, 0 otherwise).
    const out = mtoc2_tensor_alloc_nd(t.shape.length, t.shape.slice());
    for (let i = 0; i < t.data.length; i++) {
      out.data[i] = t.data[i] !== 0 ? 1 : 0;
    }
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
  squeeze_trailing(outDims);
  const out = mtoc2_tensor_alloc_nd(outDims.length, outDims);
  for (let aft = 0; aft < after; aft++) {
    for (let bef = 0; bef < before; bef++) {
      const base = aft * before * axis + bef;
      let res = emptyResult;
      for (let k = 0; k < axis; k++) {
        if (shortPredicate(t.data[base + k * before])) {
          res = emptyResult === 1 ? 0 : 1;
          break;
        }
      }
      out.data[aft * before + bef] = res;
    }
  }
  return out;
}

// ── Sum / prod / mean ───────────────────────────────────────────────
export const mtoc2_sum_all = t => sum_all_seq(t.data);
export const mtoc2_sum_dim = (t, d) => accum_dim(t, d, false, false);
export const mtoc2_prod_all = t => prod_all_seq(t.data);
export const mtoc2_prod_dim = (t, d) => accum_dim(t, d, true, false);
export const mtoc2_mean_all = t =>
  t.data.length === 0 ? NaN : sum_all_seq(t.data) / t.data.length;
export const mtoc2_mean_dim = (t, d) => accum_dim(t, d, false, true);

// ── Min / max ───────────────────────────────────────────────────────────
export const mtoc2_min_all = t => minmax_all(t, "min");
export const mtoc2_min_dim = (t, d) => minmax_dim(t, d, "min");
export const mtoc2_max_all = t => minmax_all(t, "max");
export const mtoc2_max_dim = (t, d) => minmax_dim(t, d, "max");

// ── Any / all ───────────────────────────────────────────────────────────
// any: short-circuits on a non-NaN nonzero; emptyResult = 0. NaN is
// ignored (MATLAB: any(NaN) is 0, any([0 NaN]) is 0), so `x === x`
// excludes it — without that, NaN wrongly short-circuited to true.
// (`all` needs no such guard: allShort tests `x === 0`, which NaN
// already fails, so a NaN simply doesn't force all to false.)
const anyShort = x => x !== 0 && x === x;
export const mtoc2_any_all = t => logical_all(t, 0, anyShort);
export const mtoc2_any_dim = (t, d) => logical_dim(t, d, 0, anyShort);
// all: short-circuits on zero; emptyResult = 1.
const allShort = x => x === 0;
export const mtoc2_all_all = t => logical_all(t, 1, allShort);
export const mtoc2_all_dim = (t, d) => logical_dim(t, d, 1, allShort);
