// JS helpers for scalar tensor indexing (1-based, MATLAB-style).
// Mirrors the bounds-checking entry points the C side calls
// (`jit_idx_lin` / `jit_idx_axis`) but returns the 0-based
// offset directly — no out-parameter dance.
//
// A non-integer (or NaN) index is invalid in numbl (runtime/indexPlan.ts);
// `jit_to_index` maps it to 0, which the checks reject with the
// interpreter's invalid-index error.

import {
  jit_raise_bail,
  jit_raise_error,
  jit_raise_error_deferred,
} from "../system/raise.js";

const BAD_INDEX = "Array indices must be positive integers or logical values.";

/** An index value as an integer, or 0 when it is not one. */
export function jit_to_index(k) {
  return Number.isInteger(k) ? k : 0;
}

/** Raise the interpreter's error for index `i` outside `1..bound`. `pos`
 *  is the 1-based subscript position, or 0 for linear indexing. */
export function jit_index_error(pos, i, bound) {
  if (!(i >= 1)) {
    jit_raise_error(
      "",
      pos === 0
        ? BAD_INDEX
        : `Index in position ${pos} is invalid. ${BAD_INDEX}`
    );
  }
  jit_raise_error(
    "",
    pos === 0
      ? `Index exceeds the number of array elements. Index must not exceed ${bound}.`
      : `Index in position ${pos} exceeds array bounds. Index must not exceed ${bound}.`
  );
}

/** A logical index with a true value past the extent. */
export function jit_mask_oob(pos) {
  jit_raise_error(
    "",
    pos === 0
      ? "The logical indices contain a true value outside of the array bounds."
      : `The logical indices in position ${pos} contain a true value outside of the array bounds.`
  );
}

/** Linear (column-major) 0-based offset for a 1-based MATLAB index. */
export function jit_idx_lin_js(t, k) {
  const i = jit_to_index(k);
  if (i < 1 || i > t.data.length) jit_index_error(0, i, t.data.length);
  return i - 1;
}

/** Per-axis (0-based for the axis index, 1-based for the value)
 *  bounds-checked subscript. */
export function jit_idx_axis_js(t, axis, k) {
  const i = jit_to_index(k);
  const dim = t.shape[axis] ?? 1;
  if (i < 1 || i > dim) jit_index_error(axis + 1, i, dim);
  return i - 1;
}

// Grow-aware store subscripts (JS siblings of the C `jit_idx_*_grow`
// helpers in `grow_bail.h`). An indexed STORE whose index exceeds the
// current extent would GROW the array in MATLAB; the JIT fixes the
// shape at compile time and can't model the new carrier, so it raises a
// bail (`raise.js`) and the interpreter re-runs the unit with full grow
// semantics. A sub-1 index is NOT a grow, it's a genuine error.
// (Statically-provable grows like `v(end+1) = x` on a fixed-shape
// array are declined at compile time and never reach these.)
function growBail() {
  jit_raise_bail("indexed-store array growth");
}

/** Linear (1-arg) store subscript: 0-based offset in bounds, grow-bail
 *  past the end, error on sub-1. */
export function jit_idx_lin_grow_js(t, k) {
  const i = jit_to_index(k);
  if (i < 1) jit_index_error(0, i, t.data.length);
  if (i > t.data.length) growBail();
  return i - 1;
}

/** Per-axis (N-arg) store subscript. Same contract as
 *  `jit_idx_lin_grow_js` against a single axis's dim. */
export function jit_idx_axis_grow_js(t, axis, k) {
  const i = jit_to_index(k);
  const dim = t.shape[axis] ?? 1;
  if (i < 1) jit_index_error(axis + 1, i, dim);
  if (i > dim) growBail();
  return i - 1;
}

/** A linear slice store whose RHS element count differs from the
 *  selection's. */
export function jit_numel_mismatch() {
  jit_raise_error(
    "",
    "Unable to perform assignment because the left and right sides have a different number of elements."
  );
}

/** Check a subscripted slice store's tensor RHS against the per-subscript
 *  extents `counts`: the non-singleton dimensions must agree. On a
 *  mismatch the interpreter grows an empty base (`A = []; A(:, 1) = v`),
 *  which bails, or words a size error, raised on its re-run. */
export function jit_store_shape_check(base, counts, rhs) {
  const a = counts.filter(d => d !== 1);
  const b = rhs.shape.filter(d => d !== 1);
  if (a.length === b.length && a.every((d, i) => d === b[i])) return;
  if (base.data.length === 0) jit_raise_bail("slice store into an empty array");
  jit_raise_error_deferred();
}
