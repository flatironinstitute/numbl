// JS helpers for scalar tensor indexing (1-based, MATLAB-style).
// Mirrors the bounds-checking entry points the C side calls
// (`jit_idx_lin` / `jit_idx_axis`) but returns the 0-based
// offset directly — no out-parameter dance.
//
// Index values are rounded to the nearest integer (matching numbl's
// `Math.round(idx) - 1` convention in runtime/runtimeIndexing.ts), so
// `a(2.5)` reads element 3, not 2. Truncation would diverge from
// numbl and produce subtly wrong answers on any non-integer index.

import { jit_raise_bail, jit_raise_error } from "../system/raise.js";

// Out-of-range index: the interpreter's error (see oob.h).
function rangeError() {
  jit_raise_error("", "Index exceeds array bounds");
}

/** Linear (column-major) 0-based offset for a 1-based MATLAB index. */
export function jit_idx_lin_js(t, k) {
  const i = Math.round(k);
  if (i < 1 || i > t.data.length) rangeError();
  return i - 1;
}

/** Per-axis (0-based for the axis index, 1-based for the value)
 *  bounds-checked subscript. */
export function jit_idx_axis_js(t, axis, k) {
  const i = Math.round(k);
  const dim = t.shape[axis] ?? 1;
  if (i < 1 || i > dim) rangeError();
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
  const i = Math.round(k);
  if (i < 1) rangeError();
  if (i > t.data.length) growBail();
  return i - 1;
}

/** Per-axis (N-arg) store subscript. Same contract as
 *  `jit_idx_lin_grow_js` against a single axis's dim. */
export function jit_idx_axis_grow_js(t, axis, k) {
  const i = Math.round(k);
  const dim = t.shape[axis] ?? 1;
  if (i < 1) rangeError();
  if (i > dim) growBail();
  return i - 1;
}

/** A slice store whose RHS element count does not match the slice: the
 *  interpreter grows an empty base (`A = []; A(:, 1) = v`), so that bails;
 *  otherwise it is the interpreter's dimension-mismatch error. */
export function jit_slice_store_mismatch(base) {
  if (base.data.length === 0) jit_raise_bail("slice store into an empty array");
  jit_raise_error("", "Subscripted assignment dimension mismatch");
}
