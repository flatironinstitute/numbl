// JS helpers for scalar tensor indexing (1-based, MATLAB-style).
// Mirrors the bounds-checking entry points the C side calls
// (`mtoc2_idx_lin` / `mtoc2_idx_axis`) but returns the 0-based
// offset directly — no out-parameter dance.
//
// Index values are rounded to the nearest integer (matching numbl's
// `Math.round(idx) - 1` convention in runtime/runtimeIndexing.ts), so
// `a(2.5)` reads element 3, not 2. Truncation would diverge from
// numbl and produce subtly wrong answers on any non-integer index.

function rangeError(idx, bound, where) {
  throw new RangeError(
    `Index in position ${where} (${idx}) exceeds array bounds (${bound})`
  );
}

/** Linear (column-major) 0-based offset for a 1-based MATLAB index. */
export function mtoc2_idx_lin_js(t, k) {
  const i = Math.round(k);
  if (i < 1 || i > t.data.length) rangeError(i, t.data.length, 1);
  return i - 1;
}

/** Per-axis (0-based for the axis index, 1-based for the value)
 *  bounds-checked subscript. */
export function mtoc2_idx_axis_js(t, axis, k) {
  const i = Math.round(k);
  const dim = t.shape[axis] ?? 1;
  if (i < 1 || i > dim) rangeError(i, dim, axis + 1);
  return i - 1;
}

// Grow-aware store subscripts (JS siblings of the C `mtoc2_idx_*_grow`
// helpers in `grow_bail.h`). An indexed STORE whose index exceeds the
// current extent would GROW the array in MATLAB; the JIT fixes the
// shape at compile time and can't model the new carrier, so it throws
// a tagged sentinel. The whole-scope / loop executor recognizes the
// `mtoc2GrowBail` tag, bails to the interpreter (which has full grow
// semantics), and surfaces a one-time warning. A sub-1 index is NOT a
// grow — it's a genuine error — and still throws a plain RangeError.
// (Statically-provable grows like `v(end+1) = x` on a fixed-shape
// array are declined at compile time and never reach these.)
function growBailError(idx, len) {
  const e = new Error(
    `mtoc2: indexed assignment at ${idx} would grow a ${len}-element ` +
      `array; array growth is not supported in the JIT`
  );
  e.mtoc2GrowBail = true;
  return e;
}

/** Linear (1-arg) store subscript: 0-based offset in bounds, grow-bail
 *  past the end, RangeError on sub-1. */
export function mtoc2_idx_lin_grow_js(t, k) {
  const i = Math.round(k);
  if (i < 1) rangeError(i, t.data.length, 1);
  if (i > t.data.length) throw growBailError(i, t.data.length);
  return i - 1;
}

/** Per-axis (N-arg) store subscript. Same contract as
 *  `mtoc2_idx_lin_grow_js` against a single axis's dim. */
export function mtoc2_idx_axis_grow_js(t, axis, k) {
  const i = Math.round(k);
  const dim = t.shape[axis] ?? 1;
  if (i < 1) rangeError(i, dim, axis + 1);
  if (i > dim) throw growBailError(i, dim);
  return i - 1;
}
