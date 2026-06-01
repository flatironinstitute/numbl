/**
 * Compile-time index-fold helpers. Used by two sites that need to
 * resolve a (statically-known shape, statically-known 1-based index
 * tuple) pair to a column-major flat-buffer offset:
 *
 *   1. `foldedElemType` in `lowerIndexLoad.ts` — folds `v(i, j)` to a
 *      scalar with exact data when the base is fully exact.
 *   2. `tryRefreshExactAfterIndexedWrite` in `lower.ts` — refreshes the
 *      base's exact carrier in place after a single-scalar `IndexStore`.
 *
 * Both need to map (shape, idxVals) → offset using mtoc's column-major
 * stride convention, accept either linear single-index or full N-index
 * forms, and bounds-check each axis. The shared helper keeps the two
 * call sites consistent.
 */

import { shapeNumel } from "./types.js";

/** Map an N-D shape and a 1-based index tuple to a column-major flat
 *  offset, or `undefined` if the index is out-of-range or the tuple
 *  shape doesn't match a supported form.
 *
 *  Supported forms:
 *    - `idxVals.length === 1` (linear single-index): the lone index
 *      addresses the flat buffer 1..numel. Out-of-range returns
 *      `undefined`.
 *    - `idxVals.length === shape.length` (full N-index): each slot
 *      addresses its axis 1..shape[k]. Out-of-range on any axis
 *      returns `undefined`.
 *
 *  Mixed forms (numSlots between 1 and shape.length, or numSlots >
 *  shape.length) are not supported by the fold; the caller falls back
 *  to a runtime read. */
export function columnMajorOffsetFromIndices(
  shape: ReadonlyArray<number>,
  idxVals: ReadonlyArray<number>
): number | undefined {
  if (idxVals.length === 1) {
    const total = shapeNumel(shape);
    const lin = idxVals[0];
    if (lin > total) return undefined;
    return lin - 1;
  }
  if (idxVals.length === shape.length) {
    let offset = 0;
    let stride = 1;
    for (let k = 0; k < shape.length; k++) {
      if (idxVals[k] > shape[k]) return undefined;
      offset += (idxVals[k] - 1) * stride;
      stride *= shape[k];
    }
    return offset;
  }
  return undefined;
}
