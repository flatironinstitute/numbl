/**
 * Storage-independent index planning.
 *
 * Every container kind stores its elements in a flat column-major array
 * plus a shape. `planIndex` (read), `planStore` (assignment, with growth)
 * and `planDelete` (`A(idx) = []`) turn that shape and the resolved
 * subscripts into a `Selection` of linear offsets plus the result shape,
 * applying MATLAB's rules once: logical masks, colon, ranges, linear vs
 * subscript indexing, trailing-dimension collapse, result orientation,
 * growth, and the MATLAB error messages. A container then only has to
 * gather the selected offsets, scatter values into them, and resize its
 * storage (see `forEachGrowRun`).
 */

import { RuntimeError } from "./error.js";
import {
  type RuntimeValue,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeComplexNumber,
} from "./types.js";
import { toNumber } from "./convert.js";

/** Sentinel marker for colon (:) indexing. */
export const COLON_INDEX: RuntimeValue = "__COLON__";

/** Linear offsets into column-major storage. When `offsets` is null the
 *  k-th offset is `start + k * step`; otherwise it is `offsets[k] - bias`
 *  (so raw 1-based index data can be used without copying). */
export interface Selection {
  n: number;
  start: number;
  step: number;
  offsets: ArrayLike<number> | null;
  bias: number;
}

/** Result of planning. All plans share one object layout (fields not
 *  relevant to a plan kind keep their defaults). */
export interface Plan extends Selection {
  /** Read: shape of the result. Delete: shape after deletion. */
  shape: number[];
  /** Store: new shape of the target when the assignment grows it, else
   *  null. Offsets refer to the grown layout. */
  grow: number[] | null;
  /** Delete: true when nothing is deleted and the target is unchanged;
   *  otherwise the selection lists the kept offsets. */
  noop: boolean;
}

export type ReadPlan = Plan;
export type StorePlan = Plan;
export type DeletePlan = Plan;

/** Placeholder `shape` of store plans (not meaningful for stores). */
const SCALAR_SHAPE_OUT: number[] = [1, 1];

function mkPlan(
  n: number,
  start: number,
  step: number,
  offsets: ArrayLike<number> | null,
  bias: number,
  shape: number[]
): Plan {
  return { n, start, step, offsets, bias, shape, grow: null, noop: false };
}

function selPlan(sel: Selection, shape: number[]): Plan {
  return mkPlan(sel.n, sel.start, sel.step, sel.offsets, sel.bias, shape);
}

/** The k-th offset of a selection. */
export function offsetAt(sel: Selection, k: number): number {
  return sel.offsets === null
    ? sel.start + k * sel.step
    : sel.offsets[k] - sel.bias;
}

// ── Messages ─────────────────────────────────────────────────────────────

const BAD_INDEX = "Array indices must be positive integers or logical values.";
const AMBIGUOUS_GROW = "Attempt to grow array along ambiguous dimension.";
const DELETE_OOB = "Matrix index is out of range for deletion.";
const NUMEL_MISMATCH =
  "Unable to perform assignment because the left and right sides have a different number of elements.";

function badIndex(pos: number): RuntimeError {
  return new RuntimeError(
    pos === 0 ? BAD_INDEX : `Index in position ${pos} is invalid. ${BAD_INDEX}`
  );
}

function sizeStr(shape: readonly number[]): string {
  return shape.join("-by-");
}

// ── Subscript resolution ─────────────────────────────────────────────────

/** One subscript resolved to 0-based positions along a dimension:
 *  `start .. start+count-1` when `pos` is null, else `pos[k] - bias`. */
interface Sub {
  pos: ArrayLike<number> | null;
  bias: number;
  start: number;
  count: number;
  /** 1 + the largest position (0 when empty): the extent it needs. */
  max: number;
  colon: boolean;
  logical: boolean;
  /** Shape of the index value (for linear-index result orientation). */
  ishape: readonly number[];
}

const SCALAR_SHAPE: readonly number[] = [1, 1];

function rangeSub(start: number, count: number, colon: boolean): Sub {
  return {
    pos: null,
    bias: 0,
    start,
    count,
    max: count > 0 ? start + count : 0,
    colon,
    logical: false,
    ishape: SCALAR_SHAPE,
  };
}

function maskSub(mask: ArrayLike<number>, ishape: readonly number[]): Sub {
  const len = mask.length;
  let count = 0;
  let first = -1;
  let last = -1;
  for (let i = 0; i < len; i++) {
    if (mask[i] !== 0) {
      if (first < 0) first = i;
      last = i;
      count++;
    }
  }
  let pos: Int32Array | null = null;
  if (count > 0 && last - first + 1 !== count) {
    pos = new Int32Array(count);
    let k = 0;
    for (let i = first; i <= last; i++) if (mask[i] !== 0) pos[k++] = i;
  }
  return {
    pos,
    bias: 0,
    start: first < 0 ? 0 : first,
    count,
    max: last + 1,
    colon: false,
    logical: true,
    ishape,
  };
}

/** Validate 1-based numeric indices; ranges with unit step collapse to a
 *  contiguous run so reads like `v(a:b)` copy a block. */
function numericSub(
  d: ArrayLike<number>,
  ishape: readonly number[],
  where: number
): Sub {
  const len = d.length;
  if (len === 0) {
    return {
      pos: null,
      bias: 0,
      start: 0,
      count: 0,
      max: 0,
      colon: false,
      logical: false,
      ishape,
    };
  }
  const first = d[0];
  let max = 0;
  let unit = true;
  for (let i = 0; i < len; i++) {
    const x = d[i];
    if (!Number.isInteger(x) || x < 1) throw badIndex(where);
    if (x > max) max = x;
    if (unit && x !== first + i) unit = false;
  }
  return {
    pos: unit ? null : d,
    bias: unit ? 0 : 1,
    start: first - 1,
    count: len,
    max,
    colon: false,
    logical: false,
    ishape,
  };
}

/** Resolve one subscript. `extent` sizes a colon; `where` is the 1-based
 *  subscript position for messages (0 for linear indexing). Bounds are
 *  checked by the caller, which knows whether the target may grow. */
function resolveSub(idx: RuntimeValue, extent: number, where: number): Sub {
  if (typeof idx === "number") {
    if (!Number.isInteger(idx) || idx < 1) throw badIndex(where);
    return rangeSub(idx - 1, 1, false);
  }
  if (typeof idx === "boolean") {
    const s = rangeSub(0, idx ? 1 : 0, false);
    s.logical = true;
    return s;
  }
  if (idx === COLON_INDEX) return rangeSub(0, extent, true);
  if (isRuntimeTensor(idx)) {
    return idx._isLogical
      ? maskSub(idx.data, idx.shape)
      : numericSub(idx.data, idx.shape, where);
  }
  if (isRuntimeChar(idx)) {
    if (idx.value === ":") return rangeSub(0, extent, true);
    // A char subscript indexes by its character codes (column-major).
    const rows = idx.shape ? idx.shape[0] : 1;
    const cols = idx.shape ? idx.shape[1] : idx.value.length;
    const codes = new Float64Array(rows * cols);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        codes[c * rows + r] = idx.value.charCodeAt(r * cols + c);
    return numericSub(codes, [rows, cols], where);
  }
  const x = isRuntimeComplexNumber(idx) ? idx.re : toNumber(idx);
  if (!Number.isInteger(x) || x < 1) throw badIndex(where);
  return rangeSub(x - 1, 1, false);
}

function subAt(s: Sub, j: number): number {
  return s.pos === null ? s.start + j : s.pos[j] - s.bias;
}

// ── Shape helpers ────────────────────────────────────────────────────────

function numelOf(shape: readonly number[]): number {
  let n = 1;
  for (let i = 0; i < shape.length; i++) n *= shape[i];
  return n;
}

/** A packed small-integer copy of a shape. Shapes built here always use
 *  this representation so the planner's array accesses stay monomorphic. */
function shapeCopy(shape: readonly number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < shape.length; i++) out.push(shape[i] | 0);
  return out;
}

/** Drop trailing singleton dimensions, keeping at least two. */
function trimShape(shape: number[]): number[] {
  let len = shape.length;
  while (len > 2 && shape[len - 1] === 1) len--;
  if (len !== shape.length) shape.length = len;
  return shape;
}

/** The dimensions seen by `k` subscripts: trailing dimensions collapse
 *  into the last subscript, missing ones are 1. */
function effectiveDims(shape: readonly number[], k: number): number[] {
  const dims: number[] = [];
  for (let d = 0; d < k; d++) dims.push(d < shape.length ? shape[d] | 0 : 1);
  for (let d = k; d < shape.length; d++) dims[k - 1] *= shape[d];
  return dims;
}

/** True when `k` subscripts fold non-singleton trailing dimensions. */
function collapses(shape: readonly number[], k: number): boolean {
  for (let d = k; d < shape.length; d++) if (shape[d] !== 1) return true;
  return false;
}

function isVectorShape(s: readonly number[]): boolean {
  return s.length === 2 && (s[0] === 1 || s[1] === 1);
}

function isColumnShape(s: readonly number[]): boolean {
  return s.length === 2 && s[1] === 1 && s[0] !== 1;
}

/** Non-singleton dimensions, in order. */
function squeezed(s: readonly number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < s.length; i++) if (s[i] !== 1) out.push(s[i]);
  return out;
}

function sameDims(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Result shape of `A(I)` (MATLAB): the index's shape, except that a
 *  vector indexed by a vector keeps its own orientation, `A(:)` is a
 *  column, and a logical mask acts like `find(mask)`. */
function linearResultShape(
  base: readonly number[],
  s: Sub,
  n: number
): number[] {
  if (s.colon) return [n, 1];
  let is = s.ishape;
  if (s.logical) {
    if (is.length === 2 && is[0] === 1 && is[1] === 1) {
      is = n === 1 ? SCALAR_SHAPE : [0, 0];
    } else if (is.length === 2 && is[0] === 0 && is[1] === 0) {
      is = [0, 0];
    } else {
      is = is.length === 2 && is[0] === 1 ? [1, n] : [n, 1];
    }
  }
  if (is.length < 2) is = [1, is.length === 1 ? is[0] : 1];
  if (is.length === 2 && is[0] === 0 && is[1] === 0) return [0, 0];
  if (
    isVectorShape(is) &&
    isVectorShape(base) &&
    !(base[0] === 1 && base[1] === 1)
  ) {
    return base[0] === 1 ? [1, n] : [n, 1];
  }
  return shapeCopy(is);
}

// ── Selection building ───────────────────────────────────────────────────

function subPlan(s: Sub, shape: number[]): Plan {
  return mkPlan(s.count, s.start, 1, s.pos, s.bias, shape);
}

/** Offsets of the cartesian product of per-dimension subscripts over
 *  `dims`, in column-major order. Strided and contiguous selections
 *  (columns, rows, slabs, ranges) avoid materializing offsets. */
function buildSelection(
  dims: readonly number[],
  ss: readonly Sub[]
): Selection {
  const k = ss.length;
  let n = 1;
  for (let d = 0; d < k; d++) n *= ss[d].count;
  if (n === 0) return { n: 0, start: 0, step: 1, offsets: null, bias: 0 };

  const strides: number[] = [];
  let stride = 1;
  let first = 0;
  let nMulti = 0;
  let multi = -1;
  for (let d = 0; d < k; d++) {
    strides.push(stride);
    first += subAt(ss[d], 0) * stride;
    stride *= dims[d];
    if (ss[d].count > 1) {
      nMulti++;
      multi = d;
    }
  }
  if (nMulti === 0)
    return { n: 1, start: first, step: 1, offsets: null, bias: 0 };
  if (nMulti === 1 && ss[multi].pos === null) {
    return { n, start: first, step: strides[multi], offsets: null, bias: 0 };
  }
  // Leading full dimensions, then a range, then singletons: one block.
  let d0 = 0;
  while (
    d0 < k &&
    ss[d0].pos === null &&
    ss[d0].start === 0 &&
    ss[d0].count === dims[d0]
  )
    d0++;
  if (d0 === k) return { n, start: 0, step: 1, offsets: null, bias: 0 };
  if (ss[d0].pos === null) {
    let block = true;
    for (let d = d0 + 1; d < k; d++)
      if (ss[d].count !== 1) {
        block = false;
        break;
      }
    if (block) return { n, start: first, step: 1, offsets: null, bias: 0 };
  }

  // General case: materialize. Dimension 0 is the inner loop; the rest
  // advance as an odometer.
  const dimOff: Int32Array[] = new Array(k);
  for (let d = 0; d < k; d++) {
    const s = ss[d];
    const arr = new Int32Array(s.count);
    for (let j = 0; j < s.count; j++) arr[j] = subAt(s, j) * strides[d];
    dimOff[d] = arr;
  }
  const offsets = new Int32Array(n);
  const o0 = dimOff[0];
  const c0 = o0.length;
  const idx = new Int32Array(k);
  let outer = 0;
  for (let d = 1; d < k; d++) outer += dimOff[d][0];
  let i = 0;
  for (;;) {
    for (let j = 0; j < c0; j++) offsets[i++] = outer + o0[j];
    let d = 1;
    for (; d < k; d++) {
      const od = dimOff[d];
      outer -= od[idx[d]];
      if (++idx[d] < od.length) {
        outer += od[idx[d]];
        break;
      }
      idx[d] = 0;
      outer += od[0];
    }
    if (d === k) break;
  }
  return { n, start: 0, step: 1, offsets, bias: 0 };
}

// ── Read ─────────────────────────────────────────────────────────────────

function linearOob(N: number): RuntimeError {
  return new RuntimeError(
    `Index exceeds the number of array elements. Index must not exceed ${N}.`
  );
}

/** Linear offset of `A(subs{:})` when every subscript is an in-range
 *  integer number (trailing dimensions collapse into the last subscript),
 *  else -1. `shape` must be at least 2-D. This is the scalar fast path
 *  shared by the planner and the runtime's read/store entry points. */
export function scalarSubsOffset(
  shape: readonly number[],
  numel: number,
  subs: readonly unknown[]
): number {
  const k = subs.length;
  if (k === 1) {
    const i = subs[0];
    return typeof i === "number" && Number.isInteger(i) && i >= 1 && i <= numel
      ? i - 1
      : -1;
  }
  if (k === 2) {
    const r = subs[0];
    const c = subs[1];
    if (typeof r !== "number" || typeof c !== "number") return -1;
    const rows = shape[0];
    const cols = rows > 0 ? numel / rows : 0;
    if (
      !Number.isInteger(r) ||
      !Number.isInteger(c) ||
      r < 1 ||
      c < 1 ||
      r > rows ||
      c > cols
    )
      return -1;
    return (c - 1) * rows + (r - 1);
  }
  if (k === 0) return -1;
  let lin = 0;
  let stride = 1;
  for (let d = 0; d < k; d++) {
    const x = subs[d];
    if (typeof x !== "number" || !Number.isInteger(x) || x < 1) return -1;
    let dim = d < shape.length ? shape[d] : 1;
    if (d === k - 1) for (let j = d + 1; j < shape.length; j++) dim *= shape[j];
    if (x > dim) return -1;
    lin += (x - 1) * stride;
    stride *= dim;
  }
  return lin;
}

/** Plan `A(subs{:})` for an array of the given shape. */
export function planIndex(
  shape: readonly number[],
  numel: number,
  subs: readonly RuntimeValue[]
): ReadPlan {
  const k = subs.length;
  if (k === 1) {
    const idx = subs[0];
    const N = numel;
    if (typeof idx === "number") {
      if (!Number.isInteger(idx) || idx < 1) throw badIndex(0);
      if (idx > N) throw linearOob(N);
      return mkPlan(1, idx - 1, 1, null, 0, [1, 1]);
    }
    const s = resolveSub(idx, N, 0);
    if (s.max > N) {
      throw s.logical
        ? new RuntimeError(
            "The logical indices contain a true value outside of the array bounds."
          )
        : linearOob(N);
    }
    return subPlan(s, linearResultShape(shape, s, s.count));
  }
  if (k === 0) return mkPlan(numel, 0, 1, null, 0, shapeCopy(shape));
  const lin = scalarSubsOffset(shape, numel, subs);
  if (lin >= 0) return mkPlan(1, lin, 1, null, 0, [1, 1]);
  if (k === 2 && shape.length === 2) {
    // A(:, j) and A(i, :) on a matrix.
    const a = subs[0];
    const b = subs[1];
    const rows = shape[0];
    const cols = shape[1];
    if (a === COLON_INDEX && typeof b === "number") {
      if (Number.isInteger(b) && b >= 1 && b <= cols)
        return mkPlan(rows, (b - 1) * rows, 1, null, 0, [rows, 1]);
    } else if (typeof a === "number" && b === COLON_INDEX) {
      if (Number.isInteger(a) && a >= 1 && a <= rows)
        return mkPlan(cols, a - 1, rows, null, 0, [1, cols]);
    }
  }
  const dims = effectiveDims(shape, k);
  const collapsed = collapses(shape, k);
  const ss = new Array<Sub>(k);
  const outShape: number[] = [];
  for (let d = 0; d < k; d++) {
    const s = resolveSub(subs[d], dims[d], d + 1);
    if (s.max > dims[d]) {
      if (s.logical) {
        throw new RuntimeError(
          `The logical indices in position ${d + 1} contain a true value outside of the array bounds.`
        );
      }
      throw new RuntimeError(
        collapsed && d === k - 1
          ? `Index in position ${d + 1} exceeds array bounds.`
          : `Index in position ${d + 1} exceeds array bounds. Index must not exceed ${dims[d]}.`
      );
    }
    ss[d] = s;
    outShape.push(s.count);
  }
  return selPlan(buildSelection(dims, ss), trimShape(outShape));
}

// ── Store ────────────────────────────────────────────────────────────────

/** Shape after `A(i) = x` grows a linear index past numel: vectors grow
 *  along their orientation, empties and scalars become rows, anything
 *  else is ambiguous. */
function linearGrowShape(
  shape: readonly number[],
  numel: number,
  len: number
): number[] {
  if (isColumnShape(shape)) return [len, 1];
  if (numel === 0 || (shape.length === 2 && shape[0] === 1)) return [1, len];
  throw new RuntimeError(AMBIGUOUS_GROW);
}

/** Colon extents for a store into a 0x0 array, taken from the RHS:
 *  positionally when that fits, otherwise by matching the RHS's
 *  non-singleton dimensions in order (MATLAB). */
function inferColonExtents(
  ss: (Sub | null)[],
  rhsNumel: number,
  rhsShape: readonly number[]
): number[] {
  const k = ss.length;
  const ext: number[] = [];
  for (let d = 0; d < k; d++) {
    const s = ss[d];
    ext.push(s ? s.count : d < rhsShape.length ? rhsShape[d] | 0 : 1);
  }
  if (rhsNumel === 1 || sameDims(squeezed(ext), squeezed(rhsShape))) return ext;
  const ns = squeezed(rhsShape);
  let j = 0;
  for (let d = 0; d < k; d++) {
    const s = ss[d];
    if (!s) {
      ext[d] = j < ns.length ? ns[j++] : 1;
    } else if (s.count !== 1) {
      if (j < ns.length && ns[j] === s.count) j++;
      else j = -1;
      if (j < 0) break;
    }
  }
  if (j !== ns.length) {
    throw new RuntimeError(
      "Unable to perform assignment because the indices on the left side are not compatible with the size of the right side."
    );
  }
  return ext;
}

/** Plan `A(subs{:}) = rhs` where the RHS has `rhsNumel` elements and
 *  shape `rhsShape`. A one-element RHS is scalar-expanded; otherwise its
 *  element count (linear indexing) or non-singleton dimensions (subscript
 *  indexing) must match the selection. Out-of-range subscripts grow the
 *  target (`grow`). */
export function planStore(
  shape: readonly number[],
  numel: number,
  subs: readonly RuntimeValue[],
  rhsNumel: number,
  rhsShape: readonly number[]
): StorePlan {
  const k = subs.length;
  if (k === 1) {
    const idx = subs[0];
    const N = numel;
    if (typeof idx === "number") {
      if (!Number.isInteger(idx) || idx < 1) throw badIndex(0);
      if (rhsNumel !== 1) throw new RuntimeError(NUMEL_MISMATCH);
      const plan = mkPlan(1, idx - 1, 1, null, 0, SCALAR_SHAPE_OUT);
      if (idx > N) plan.grow = linearGrowShape(shape, N, idx);
      return plan;
    }
    const s = resolveSub(idx, N, 0);
    if (rhsNumel !== 1 && rhsNumel !== s.count)
      throw new RuntimeError(NUMEL_MISMATCH);
    const plan = subPlan(s, SCALAR_SHAPE_OUT);
    if (s.bias !== 0 && s.pos !== null) {
      // Copy raw index data: the index may alias the target (x(x) = v).
      const src = s.pos;
      const offs = new Int32Array(s.count);
      for (let j = 0; j < offs.length; j++) offs[j] = src[j] - 1;
      plan.offsets = offs;
      plan.bias = 0;
    }
    if (s.max > N) plan.grow = linearGrowShape(shape, N, s.max);
    return plan;
  }
  if (rhsNumel === 1 && k > 1) {
    const lin = scalarSubsOffset(shape, numel, subs);
    if (lin >= 0) return mkPlan(1, lin, 1, null, 0, SCALAR_SHAPE_OUT);
  }
  if (k === 0) throw new RuntimeError("Assignment requires at least one index");

  const dims = effectiveDims(shape, k);
  const collapsed = collapses(shape, k);
  const ss = new Array<Sub | null>(k);
  let anyColon = false;
  for (let d = 0; d < k; d++) {
    const idx = subs[d];
    if (idx === COLON_INDEX) {
      ss[d] = null;
      anyColon = true;
    } else {
      ss[d] = resolveSub(idx, dims[d], d + 1);
    }
  }
  if (anyColon) {
    const is0x0 = shape.length === 2 && shape[0] === 0 && shape[1] === 0;
    const ext = is0x0 ? inferColonExtents(ss, rhsNumel, rhsShape) : dims;
    for (let d = 0; d < k; d++)
      if (ss[d] === null) ss[d] = rangeSub(0, ext[d], true);
  }
  const subsR = ss as Sub[];

  const newDims = dims.slice();
  let grows = false;
  for (let d = 0; d < k; d++) {
    const m = subsR[d].max;
    if (m > newDims[d]) {
      if (collapsed && d === k - 1) throw new RuntimeError(AMBIGUOUS_GROW);
      newDims[d] = m;
      grows = true;
    }
  }

  if (rhsNumel !== 1) {
    const lhs: number[] = [];
    for (let d = 0; d < k; d++) lhs.push(subsR[d].count);
    if (!sameDims(squeezed(lhs), squeezed(rhsShape))) {
      throw new RuntimeError(
        `Unable to perform assignment because the size of the left side is ${sizeStr(trimShape(lhs))} and the size of the right side is ${sizeStr(rhsShape)}.`
      );
    }
  }

  const plan = selPlan(buildSelection(newDims, subsR), SCALAR_SHAPE_OUT);
  if (grows) {
    const full = collapsed
      ? newDims.slice(0, k - 1).concat(shape.slice(k - 1))
      : newDims;
    plan.grow = trimShape(full);
  }
  return plan;
}

// ── Delete ───────────────────────────────────────────────────────────────

/** Mark the positions of `s` in a 0/1 array of length `len`; returns the
 *  number of distinct marked positions. */
function markPositions(s: Sub, del: Uint8Array): number {
  let nDel = 0;
  for (let j = 0; j < s.count; j++) {
    const p = subAt(s, j);
    if (del[p] === 0) {
      del[p] = 1;
      nDel++;
    }
  }
  return nDel;
}

function keptPositions(del: Uint8Array, nKeep: number): Int32Array {
  const keep = new Int32Array(nKeep);
  let k = 0;
  for (let i = 0; i < del.length; i++) if (del[i] === 0) keep[k++] = i;
  return keep;
}

function noopDelete(shape: readonly number[]): DeletePlan {
  const plan = mkPlan(0, 0, 1, null, 0, shapeCopy(shape));
  plan.noop = true;
  return plan;
}

/** Plan `A(subs{:}) = []`. Linear deletion keeps a vector's orientation
 *  and turns anything else into a row; subscript deletion removes
 *  slices along the single non-colon subscript. */
export function planDelete(
  shape: readonly number[],
  numel: number,
  subs: readonly RuntimeValue[]
): DeletePlan {
  const k = subs.length;
  if (k === 1) {
    const N = numel;
    if (subs[0] === COLON_INDEX) return mkPlan(0, 0, 1, null, 0, [0, 0]);
    const s = resolveSub(subs[0], N, 0);
    if (s.max > N) throw new RuntimeError(DELETE_OOB);
    const del = new Uint8Array(N);
    const nDel = markPositions(s, del);
    const isCol = isColumnShape(shape);
    const isRow = shape.length === 2 && shape[0] === 1;
    // MATLAB reshapes a matrix to a row even for an empty numeric index;
    // an all-false mask leaves it alone.
    if (nDel === 0 && (isCol || isRow || s.logical || N === 0))
      return noopDelete(shape);
    const nKeep = N - nDel;
    return mkPlan(
      nKeep,
      0,
      1,
      keptPositions(del, nKeep),
      0,
      isCol ? [nKeep, 1] : [1, nKeep]
    );
  }
  if (k === 0) throw new RuntimeError("Deletion requires at least one index");

  const dims = effectiveDims(shape, k);
  let target = -1;
  let nonColon = 0;
  for (let d = 0; d < k; d++) {
    if (subs[d] !== COLON_INDEX) {
      nonColon++;
      if (target < 0) target = d;
    }
  }
  if (nonColon === 0) {
    const out = dims.slice();
    out[0] = 0;
    return mkPlan(0, 0, 1, null, 0, trimShape(out));
  }
  if (nonColon > 1) {
    // Deleting nothing is harmless even with several subscripts.
    for (let d = 0; d < k; d++) {
      if (subs[d] !== COLON_INDEX) {
        const s = resolveSub(subs[d], dims[d], d + 1);
        if (s.count === 0) return noopDelete(shape);
      }
    }
    throw new RuntimeError(
      "A null assignment can have only one non-colon index."
    );
  }
  const s = resolveSub(subs[target], dims[target], target + 1);
  if (s.max > dims[target]) throw new RuntimeError(DELETE_OOB);
  const del = new Uint8Array(dims[target]);
  const nDel = markPositions(s, del);
  if (nDel === 0) return noopDelete(shape);
  const nKeep = dims[target] - nDel;
  const ss = new Array<Sub>(k);
  for (let d = 0; d < k; d++) ss[d] = rangeSub(0, dims[d], true);
  ss[target] = {
    pos: keptPositions(del, nKeep),
    bias: 0,
    start: 0,
    count: nKeep,
    max: 0,
    colon: false,
    logical: false,
    ishape: SCALAR_SHAPE,
  };
  if (nKeep > 0) ss[target].start = ss[target].pos![0];
  const out = dims.slice();
  out[target] = nKeep;
  return selPlan(buildSelection(dims, ss), trimShape(out));
}

// ── Resize ───────────────────────────────────────────────────────────────

/** True when growing `oldShape` to `newShape` keeps every old element at
 *  its linear offset (e.g. appending to a vector or adding columns). */
export function growsInPlace(
  oldShape: readonly number[],
  newShape: readonly number[]
): boolean {
  const nd = Math.max(oldShape.length, newShape.length);
  let d = 0;
  while (d < nd && (oldShape[d] ?? 1) === (newShape[d] ?? 1)) d++;
  for (d++; d < nd; d++) if ((oldShape[d] ?? 1) !== 1) return false;
  return true;
}

/** Copy plan for growing column-major storage from `oldShape` to
 *  `newShape` (every dimension at least as large): calls
 *  `fn(srcOffset, dstOffset, length)` for each contiguous run of old
 *  elements. Leading dimensions that do not change are merged into a
 *  single run, so appending to a vector or adding columns is one copy. */
export function forEachGrowRun(
  oldShape: readonly number[],
  newShape: readonly number[],
  fn: (src: number, dst: number, len: number) => void
): void {
  const nd = Math.max(oldShape.length, newShape.length);
  const o: number[] = [];
  const w: number[] = [];
  for (let d = 0; d < nd; d++) {
    o.push(d < oldShape.length ? oldShape[d] | 0 : 1);
    w.push(d < newShape.length ? newShape[d] | 0 : 1);
  }
  const total = numelOf(o);
  if (total === 0) return;
  // Run = dims 0..m where m is the first dimension that changes.
  let m = 0;
  while (m < nd - 1 && o[m] === w[m]) m++;
  let runLen = 1;
  for (let d = 0; d <= m; d++) runLen *= o[d];
  const nRuns = total / runLen;
  if (nRuns === 1) {
    fn(0, 0, runLen);
    return;
  }
  const strides: number[] = [];
  const idx: number[] = [];
  let st = 1;
  for (let d = 0; d < nd; d++) {
    strides.push(st);
    idx.push(0);
    st *= w[d];
  }
  let dst = 0;
  for (let r = 0; r < nRuns; r++) {
    fn(r * runLen, dst, runLen);
    for (let d = m + 1; d < nd; d++) {
      idx[d]++;
      dst += strides[d];
      if (idx[d] < o[d]) break;
      dst -= idx[d] * strides[d];
      idx[d] = 0;
    }
  }
}
