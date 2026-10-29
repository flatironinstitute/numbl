/**
 * Indexing: per-container gather / scatter / resize on top of the
 * storage-independent planner in `indexPlan.ts`.
 *
 * Every container uses the planner for read, store (with growth) and
 * delete: tensors (real, complex, logical), scalars, char arrays, cell
 * arrays, struct arrays, string arrays, class-instance arrays and sparse
 * matrices.
 */

import {
  type RuntimeValue,
  RuntimeTensor,
  RuntimeCell,
  type RuntimeSparseMatrix,
  type RuntimeComplexNumber,
  type RuntimeStruct,
  type RuntimeStructArray,
  type RuntimeClassInstance,
  RuntimeClassInstanceArray,
  RuntimeChar,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeSparseMatrix,
  isRuntimeStructArray,
  isRuntimeStruct,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeChar,
  isRuntimeStringArray,
  type RuntimeStringArray,
  stringArrayValue,
  structArrayValue,
} from "./types.js";
import { assertNever, kstr, valueClass } from "./kinds.js";
import { RuntimeError } from "./error.js";
import { RTV } from "./constructors.js";
import { type RefcountRuntime, isShared } from "./refcount.js";
import { allocFloat64Array } from "./alloc.js";
import { matlabNumToString } from "./utils.js";
import {
  COLON_INDEX,
  type Selection,
  planIndex,
  planStore,
  planDelete,
  planIndexDims,
  planStoreDims,
  planDeleteDims,
  offsetAt,
  forEachGrowRun,
  growsInPlace,
} from "./indexPlan.js";

export { COLON_INDEX };

/** Runtime surface needed by index-store mutations. The full Runtime
 *  class satisfies this structurally. */
type StoreRuntime = RefcountRuntime;

// ── Shared helpers ───────────────────────────────────────────────────────

/** Shape of a char array as seen by indexing. */
function charShape(c: RuntimeChar): readonly number[] {
  return c.shape ?? (c.value.length === 0 ? [0, 0] : [1, c.value.length]);
}

/** Convert a char value to a numeric (column-major) tensor of its code
 *  points. A RuntimeChar stores its `value` row-major (each of `shape[0]`
 *  rows is `shape[1]` chars, concatenated), so we transpose into the
 *  tensor's column-major layout. Used so indexed assignment into / from a
 *  char array runs on the numeric tensor machinery (MATLAB treats char as
 *  numeric for indexed assignment). */
export function charToNumericTensor(c: RuntimeChar): RuntimeTensor {
  const [rows, cols] = charShape(c);
  const data = allocFloat64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < cols; col++) {
      data[col * rows + r] = c.value.charCodeAt(r * cols + col);
    }
  }
  return RTV.tensor(data, [rows, cols]);
}

/** Inverse of charToNumericTensor: read a numeric tensor's column-major
 *  data into a row-major char value, rounding codes to integers. */
export function numericTensorToChar(t: RuntimeTensor): RuntimeChar {
  const rows = t.shape.length >= 2 ? (t.shape[0] ?? 0) : 1;
  const cols = t.shape.length >= 2 ? (t.shape[1] ?? 0) : t.data.length;
  let s = "";
  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < cols; col++) {
      s += String.fromCharCode(Math.round(t.data[col * rows + r]));
    }
  }
  if (rows === 1 && cols > 0) return RTV.char(s);
  if (rows === 0 && cols === 0) return RTV.char("");
  return new RuntimeChar(s, [rows, cols]);
}

// ── Gather / scatter primitives ──────────────────────────────────────────

const SCALAR_SHAPE: readonly number[] = [1, 1];

/** Shape of a tensor or cell as seen by indexing (at least 2-D). */
function arrayShape(shape: number[]): readonly number[] {
  if (shape.length >= 2) return shape;
  return shape.length === 1 ? [1, shape[0]] : SCALAR_SHAPE;
}

function isScalarShape(shape: readonly number[]): boolean {
  return shape.length === 2 && shape[0] === 1 && shape[1] === 1;
}

/** out[k] = src[offset k]. */
function gatherNumbers(
  src: Float64Array,
  sel: Selection,
  out: Float64Array
): void {
  const n = sel.n;
  if (sel.offsets === null) {
    const start = sel.start;
    const step = sel.step;
    if (step === 1 && n > 16) {
      out.set(src.subarray(start, start + n));
    } else {
      for (let k = 0, o = start; k < n; k++, o += step) out[k] = src[o];
    }
  } else {
    const off = sel.offsets;
    const b = sel.bias;
    for (let k = 0; k < n; k++) out[k] = src[off[k] - b];
  }
}

/** dst[offset k] = v. */
function fillNumbers(dst: Float64Array, sel: Selection, v: number): void {
  const n = sel.n;
  if (n === 1) {
    dst[offsetAt(sel, 0)] = v;
    return;
  }
  if (sel.offsets === null) {
    const step = sel.step;
    if (step === 1) dst.fill(v, sel.start, sel.start + n);
    else for (let k = 0, o = sel.start; k < n; k++, o += step) dst[o] = v;
  } else {
    const off = sel.offsets;
    const b = sel.bias;
    for (let k = 0; k < n; k++) dst[off[k] - b] = v;
  }
}

/** dst[offset k] = src[k]. */
function scatterNumbers(
  dst: Float64Array,
  sel: Selection,
  src: Float64Array
): void {
  const n = sel.n;
  if (sel.offsets === null) {
    const step = sel.step;
    if (step === 1 && n > 16) dst.set(src.subarray(0, n), sel.start);
    else for (let k = 0, o = sel.start; k < n; k++, o += step) dst[o] = src[k];
  } else {
    const off = sel.offsets;
    const b = sel.bias;
    for (let k = 0; k < n; k++) dst[off[k] - b] = src[k];
  }
}

/** The selected elements of a generic element array. */
function gatherElements<T>(data: readonly T[], sel: Selection): T[] {
  const n = sel.n;
  const out = new Array<T>(n);
  if (sel.offsets === null) {
    const step = sel.step;
    for (let k = 0, o = sel.start; k < n; k++, o += step) out[k] = data[o];
  } else {
    const off = sel.offsets;
    const b = sel.bias;
    for (let k = 0; k < n; k++) out[k] = data[off[k] - b];
  }
  return out;
}

function hasNonZero(a: Float64Array): boolean {
  for (let i = 0; i < a.length; i++) if (a[i] !== 0) return true;
  return false;
}

/** A 0x0 RHS (`[]` or `''`) turns an indexed assignment into deletion. */
function isDeletionRhs(rhs: RuntimeValue): boolean {
  if (isRuntimeTensor(rhs)) {
    const s = rhs.shape;
    return rhs.data.length === 0 && s.length === 2 && s[0] === 0 && s[1] === 0;
  }
  if (isRuntimeChar(rhs)) {
    const s = charShape(rhs);
    return s[0] === 0 && s[1] === 0;
  }
  return false;
}

// ── Tensors and numeric scalars ──────────────────────────────────────────

/** The selected elements of a tensor. A 1x1 result is returned as a
 *  scalar; an all-zero imaginary part is dropped (MATLAB). */
function gatherTensor(
  base: RuntimeTensor,
  sel: Selection,
  shape: number[]
): RuntimeValue {
  if (sel.n === 1 && isScalarShape(shape)) {
    const o = offsetAt(sel, 0);
    if (base.imag !== undefined && base.imag[o] !== 0)
      return RTV.complex(base.data[o], base.imag[o]);
    if (base.cls === "logical") return base.data[o] !== 0;
    return base.data[o];
  }
  const data = allocFloat64Array(sel.n);
  gatherNumbers(base.data, sel, data);
  let imag: Float64Array | undefined;
  if (base.imag !== undefined) {
    imag = allocFloat64Array(sel.n);
    gatherNumbers(base.imag, sel, imag);
    if (!hasNonZero(imag)) imag = undefined;
  }
  return new RuntimeTensor(data, shape, imag, base.cls);
}

function indexIntoTensor(
  base: RuntimeTensor,
  indices: RuntimeValue[]
): RuntimeValue {
  const plan = planIndex(arrayShape(base.shape), base.data.length, indices);
  return gatherTensor(base, plan, plan.shape);
}

/** Index a number, logical or complex scalar (a 1x1 array). */
function indexIntoScalar(
  base: number | boolean | RuntimeComplexNumber,
  indices: RuntimeValue[]
): RuntimeValue {
  const plan = planIndex(SCALAR_SHAPE, 1, indices);
  if (plan.n === 1 && isScalarShape(plan.shape)) return base;
  const n = plan.n;
  const data = allocFloat64Array(n);
  let imag: Float64Array | undefined;
  if (typeof base === "number") data.fill(base);
  else if (typeof base === "boolean") data.fill(base ? 1 : 0);
  else {
    data.fill(base.re);
    if (base.im !== 0) {
      imag = allocFloat64Array(n);
      imag.fill(base.im);
    }
  }
  return new RuntimeTensor(
    data,
    plan.shape,
    imag,
    typeof base === "boolean" ? "logical" : "double"
  );
}

/** A tensor of `newShape` holding `base`'s elements at their subscripts,
 *  zero elsewhere. */
function growTensor(
  base: RuntimeTensor,
  oldShape: readonly number[],
  newShape: number[]
): RuntimeTensor {
  let total = 1;
  for (const d of newShape) total *= d;
  const data = allocFloat64Array(total);
  const src = base.data;
  const srcIm = base.imag;
  const imag = srcIm ? allocFloat64Array(total) : undefined;
  if (growsInPlace(oldShape, newShape)) {
    data.set(src);
    if (imag) imag.set(srcIm!);
  } else {
    forEachGrowRun(oldShape, newShape, (s, d, len) => {
      data.set(src.subarray(s, s + len), d);
      if (imag) imag.set(srcIm!.subarray(s, s + len), d);
    });
  }
  return new RuntimeTensor(data, newShape, imag, base.cls);
}

/** Value assigned into a logical array: must be real and not NaN. */
function toLogicalValue(re: number, im: number): number {
  if (im !== 0)
    throw new RuntimeError("Complex values cannot be converted to logicals.");
  if (re !== re)
    throw new RuntimeError("NaN values cannot be converted to logicals.");
  return re !== 0 ? 1 : 0;
}

/** base(indices) = rhs for a numeric or logical tensor. Mutates `base`
 *  in place unless it is shared or grows. A logical target stays logical
 *  (MATLAB converts the assigned values). */
function storeIntoTensor(
  base: RuntimeTensor,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeValue {
  let rhsT: RuntimeTensor | null = null;
  let re = 0;
  let im = 0;
  if (typeof rhs === "number") re = rhs;
  else if (typeof rhs === "boolean") re = rhs ? 1 : 0;
  else if (isRuntimeComplexNumber(rhs)) {
    re = rhs.re;
    im = rhs.im;
  } else if (isRuntimeTensor(rhs)) {
    if (rhs.data.length === 1) {
      re = rhs.data[0];
      im = rhs.imag ? rhs.imag[0] : 0;
    } else {
      rhsT = rhs;
    }
  } else {
    throw new RuntimeError(
      `Conversion to ${valueClass(base)} from ${valueClass(rhs)} is not possible.`
    );
  }

  const shape = arrayShape(base.shape);
  const plan = rhsT
    ? planStore(
        shape,
        base.data.length,
        indices,
        rhsT.data.length,
        arrayShape(rhsT.shape)
      )
    : planStore(shape, base.data.length, indices, 1, SCALAR_SHAPE);

  const logical = base.cls === "logical";
  let rhsRe: Float64Array | null = rhsT ? rhsT.data : null;
  let rhsIm: Float64Array | undefined = rhsT ? rhsT.imag : undefined;
  if (logical) {
    if (rhsT) {
      if (rhsT.cls !== "logical") {
        const conv = allocFloat64Array(rhsT.data.length);
        for (let i = 0; i < conv.length; i++)
          conv[i] = toLogicalValue(rhsT.data[i], rhsIm ? rhsIm[i] : 0);
        rhsRe = conv;
      }
      rhsIm = undefined;
    } else {
      re = toLogicalValue(re, im);
      im = 0;
    }
  }

  let target: RuntimeTensor;
  if (plan.grow) target = growTensor(base, shape, plan.grow);
  else if (isShared(base)) target = base.shallowClone();
  else target = base;
  if (plan.n === 0) return target;

  // x(p) = x: the RHS aliases the storage being written.
  if (rhsRe === target.data) rhsRe = allocFloat64Array(rhsRe);
  if (rhsIm !== undefined && rhsIm === target.imag)
    rhsIm = allocFloat64Array(rhsIm);

  const needImag = rhsT ? rhsIm !== undefined : im !== 0;
  if (needImag && target.imag === undefined)
    target.imag = allocFloat64Array(target.data.length);
  if (rhsRe === null) {
    fillNumbers(target.data, plan, re);
    if (target.imag) fillNumbers(target.imag, plan, im);
  } else {
    scatterNumbers(target.data, plan, rhsRe);
    if (target.imag) {
      if (rhsIm) scatterNumbers(target.imag, plan, rhsIm);
      else fillNumbers(target.imag, plan, 0);
    }
  }
  return target;
}

/** base(indices) = [] for a tensor. */
function deleteFromTensor(
  base: RuntimeTensor,
  indices: RuntimeValue[]
): RuntimeValue {
  const plan = planDelete(arrayShape(base.shape), base.data.length, indices);
  if (plan.noop) return base;
  const data = allocFloat64Array(plan.n);
  gatherNumbers(base.data, plan, data);
  let imag: Float64Array | undefined;
  if (base.imag !== undefined) {
    imag = allocFloat64Array(plan.n);
    gatherNumbers(base.imag, plan, imag);
    if (!hasNonZero(imag)) imag = undefined;
  }
  return new RuntimeTensor(data, plan.shape, imag, base.cls);
}

// ── Char arrays ──────────────────────────────────────────────────────────

function codesToString(codes: number[]): string {
  let s = "";
  for (let i = 0; i < codes.length; i += 8192)
    s += String.fromCharCode.apply(null, codes.slice(i, i + 8192));
  return s;
}

/** Char result from selected elements. RuntimeChar stores rows
 *  concatenated (row-major), so offsets map through (row, col). */
function gatherChar(
  base: RuntimeChar,
  sel: Selection,
  shape: number[]
): RuntimeValue {
  if (shape.length > 2)
    throw new RuntimeError("N-dimensional char arrays are not supported");
  const v = base.value;
  const n = sel.n;
  const R = shape[0];
  const C = shape[1];
  const rows = base.shape ? base.shape[0] : v.length === 0 ? 0 : 1;
  const cols = base.shape ? base.shape[1] : v.length;
  if (n === 1 && R === 1 && C === 1) {
    const o = offsetAt(sel, 0);
    if (rows === 1) return RTV.char(v[o]);
    const r = o % rows;
    return RTV.char(v[r * cols + (o - r) / rows]);
  }
  if (n === 0) {
    if (R === 0 && C === 0) return RTV.char("");
    return new RuntimeChar("", [R, C]);
  }
  let colMajor: string;
  if (rows === 1 && sel.offsets === null && sel.step === 1) {
    colMajor = v.substring(sel.start, sel.start + n);
  } else {
    const codes = new Array<number>(n);
    for (let k = 0; k < n; k++) {
      const o = offsetAt(sel, k);
      const r = o % rows;
      codes[k] = v.charCodeAt(r * cols + (o - r) / rows);
    }
    colMajor = codesToString(codes);
  }
  if (R === 1) return RTV.char(colMajor);
  if (C === 1) return new RuntimeChar(colMajor, [R, 1]);
  const codes = new Array<number>(n);
  for (let r = 0; r < R; r++)
    for (let c = 0; c < C; c++)
      codes[r * C + c] = colMajor.charCodeAt(c * R + r);
  return new RuntimeChar(codesToString(codes), [R, C]);
}

function indexIntoChar(
  base: RuntimeChar,
  indices: RuntimeValue[]
): RuntimeValue {
  const plan = planIndex(charShape(base), base.value.length, indices);
  return gatherChar(base, plan, plan.shape);
}

// ── Cell arrays ──────────────────────────────────────────────────────────

function indexIntoCell(
  base: RuntimeCell,
  indices: RuntimeValue[]
): RuntimeValue {
  const plan = planIndex(arrayShape(base.shape), base.data.length, indices);
  return new RuntimeCell(gatherElements(base.data, plan), plan.shape);
}

/** The elements selected by c{indices} (a comma-separated list). */
export function cellBraceElements(
  base: RuntimeCell,
  indices: RuntimeValue[]
): RuntimeValue[] {
  return gatherElements(
    base.data,
    planIndex(arrayShape(base.shape), base.data.length, indices)
  );
}

/** Replace cell.data[pos] with value, decref-old / incref-new; without a
 *  runtime, refcount work is skipped (lax mode). */
function setCellElement(
  cell: RuntimeCell,
  pos: number,
  value: RuntimeValue,
  rt?: RefcountRuntime
): void {
  if (rt) cell.bindElement(rt, pos, value);
  else cell.data[pos] = value;
}

/** Resize a cell in place to `newShape`, padding with `[]`. Existing
 *  elements move without refcount changes. */
function growCell(
  cell: RuntimeCell,
  oldShape: readonly number[],
  newShape: number[],
  rt?: RefcountRuntime
): void {
  let total = 1;
  for (const d of newShape) total *= d;
  const empty = RTV.tensor(allocFloat64Array(0), [0, 0]);
  const data = new Array<RuntimeValue>(total).fill(empty);
  const src = cell.data;
  if (growsInPlace(oldShape, newShape)) {
    for (let j = 0; j < src.length; j++) data[j] = src[j];
  } else {
    forEachGrowRun(oldShape, newShape, (s, d, len) => {
      for (let j = 0; j < len; j++) data[d + j] = src[s + j];
    });
  }
  if (rt) for (let i = src.length; i < total; i++) empty.incref();
  cell.data = data;
  cell.shape = newShape;
}

/** c(indices) = rhs (paren, `rhs` must be a cell) or c{indices} = rhs
 *  (brace, exactly one element). */
function storeIntoCell(
  base: RuntimeCell,
  indices: RuntimeValue[],
  rhs: RuntimeValue,
  brace: boolean,
  rt?: StoreRuntime
): RuntimeValue {
  let single: RuntimeValue = rhs;
  let elems: RuntimeValue[] | null = null;
  const shape = arrayShape(base.shape);
  let plan;
  if (brace) {
    plan = planStore(shape, base.data.length, indices, 1, SCALAR_SHAPE);
    if (plan.n !== 1) {
      if (plan.n === 0 && !plan.grow) return base;
      throw new RuntimeError(
        `Assigning to ${plan.n} elements using a simple assignment statement is not supported. Consider using comma-separated list assignment.`
      );
    }
  } else {
    if (!isRuntimeCell(rhs))
      throw new RuntimeError(
        `Conversion to cell from ${valueClass(rhs)} is not possible.`
      );
    if (rhs.data.length === 1) {
      single = rhs.data[0];
      plan = planStore(shape, base.data.length, indices, 1, SCALAR_SHAPE);
    } else {
      elems = rhs === base ? rhs.data.slice() : rhs.data;
      plan = planStore(
        shape,
        base.data.length,
        indices,
        elems.length,
        arrayShape(rhs.shape)
      );
    }
  }
  if (isShared(base)) base = base.shallowClone();
  if (plan.grow) growCell(base, shape, plan.grow, rt);
  for (let k = 0; k < plan.n; k++)
    setCellElement(base, offsetAt(plan, k), elems ? elems[k] : single, rt);
  return base;
}

/** c(indices) = [] */
function deleteFromCell(
  base: RuntimeCell,
  indices: RuntimeValue[]
): RuntimeValue {
  const plan = planDelete(arrayShape(base.shape), base.data.length, indices);
  if (plan.noop) return base;
  return RTV.cell(gatherElements(base.data, plan), plan.shape);
}

// ── Element-array helpers ────────────────────────────────────────────────

/** A shape for a container that is at most 2-D (string and object
 *  arrays, sparse matrices). */
function shape2d(shape: readonly number[], what: string): [number, number] {
  let len = shape.length;
  while (len > 2 && shape[len - 1] === 1) len--;
  if (len > 2)
    throw new RuntimeError(`N-dimensional ${what} are not supported`);
  return [shape[0], shape[1]];
}

function numelOf(shape: readonly number[]): number {
  let n = 1;
  for (let i = 0; i < shape.length; i++) n *= shape[i];
  return n;
}

/** `src` (laid out as `oldShape`) moved into a new array of `newShape`,
 *  with `fill(i)` at the new positions. */
function growElements<T>(
  src: readonly T[],
  oldShape: readonly number[],
  newShape: readonly number[],
  fill: () => T
): T[] {
  const total = numelOf(newShape);
  const out = new Array<T>(total);
  if (growsInPlace(oldShape, newShape)) {
    for (let j = 0; j < src.length; j++) out[j] = src[j];
    for (let j = src.length; j < total; j++) out[j] = fill();
    return out;
  }
  const set = new Uint8Array(total);
  forEachGrowRun(oldShape, newShape, (s, d, len) => {
    for (let j = 0; j < len; j++) {
      out[d + j] = src[s + j];
      set[d + j] = 1;
    }
  });
  for (let j = 0; j < total; j++) if (set[j] === 0) out[j] = fill();
  return out;
}

/** dst[offset k] = src[k] (or `single` for every k when src is null). */
function scatterElements<T>(
  dst: T[],
  sel: Selection,
  src: readonly T[] | null,
  single: T
): void {
  for (let k = 0; k < sel.n; k++)
    dst[offsetAt(sel, k)] = src === null ? single : src[k];
}

// ── String arrays ────────────────────────────────────────────────────────

function stringResult(data: string[], shape: readonly number[]): RuntimeValue {
  return stringArrayValue(data, shape2d(shape, "string arrays"));
}

function indexIntoStringArray(
  base: string | RuntimeStringArray,
  indices: RuntimeValue[]
): RuntimeValue {
  if (typeof base === "string") {
    const plan = planIndex(SCALAR_SHAPE, 1, indices);
    if (plan.n === 1 && isScalarShape(plan.shape)) return base;
    return stringResult(new Array<string>(plan.n).fill(base), plan.shape);
  }
  const plan = planIndex(base.shape, base.data.length, indices);
  if (plan.n === 1 && isScalarShape(plan.shape))
    return base.data[offsetAt(plan, 0)];
  return stringResult(gatherElements(base.data, plan), plan.shape);
}

/** The elements selected by str{indices}, as char vectors. */
export function stringBraceElements(
  base: string | RuntimeStringArray,
  indices: RuntimeValue[]
): RuntimeValue[] {
  const data = typeof base === "string" ? [base] : base.data;
  const shape = typeof base === "string" ? SCALAR_SHAPE : base.shape;
  return gatherElements(data, planIndex(shape, data.length, indices)).map(s =>
    RTV.char(s)
  );
}

/** The string elements (column-major) and shape of an assigned value. */
function rhsStrings(rhs: RuntimeValue): {
  data: string[];
  shape: readonly number[];
} {
  if (typeof rhs === "string") return { data: [rhs], shape: SCALAR_SHAPE };
  if (isRuntimeStringArray(rhs)) return { data: rhs.data, shape: rhs.shape };
  if (isRuntimeChar(rhs)) {
    // Each row of a char matrix is one string.
    const [rows, cols] = charShape(rhs);
    if (rows <= 1) return { data: [rhs.value], shape: SCALAR_SHAPE };
    const out: string[] = [];
    for (let r = 0; r < rows; r++)
      out.push(rhs.value.slice(r * cols, (r + 1) * cols));
    return { data: out, shape: [rows, 1] };
  }
  if (typeof rhs === "number")
    return { data: [matlabNumToString(rhs)], shape: SCALAR_SHAPE };
  if (typeof rhs === "boolean")
    return { data: [rhs ? "true" : "false"], shape: SCALAR_SHAPE };
  if (isRuntimeTensor(rhs))
    return {
      data: Array.from(rhs.data, d => matlabNumToString(d)),
      shape: arrayShape(rhs.shape),
    };
  throw new RuntimeError(
    `Conversion to string from ${valueClass(rhs)} is not possible.`
  );
}

/** base(indices) = rhs where base is a string, a string array, or an
 *  empty numeric array becoming one. Growth pads with "" (numbl has no
 *  <missing>). */
function storeIntoStringArray(
  base: RuntimeValue,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeValue {
  let data: string[];
  let shape: readonly number[];
  if (typeof base === "string") {
    data = [base];
    shape = SCALAR_SHAPE;
  } else if (isRuntimeStringArray(base)) {
    data = base.data;
    shape = base.shape;
  } else {
    data = [];
    shape = isRuntimeTensor(base) ? arrayShape(base.shape) : [0, 0];
  }
  const r = rhsStrings(rhs);
  const plan =
    r.data.length === 1
      ? planStore(shape, data.length, indices, 1, SCALAR_SHAPE)
      : planStore(shape, data.length, indices, r.data.length, r.shape);
  const src = r.data.length === 1 ? null : r.data.slice();
  if (
    isRuntimeStringArray(base) &&
    !plan.grow &&
    !isShared(base) &&
    data.length !== 1
  ) {
    scatterElements(base.data, plan, src, r.data[0]);
    return base;
  }
  const out = plan.grow
    ? growElements(data, shape, plan.grow, () => "")
    : data.slice();
  scatterElements(out, plan, src, r.data[0]);
  return stringResult(out, plan.grow ?? shape);
}

function deleteFromStringArray(
  base: string | RuntimeStringArray,
  indices: RuntimeValue[]
): RuntimeValue {
  const data = typeof base === "string" ? [base] : base.data;
  const shape = typeof base === "string" ? SCALAR_SHAPE : base.shape;
  const plan = planDelete(shape, data.length, indices);
  if (plan.noop) return base;
  return stringResult(gatherElements(data, plan), plan.shape);
}

// ── Struct arrays ────────────────────────────────────────────────────────

const DISSIMILAR = "Subscripted assignment between dissimilar structures.";

function emptyValue(): RuntimeValue {
  return RTV.tensor(allocFloat64Array(0), [0, 0]);
}

/** True when `s` has exactly the fields `names`, in that order. */
export function hasFieldOrder(
  s: RuntimeStruct,
  names: readonly string[]
): boolean {
  if (s.fields.size !== names.length) return false;
  let i = 0;
  for (const k of s.fields.keys()) if (k !== names[i++]) return false;
  return true;
}

/** `s` with exactly the fields `names`, in that order; missing fields are
 *  `[]`. Returns `s` itself when it already conforms. */
function conformStruct(
  s: RuntimeStruct,
  names: readonly string[]
): RuntimeStruct {
  if (hasFieldOrder(s, names)) return s;
  const fields = new Map<string, RuntimeValue>();
  for (const f of names) fields.set(f, s.fields.get(f) ?? emptyValue());
  return RTV.struct(fields);
}

function emptyStruct(names: readonly string[]): RuntimeStruct {
  const fields = new Map<string, RuntimeValue>();
  for (const f of names) fields.set(f, emptyValue());
  return RTV.struct(fields);
}

/** A scalar struct seen as a 1x1 struct array. */
function structParts(base: RuntimeStruct | RuntimeStructArray): {
  names: string[];
  elems: RuntimeStruct[];
  shape: readonly number[];
} {
  return isRuntimeStruct(base)
    ? { names: [...base.fields.keys()], elems: [base], shape: SCALAR_SHAPE }
    : { names: base.fieldNames, elems: base.elements, shape: base.shape };
}

function indexIntoStructArray(
  base: RuntimeStruct | RuntimeStructArray,
  indices: RuntimeValue[]
): RuntimeValue {
  const { names, elems, shape } = structParts(base);
  const plan = planIndex(shape, elems.length, indices);
  if (plan.n === 1 && isScalarShape(plan.shape))
    return elems[offsetAt(plan, 0)];
  return RTV.structArray(
    names.slice(),
    gatherElements(elems, plan),
    plan.shape
  );
}

/** base(indices) = rhs for a struct target (a struct, a struct array, or
 *  an empty numeric array becoming one). With `sameFields`, the RHS must
 *  have the target's fields (MATLAB `s(k) = t`); otherwise new RHS fields
 *  are added to every element (the store-back of `s(k).f = v`). Growth
 *  pads with structs whose fields are `[]`. */
function storeIntoStructArray(
  base: RuntimeValue,
  indices: RuntimeValue[],
  rhs: RuntimeStruct | RuntimeStructArray,
  sameFields: boolean,
  rt?: StoreRuntime
): RuntimeValue {
  let names: string[];
  let elems: RuntimeStruct[];
  let shape: readonly number[];
  if (isRuntimeStruct(base) || isRuntimeStructArray(base)) {
    ({ names, elems, shape } = structParts(base));
  } else {
    names = [];
    elems = [];
    shape = isRuntimeTensor(base) ? arrayShape(base.shape) : [0, 0];
  }
  const isStructTarget = isRuntimeStruct(base) || isRuntimeStructArray(base);
  const r = structParts(rhs);
  let allNames = names;
  if (isStructTarget && sameFields) {
    if (
      r.names.length !== names.length ||
      r.names.some(f => !names.includes(f))
    )
      throw new RuntimeError(DISSIMILAR);
  } else if (!isStructTarget) {
    allNames = r.names;
  } else {
    for (const f of r.names)
      if (!names.includes(f)) {
        if (allNames === names) allNames = names.slice();
        allNames.push(f);
      }
  }
  const multi = r.elems.length !== 1;
  const plan = multi
    ? planStore(shape, elems.length, indices, r.elems.length, r.shape)
    : planStore(shape, elems.length, indices, 1, SCALAR_SHAPE);
  const single = conformStruct(r.elems[0] ?? emptyStruct(allNames), allNames);
  const src = multi ? r.elems.map(e => conformStruct(e, allNames)) : null;
  const newShape = plan.grow ?? shape;
  if (numelOf(newShape) === 1 && plan.n === 1) return src ? src[0] : single;

  // Update in place when the target is an unshared struct array whose
  // fields and size do not change.
  if (
    isRuntimeStructArray(base) &&
    !isShared(base) &&
    !plan.grow &&
    allNames === names
  ) {
    for (let k = 0; k < plan.n; k++) {
      const o = offsetAt(plan, k);
      const v = src === null ? single : src[k];
      if (rt) base.bindElement(rt, o, v);
      else base.elements[o] = v;
    }
    return base;
  }
  let out =
    allNames === names
      ? elems.slice()
      : elems.map(e => conformStruct(e, allNames));
  if (plan.grow)
    out = growElements(out, shape, plan.grow, () => emptyStruct(allNames));
  scatterElements(out, plan, src, single);
  return structArrayValue(allNames.slice(), out, newShape.slice());
}

function deleteFromStructArray(
  base: RuntimeStruct | RuntimeStructArray,
  indices: RuntimeValue[]
): RuntimeValue {
  const { names, elems, shape } = structParts(base);
  const plan = planDelete(shape, elems.length, indices);
  if (plan.noop) return base;
  return structArrayValue(
    names.slice(),
    gatherElements(elems, plan),
    plan.shape
  );
}

// ── Class-instance arrays ────────────────────────────────────────────────

/** Object-array result; a single element is the instance itself. */
function objectResult(
  cls: string,
  elems: RuntimeClassInstance[],
  shape: readonly number[]
): RuntimeValue {
  if (elems.length === 1 && isScalarShape(shape)) return elems[0];
  return new RuntimeClassInstanceArray(
    cls,
    elems,
    shape2d(shape, "object arrays")
  );
}

function objectParts(base: RuntimeClassInstance | RuntimeClassInstanceArray): {
  elems: RuntimeClassInstance[];
  shape: readonly number[];
} {
  return isRuntimeClassInstance(base)
    ? { elems: [base], shape: SCALAR_SHAPE }
    : { elems: base.elements, shape: base.shape };
}

function indexIntoObjectArray(
  base: RuntimeClassInstance | RuntimeClassInstanceArray,
  indices: RuntimeValue[]
): RuntimeValue {
  const { elems, shape } = objectParts(base);
  const plan = planIndex(shape, elems.length, indices);
  if (plan.n === 1 && isScalarShape(plan.shape))
    return elems[offsetAt(plan, 0)];
  return objectResult(base.className, gatherElements(elems, plan), plan.shape);
}

/** base(indices) = rhs for an object target (an instance, an object
 *  array, or an empty numeric array becoming one). Growth pads with
 *  instances whose properties are `[]`. */
function storeIntoObjectArray(
  base: RuntimeValue,
  indices: RuntimeValue[],
  rhs: RuntimeClassInstance | RuntimeClassInstanceArray,
  rt?: StoreRuntime
): RuntimeValue {
  let elems: RuntimeClassInstance[];
  let shape: readonly number[];
  let cls: string;
  if (isRuntimeClassInstance(base) || isRuntimeClassInstanceArray(base)) {
    ({ elems, shape } = objectParts(base));
    cls = base.className;
  } else {
    elems = [];
    shape = isRuntimeTensor(base) ? arrayShape(base.shape) : [0, 0];
    cls = rhs.className;
  }
  const r = objectParts(rhs);
  const multi = r.elems.length !== 1;
  const plan = multi
    ? planStore(shape, elems.length, indices, r.elems.length, r.shape)
    : planStore(shape, elems.length, indices, 1, SCALAR_SHAPE);
  const single = r.elems[0];
  const src = multi ? r.elems.slice() : null;
  const newShape = plan.grow ?? shape;
  if (numelOf(newShape) === 1 && plan.n === 1) return src ? src[0] : single;
  if (isRuntimeClassInstanceArray(base) && !isShared(base) && !plan.grow) {
    for (let k = 0; k < plan.n; k++) {
      const o = offsetAt(plan, k);
      const v = src === null ? single : src[k];
      if (rt) base.bindElement(rt, o, v);
      else base.elements[o] = v;
    }
    return base;
  }
  const template = single ?? elems[0];
  const out = plan.grow
    ? growElements(elems, shape, plan.grow, () =>
        RTV.classInstance(cls, template ? [...template.fields.keys()] : [])
      )
    : elems.slice();
  scatterElements(out, plan, src, single);
  return objectResult(cls, out, newShape);
}

function deleteFromObjectArray(
  base: RuntimeClassInstance | RuntimeClassInstanceArray,
  indices: RuntimeValue[]
): RuntimeValue {
  const { elems, shape } = objectParts(base);
  const plan = planDelete(shape, elems.length, indices);
  if (plan.noop) return base;
  return objectResult(base.className, gatherElements(elems, plan), plan.shape);
}

// ── Sparse matrices ──────────────────────────────────────────────────────
//
// Sparse matrices keep CSC storage: reads and deletions with one subscript
// walk the plan's linear offsets, with two or more they use the per-
// dimension positions (`planIndexDims`/`planDeleteDims`), so selecting
// rows and columns of a large matrix never enumerates its dense size.
// Stores collect the assigned (row, column, value) triples and merge them
// into the columns. A read of one element returns a number (MATLAB returns
// a 1x1 sparse matrix).

/** Position of (row, col) in `ir`/`pr`, or -1 for a structural zero. */
function sparseFind(s: RuntimeSparseMatrix, row: number, col: number): number {
  let lo = s.jc[col];
  let hi = s.jc[col + 1] - 1;
  const ir = s.ir;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const r = ir[mid];
    if (r === row) return mid;
    if (r < row) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

function sparseScalar(s: RuntimeSparseMatrix, k: number): RuntimeValue {
  if (k < 0) return 0;
  if (s.pi !== undefined && s.pi[k] !== 0) return RTV.complex(s.pr[k], s.pi[k]);
  return s.pr[k];
}

/** Growable CSC output, filled column by column. */
class CscBuilder {
  ir: number[] = [];
  pr: number[] = [];
  pi: number[] | null;
  jc: Int32Array;
  constructor(
    readonly m: number,
    readonly n: number,
    complex: boolean
  ) {
    this.pi = complex ? [] : null;
    this.jc = new Int32Array(n + 1);
  }
  push(row: number, re: number, im: number): void {
    this.ir.push(row);
    this.pr.push(re);
    if (this.pi) this.pi.push(im);
  }
  /** Mark the start of columns up to and including `col`. */
  startColumn(col: number, from: number): void {
    for (let c = from; c <= col; c++) this.jc[c] = this.ir.length;
  }
  finish(): RuntimeSparseMatrix {
    this.jc[this.n] = this.ir.length;
    let pi: Float64Array | undefined;
    if (this.pi) {
      let any = false;
      for (const v of this.pi) if (v !== 0) any = true;
      if (any) pi = allocFloat64Array(this.pi);
    }
    return RTV.sparseMatrix(
      this.m,
      this.n,
      new Int32Array(this.ir),
      this.jc,
      allocFloat64Array(this.pr),
      pi
    );
  }
}

/** The elements at the selected linear offsets, as an R x C sparse
 *  matrix in column-major order. */
function gatherSparseLinear(
  s: RuntimeSparseMatrix,
  sel: Selection,
  R: number,
  C: number
): RuntimeSparseMatrix {
  const out = new CscBuilder(R, C, s.pi !== undefined);
  const m = s.m;
  let col = 0;
  const emit = (k: number, src: number) => {
    const c = R > 0 ? Math.floor(k / R) : 0;
    if (c > col) {
      out.startColumn(c, col + 1);
      col = c;
    }
    const im = s.pi !== undefined ? s.pi[src] : 0;
    if (s.pr[src] !== 0 || im !== 0) out.push(k - c * R, s.pr[src], im);
  };
  out.startColumn(0, 0);
  if (sel.offsets === null && sel.step === 1) {
    // A contiguous run: walk the stored entries in range.
    const lo = sel.start;
    const hi = sel.start + sel.n;
    if (sel.n > 0 && m > 0) {
      for (let c = Math.floor(lo / m); c < s.n && c * m < hi; c++) {
        for (let p = s.jc[c]; p < s.jc[c + 1]; p++) {
          const o = c * m + s.ir[p];
          if (o >= lo && o < hi) emit(o - lo, p);
        }
      }
    }
  } else {
    for (let k = 0; k < sel.n; k++) {
      const o = offsetAt(sel, k);
      const c = Math.floor(o / m);
      const p = sparseFind(s, o - c * m, c);
      if (p >= 0) emit(k, p);
    }
  }
  out.startColumn(C - 1, col + 1);
  return out.finish();
}

/** S(rows, cols) for 0-based position lists. */
function sparseSubmatrix(
  s: RuntimeSparseMatrix,
  rows: Int32Array,
  cols: Int32Array
): RuntimeSparseMatrix {
  const R = rows.length;
  const C = cols.length;
  const out = new CscBuilder(R, C, s.pi !== undefined);
  let identity = R === s.m;
  for (let i = 0; identity && i < R; i++) if (rows[i] !== i) identity = false;
  let increasing = true;
  for (let i = 1; increasing && i < R; i++)
    if (rows[i] <= rows[i - 1]) increasing = false;
  // Source row -> output rows (a linked list, in increasing order).
  let head: Int32Array | null = null;
  let next: Int32Array | null = null;
  if (!identity) {
    head = new Int32Array(s.m).fill(-1);
    next = new Int32Array(R);
    for (let i = R - 1; i >= 0; i--) {
      next[i] = head[rows[i]];
      head[rows[i]] = i;
    }
  }
  const tmp: { row: number; p: number }[] = [];
  for (let c = 0; c < C; c++) {
    out.jc[c] = out.ir.length;
    const sc = cols[c];
    for (let p = s.jc[sc]; p < s.jc[sc + 1]; p++) {
      const im = s.pi !== undefined ? s.pi[p] : 0;
      if (identity) {
        out.push(s.ir[p], s.pr[p], im);
      } else if (increasing) {
        const i = head![s.ir[p]];
        if (i >= 0) out.push(i, s.pr[p], im);
      } else {
        for (let i = head![s.ir[p]]; i >= 0; i = next![i])
          tmp.push({ row: i, p });
      }
    }
    if (tmp.length > 0) {
      tmp.sort((a, b) => a.row - b.row);
      for (const e of tmp)
        out.push(e.row, s.pr[e.p], s.pi !== undefined ? s.pi[e.p] : 0);
      tmp.length = 0;
    }
  }
  return out.finish();
}

/** Two or more subscripts on a (2-D) sparse matrix: the row and column
 *  positions, after checking that the remaining subscripts select only
 *  the first page. */
function sparseRowsCols(
  pos: Int32Array[],
  shape: readonly number[]
): [Int32Array, Int32Array] {
  for (let d = 2; d < pos.length; d++)
    if (pos[d].length !== 1 || (shape[d] ?? 1) !== 1)
      throw new RuntimeError("N-dimensional sparse arrays are not supported");
  return [pos[0], pos[1]];
}

function indexIntoSparse(
  base: RuntimeSparseMatrix,
  indices: RuntimeValue[]
): RuntimeValue {
  const shape = [base.m, base.n];
  if (indices.length === 1) {
    const plan = planIndex(shape, base.m * base.n, indices);
    if (plan.n === 1 && isScalarShape(plan.shape)) {
      const o = offsetAt(plan, 0);
      const c = Math.floor(o / base.m);
      return sparseScalar(base, sparseFind(base, o - c * base.m, c));
    }
    const [R, C] = shape2d(plan.shape, "sparse arrays");
    return gatherSparseLinear(base, plan, R, C);
  }
  if (indices.length === 0) return base;
  const plan = planIndexDims(shape, indices);
  const [rows, cols] = sparseRowsCols(plan.pos, plan.shape);
  if (rows.length === 1 && cols.length === 1)
    return sparseScalar(base, sparseFind(base, rows[0], cols[0]));
  return sparseSubmatrix(base, rows, cols);
}

function deleteFromSparse(
  base: RuntimeSparseMatrix,
  indices: RuntimeValue[]
): RuntimeValue {
  const shape = [base.m, base.n];
  if (indices.length === 1) {
    const plan = planDelete(shape, base.m * base.n, indices);
    if (plan.noop) return base;
    const [R, C] = shape2d(plan.shape, "sparse arrays");
    return gatherSparseLinear(base, plan, R, C);
  }
  const plan = planDeleteDims(shape, indices);
  if (plan.noop) return base;
  shape2d(plan.shape, "sparse arrays");
  const [rows, cols] = sparseRowsCols(plan.pos, shape);
  return sparseSubmatrix(base, rows, cols);
}

/** base(indices) = rhs for a sparse matrix; returns a new matrix. */
function storeIntoSparse(
  base: RuntimeSparseMatrix,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeSparseMatrix {
  let rRe: Float64Array;
  let rIm: Float64Array | undefined;
  let rShape: readonly number[] = SCALAR_SHAPE;
  if (typeof rhs === "number") rRe = new Float64Array([rhs]);
  else if (typeof rhs === "boolean") rRe = new Float64Array([rhs ? 1 : 0]);
  else if (isRuntimeComplexNumber(rhs)) {
    rRe = new Float64Array([rhs.re]);
    rIm = new Float64Array([rhs.im]);
  } else if (isRuntimeTensor(rhs)) {
    rRe = rhs.data;
    rIm = rhs.imag;
    rShape = arrayShape(rhs.shape);
  } else if (isRuntimeSparseMatrix(rhs)) {
    rRe = new Float64Array(rhs.m * rhs.n);
    if (rhs.pi) rIm = new Float64Array(rhs.m * rhs.n);
    for (let c = 0; c < rhs.n; c++)
      for (let p = rhs.jc[c]; p < rhs.jc[c + 1]; p++) {
        rRe[c * rhs.m + rhs.ir[p]] = rhs.pr[p];
        if (rIm) rIm[c * rhs.m + rhs.ir[p]] = rhs.pi![p];
      }
    rShape = [rhs.m, rhs.n];
  } else {
    throw new RuntimeError(
      `Conversion to sparse from ${valueClass(rhs)} is not possible.`
    );
  }
  const rN = rRe.length;
  const shape = [base.m, base.n];
  let m2: number;
  let n2: number;
  let upRow: Int32Array;
  let upCol: Int32Array;
  if (indices.length === 1) {
    const plan = planStore(shape, base.m * base.n, indices, rN, rShape);
    [m2, n2] = plan.grow
      ? shape2d(plan.grow, "sparse arrays")
      : [base.m, base.n];
    upRow = new Int32Array(plan.n);
    upCol = new Int32Array(plan.n);
    for (let k = 0; k < plan.n; k++) {
      const o = offsetAt(plan, k);
      const c = Math.floor(o / m2);
      upCol[k] = c;
      upRow[k] = o - c * m2;
    }
  } else {
    const plan = planStoreDims(shape, indices, rN, rShape);
    const [rows, cols] = sparseRowsCols(plan.pos, plan.shape);
    m2 = plan.shape[0];
    n2 = plan.shape[1];
    const u = rows.length * cols.length;
    upRow = new Int32Array(u);
    upCol = new Int32Array(u);
    let k = 0;
    for (let j = 0; j < cols.length; j++)
      for (let i = 0; i < rows.length; i++, k++) {
        upRow[k] = rows[i];
        upCol[k] = cols[j];
      }
  }
  return sparseScatter(base, m2, n2, upRow, upCol, rRe, rIm, rN === 1);
}

/** Merge assignments (row, col, value) into a copy of `s` resized to
 *  m2 x n2. Later assignments to the same element win; zeros remove
 *  stored entries. With `scalar`, every assignment uses value 0. */
function sparseScatter(
  s: RuntimeSparseMatrix,
  m2: number,
  n2: number,
  upRow: Int32Array,
  upCol: Int32Array,
  re: Float64Array,
  im: Float64Array | undefined,
  scalar: boolean
): RuntimeSparseMatrix {
  const u = upRow.length;
  const order = new Int32Array(u);
  let sorted = true;
  for (let k = 0; k < u; k++) {
    order[k] = k;
    if (
      k > 0 &&
      (upCol[k] < upCol[k - 1] ||
        (upCol[k] === upCol[k - 1] && upRow[k] <= upRow[k - 1]))
    )
      sorted = false;
  }
  if (!sorted)
    order.sort((a, b) => upCol[a] - upCol[b] || upRow[a] - upRow[b] || a - b);
  let complex = s.pi !== undefined;
  if (!complex && im)
    for (let k = 0; k < im.length; k++) if (im[k] !== 0) complex = true;
  const out = new CscBuilder(m2, n2, complex);
  let q = 0;
  for (let c = 0; c < n2; c++) {
    out.jc[c] = out.ir.length;
    let p = c < s.n ? s.jc[c] : 0;
    const pEnd = c < s.n ? s.jc[c + 1] : 0;
    for (;;) {
      // The next assignment in this column (the last of any duplicates).
      let k = -1;
      if (q < u && upCol[order[q]] === c) {
        while (
          q + 1 < u &&
          upCol[order[q + 1]] === c &&
          upRow[order[q + 1]] === upRow[order[q]]
        )
          q++;
        k = order[q];
      }
      if (k < 0 && p >= pEnd) break;
      if (k >= 0 && (p >= pEnd || upRow[k] <= s.ir[p])) {
        if (p < pEnd && upRow[k] === s.ir[p]) p++;
        const j = scalar ? 0 : k;
        const vIm = im ? im[j] : 0;
        if (re[j] !== 0 || vIm !== 0) out.push(upRow[k], re[j], vIm);
        q++;
      } else {
        out.push(s.ir[p], s.pr[p], s.pi ? s.pi[p] : 0);
        p++;
      }
    }
  }
  return out.finish();
}

// ── Index read: main dispatcher ──────────────────────────────────────────

/** Index into a value: v(i1, i2, ...) */
export function indexIntoRTValue(
  base: RuntimeValue,
  indices: RuntimeValue[]
): RuntimeValue {
  if (typeof base === "number" || typeof base === "boolean")
    return indexIntoScalar(base, indices);
  if (typeof base === "string") return indexIntoStringArray(base, indices);
  switch (base.kind) {
    case "complex_number":
      return indexIntoScalar(base, indices);
    case "tensor":
      return indexIntoTensor(base, indices);
    case "cell":
      return indexIntoCell(base, indices);
    case "char":
      return indexIntoChar(base, indices);
    case "struct":
    case "struct_array":
      return indexIntoStructArray(base, indices);
    case "string_array":
      return indexIntoStringArray(base, indices);
    case "class_instance":
    case "class_instance_array":
      return indexIntoObjectArray(base, indices);
    case "sparse_matrix":
      return indexIntoSparse(base, indices);
    case "function":
    case "dictionary":
    case "dummy_handle":
    case "graphics_handle":
      throw new RuntimeError(`Cannot index into ${kstr(base)}`);
    default:
      return assertNever(base);
  }
}

// ── Store: main dispatcher ───────────────────────────────────────────────

/** Store into indexed position: base(indices) = rhs (or base{indices} = rhs
 *  for cells when `parenAssign` is false). A 0x0 RHS in a paren store
 *  deletes. Shared targets are copied before mutation. `storeBack` marks
 *  the store-back of `s(k).f = v`, which may add fields to a struct
 *  array. */
export function storeIntoRTValueIndex(
  base: RuntimeValue,
  indices: RuntimeValue[],
  rhs: RuntimeValue,
  parenAssign = false,
  rt?: StoreRuntime,
  storeBack = false
): RuntimeValue {
  if (isRuntimeSparseMatrix(base)) {
    if (isDeletionRhs(rhs)) return deleteFromSparse(base, indices);
    return storeIntoSparse(base, indices, rhs);
  }

  // Auto-convert sparse RHS to dense when assigning into a dense tensor
  if (isRuntimeSparseMatrix(rhs) && isRuntimeTensor(base)) {
    const S = rhs;
    const data = allocFloat64Array(S.m * S.n);
    const imag = S.pi ? allocFloat64Array(S.m * S.n) : undefined;
    for (let col = 0; col < S.n; col++) {
      for (let k = S.jc[col]; k < S.jc[col + 1]; k++) {
        const idx = col * S.m + S.ir[k];
        data[idx] = S.pr[k];
        if (imag && S.pi) imag[idx] = S.pi[k];
      }
    }
    rhs = RTV.tensor(data, [S.m, S.n], imag);
  }

  if (isRuntimeTensor(base)) {
    if (isDeletionRhs(rhs)) return deleteFromTensor(base, indices);
    // An empty numeric target takes the class of a struct, object or
    // string RHS.
    if (base.data.length === 0) {
      if (isRuntimeStruct(rhs) || isRuntimeStructArray(rhs))
        return storeIntoStructArray(base, indices, rhs, false, rt);
      if (isRuntimeClassInstance(rhs) || isRuntimeClassInstanceArray(rhs))
        return storeIntoObjectArray(base, indices, rhs, rt);
      if (isRuntimeString(rhs) || isRuntimeStringArray(rhs))
        return storeIntoStringArray(base, indices, rhs);
    }
    return storeIntoTensor(base, indices, rhs);
  }

  if (isRuntimeCell(base)) {
    if (parenAssign && isDeletionRhs(rhs)) return deleteFromCell(base, indices);
    return storeIntoCell(base, indices, rhs, !parenAssign, rt);
  }

  if (isRuntimeStruct(base) || isRuntimeStructArray(base)) {
    if (isDeletionRhs(rhs)) return deleteFromStructArray(base, indices);
    if (isRuntimeStruct(rhs) || isRuntimeStructArray(rhs))
      return storeIntoStructArray(base, indices, rhs, !storeBack, rt);
    throw new RuntimeError("Assignment between unlike types is not allowed.");
  }

  if (isRuntimeClassInstance(base) || isRuntimeClassInstanceArray(base)) {
    if (isDeletionRhs(rhs)) return deleteFromObjectArray(base, indices);
    if (isRuntimeClassInstance(rhs) || isRuntimeClassInstanceArray(rhs))
      return storeIntoObjectArray(base, indices, rhs, rt);
    throw new RuntimeError(
      `Conversion to ${base.className} from ${valueClass(rhs)} is not possible.`
    );
  }

  if (isRuntimeString(base) || isRuntimeStringArray(base)) {
    if (isDeletionRhs(rhs)) return deleteFromStringArray(base, indices);
    return storeIntoStringArray(base, indices, rhs);
  }

  throw new RuntimeError(`Cannot index-assign into ${kstr(base)}`);
}
