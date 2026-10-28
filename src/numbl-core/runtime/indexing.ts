/**
 * Indexing: per-container gather / scatter / resize on top of the
 * storage-independent planner in `indexPlan.ts`.
 *
 * Tensors (real, complex, logical), scalars, char arrays and cell arrays
 * use the planner for read, store (with growth) and delete. Sparse
 * matrices, string arrays and struct arrays still have their own
 * resolution code.
 */

import {
  type RuntimeValue,
  RuntimeTensor,
  RuntimeCell,
  type RuntimeSparseMatrix,
  type RuntimeComplexNumber,
  RuntimeChar,
  isRuntimeTensor,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeComplexNumber,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeSparseMatrix,
  kstr,
  isRuntimeStructArray,
  isRuntimeStruct,
  isRuntimeClassInstance,
  isRuntimeChar,
  isRuntimeStringArray,
  RuntimeStringArray,
  stringArrayValue,
} from "./types.js";
import { RuntimeError } from "./error.js";
import { RTV } from "./constructors.js";
import { toNumber } from "./convert.js";
import { type RefcountRuntime, isShared } from "./refcount.js";
import { allocFloat64Array } from "./alloc.js";
import {
  COLON_INDEX,
  type Selection,
  planIndex,
  planStore,
  planDelete,
  offsetAt,
  forEachGrowRun,
  growsInPlace,
} from "./indexPlan.js";

export { COLON_INDEX };

/** Runtime surface needed by index-store mutations. The full Runtime
 *  class satisfies this structurally. */
type StoreRuntime = RefcountRuntime;

/** Check if an index is a colon sentinel */
function isColonIndex(v: RuntimeValue): boolean {
  return v === COLON_INDEX;
}

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

// ── Legacy subscript helpers (sparse only) ──────────────────────────────

/** Convert a RuntimeValue index to an array of 0-based numeric indices.
 * Handles: colon (all), logical scalar, logical tensor, numeric tensor, numeric scalar.
 * If `boundsLimit` > 0, throws on out-of-bounds; if 0, skips upper-bound check. */
function resolveIndex(
  idx: RuntimeValue,
  dimSize: number,
  boundsLimit: number = dimSize
): number[] {
  if (isColonIndex(idx)) {
    return Array.from({ length: dimSize }, (_, i) => i);
  }
  if (isRuntimeLogical(idx)) {
    return idx ? [0] : [];
  }
  if (isRuntimeTensor(idx)) {
    if (idx._isLogical) {
      const result: number[] = [];
      for (let i = 0; i < idx.data.length; i++) {
        if (idx.data[i] !== 0) result.push(i);
      }
      return result;
    }
    return Array.from(idx.data).map(v => {
      const i = Math.round(v) - 1;
      if (i < 0 || (boundsLimit > 0 && i >= boundsLimit))
        throw new RuntimeError("Index exceeds array bounds");
      return i;
    });
  }
  const i = Math.round(toNumber(idx)) - 1;
  if (i < 0 || (boundsLimit > 0 && i >= boundsLimit))
    throw new RuntimeError("Index exceeds array bounds");
  return [i];
}

/** Extract RHS values as individual RuntimeValues for element-wise assignment. */
function extractRhsValues(rhs: RuntimeValue, count: number): RuntimeValue[] {
  if (
    isRuntimeNumber(rhs) ||
    isRuntimeLogical(rhs) ||
    isRuntimeComplexNumber(rhs)
  ) {
    return new Array(count).fill(rhs);
  }
  if (isRuntimeTensor(rhs)) {
    const result: RuntimeValue[] = [];
    for (let i = 0; i < count; i++) {
      if (rhs.imag) {
        result.push(RTV.complex(rhs.data[i], rhs.imag[i]));
      } else {
        result.push(RTV.num(rhs.data[i]));
      }
    }
    return result;
  }
  return new Array(count).fill(rhs);
}

// ── Sparse indexing helpers ──────────────────────────────────────────────

/** Look up a single value in a sparse matrix by 0-based (row, col). */
function sparseElementAt(
  s: RuntimeSparseMatrix,
  row: number,
  col: number
): { re: number; im: number } {
  for (let k = s.jc[col]; k < s.jc[col + 1]; k++) {
    if (s.ir[k] === row) {
      return { re: s.pr[k], im: s.pi ? s.pi[k] : 0 };
    }
  }
  return { re: 0, im: 0 };
}

/** Build a sparse matrix from selected rows and columns of another sparse matrix.
 *  rowIdx and colIdx are 0-based. */
function sparseSubmatrix(
  s: RuntimeSparseMatrix,
  rowIdx: number[],
  colIdx: number[]
): RuntimeSparseMatrix {
  const nRows = rowIdx.length;
  const nCols = colIdx.length;
  const rowLookup = new Map<number, number[]>();
  for (let i = 0; i < nRows; i++) {
    const orig = rowIdx[i];
    const list = rowLookup.get(orig);
    if (list) list.push(i);
    else rowLookup.set(orig, [i]);
  }

  const isComplex = s.pi !== undefined;
  const irArr: number[] = [];
  const prArr: number[] = [];
  const piArr: number[] = [];
  const jcNew = new Int32Array(nCols + 1);

  for (let ci = 0; ci < nCols; ci++) {
    const origCol = colIdx[ci];
    jcNew[ci] = irArr.length;
    const entries: { newRow: number; re: number; im: number }[] = [];
    for (let k = s.jc[origCol]; k < s.jc[origCol + 1]; k++) {
      const newRows = rowLookup.get(s.ir[k]);
      if (newRows !== undefined) {
        const re = s.pr[k];
        const im = isComplex ? s.pi![k] : 0;
        for (const newRow of newRows) {
          entries.push({ newRow, re, im });
        }
      }
    }
    entries.sort((a, b) => a.newRow - b.newRow);
    for (const e of entries) {
      irArr.push(e.newRow);
      prArr.push(e.re);
      if (isComplex) piArr.push(e.im);
    }
  }
  jcNew[nCols] = irArr.length;

  return RTV.sparseMatrix(
    nRows,
    nCols,
    new Int32Array(irArr),
    jcNew,
    allocFloat64Array(prArr),
    isComplex ? allocFloat64Array(piArr) : undefined
  );
}

// ── Sparse store ─────────────────────────────────────────────────────────

/** Assign into a sparse matrix: S(rows, cols) = rhs.
 *  Returns a new sparse matrix (COW). */
function storeIntoSparse(
  base: RuntimeSparseMatrix,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeSparseMatrix {
  if (indices.length === 1) {
    const totalLen = base.m * base.n;
    const linIdx = resolveIndex(indices[0], totalLen);
    const rowIdx = linIdx.map(k => k % base.m);
    const colIdx = linIdx.map(k => Math.floor(k / base.m));
    let result = base;
    const rhsVals = extractRhsValues(rhs, linIdx.length);
    for (let i = 0; i < linIdx.length; i++) {
      result = storeIntoSparse(
        result,
        [RTV.num(rowIdx[i] + 1), RTV.num(colIdx[i] + 1)],
        rhsVals[i]
      );
    }
    return result;
  }
  if (indices.length !== 2) {
    throw new RuntimeError("Sparse matrix assignment requires 1 or 2 indices");
  }

  const rowIdx = resolveIndex(indices[0], base.m);
  const colIdx = resolveIndex(indices[1], base.n);

  const nR = rowIdx.length;
  const nC = colIdx.length;
  const rhsRe = allocFloat64Array(nR * nC);
  const rhsIm = allocFloat64Array(nR * nC);
  let rhsHasImag = false;

  if (isRuntimeNumber(rhs)) {
    rhsRe.fill(rhs);
  } else if (isRuntimeLogical(rhs)) {
    rhsRe.fill(rhs ? 1 : 0);
  } else if (isRuntimeComplexNumber(rhs)) {
    rhsRe.fill(rhs.re);
    rhsIm.fill(rhs.im);
    rhsHasImag = rhs.im !== 0;
  } else if (isRuntimeTensor(rhs)) {
    for (let i = 0; i < rhs.data.length; i++) rhsRe[i] = rhs.data[i];
    if (rhs.imag) {
      for (let i = 0; i < rhs.imag.length; i++) rhsIm[i] = rhs.imag[i];
      rhsHasImag = true;
    }
  } else if (isRuntimeSparseMatrix(rhs)) {
    for (let c = 0; c < rhs.n; c++) {
      for (let k = rhs.jc[c]; k < rhs.jc[c + 1]; k++) {
        const idx = c * nR + rhs.ir[k];
        rhsRe[idx] = rhs.pr[k];
        if (rhs.pi) {
          rhsIm[idx] = rhs.pi[k];
          rhsHasImag = true;
        }
      }
    }
  } else {
    throw new RuntimeError(
      "Cannot assign non-numeric value into sparse matrix"
    );
  }

  const isComplex = base.pi !== undefined || rhsHasImag;

  type Entry = { row: number; re: number; im: number };
  const colEntries = new Map<number, Entry[]>();

  for (let c = 0; c < base.n; c++) {
    const entries: Entry[] = [];
    for (let k = base.jc[c]; k < base.jc[c + 1]; k++) {
      entries.push({
        row: base.ir[k],
        re: base.pr[k],
        im: base.pi ? base.pi[k] : 0,
      });
    }
    if (entries.length > 0) colEntries.set(c, entries);
  }

  for (let ci = 0; ci < nC; ci++) {
    const col = colIdx[ci];
    let entries = colEntries.get(col);
    if (!entries) {
      entries = [];
      colEntries.set(col, entries);
    }
    for (let ri = 0; ri < nR; ri++) {
      const row = rowIdx[ri];
      const re = rhsRe[ci * nR + ri];
      const im = rhsIm[ci * nR + ri];
      let found = false;
      for (let e = 0; e < entries.length; e++) {
        if (entries[e].row === row) {
          if (re === 0 && im === 0) {
            entries.splice(e, 1);
          } else {
            entries[e].re = re;
            entries[e].im = im;
          }
          found = true;
          break;
        }
      }
      if (!found && (re !== 0 || im !== 0)) {
        entries.push({ row, re, im });
      }
    }
    entries.sort((a, b) => a.row - b.row);
    if (entries.length === 0) colEntries.delete(col);
  }

  const irArr: number[] = [];
  const prArr: number[] = [];
  const piArr: number[] = [];
  const jc = new Int32Array(base.n + 1);

  for (let c = 0; c < base.n; c++) {
    jc[c] = irArr.length;
    const entries = colEntries.get(c);
    if (entries) {
      for (const e of entries) {
        irArr.push(e.row);
        prArr.push(e.re);
        if (isComplex) piArr.push(e.im);
      }
    }
  }
  jc[base.n] = irArr.length;

  return RTV.sparseMatrix(
    base.m,
    base.n,
    new Int32Array(irArr),
    jc,
    allocFloat64Array(prArr),
    isComplex ? allocFloat64Array(piArr) : undefined
  );
}

// ── Sparse index (read) ─────────────────────────────────────────────────

function indexIntoSparse(
  base: RuntimeSparseMatrix,
  indices: RuntimeValue[]
): RuntimeValue {
  if (indices.length === 1) {
    const idx = indices[0];

    if (isColonIndex(idx)) {
      const totalLen = base.m * base.n;
      const irArr: number[] = [];
      const prArr: number[] = [];
      const piArr: number[] = [];
      const isComplex = base.pi !== undefined;

      for (let col = 0; col < base.n; col++) {
        for (let k = base.jc[col]; k < base.jc[col + 1]; k++) {
          const linIdx = col * base.m + base.ir[k];
          irArr.push(linIdx);
          prArr.push(base.pr[k]);
          if (isComplex) piArr.push(base.pi![k]);
        }
      }

      const jcNew = new Int32Array([0, irArr.length]);
      return RTV.sparseMatrix(
        totalLen,
        1,
        new Int32Array(irArr),
        jcNew,
        allocFloat64Array(prArr),
        isComplex ? allocFloat64Array(piArr) : undefined
      );
    }

    if (isRuntimeNumber(idx)) {
      const k = Math.round(idx) - 1;
      if (k < 0 || k >= base.m * base.n)
        throw new RuntimeError("Index exceeds array bounds");
      const col = Math.floor(k / base.m);
      const row = k % base.m;
      const val = sparseElementAt(base, row, col);
      if (base.pi !== undefined && val.im !== 0) {
        return RTV.complex(val.re, val.im);
      }
      return val.re;
    }

    if (isRuntimeTensor(idx) || isRuntimeLogical(idx)) {
      const totalLen = base.m * base.n;
      const linIdx = resolveIndex(idx, totalLen);
      const n = linIdx.length;
      const isComplex = base.pi !== undefined;
      const irArr: number[] = [];
      const prArr: number[] = [];
      const piArr: number[] = [];
      const jcNew = new Int32Array(n + 1);
      for (let i = 0; i < n; i++) {
        jcNew[i] = irArr.length;
        const k = linIdx[i];
        const col = Math.floor(k / base.m);
        const row = k % base.m;
        const val = sparseElementAt(base, row, col);
        if (val.re !== 0 || val.im !== 0) {
          irArr.push(0);
          prArr.push(val.re);
          if (isComplex) piArr.push(val.im);
        }
      }
      jcNew[n] = irArr.length;
      return RTV.sparseMatrix(
        1,
        n,
        new Int32Array(irArr),
        jcNew,
        allocFloat64Array(prArr),
        isComplex ? allocFloat64Array(piArr) : undefined
      );
    }
  }

  if (indices.length === 2) {
    // Fast path: sp(:, col) — extract full column(s) directly from CSC
    if (isColonIndex(indices[0])) {
      const colIdx = resolveIndex(indices[1], base.n);
      if (colIdx.length === 1) {
        // Single column extraction — O(nnz_in_col) not O(m)
        const col = colIdx[0];
        const start = base.jc[col];
        const end = base.jc[col + 1];
        const nnz = end - start;
        const ir = new Int32Array(nnz);
        const pr = allocFloat64Array(nnz);
        let pi: Float64Array | undefined;
        for (let k = 0; k < nnz; k++) {
          ir[k] = base.ir[start + k];
          pr[k] = base.pr[start + k];
        }
        if (base.pi !== undefined) {
          pi = allocFloat64Array(nnz);
          for (let k = 0; k < nnz; k++) pi[k] = base.pi[start + k];
        }
        const jc = new Int32Array([0, nnz]);
        return RTV.sparseMatrix(base.m, 1, ir, jc, pr, pi);
      }
      // Multi-column extraction with colon rows — avoid resolving colon
      const nCols = colIdx.length;
      const isComplex = base.pi !== undefined;
      const irArr: number[] = [];
      const prArr: number[] = [];
      const piArr: number[] = [];
      const jcNew = new Int32Array(nCols + 1);
      for (let ci = 0; ci < nCols; ci++) {
        jcNew[ci] = irArr.length;
        const origCol = colIdx[ci];
        for (let k = base.jc[origCol]; k < base.jc[origCol + 1]; k++) {
          irArr.push(base.ir[k]);
          prArr.push(base.pr[k]);
          if (isComplex) piArr.push(base.pi![k]);
        }
      }
      jcNew[nCols] = irArr.length;
      return RTV.sparseMatrix(
        base.m,
        nCols,
        new Int32Array(irArr),
        jcNew,
        allocFloat64Array(prArr),
        isComplex ? allocFloat64Array(piArr) : undefined
      );
    }

    const rowIdx = resolveIndex(indices[0], base.m);
    const colIdx = resolveIndex(indices[1], base.n);

    if (rowIdx.length === 1 && colIdx.length === 1) {
      const val = sparseElementAt(base, rowIdx[0], colIdx[0]);
      if (base.pi !== undefined && val.im !== 0) {
        return RTV.complex(val.re, val.im);
      }
      return val.re;
    }

    return sparseSubmatrix(base, rowIdx, colIdx);
  }

  throw new RuntimeError("Sparse matrix supports 1 or 2 index dimensions");
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
    if (base._isLogical === true) return base.data[o] !== 0;
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
  return new RuntimeTensor(
    data,
    shape,
    imag,
    base._isLogical === true ? true : undefined
  );
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
    typeof base === "boolean" ? true : undefined
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
  return new RuntimeTensor(data, newShape, imag, base._isLogical);
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
      `Conversion to ${base._isLogical === true ? "logical" : "double"} from ${className(rhs)} is not possible.`
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

  const logical = base._isLogical === true;
  let rhsRe: Float64Array | null = rhsT ? rhsT.data : null;
  let rhsIm: Float64Array | undefined = rhsT ? rhsT.imag : undefined;
  if (logical) {
    if (rhsT) {
      if (rhsT._isLogical !== true) {
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
  return new RuntimeTensor(data, plan.shape, imag, base._isLogical);
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

function className(v: RuntimeValue): string {
  if (typeof v === "number" || isRuntimeTensor(v) || isRuntimeComplexNumber(v))
    return isRuntimeTensor(v) && v._isLogical ? "logical" : "double";
  if (typeof v === "boolean") return "logical";
  if (isRuntimeChar(v)) return "char";
  if (isRuntimeStruct(v) || isRuntimeStructArray(v)) return "struct";
  if (typeof v === "string" || isRuntimeStringArray(v)) return "string";
  return kstr(v);
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
        `Conversion to cell from ${className(rhs)} is not possible.`
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

// ── String arrays ────────────────────────────────────────────────────────

/** Resolve one subscript against a dimension of extent `extent` to 0-based
 *  positions. Handles colon, scalars, numeric vectors, and logical masks. */
export function stringArraySubscript(
  idx: RuntimeValue,
  extent: number
): number[] {
  if (isColonIndex(idx)) {
    return Array.from({ length: extent }, (_, i) => i);
  }
  if (isRuntimeTensor(idx)) {
    if (idx._isLogical) {
      const out: number[] = [];
      for (let i = 0; i < idx.data.length; i++)
        if (idx.data[i] !== 0) out.push(i);
      return out;
    }
    return Array.from(idx.data, (v: number) => Math.round(v) - 1);
  }
  if (isRuntimeLogical(idx)) return idx ? [0] : [];
  return [Math.round(toNumber(idx)) - 1];
}

function indexIntoStringArray(
  base: RuntimeStringArray,
  indices: RuntimeValue[]
): RuntimeValue {
  const [rows, cols] = base.shape;
  const total = base.data.length;

  if (indices.length === 2) {
    const ri = stringArraySubscript(indices[0], rows);
    const ci = stringArraySubscript(indices[1], cols);
    const out: string[] = [];
    for (const c of ci) {
      if (c < 0 || c >= cols)
        throw new RuntimeError("Index exceeds array bounds");
      for (const r of ri) {
        if (r < 0 || r >= rows)
          throw new RuntimeError("Index exceeds array bounds");
        out.push(base.data[c * rows + r]);
      }
    }
    return stringArrayValue(out, [ri.length, ci.length]);
  }

  if (indices.length !== 1) {
    throw new RuntimeError("String array indexing supports 1 or 2 subscripts");
  }

  const idx = indices[0];
  if (isColonIndex(idx)) {
    return stringArrayValue(base.data.slice(), [total, 1]);
  }
  const linear = stringArraySubscript(idx, total);
  const out: string[] = [];
  for (const i of linear) {
    if (i < 0 || i >= total)
      throw new RuntimeError("Index exceeds array bounds");
    out.push(base.data[i]);
  }
  // Orientation: vector sources keep their orientation; a matrix source
  // takes the shape of a tensor index (MATLAB rule), else a row.
  let shape: [number, number];
  if (rows === 1 || cols === 1) {
    shape = rows === 1 ? [1, out.length] : [out.length, 1];
  } else if (isRuntimeTensor(idx) && !idx._isLogical) {
    const is = idx.shape;
    const im = is.length >= 2 ? is[0] : 1;
    shape = [im, out.length / (im || 1)];
  } else if (isRuntimeTensor(idx) && idx._isLogical) {
    shape = [out.length, 1];
  } else {
    shape = [1, out.length];
  }
  return stringArrayValue(out, shape);
}

// ── Index read: main dispatcher ──────────────────────────────────────────

/** Index into a value: v(i1, i2, ...) */
export function indexIntoRTValue(
  base: RuntimeValue,
  indices: RuntimeValue[]
): RuntimeValue {
  if (
    typeof base === "number" ||
    typeof base === "boolean" ||
    isRuntimeComplexNumber(base)
  ) {
    return indexIntoScalar(base, indices);
  }
  if (isRuntimeTensor(base)) {
    return indexIntoTensor(base, indices);
  }
  if (isRuntimeCell(base)) {
    return indexIntoCell(base, indices);
  }
  if (isRuntimeChar(base)) {
    return indexIntoChar(base, indices);
  }
  if (isRuntimeString(base)) {
    if (indices.length === 1) {
      const i = Math.round(toNumber(indices[0]));
      if (i !== 1) throw new RuntimeError("Index exceeds string dimensions");
      return base;
    }
  }
  if (isRuntimeStringArray(base)) {
    return indexIntoStringArray(base, indices);
  }
  if (isRuntimeStruct(base) || isRuntimeClassInstance(base)) {
    // A scalar (1x1) struct / object: any all-ones subscripting — s(1),
    // s(1,1), s(true), ... — refers to the element itself, like MATLAB.
    const allOnes =
      indices.length >= 1 &&
      indices.every(idx => {
        if (isRuntimeNumber(idx)) return Math.round(idx) === 1;
        if (isRuntimeLogical(idx)) return idx === true;
        if (isRuntimeTensor(idx))
          return idx.data.length === 1 && Math.round(idx.data[0]) === 1;
        if (isColonIndex(idx)) return true; // s(:) on 1x1 → itself
        return false;
      });
    if (allOnes) return base;
    throw new RuntimeError("Index exceeds struct dimensions");
  }
  if (isRuntimeSparseMatrix(base)) {
    return indexIntoSparse(base, indices);
  }
  if (isRuntimeStructArray(base)) {
    if (indices.length === 1) {
      const idx = indices[0];
      // s(:) → the whole struct array (as a column, but struct arrays
      // are stored flat so the element list is unchanged)
      if (isColonIndex(idx)) {
        return RTV.structArray(base.fieldNames, [...base.elements]);
      }
      if (isRuntimeNumber(idx)) {
        const i = Math.round(idx) - 1;
        if (i < 0 || i >= base.elements.length)
          throw new RuntimeError("Index exceeds struct array bounds");
        return base.elements[i];
      }
      if (isRuntimeTensor(idx)) {
        // Logical mask: select the elements at true positions.
        if (idx._isLogical) {
          const newElements = [];
          for (let k = 0; k < idx.data.length; k++) {
            if (idx.data[k] !== 0) {
              if (k >= base.elements.length)
                throw new RuntimeError("Index exceeds struct array bounds");
              newElements.push(base.elements[k]);
            }
          }
          return RTV.structArray(base.fieldNames, newElements);
        }
        const newElements = [];
        for (let k = 0; k < idx.data.length; k++) {
          const i = Math.round(idx.data[k]) - 1;
          if (i < 0 || i >= base.elements.length)
            throw new RuntimeError("Index exceeds struct array bounds");
          newElements.push(base.elements[i]);
        }
        return RTV.structArray(base.fieldNames, newElements);
      }
      if (isRuntimeLogical(idx)) {
        const i = idx ? 0 : -1;
        if (i < 0 || i >= base.elements.length)
          throw new RuntimeError("Index exceeds struct array bounds");
        return base.elements[i];
      }
      throw new RuntimeError("Invalid index type for struct array");
    }
    throw new RuntimeError("Struct array only supports single-index access");
  }

  throw new RuntimeError(`Cannot index into ${kstr(base)}`);
}

// ── Store: main dispatcher ───────────────────────────────────────────────

/** Store into indexed position: base(indices) = rhs (or base{indices} = rhs
 *  for cells when `parenAssign` is false). A 0x0 RHS in a paren store
 *  deletes. Shared targets are copied before mutation. */
export function storeIntoRTValueIndex(
  base: RuntimeValue,
  indices: RuntimeValue[],
  rhs: RuntimeValue,
  parenAssign = false,
  rt?: StoreRuntime
): RuntimeValue {
  if (isRuntimeSparseMatrix(base)) {
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
    return storeIntoTensor(base, indices, rhs);
  }

  if (isRuntimeCell(base)) {
    if (parenAssign && isDeletionRhs(rhs)) return deleteFromCell(base, indices);
    return storeIntoCell(base, indices, rhs, !parenAssign, rt);
  }

  throw new RuntimeError(`Cannot index-assign into ${kstr(base)}`);
}
