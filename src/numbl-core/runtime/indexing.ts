/**
 * Indexing, struct field access, range creation, and tensor concatenation.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  type RuntimeCell,
  type RuntimeSparseMatrix,
  FloatXArray,
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
  type RuntimeChar,
} from "./types.js";
import { RuntimeError } from "./error.js";
import { RTV } from "./constructors.js";
import {
  tensorSize2D,
  colMajorIndex,
  sub2ind,
  shareRuntimeValue,
} from "./utils.js";
import { toNumber } from "./convert.js";

// ── Colon index sentinel ─────────────────────────────────────────────────

/** Sentinel marker for colon (:) indexing — means "all indices in this dimension" */
export const COLON_INDEX: RuntimeValue = "__COLON__";

/** Check if an index is a colon sentinel */
function isColonIndex(v: RuntimeValue): boolean {
  return isRuntimeString(v) && v === "__COLON__";
}

// ── Shared helpers ───────────────────────────────────────────────────────

/** Extract real and imaginary parts from a value for assignment into a tensor. */
function toReIm(v: RuntimeValue): { re: number; im: number } {
  if (isRuntimeNumber(v)) return { re: v, im: 0 };
  if (isRuntimeLogical(v)) return { re: v ? 1 : 0, im: 0 };
  if (isRuntimeComplexNumber(v)) return { re: v.re, im: v.im };
  if (isRuntimeTensor(v) && v.data.length === 1) {
    return { re: v.data[0], im: v.imag ? v.imag[0] : 0 };
  }
  if (isRuntimeSparseMatrix(v) && v.m === 1 && v.n === 1) {
    const re = v.pr.length > 0 ? v.pr[0] : 0;
    const im = v.pi && v.pi.length > 0 ? v.pi[0] : 0;
    return { re, im };
  }
  throw new RuntimeError(`Cannot convert ${kstr(v)} to number for assignment`);
}

/** Ensure a tensor has an imag array (allocate if needed). */
function ensureImag(t: RuntimeTensor): void {
  if (!t.imag) {
    t.imag = new FloatXArray(t.data.length);
  }
}

/** Return a scalar (number or complex) from a tensor at linear index `i`. */
function extractTensorElement(base: RuntimeTensor, i: number): RuntimeValue {
  if (base.imag !== undefined) {
    const im = base.imag[i];
    return im === 0 ? RTV.num(base.data[i]) : RTV.complex(base.data[i], im);
  }
  return RTV.num(base.data[i]);
}

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

/** Resolve a pair of row/col indices to 0-based arrays.
 *  Works for tensor, colon, logical, and scalar indices. */
function resolveIndex2D(
  rowIdx: RuntimeValue,
  colIdx: RuntimeValue,
  rows: number,
  cols: number
): { rowIndices: number[]; colIndices: number[] } {
  return {
    rowIndices: resolveIndex(rowIdx, rows),
    colIndices: resolveIndex(colIdx, cols),
  };
}

/**
 * Grow a tensor to at least `newRows × newCols`, copying old column-major data.
 * Returns the new tensor. If no growth is needed, returns the original.
 */
function growTensor2D(
  base: RuntimeTensor,
  newRows: number,
  newCols: number
): RuntimeTensor {
  const [curRows, curCols] = tensorSize2D(base);
  if (newRows <= curRows && newCols <= curCols) return base;
  const nr = Math.max(curRows, newRows);
  const nc = Math.max(curCols, newCols);
  const newData = new FloatXArray(nr * nc);
  const hasImag = base.imag !== undefined;
  const newIm = hasImag ? new FloatXArray(nr * nc) : undefined;
  for (let c = 0; c < curCols; c++) {
    for (let r = 0; r < curRows; r++) {
      newData[r + c * nr] = base.data[r + c * curRows];
      if (newIm && base.imag) newIm[r + c * nr] = base.imag[r + c * curRows];
    }
  }
  return RTV.tensor(newData, [nr, nc], newIm);
}

/**
 * Assign a scalar (re, im) or tensor RHS into a tensor slice defined by
 * resolved 0-based rowIndices and colIndices. Mutates `base` in place.
 */
function assignSlice(
  base: RuntimeTensor,
  rowIndices: number[],
  colIndices: number[],
  rhs: RuntimeValue,
  curRows: number
): void {
  if (isRuntimeTensor(rhs)) {
    const [rhsRows, rhsCols] = tensorSize2D(rhs);
    if (rhsRows !== rowIndices.length || rhsCols !== colIndices.length) {
      throw new RuntimeError("Subscripted assignment dimension mismatch");
    }
    if (rhs.imag || base.imag) ensureImag(base);
    for (let ri = 0; ri < rowIndices.length; ri++) {
      for (let ci = 0; ci < colIndices.length; ci++) {
        const dstLi = colMajorIndex(rowIndices[ri], colIndices[ci], curRows);
        const srcLi = colMajorIndex(ri, ci, rhsRows);
        base.data[dstLi] = rhs.data[srcLi];
        if (base.imag) base.imag[dstLi] = rhs.imag ? rhs.imag[srcLi] : 0;
      }
    }
  } else {
    const { re, im } = toReIm(rhs);
    if (im !== 0 || base.imag) ensureImag(base);
    for (const r of rowIndices) {
      for (const c of colIndices) {
        const li = colMajorIndex(r, c, curRows);
        base.data[li] = re;
        if (base.imag) base.imag[li] = im;
      }
    }
  }
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
    new Float64Array(prArr),
    isComplex ? new Float64Array(piArr) : undefined
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
  const rhsRe = new Float64Array(nR * nC);
  const rhsIm = new Float64Array(nR * nC);
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
    new Float64Array(prArr),
    isComplex ? new Float64Array(piArr) : undefined
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
        new Float64Array(prArr),
        isComplex ? new Float64Array(piArr) : undefined
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
        new Float64Array(prArr),
        isComplex ? new Float64Array(piArr) : undefined
      );
    }
  }

  if (indices.length === 2) {
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

// ── Index read: per-type handlers ────────────────────────────────────────

function indexIntoScalar(
  base: RuntimeValue,
  indices: RuntimeValue[]
): RuntimeValue {
  // If any index is an empty tensor, result is empty
  if (indices.some(idx => isRuntimeTensor(idx) && idx.data.length === 0)) {
    if (indices.length === 2) {
      const dimSize = (idx: RuntimeValue): number => {
        if (isColonIndex(idx)) return 1;
        if (isRuntimeTensor(idx)) return idx.data.length;
        return 1;
      };
      return RTV.tensor(new FloatXArray(0), [
        dimSize(indices[0]),
        dimSize(indices[1]),
      ]);
    }
    const idx = indices[0];
    if (isRuntimeTensor(idx)) {
      const is0x0 =
        idx.shape.length === 2 && idx.shape[0] === 0 && idx.shape[1] === 0;
      const outShape = is0x0 ? [0, 0] : [...idx.shape];
      return RTV.tensor(new FloatXArray(0), outShape);
    }
  }
  // Logical indexing on scalar
  if (indices.length === 1) {
    const idx = indices[0];
    if (isRuntimeLogical(idx)) {
      if (!idx) return RTV.tensor(new FloatXArray(0), [0, 0]);
      return base;
    }
    if (isRuntimeTensor(idx) && idx._isLogical) {
      let count = 0;
      for (let j = 0; j < idx.data.length; j++) if (idx.data[j]) count++;
      if (count === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
      const out = new FloatXArray(count);
      out.fill(isRuntimeNumber(base) ? base : (base as { re: number }).re);
      if (isRuntimeComplexNumber(base) && base.im !== 0) {
        const imOut = new FloatXArray(count);
        imOut.fill(base.im);
        return RTV.tensor(out, [1, count], imOut);
      }
      return RTV.tensor(out, [1, count]);
    }
  }
  // Scalar indexing: only valid with (1) or (1,1) or (:)
  for (const idx of indices) {
    if (isColonIndex(idx)) continue;
    const i = toNumber(idx);
    if (i !== 1) throw new RuntimeError("Index exceeds array bounds");
  }
  return base;
}

function indexIntoTensor(
  base: RuntimeTensor,
  indices: RuntimeValue[]
): RuntimeValue {
  if (indices.length === 1) {
    return indexIntoTensor1D(base, indices[0]);
  }
  // MATLAB: when fewer indices are supplied than the tensor's rank, the
  // last index linearizes all trailing dimensions (column-major).  E.g.
  // A(i,j) on a [2 16 4] tensor behaves as if A were [2 64].
  if (base.shape.length > indices.length) {
    const collapsedShape: number[] = [];
    for (let d = 0; d < indices.length - 1; d++) {
      collapsedShape.push(base.shape[d]);
    }
    let tail = 1;
    for (let d = indices.length - 1; d < base.shape.length; d++) {
      tail *= base.shape[d];
    }
    collapsedShape.push(tail);
    const view: RuntimeTensor = { ...base, shape: collapsedShape };
    if (indices.length === 2) {
      return indexIntoTensor2D(view, indices[0], indices[1]);
    }
    return indexIntoTensorND(view, indices);
  }
  if (indices.length === 2) {
    return indexIntoTensor2D(base, indices[0], indices[1]);
  }
  // N-dimensional indexing (3+)
  return indexIntoTensorND(base, indices);
}

function indexIntoTensor1D(
  base: RuntimeTensor,
  idx: RuntimeValue
): RuntimeValue {
  // Colon: base(:) → column vector
  if (isColonIndex(idx)) {
    const imag = base.imag ? new FloatXArray(base.imag) : undefined;
    return RTV.tensor(new FloatXArray(base.data), [base.data.length, 1], imag);
  }

  if (isRuntimeLogical(idx)) {
    if (!idx) return RTV.tensor(new FloatXArray(0), [0, 0]);
    return extractTensorElement(base, 0);
  }

  if (isRuntimeNumber(idx)) {
    const i = Math.round(toNumber(idx)) - 1;
    if (i < 0 || i >= base.data.length) {
      throw new RuntimeError("Index exceeds array bounds");
    }
    return extractTensorElement(base, i);
  }

  if (isRuntimeTensor(idx)) {
    return indexIntoTensorWithTensor(base, idx);
  }

  throw new RuntimeError(`Invalid index type for tensor`);
}

function indexIntoTensor2D(
  base: RuntimeTensor,
  rowIdx: RuntimeValue,
  colIdx: RuntimeValue
): RuntimeValue {
  const [rows, cols] = tensorSize2D(base);

  // Fast path: single-row extraction c(k,:) — avoids allocating a cols-length index array
  if (isRuntimeNumber(rowIdx) && isColonIndex(colIdx)) {
    const r = Math.round(rowIdx as number) - 1;
    if (r < 0 || r >= rows)
      throw new RuntimeError("Index exceeds array bounds");
    const resultData = new FloatXArray(cols);
    const resultImag = base.imag ? new FloatXArray(cols) : undefined;
    for (let ci = 0; ci < cols; ci++) {
      resultData[ci] = base.data[r + ci * rows];
      if (resultImag && base.imag) resultImag[ci] = base.imag[r + ci * rows];
    }
    return RTV.tensor(resultData, [1, cols], resultImag);
  }

  // Fast path: single-column extraction c(:,k) — column is contiguous in col-major
  if (isColonIndex(rowIdx) && isRuntimeNumber(colIdx)) {
    const c = Math.round(colIdx as number) - 1;
    if (c < 0 || c >= cols)
      throw new RuntimeError("Index exceeds array bounds");
    const offset = c * rows;
    const resultData = new FloatXArray(rows);
    for (let ri = 0; ri < rows; ri++) resultData[ri] = base.data[offset + ri];
    const resultImag = base.imag ? new FloatXArray(rows) : undefined;
    if (resultImag && base.imag)
      for (let ri = 0; ri < rows; ri++) resultImag[ri] = base.imag[offset + ri];
    return RTV.tensor(resultData, [rows, 1], resultImag);
  }

  const rowIdxArr = resolveIndex(rowIdx, rows);
  const colIdxArr = resolveIndex(colIdx, cols);

  const numR = rowIdxArr.length;
  const numC = colIdxArr.length;

  if (numR === 1 && numC === 1) {
    const linearIdx = colMajorIndex(rowIdxArr[0], colIdxArr[0], rows);
    return extractTensorElement(base, linearIdx);
  }

  const resultData = new FloatXArray(numR * numC);
  const resultImag = base.imag ? new FloatXArray(numR * numC) : undefined;
  for (let ci = 0; ci < numC; ci++) {
    for (let ri = 0; ri < numR; ri++) {
      const srcIdx = colMajorIndex(rowIdxArr[ri], colIdxArr[ci], rows);
      const dstIdx = colMajorIndex(ri, ci, numR);
      resultData[dstIdx] = base.data[srcIdx];
      if (resultImag && base.imag) {
        resultImag[dstIdx] = base.imag[srcIdx];
      }
    }
  }
  return RTV.tensor(resultData, [numR, numC], resultImag);
}

function indexIntoTensorND(
  base: RuntimeTensor,
  indices: RuntimeValue[]
): RuntimeValue {
  const shape = base.shape;
  const dimIndices: number[][] = indices.map((idx, dim) => {
    const dimSize = dim < shape.length ? shape[dim] : 1;
    return resolveIndex(idx, dimSize);
  });

  const allScalar = dimIndices.every(d => d.length === 1);
  if (allScalar) {
    const subs = dimIndices.map(d => d[0]);
    const linearIdx = sub2ind(shape, subs);
    return extractTensorElement(base, linearIdx);
  }

  const resultShape = dimIndices.map(d => d.length);
  while (resultShape.length > 2 && resultShape[resultShape.length - 1] === 1) {
    resultShape.pop();
  }
  const totalElems = resultShape.reduce((a, b) => a * b, 1);
  const resultData = new FloatXArray(totalElems);
  const resultImag = base.imag ? new FloatXArray(totalElems) : undefined;

  const ndimIdx = dimIndices.length;
  const srcStrides = new Array(ndimIdx);
  srcStrides[0] = 1;
  for (let d = 1; d < ndimIdx; d++)
    srcStrides[d] =
      srcStrides[d - 1] * (d - 1 < shape.length ? shape[d - 1] : 1);

  const dimOffsets: number[][] = dimIndices.map((indices, d) =>
    indices.map(idx => idx * srcStrides[d])
  );

  const subs = new Array(ndimIdx).fill(0);
  let srcLinear = 0;
  for (let d = 0; d < ndimIdx; d++) srcLinear += dimOffsets[d][0];
  for (let i = 0; i < totalElems; i++) {
    resultData[i] = base.data[srcLinear];
    if (resultImag && base.imag) {
      resultImag[i] = base.imag[srcLinear];
    }
    for (let d = 0; d < ndimIdx; d++) {
      const prev = subs[d];
      subs[d]++;
      if (subs[d] < dimIndices[d].length) {
        srcLinear += dimOffsets[d][subs[d]] - dimOffsets[d][prev];
        break;
      }
      srcLinear -= dimOffsets[d][prev] - dimOffsets[d][0];
      subs[d] = 0;
    }
  }

  return RTV.tensor(resultData, resultShape, resultImag);
}

function indexIntoCell(
  base: RuntimeCell,
  indices: RuntimeValue[]
): RuntimeValue {
  if (indices.length === 1) {
    return indexIntoCell1D(base, indices[0]);
  }
  if (indices.length === 2) {
    return indexIntoCell2D(base, indices[0], indices[1]);
  }
  throw new RuntimeError("Cell indexing supports 1 or 2 dimensions");
}

function indexIntoCell1D(base: RuntimeCell, idx: RuntimeValue): RuntimeValue {
  if (isColonIndex(idx)) {
    return RTV.cell([...base.data], [...base.shape]);
  }

  if (isRuntimeTensor(idx)) {
    const selected: RuntimeValue[] = [];
    if (idx._isLogical) {
      for (let k = 0; k < idx.data.length; k++) {
        if (idx.data[k] !== 0) {
          if (k >= base.data.length)
            throw new RuntimeError("Cell index exceeds bounds");
          selected.push(base.data[k]);
        }
      }
    } else {
      for (let k = 0; k < idx.data.length; k++) {
        const i = Math.round(idx.data[k]) - 1;
        if (i < 0 || i >= base.data.length)
          throw new RuntimeError("Cell index exceeds bounds");
        selected.push(base.data[i]);
      }
    }
    const isVector =
      idx.shape.length <= 2 &&
      (idx.shape[0] === 1 || (idx.shape.length === 2 && idx.shape[1] === 1));
    if (idx._isLogical || isVector) {
      if (base.shape[1] === 1 && base.shape[0] > 1) {
        return RTV.cell(selected, [selected.length, 1]);
      }
      return RTV.cell(selected, [1, selected.length]);
    }
    return RTV.cell(selected, [...idx.shape]);
  }

  // Scalar index — paren indexing on cells returns a 1x1 cell
  const i = Math.round(toNumber(idx)) - 1;
  if (i < 0 || i >= base.data.length)
    throw new RuntimeError("Cell index exceeds bounds");
  return RTV.cell([base.data[i]], [1, 1]);
}

function indexIntoCell2D(
  base: RuntimeCell,
  rowIdx: RuntimeValue,
  colIdx: RuntimeValue
): RuntimeValue {
  const rows = base.shape[0];
  const cols = base.shape.length >= 2 ? base.shape[1] : 1;

  const { rowIndices, colIndices } = resolveIndex2D(rowIdx, colIdx, rows, cols);

  // When both indices are plain scalars (no colon/tensor), return a 1x1 cell.
  if (
    !isColonIndex(rowIdx) &&
    !isColonIndex(colIdx) &&
    !isRuntimeTensor(rowIdx) &&
    !isRuntimeTensor(colIdx)
  ) {
    const linearIdx = colIndices[0] * rows + rowIndices[0];
    if (linearIdx < 0 || linearIdx >= base.data.length)
      throw new RuntimeError("Cell index exceeds bounds");
    return RTV.cell([base.data[linearIdx]], [1, 1]);
  }

  const result: RuntimeValue[] = [];
  for (let cj = 0; cj < colIndices.length; cj++) {
    for (let ri = 0; ri < rowIndices.length; ri++) {
      const linearIdx = colIndices[cj] * rows + rowIndices[ri];
      if (linearIdx < 0 || linearIdx >= base.data.length)
        throw new RuntimeError("Cell index exceeds bounds");
      result.push(base.data[linearIdx]);
    }
  }
  return RTV.cell(result, [rowIndices.length, colIndices.length]);
}

function indexIntoChar(
  base: RuntimeChar,
  indices: RuntimeValue[]
): RuntimeValue {
  const charShape =
    base.shape ?? (base.value.length === 0 ? [0, 0] : [1, base.value.length]);
  const nRows = charShape[0];
  const nCols = charShape[1] || 0;

  if (indices.length === 1) {
    const idx = indices[0];
    if (isColonIndex(idx)) return base;
    if (isRuntimeTensor(idx)) {
      let result = "";
      for (let k = 0; k < idx.data.length; k++) {
        const i = Math.round(idx.data[k]) - 1;
        if (i < 0 || i >= base.value.length)
          throw new RuntimeError("Index exceeds char array length");
        result += base.value[i];
      }
      return RTV.char(result);
    }
    const i = Math.round(toNumber(idx)) - 1;
    if (i < 0 || i >= base.value.length)
      throw new RuntimeError("Index exceeds char array length");
    return RTV.char(base.value[i]);
  }

  if (indices.length === 2) {
    const rowIdx = indices[0];
    const colIdx = indices[1];

    let rows: number[];
    if (isColonIndex(rowIdx)) {
      rows = Array.from({ length: nRows }, (_, i) => i);
    } else if (isRuntimeTensor(rowIdx)) {
      rows = Array.from(rowIdx.data, (v: number) => Math.round(v) - 1);
    } else {
      rows = [Math.round(toNumber(rowIdx)) - 1];
    }

    let cols: number[];
    if (isColonIndex(colIdx)) {
      cols = Array.from({ length: nCols }, (_, i) => i);
    } else if (isRuntimeTensor(colIdx)) {
      cols = Array.from(colIdx.data, (v: number) => Math.round(v) - 1);
    } else {
      cols = [Math.round(toNumber(colIdx)) - 1];
    }

    let result = "";
    for (const r of rows) {
      for (const c of cols) {
        if (r < 0 || r >= nRows || c < 0 || c >= nCols)
          throw new RuntimeError("Index exceeds char array dimensions");
        result += base.value[r * nCols + c];
      }
    }

    if (rows.length <= 1) return RTV.char(result);
    const resultChar: RuntimeChar = {
      kind: "char",
      value: result,
      shape: [rows.length, cols.length],
    };
    return resultChar;
  }

  throw new RuntimeError("Char indexing supports 1 or 2 dimensions");
}

function indexIntoLogical(
  base: boolean,
  indices: RuntimeValue[]
): RuntimeValue {
  if (indices.length === 1) {
    const idx = indices[0];
    if (isColonIndex(idx)) return base;
    if (isRuntimeTensor(idx) && idx.data.length === 0) {
      return RTV.tensor(new FloatXArray(0), [0, 0]);
    }
    if (isRuntimeTensor(idx)) {
      for (let i = 0; i < idx.data.length; i++) {
        const vi = Math.round(idx.data[i]);
        if (vi !== 1) throw new RuntimeError("Index exceeds array bounds");
      }
      const data = new FloatXArray(idx.data.length);
      data.fill(base ? 1 : 0);
      const result = RTV.tensor(data, [1, idx.data.length]);
      result._isLogical = true;
      return result;
    }
    if (isRuntimeLogical(idx)) {
      if (!idx) return RTV.tensor(new FloatXArray(0), [0, 0]);
      return base;
    }
    const i = Math.round(toNumber(idx));
    if (i !== 1) throw new RuntimeError("Index exceeds array bounds");
    return base;
  }
  // Multi-dimensional: each index must select only position 1
  for (const idx of indices) {
    if (isColonIndex(idx)) continue;
    if (isRuntimeLogical(idx)) {
      if (!idx) return RTV.tensor(new FloatXArray(0), [0, 0]);
      continue;
    }
    const i = Math.round(toNumber(idx));
    if (i !== 1) throw new RuntimeError("Index exceeds array bounds");
  }
  return base;
}

function indexIntoTensorWithTensor(
  base: RuntimeTensor,
  idx: RuntimeTensor
): RuntimeValue {
  if (idx._isLogical) {
    const selected: number[] = [];
    const selectedIm: number[] = [];
    const hasImag = base.imag !== undefined;
    for (let i = 0; i < idx.data.length; i++) {
      if (idx.data[i] !== 0) {
        selected.push(base.data[i]);
        if (hasImag) selectedIm.push(base.imag![i]);
      }
    }
    if (selected.length === 1) {
      if (hasImag && selectedIm[0] !== 0)
        return RTV.complex(selected[0], selectedIm[0]);
      return RTV.num(selected[0]);
    }
    const imOut =
      hasImag && selectedIm.some(x => x !== 0)
        ? new FloatXArray(selectedIm)
        : undefined;
    const isRow = base.shape.length === 2 && base.shape[0] === 1;
    const outShape: number[] = isRow
      ? [1, selected.length]
      : [selected.length, 1];
    return RTV.tensor(new FloatXArray(selected), outShape, imOut);
  }
  // Numeric indexing
  const resultData: number[] = [];
  const hasImag = base.imag !== undefined;
  const imIndices: number[] = [];
  for (let i = 0; i < idx.data.length; i++) {
    const k = Math.round(idx.data[i]) - 1;
    if (k < 0 || k >= base.data.length)
      throw new RuntimeError("Index exceeds array bounds");
    resultData.push(base.data[k]);
    if (hasImag) imIndices.push(base.imag![k]);
  }
  const idxIs0x0 =
    idx.data.length === 0 &&
    idx.shape.length === 2 &&
    idx.shape[0] === 0 &&
    idx.shape[1] === 0;
  const baseIsVector =
    base.shape.length <= 2 &&
    (base.shape[0] === 1 || base.shape[1] === 1 || base.shape.length === 1);
  const outShape = idxIs0x0
    ? [0, 0]
    : baseIsVector
      ? base.shape[0] === 1
        ? [1, resultData.length]
        : [resultData.length, 1]
      : idx.shape;
  const imOut =
    hasImag && imIndices.some(x => x !== 0)
      ? new FloatXArray(imIndices)
      : undefined;
  return RTV.tensor(new FloatXArray(resultData), outShape, imOut);
}

// ── Index read: main dispatcher ──────────────────────────────────────────

/** Index into a value: v(i1, i2, ...) */
export function indexIntoRTValue(
  base: RuntimeValue,
  indices: RuntimeValue[]
): RuntimeValue {
  if (isRuntimeNumber(base) || isRuntimeComplexNumber(base)) {
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
  if (isRuntimeLogical(base)) {
    return indexIntoLogical(base, indices);
  }
  if (isRuntimeStruct(base) || isRuntimeClassInstance(base)) {
    if (indices.length === 1) {
      const i = Math.round(toNumber(indices[0]));
      if (i !== 1) throw new RuntimeError("Index exceeds struct dimensions");
      return base;
    }
  }
  if (isRuntimeSparseMatrix(base)) {
    return indexIntoSparse(base, indices);
  }
  if (isRuntimeStructArray(base)) {
    if (indices.length === 1) {
      const idx = indices[0];
      if (isRuntimeNumber(idx)) {
        const i = Math.round(idx) - 1;
        if (i < 0 || i >= base.elements.length)
          throw new RuntimeError("Index exceeds struct array bounds");
        return base.elements[i];
      }
      if (isRuntimeTensor(idx)) {
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

// ── Store: per-type handlers ─────────────────────────────────────────────

function storeIntoTensor(
  base: RuntimeTensor,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeValue {
  // Element deletion: base(idx) = []
  if (isRuntimeTensor(rhs) && rhs.data.length === 0 && indices.length === 1) {
    return deleteTensorElements(base, indices[0]);
  }

  // Row/column deletion: base(idx,:) = [] or base(:,idx) = []
  if (isRuntimeTensor(rhs) && rhs.data.length === 0 && indices.length === 2) {
    // If either index selects zero elements, it's a no-op, not a deletion
    const nrows = base.shape[0] ?? 1;
    const ncols = base.shape.length >= 2 ? base.shape[1] : 1;
    const rowCount = isColonIndex(indices[0])
      ? nrows
      : resolveIndex(indices[0], nrows).length;
    const colCount = isColonIndex(indices[1])
      ? ncols
      : resolveIndex(indices[1], ncols).length;
    if (rowCount === 0 || colCount === 0) return base;
    return deleteTensorRowsOrCols(base, indices);
  }

  // COW: if data is shared, copy before mutating
  if (base._rc > 1) {
    base._rc--;
    const cowImag = base.imag ? new FloatXArray(base.imag) : undefined;
    base = RTV.tensor(new FloatXArray(base.data), [...base.shape], cowImag);
  }

  if (indices.length === 1) {
    return storeIntoTensor1D(base, indices[0], rhs);
  }
  if (indices.length === 2) {
    return storeIntoTensor2D(base, indices, rhs);
  }
  if (indices.length >= 3) {
    return storeIntoTensorND(base, indices, rhs);
  }

  throw new RuntimeError("Invalid number of indices for tensor assignment");
}

function deleteTensorElements(
  base: RuntimeTensor,
  idx: RuntimeValue
): RuntimeValue {
  const toDelete = new Set<number>();
  if (isRuntimeNumber(idx)) {
    toDelete.add(Math.round(idx) - 1);
  } else if (isRuntimeLogical(idx)) {
    if (idx) toDelete.add(0);
  } else if (isRuntimeTensor(idx) && idx._isLogical) {
    for (let i = 0; i < idx.data.length; i++) {
      if (idx.data[i] !== 0) toDelete.add(i);
    }
  } else if (isRuntimeTensor(idx)) {
    for (let i = 0; i < idx.data.length; i++) {
      toDelete.add(Math.round(idx.data[i]) - 1);
    }
  } else if (isColonIndex(idx)) {
    return RTV.tensor(new FloatXArray(0), [0, 0]);
  }
  const newData: number[] = [];
  const newIm: number[] = [];
  const hasImag = base.imag !== undefined;
  for (let i = 0; i < base.data.length; i++) {
    if (!toDelete.has(i)) {
      newData.push(base.data[i]);
      if (hasImag) newIm.push(base.imag![i]);
    }
  }
  const baseIsColVec =
    base.shape.length >= 2 && base.shape[1] === 1 && base.shape[0] !== 1;
  const outShape = baseIsColVec ? [newData.length, 1] : [1, newData.length];
  const imOut =
    hasImag && newIm.some(x => x !== 0) ? new FloatXArray(newIm) : undefined;
  return RTV.tensor(new FloatXArray(newData), outShape, imOut);
}

/** Collect a set of 0-based indices to delete from a dimension. */
function collectDelIndices(idx: RuntimeValue, dimLen: number): Set<number> {
  const s = new Set<number>();
  if (isRuntimeNumber(idx)) {
    s.add(Math.round(idx) - 1);
  } else if (isRuntimeLogical(idx)) {
    if (idx) s.add(0);
  } else if (isRuntimeTensor(idx) && idx._isLogical) {
    for (let i = 0; i < idx.data.length; i++) {
      if (idx.data[i] !== 0) s.add(i);
    }
  } else if (isRuntimeTensor(idx)) {
    for (let i = 0; i < idx.data.length; i++)
      s.add(Math.round(idx.data[i]) - 1);
  } else if (isColonIndex(idx)) {
    for (let i = 0; i < dimLen; i++) s.add(i);
  }
  return s;
}

function deleteTensorRowsOrCols(
  base: RuntimeTensor,
  indices: RuntimeValue[]
): RuntimeValue {
  const nrows = base.shape[0] ?? 1;
  const ncols = base.shape.length >= 2 ? base.shape[1] : 1;
  const hasImag = base.imag !== undefined;

  if (isColonIndex(indices[1])) {
    // base(rowIdx,:) = [] — delete rows
    const delRows = collectDelIndices(indices[0], nrows);
    const keepRows = Array.from({ length: nrows }, (_, i) => i).filter(
      i => !delRows.has(i)
    );
    const newNrows = keepRows.length;
    const newData = new FloatXArray(newNrows * ncols);
    const newIm = hasImag ? new FloatXArray(newNrows * ncols) : undefined;
    for (let j = 0; j < ncols; j++) {
      for (let ki = 0; ki < keepRows.length; ki++) {
        const srcIdx = keepRows[ki] + j * nrows;
        const dstIdx = ki + j * newNrows;
        newData[dstIdx] = base.data[srcIdx];
        if (newIm && base.imag) newIm[dstIdx] = base.imag[srcIdx];
      }
    }
    return RTV.tensor(newData, [newNrows, ncols], newIm);
  }
  if (isColonIndex(indices[0])) {
    // base(:,colIdx) = [] — delete columns
    const delCols = collectDelIndices(indices[1], ncols);
    const keepCols = Array.from({ length: ncols }, (_, j) => j).filter(
      j => !delCols.has(j)
    );
    const newNcols = keepCols.length;
    const newData = new FloatXArray(nrows * newNcols);
    const newIm = hasImag ? new FloatXArray(nrows * newNcols) : undefined;
    for (let ki = 0; ki < keepCols.length; ki++) {
      const srcCol = keepCols[ki];
      for (let i = 0; i < nrows; i++) {
        const srcIdx = i + srcCol * nrows;
        const dstIdx = i + ki * nrows;
        newData[dstIdx] = base.data[srcIdx];
        if (newIm && base.imag) newIm[dstIdx] = base.imag[srcIdx];
      }
    }
    return RTV.tensor(newData, [nrows, newNcols], newIm);
  }
  throw new RuntimeError("Cannot delete from both row and column dimensions");
}

function storeIntoTensor1D(
  base: RuntimeTensor,
  idx: RuntimeValue,
  rhs: RuntimeValue
): RuntimeValue {
  // Colon: base(:) = rhs
  if (isColonIndex(idx)) {
    if (isRuntimeTensor(rhs)) {
      if (rhs.data.length === 1) {
        // 1×1 tensor acts as scalar — broadcast to all elements
        rhs =
          rhs.imag && rhs.imag[0] !== 0
            ? RTV.complex(rhs.data[0], rhs.imag[0])
            : RTV.num(rhs.data[0]);
      } else {
        if (rhs.data.length !== base.data.length) {
          throw new RuntimeError("Subscripted assignment dimension mismatch");
        }
        base.data.set(rhs.data);
        if (rhs.imag || base.imag) {
          ensureImag(base);
          if (rhs.imag) base.imag!.set(rhs.imag);
          else base.imag!.fill(0);
        }
        return base;
      }
    }
    const { re, im } = toReIm(rhs);
    base.data.fill(re);
    if (im !== 0 || base.imag) {
      ensureImag(base);
      base.imag!.fill(im);
    }
    return base;
  }

  // Vector index: base(idx) = rhs where idx is a tensor
  if (isRuntimeTensor(idx)) {
    return storeIntoTensorByVector(base, idx, rhs);
  }

  // Logical scalar
  if (isRuntimeLogical(idx)) {
    if (!idx) return base;
    const { re: rhsRe, im: rhsIm } = toReIm(rhs);
    if (base.data.length < 1)
      throw new RuntimeError("Index exceeds array bounds");
    base.data[0] = rhsRe;
    if (rhsIm !== 0 || base.imag) {
      ensureImag(base);
      base.imag![0] = rhsIm;
    }
    return base;
  }

  // Scalar index with auto-grow
  const { re: rhsRe, im: rhsIm } = toReIm(rhs);
  const i = Math.round(toNumber(idx)) - 1;
  if (i < 0) throw new RuntimeError("Index exceeds array bounds");
  if (i >= base.data.length) {
    const grown = new FloatXArray(i + 1);
    grown.set(base.data);
    grown[i] = rhsRe;
    let grownImag: InstanceType<typeof FloatXArray> | undefined;
    if (rhsIm !== 0 || base.imag) {
      grownImag = new FloatXArray(i + 1);
      if (base.imag) grownImag.set(base.imag);
      grownImag[i] = rhsIm;
    }
    return RTV.tensor(grown, [1, i + 1], grownImag);
  }
  base.data[i] = rhsRe;
  if (rhsIm !== 0 || base.imag) {
    ensureImag(base);
    base.imag![i] = rhsIm;
  }
  return base;
}

function storeIntoTensorByVector(
  base: RuntimeTensor,
  idx: RuntimeTensor,
  rhs: RuntimeValue
): RuntimeValue {
  // Logical indexing: base(mask) = rhs
  if (idx._isLogical) {
    const { re: rhsRe, im: rhsIm } = isRuntimeTensor(rhs)
      ? { re: null, im: null }
      : toReIm(rhs);
    let k = 0;
    for (let i = 0; i < idx.data.length; i++) {
      if (idx.data[i] !== 0) {
        if (isRuntimeTensor(rhs)) {
          base.data[i] = rhs.data[k];
          if (rhs.imag || base.imag) {
            ensureImag(base);
            base.imag![i] = rhs.imag ? rhs.imag[k] : 0;
          }
        } else {
          base.data[i] = rhsRe!;
          if (rhsIm !== 0 || base.imag) {
            ensureImag(base);
            base.imag![i] = rhsIm!;
          }
        }
        k++;
      }
    }
    if (isRuntimeTensor(rhs) && k !== rhs.data.length) {
      throw new RuntimeError("Subscripted assignment dimension mismatch");
    }
    return base;
  }

  // Auto-grow if any index exceeds bounds
  let maxLi = -1;
  for (let i = 0; i < idx.data.length; i++) {
    const li = Math.round(idx.data[i]) - 1;
    if (li < 0) throw new RuntimeError("Index exceeds array bounds");
    if (li > maxLi) maxLi = li;
  }
  if (maxLi >= base.data.length) {
    const newLen = maxLi + 1;
    const grown = new FloatXArray(newLen);
    grown.set(base.data);
    let grownImag: InstanceType<typeof FloatXArray> | undefined;
    if (base.imag) {
      grownImag = new FloatXArray(newLen);
      grownImag.set(base.imag);
    }
    const isColVec =
      base.shape.length >= 2 && base.shape[1] === 1 && base.shape[0] > 1;
    const newShape = isColVec ? [newLen, 1] : [1, newLen];
    base = RTV.tensor(grown, newShape, grownImag);
  }

  if (isRuntimeTensor(rhs)) {
    if (idx.data.length !== rhs.data.length) {
      throw new RuntimeError("Subscripted assignment dimension mismatch");
    }
    if (rhs.imag || base.imag) ensureImag(base);
    for (let i = 0; i < idx.data.length; i++) {
      const li = Math.round(idx.data[i]) - 1;
      base.data[li] = rhs.data[i];
      if (base.imag) base.imag[li] = rhs.imag ? rhs.imag[i] : 0;
    }
    return base;
  }
  // Scalar RHS assigned to multiple positions
  const { re, im } = toReIm(rhs);
  if (im !== 0 || base.imag) ensureImag(base);
  for (let i = 0; i < idx.data.length; i++) {
    const li = Math.round(idx.data[i]) - 1;
    base.data[li] = re;
    if (base.imag) base.imag[li] = im;
  }
  return base;
}

function storeIntoTensor2D(
  base: RuntimeTensor,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeValue {
  const [rows, cols] = tensorSize2D(base);
  const rowIsColon = isColonIndex(indices[0]);
  const colIsColon = isColonIndex(indices[1]);
  const rowIsTensor = isRuntimeTensor(indices[0]);
  const colIsTensor = isRuntimeTensor(indices[1]);

  if (rowIsColon && colIsColon) {
    // base(:, :) = rhs — assign entire matrix
    if (isRuntimeTensor(rhs)) {
      if (rhs.data.length !== base.data.length) {
        throw new RuntimeError("Subscripted assignment dimension mismatch");
      }
      base.data.set(rhs.data);
      if (rhs.imag || base.imag) {
        ensureImag(base);
        if (rhs.imag) base.imag!.set(rhs.imag);
        else base.imag!.fill(0);
      }
      return base;
    }
    const { re, im } = toReIm(rhs);
    base.data.fill(re);
    if (im !== 0 || base.imag) {
      ensureImag(base);
      base.imag!.fill(im);
    }
    return base;
  }

  if (rowIsColon && !colIsColon && !colIsTensor) {
    return storeIntoTensorColonRow(base, indices[1], rhs, rows, cols);
  }

  if (!rowIsColon && !rowIsTensor && colIsColon) {
    return storeIntoTensorColonCol(base, indices[0], rhs, rows, cols);
  }

  // Handle tensor indices (ranges): base(rowRange, colRange) = rhs
  if (rowIsTensor || colIsTensor) {
    const rowIndices = resolveIndex(indices[0], rows, 0);
    const colIndices = resolveIndex(indices[1], cols, 0);

    const maxRow = rowIndices.length > 0 ? Math.max(...rowIndices) + 1 : rows;
    const maxCol = colIndices.length > 0 ? Math.max(...colIndices) + 1 : cols;
    base = growTensor2D(base, maxRow, maxCol);
    const [curRows] = tensorSize2D(base);

    assignSlice(base, rowIndices, colIndices, rhs, curRows);
    return base;
  }

  // Both are scalar indices
  if (
    (isRuntimeLogical(indices[0]) && !indices[0]) ||
    (isRuntimeLogical(indices[1]) && !indices[1])
  ) {
    return base;
  }
  const { re: rhsRe, im: rhsIm } = toReIm(rhs);
  const r = isRuntimeLogical(indices[0])
    ? 0
    : Math.round(toNumber(indices[0])) - 1;
  const c = isRuntimeLogical(indices[1])
    ? 0
    : Math.round(toNumber(indices[1])) - 1;
  if (r < 0 || c < 0) throw new RuntimeError("Index exceeds array bounds");

  base = growTensor2D(base, r + 1, c + 1);
  const [scRows] = tensorSize2D(base);
  const linearIdx = colMajorIndex(r, c, scRows);
  base.data[linearIdx] = rhsRe;
  if (rhsIm !== 0 || base.imag) {
    ensureImag(base);
    base.imag![linearIdx] = rhsIm;
  }
  return base;
}

/**
 * Assign a stripe of values along one dimension of a tensor.
 * Unlike assignSlice, this checks element count rather than shape — MATLAB
 * allows `A(:,c) = row_vector` and `A(r,:) = col_vector`.
 * `positions` are the 0-based linear indices into `base.data`.
 */
function assignStripe(
  base: RuntimeTensor,
  positions: number[],
  rhs: RuntimeValue
): void {
  if (isRuntimeTensor(rhs)) {
    if (rhs.data.length !== positions.length) {
      throw new RuntimeError("Subscripted assignment dimension mismatch");
    }
    if (rhs.imag || base.imag) ensureImag(base);
    for (let i = 0; i < positions.length; i++) {
      base.data[positions[i]] = rhs.data[i];
      if (base.imag) base.imag[positions[i]] = rhs.imag ? rhs.imag[i] : 0;
    }
  } else {
    const { re, im } = toReIm(rhs);
    if (im !== 0 || base.imag) ensureImag(base);
    for (const li of positions) {
      base.data[li] = re;
      if (base.imag) base.imag[li] = im;
    }
  }
}

/** base(:, c) = rhs — assign/grow entire column */
function storeIntoTensorColonRow(
  base: RuntimeTensor,
  colIdx: RuntimeValue,
  rhs: RuntimeValue,
  rows: number,
  cols: number
): RuntimeValue {
  // Logical scalar column: false → no-op, true → col 1
  if (isRuntimeLogical(colIdx)) {
    if (!colIdx) return base;
    const positions = Array.from({ length: rows }, (_, r) =>
      colMajorIndex(r, 0, rows)
    );
    assignStripe(base, positions, rhs);
    return base;
  }
  const c = Math.round(toNumber(colIdx)) - 1;
  if (c < 0) throw new RuntimeError("Index exceeds array bounds");
  let curRows = rows;
  let curCols = cols;
  // When base is empty, infer dimensions from RHS
  if (curRows === 0 && isRuntimeTensor(rhs) && rhs.data.length > 0) {
    curRows = rhs.data.length;
    const newCols = c + 1;
    base = RTV.tensor(new FloatXArray(curRows * newCols), [curRows, newCols]);
    curCols = newCols;
  } else if (curRows === 0 && (isRuntimeNumber(rhs) || isRuntimeLogical(rhs))) {
    curRows = 1;
    const newCols = c + 1;
    base = RTV.tensor(new FloatXArray(curRows * newCols), [curRows, newCols]);
    curCols = newCols;
  }
  if (c >= curCols) {
    base = growTensor2D(base, curRows, c + 1);
  }
  const [finalRows] = tensorSize2D(base);
  const positions = Array.from({ length: finalRows }, (_, r) =>
    colMajorIndex(r, c, finalRows)
  );
  assignStripe(base, positions, rhs);
  return base;
}

/** base(r, :) = rhs — assign/grow entire row */
function storeIntoTensorColonCol(
  base: RuntimeTensor,
  rowIdx: RuntimeValue,
  rhs: RuntimeValue,
  rows: number,
  cols: number
): RuntimeValue {
  // Logical scalar row: false → no-op, true → row 1
  if (isRuntimeLogical(rowIdx)) {
    if (!rowIdx) return base;
    const positions = Array.from({ length: cols }, (_, c) =>
      colMajorIndex(0, c, rows)
    );
    assignStripe(base, positions, rhs);
    return base;
  }
  const r = Math.round(toNumber(rowIdx)) - 1;
  if (r < 0) throw new RuntimeError("Index exceeds array bounds");
  let curRows = rows;
  let curCols = cols;
  // When base is empty, infer dimensions from RHS
  if (curCols === 0 && isRuntimeTensor(rhs) && rhs.data.length > 0) {
    curCols = rhs.data.length;
    const newRows = r + 1;
    base = RTV.tensor(new FloatXArray(newRows * curCols), [newRows, curCols]);
    curRows = newRows;
  } else if (curCols === 0 && (isRuntimeNumber(rhs) || isRuntimeLogical(rhs))) {
    curCols = 1;
    const newRows = r + 1;
    base = RTV.tensor(new FloatXArray(newRows * curCols), [newRows, curCols]);
    curRows = newRows;
  }
  if (r >= curRows) {
    base = growTensor2D(base, r + 1, curCols);
  }
  const [finalRows, finalCols] = tensorSize2D(base);
  const positions = Array.from({ length: finalCols }, (_, c) =>
    colMajorIndex(r, c, finalRows)
  );
  assignStripe(base, positions, rhs);
  return base;
}

function storeIntoTensorND(
  base: RuntimeTensor,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeValue {
  let shape = [...base.shape];

  const rhsTensor = isRuntimeTensor(rhs) ? rhs : null;
  const rhsShape = rhsTensor ? rhsTensor.shape : null;

  let rhsDimCursor = 0;
  const dimIndices: number[][] = indices.map((idx, dim) => {
    const dimSize = dim < shape.length ? shape[dim] : 1;
    if (isColonIndex(idx)) {
      if (dimSize === 0 && rhsShape) {
        const rDim =
          rhsDimCursor < rhsShape.length ? rhsShape[rhsDimCursor] : 1;
        rhsDimCursor++;
        return Array.from({ length: rDim }, (_, i) => i);
      }
      rhsDimCursor++;
      return Array.from({ length: dimSize }, (_, i) => i);
    }
    return resolveIndex(idx, dimSize, 0);
  });

  // Grow base if any index exceeds current dims
  const requiredShape = [...shape];
  while (requiredShape.length < indices.length) requiredShape.push(1);
  let needGrow = false;
  for (let d = 0; d < dimIndices.length; d++) {
    for (const idx of dimIndices[d]) {
      if (idx >= requiredShape[d]) {
        requiredShape[d] = idx + 1;
        needGrow = true;
      }
    }
  }
  if (needGrow || base.data.length === 0) {
    const newTotal = requiredShape.reduce((a, b) => a * b, 1);
    const newData = new FloatXArray(newTotal);
    const newImag = base.imag ? new FloatXArray(newTotal) : undefined;
    if (base.data.length > 0) {
      const oldShape = [...shape];
      while (oldShape.length < requiredShape.length) oldShape.push(1);
      const oldTotal = base.data.length;
      const ndimGrow = oldShape.length;
      const newStrides = new Array(ndimGrow);
      newStrides[0] = 1;
      for (let d = 1; d < ndimGrow; d++)
        newStrides[d] = newStrides[d - 1] * requiredShape[d - 1];
      const growSubs = new Array(ndimGrow).fill(0);
      let dstIdx = 0;
      for (let i = 0; i < oldTotal; i++) {
        newData[dstIdx] = base.data[i];
        if (newImag && base.imag) newImag[dstIdx] = base.imag[i];
        for (let d = 0; d < ndimGrow; d++) {
          growSubs[d]++;
          dstIdx += newStrides[d];
          if (growSubs[d] < oldShape[d]) break;
          dstIdx -= growSubs[d] * newStrides[d];
          growSubs[d] = 0;
        }
      }
    }
    while (
      requiredShape.length > 2 &&
      requiredShape[requiredShape.length - 1] === 1
    ) {
      requiredShape.pop();
    }
    base = RTV.tensor(newData, requiredShape, newImag);
    shape = requiredShape;
  }

  const resultShape = dimIndices.map(d => d.length);
  const totalElems = resultShape.reduce((a, b) => a * b, 1);

  const ndimStore = dimIndices.length;
  const dstStrides = new Array(ndimStore);
  dstStrides[0] = 1;
  for (let d = 1; d < ndimStore; d++)
    dstStrides[d] =
      dstStrides[d - 1] * (d - 1 < shape.length ? shape[d - 1] : 1);
  const dimOffsetsStore: number[][] = dimIndices.map((indices, d) =>
    indices.map(idx => idx * dstStrides[d])
  );

  // Iterate through N-D subscripts writing values
  const iterateND = (writeFn: (dstLinear: number, srcIdx: number) => void) => {
    const subs = new Array(ndimStore).fill(0);
    let dstLinear = 0;
    for (let d = 0; d < ndimStore; d++) dstLinear += dimOffsetsStore[d][0];
    for (let i = 0; i < totalElems; i++) {
      writeFn(dstLinear, i);
      for (let d = 0; d < ndimStore; d++) {
        const prev = subs[d];
        subs[d]++;
        if (subs[d] < dimIndices[d].length) {
          dstLinear += dimOffsetsStore[d][subs[d]] - dimOffsetsStore[d][prev];
          break;
        }
        dstLinear -= dimOffsetsStore[d][prev] - dimOffsetsStore[d][0];
        subs[d] = 0;
      }
    }
  };

  if (isRuntimeTensor(rhs)) {
    if (rhs.data.length !== totalElems) {
      throw new RuntimeError("Subscripted assignment dimension mismatch");
    }
    const hasRhsImag = rhs.imag !== undefined;
    if (hasRhsImag || base.imag) ensureImag(base);
    iterateND((dstLinear, srcIdx) => {
      base.data[dstLinear] = rhs.data[srcIdx];
      if (base.imag) base.imag[dstLinear] = hasRhsImag ? rhs.imag![srcIdx] : 0;
    });
  } else {
    const { re, im } = toReIm(rhs);
    if (im !== 0 || base.imag) ensureImag(base);
    iterateND(dstLinear => {
      base.data[dstLinear] = re;
      if (base.imag) base.imag[dstLinear] = im;
    });
  }
  return base;
}

function storeIntoCell(
  base: RuntimeCell,
  indices: RuntimeValue[],
  rhs: RuntimeValue,
  parenAssign = false
): RuntimeValue {
  // COW
  if (base._rc > 1) {
    base._rc--;
    const sharedData = base.data.map(elem => shareRuntimeValue(elem));
    base = RTV.cell(sharedData, [...base.shape]);
  }

  const updateShapeAfterLinearAssign = (oldLen: number) => {
    const c = base as RuntimeCell;
    const newLen = c.data.length;
    if (newLen === oldLen) return;
    const isColVec = c.shape[0] > 1 && c.shape[1] === 1;
    c.shape = isColVec ? [newLen, 1] : [1, newLen];
  };

  if (indices.length === 1) {
    return storeIntoCell1D(
      base,
      indices[0],
      rhs,
      updateShapeAfterLinearAssign,
      parenAssign
    );
  }
  if (indices.length === 2) {
    return storeIntoCell2D(base, indices, rhs, parenAssign);
  }

  throw new RuntimeError(`Cannot index-assign into cell`);
}

function storeIntoCell1D(
  base: RuntimeCell,
  idx: RuntimeValue,
  rhs: RuntimeValue,
  updateShape: (oldLen: number) => void,
  parenAssign = false
): RuntimeValue {
  // Vector index
  if (isRuntimeTensor(idx)) {
    let positions: number[];
    if (idx._isLogical) {
      positions = [];
      for (let i = 0; i < idx.data.length; i++) {
        if (idx.data[i] !== 0) positions.push(i);
      }
    } else {
      positions = Array.from(idx.data).map((v: number) => Math.round(v) - 1);
    }

    // Single-element index with non-cell RHS: treat as scalar assignment
    if (positions.length === 1 && !isRuntimeCell(rhs)) {
      const pos = positions[0];
      if (pos < 0) throw new RuntimeError("Cell index exceeds bounds");
      const oldLen = base.data.length;
      while (base.data.length <= pos)
        base.data.push(RTV.tensor(new FloatXArray(0), [0, 0]));
      base.data[pos] = rhs;
      updateShape(oldLen);
      return base;
    }
    if (
      !isRuntimeCell(rhs) ||
      (rhs.data.length !== positions.length && rhs.data.length !== 1)
    ) {
      throw new RuntimeError("Subscripted assignment dimension mismatch");
    }
    const scalarExpand = rhs.data.length === 1;
    const oldLen = base.data.length;
    let maxIdx = -1;
    for (let j = 0; j < positions.length; j++) {
      const pos = positions[j];
      if (pos < 0) throw new RuntimeError("Cell index exceeds bounds");
      if (pos > maxIdx) maxIdx = pos;
    }
    while (base.data.length <= maxIdx)
      base.data.push(RTV.tensor(new FloatXArray(0), [0, 0]));
    for (let j = 0; j < positions.length; j++) {
      const pos = positions[j];
      base.data[pos] = scalarExpand ? rhs.data[0] : rhs.data[j];
    }
    updateShape(oldLen);
    return base;
  }

  // Scalar index
  const i = Math.round(toNumber(idx)) - 1;
  if (i < 0) throw new RuntimeError("Cell index exceeds bounds");
  const oldLen = base.data.length;
  if (i >= base.data.length) {
    while (base.data.length <= i)
      base.data.push(RTV.tensor(new FloatXArray(0), [0, 0]));
  }
  // Unwrap 1x1 cell RHS only for paren assignment: c(i) = {val} stores val
  base.data[i] =
    parenAssign && isRuntimeCell(rhs) && rhs.data.length === 1
      ? rhs.data[0]
      : rhs;
  updateShape(oldLen);
  return base;
}

function storeIntoCell2D(
  base: RuntimeCell,
  indices: RuntimeValue[],
  rhs: RuntimeValue,
  parenAssign = false
): RuntimeValue {
  let rows = base.shape[0];
  let cols = base.shape.length >= 2 ? base.shape[1] : 1;

  // Resolve with boundsLimit=0 to allow auto-grow beyond current dims
  const rowIndices = resolveIndex(indices[0], rows, 0);
  const colIndices = resolveIndex(indices[1], cols, 0);

  // Auto-grow cell if needed
  const maxRow = Math.max(...rowIndices) + 1;
  const maxCol = Math.max(...colIndices) + 1;
  const newRows = Math.max(rows, maxRow);
  const newCols = Math.max(cols, maxCol);
  if (newRows > rows || newCols > cols) {
    const emptyVal = () => RTV.tensor(new FloatXArray(0), [0, 0]);
    const newData: RuntimeValue[] = new Array(newRows * newCols);
    for (let k = 0; k < newData.length; k++) newData[k] = emptyVal();
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        newData[j * newRows + i] = base.data[j * rows + i];
      }
    }
    base.data = newData;
    base.shape = [newRows, newCols];
    rows = newRows;
    cols = newCols;
  }

  const nSelectedRows = rowIndices.length;
  const nSelectedCols = colIndices.length;
  const totalSelected = nSelectedRows * nSelectedCols;

  if (parenAssign && isRuntimeCell(rhs)) {
    if (rhs.data.length !== totalSelected && rhs.data.length !== 1) {
      throw new RuntimeError("Subscripted assignment dimension mismatch");
    }
    const scalarExpand = rhs.data.length === 1;
    let k = 0;
    for (let cj = 0; cj < nSelectedCols; cj++) {
      for (let ri = 0; ri < nSelectedRows; ri++) {
        const linearIdx = colIndices[cj] * rows + rowIndices[ri];
        base.data[linearIdx] = scalarExpand ? rhs.data[0] : rhs.data[k++];
      }
    }
  } else {
    const linearIdx = colIndices[0] * rows + rowIndices[0];
    base.data[linearIdx] = rhs;
  }
  return base;
}

// ── Store: main dispatcher ───────────────────────────────────────────────

/** Store into indexed position: base(indices) = rhs — uses copy-on-write for shared data */
export function storeIntoRTValueIndex(
  base: RuntimeValue,
  indices: RuntimeValue[],
  rhs: RuntimeValue,
  parenAssign = false
): RuntimeValue {
  if (isRuntimeSparseMatrix(base)) {
    return storeIntoSparse(base, indices, rhs);
  }

  // Auto-convert sparse RHS to dense when assigning into a dense tensor
  if (isRuntimeSparseMatrix(rhs) && isRuntimeTensor(base)) {
    const S = rhs;
    const data = new FloatXArray(S.m * S.n);
    const imag = S.pi ? new FloatXArray(S.m * S.n) : undefined;
    for (let col = 0; col < S.n; col++) {
      for (let k = S.jc[col]; k < S.jc[col + 1]; k++) {
        const idx = col * S.m + S.ir[k];
        data[idx] = S.pr[k];
        if (imag && S.pi) imag[idx] = S.pi[k];
      }
    }
    rhs = { kind: "tensor", data, imag, shape: [S.m, S.n], _rc: 1 };
  }

  if (isRuntimeTensor(base)) {
    return storeIntoTensor(base, indices, rhs);
  }

  if (isRuntimeCell(base)) {
    return storeIntoCell(base, indices, rhs, parenAssign);
  }

  throw new RuntimeError(`Cannot index-assign into ${kstr(base)}`);
}
