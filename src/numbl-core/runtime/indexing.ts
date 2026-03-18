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

/** Check if an index is a colon sentinel */
export function isColonIndex(v: RuntimeValue): boolean {
  return isRuntimeString(v) && v === "__COLON__";
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
  // Build reverse lookup: original row → list of new row positions
  // (handles duplicate row indices correctly)
  const rowLookup = new Map<number, number[]>();
  for (let i = 0; i < nRows; i++) {
    const orig = rowIdx[i];
    const list = rowLookup.get(orig);
    if (list) list.push(i);
    else rowLookup.set(orig, [i]);
  }

  const isComplex = s.pi !== undefined;

  // Collect triplets
  const irArr: number[] = [];
  const prArr: number[] = [];
  const piArr: number[] = [];
  const jcNew = new Int32Array(nCols + 1);

  for (let ci = 0; ci < nCols; ci++) {
    const origCol = colIdx[ci];
    jcNew[ci] = irArr.length;
    // Collect entries in this column matching requested rows, in row order
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
    // Sort by new row index to maintain CSC invariant
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

/** Assign into a sparse matrix: S(rows, cols) = rhs.
 *  Supports scalar, vector, and submatrix assignment.
 *  Returns a new sparse matrix (COW). */
function storeIntoSparse(
  base: RuntimeSparseMatrix,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeSparseMatrix {
  if (indices.length === 1) {
    // Linear indexing: convert to 2D (row, col) pairs
    const totalLen = base.m * base.n;
    const linIdx = resolveIndex(indices[0], totalLen);
    // Convert each linear index to (row, col) and do element-wise assignment
    const rowIdx = linIdx.map(k => k % base.m);
    const colIdx = linIdx.map(k => Math.floor(k / base.m));
    // For linear indexing, each element maps independently, so we need
    // to assign one at a time to handle non-contiguous row/col pairs
    let result = base;
    const rhsVals = extractRhsValues(rhs, linIdx.length);
    for (let i = 0; i < linIdx.length; i++) {
      const r = rowIdx[i];
      const c = colIdx[i];
      result = storeIntoSparse(
        result,
        [RTV.num(r + 1), RTV.num(c + 1)],
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

  // Extract RHS values into a dense grid (rowIdx.length × colIdx.length, col-major)
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
    // Scatter sparse RHS into dense grid
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

  // Strategy: convert base to a map, apply updates, rebuild CSC.
  // For large matrices this could be optimized, but correctness first.
  // Build a map: col -> sorted array of { row, re, im }
  type Entry = { row: number; re: number; im: number };
  const colEntries = new Map<number, Entry[]>();

  // Load existing entries
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

  // Apply updates
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
      // Find existing entry
      let found = false;
      for (let e = 0; e < entries.length; e++) {
        if (entries[e].row === row) {
          if (re === 0 && im === 0) {
            entries.splice(e, 1); // remove zero entry
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
    // Re-sort by row
    entries.sort((a, b) => a.row - b.row);
    if (entries.length === 0) colEntries.delete(col);
  }

  // Rebuild CSC arrays
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

/** Index into a sparse matrix. Returns sparse results (matching MATLAB behavior). */
function indexIntoSparse(
  base: RuntimeSparseMatrix,
  indices: RuntimeValue[]
): RuntimeValue {
  if (indices.length === 1) {
    const idx = indices[0];

    // S(:) → reshape to column vector (m*n × 1)
    if (isColonIndex(idx)) {
      const totalLen = base.m * base.n;
      const irArr: number[] = [];
      const prArr: number[] = [];
      const piArr: number[] = [];
      const isComplex = base.pi !== undefined;

      // Iterate in column-major order, mapping (row,col) → linear index
      for (let col = 0; col < base.n; col++) {
        for (let k = base.jc[col]; k < base.jc[col + 1]; k++) {
          const linIdx = col * base.m + base.ir[k];
          irArr.push(linIdx);
          prArr.push(base.pr[k]);
          if (isComplex) piArr.push(base.pi![k]);
        }
      }

      // Build single-column sparse: m*n × 1
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

    // S(k) — linear indexing (single scalar) → return plain double
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

    // S(vector) or S(logical_vector) — linear vector indexing
    if (isRuntimeTensor(idx) || isRuntimeLogical(idx)) {
      const totalLen = base.m * base.n;
      const linIdx = resolveIndex(idx, totalLen);
      const n = linIdx.length;
      const isComplex = base.pi !== undefined;
      const irArr: number[] = [];
      const prArr: number[] = [];
      const piArr: number[] = [];
      const jcNew = new Int32Array(n + 1);
      // Result is 1 × n sparse (row vector, one column per index)
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

    // Scalar indices → return plain double (matching MATLAB)
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

// ── Indexing ─────────────────────────────────────────────────────────────

/** Index into a value: v(i1, i2, ...) */
export function indexIntoRTValue(
  base: RuntimeValue,
  indices: RuntimeValue[]
): RuntimeValue {
  if (isRuntimeNumber(base) || isRuntimeComplexNumber(base)) {
    // If any index is an empty tensor, result is empty (e.g. scalar([], :) → empty)
    if (indices.some(idx => isRuntimeTensor(idx) && idx.data.length === 0)) {
      if (indices.length === 2) {
        // For 2D indexing, compute each dimension's size independently
        // scalar is implicitly 1x1, so colon or scalar index gives size 1
        const dimSize = (idx: RuntimeValue): number => {
          if (isColonIndex(idx)) return 1;
          if (isRuntimeTensor(idx)) return idx.data.length;
          return 1; // scalar index
        };
        return RTV.tensor(new FloatXArray(0), [
          dimSize(indices[0]),
          dimSize(indices[1]),
        ]);
      }
      // Single index case: preserve index shape
      const idx = indices[0];
      if (isRuntimeTensor(idx)) {
        const is0x0 =
          idx.shape.length === 2 && idx.shape[0] === 0 && idx.shape[1] === 0;
        const outShape = is0x0 ? [0, 0] : [...idx.shape];
        return RTV.tensor(new FloatXArray(0), outShape);
      }
    }
    // Handle logical indexing on scalar: scalar(false) → empty, scalar(true) → scalar
    if (indices.length === 1) {
      const idx = indices[0];
      if (isRuntimeLogical(idx)) {
        if (!idx) return RTV.tensor(new FloatXArray(0), [0, 0]);
        return base;
      }
      if (isRuntimeTensor(idx) && idx._isLogical) {
        // Logical tensor index on scalar: select element where mask is true
        let count = 0;
        for (let j = 0; j < idx.data.length; j++) if (idx.data[j]) count++;
        if (count === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
        const out = new FloatXArray(count);
        out.fill(isRuntimeNumber(base) ? base : base.re);
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

  if (isRuntimeTensor(base)) {
    if (indices.length === 1) {
      // Colon indexing: base(:) → column vector of all elements
      if (isColonIndex(indices[0])) {
        const imag = base.imag ? new FloatXArray(base.imag) : undefined;
        return RTV.tensor(
          new FloatXArray(base.data),
          [base.data.length, 1],
          imag
        );
      }

      // Linear indexing
      const idx = indices[0];
      if (isRuntimeLogical(idx)) {
        // Logical scalar: true → first element, false → empty
        if (!idx) {
          return RTV.tensor(new FloatXArray(0), [0, 0]);
        }
        // true → index 1 (first element)
        if (base.imag !== undefined) {
          const im = base.imag[0];
          return im === 0
            ? RTV.num(base.data[0])
            : RTV.complex(base.data[0], im);
        }
        return RTV.num(base.data[0]);
      }
      if (isRuntimeNumber(idx)) {
        const i = Math.round(toNumber(idx)) - 1;
        if (i < 0 || i >= base.data.length) {
          throw new RuntimeError("Index exceeds array bounds");
        }
        // Return complex scalar if tensor has imaginary part
        if (base.imag !== undefined) {
          const im = base.imag[i];
          return im === 0
            ? RTV.num(base.data[i])
            : RTV.complex(base.data[i], im);
        }
        return RTV.num(base.data[i]);
      }
      // Logical or tensor indexing → extract subset
      if (isRuntimeTensor(idx)) {
        return indexIntoTensorWithTensor(base, idx);
      }
    } else if (indices.length === 2) {
      const [rows, cols] = tensorSize2D(base);
      const rowIsColon = isColonIndex(indices[0]);
      const colIsColon = isColonIndex(indices[1]);

      if (rowIsColon && colIsColon) {
        // base(:, :) — return copy of entire matrix
        const imag = base.imag ? new FloatXArray(base.imag) : undefined;
        return RTV.tensor(new FloatXArray(base.data), [...base.shape], imag);
      }

      if (rowIsColon && !colIsColon) {
        // base(:, c) or base(:, [c1 c2 ...])
        if (isRuntimeTensor(indices[1])) {
          // Multiple column indices: base(:, [c1 c2 ...])
          // If the index tensor is a logical mask, convert to 1-based indices.
          const rawColIdx = indices[1];
          const colIndices: number[] = [];
          if (rawColIdx._isLogical) {
            for (let i = 0; i < rawColIdx.data.length; i++) {
              if (rawColIdx.data[i] !== 0) colIndices.push(i + 1);
            }
          } else {
            for (let i = 0; i < rawColIdx.data.length; i++) {
              colIndices.push(rawColIdx.data[i]);
            }
          }
          const numCols = colIndices.length;
          const resultData = new FloatXArray(rows * numCols);
          const resultImag = base.imag
            ? new FloatXArray(rows * numCols)
            : undefined;
          for (let ci = 0; ci < numCols; ci++) {
            const c = Math.round(colIndices[ci]) - 1;
            if (c < 0 || c >= cols)
              throw new RuntimeError("Index exceeds array bounds");
            for (let r = 0; r < rows; r++) {
              const srcIdx = colMajorIndex(r, c, rows);
              const dstIdx = colMajorIndex(r, ci, rows);
              resultData[dstIdx] = base.data[srcIdx];
              if (resultImag && base.imag) {
                resultImag[dstIdx] = base.imag[srcIdx];
              }
            }
          }
          return RTV.tensor(resultData, [rows, numCols], resultImag);
        }
        // Logical scalar column index: true → col 1, false → empty
        if (isRuntimeLogical(indices[1])) {
          if (!indices[1]) {
            return RTV.tensor(new FloatXArray(0), [rows, 0]);
          }
          // true → first column (index 0)
          const colData = new FloatXArray(rows);
          const colImag = base.imag ? new FloatXArray(rows) : undefined;
          for (let r = 0; r < rows; r++) {
            const idx = colMajorIndex(r, 0, rows);
            colData[r] = base.data[idx];
            if (colImag && base.imag) {
              colImag[r] = base.imag[idx];
            }
          }
          return RTV.tensor(colData, [rows, 1], colImag);
        }
        // Single column: base(:, c)
        const c = Math.round(toNumber(indices[1])) - 1;
        if (c < 0 || c >= cols)
          throw new RuntimeError("Index exceeds array bounds");
        const colData = new FloatXArray(rows);
        const colImag = base.imag ? new FloatXArray(rows) : undefined;
        for (let r = 0; r < rows; r++) {
          const idx = colMajorIndex(r, c, rows);
          colData[r] = base.data[idx];
          if (colImag && base.imag) {
            colImag[r] = base.imag[idx];
          }
        }
        return RTV.tensor(colData, [rows, 1], colImag);
      }

      if (!rowIsColon && colIsColon) {
        // base(r, :) or base([r1 r2 ...], :)
        if (isRuntimeTensor(indices[0])) {
          // Multiple row indices: base([r1 r2 ...], :)
          // If the index tensor is a logical mask, convert to 1-based indices.
          const rawRowIdx = indices[0];
          const rowIndices: number[] = [];
          if (rawRowIdx._isLogical) {
            for (let i = 0; i < rawRowIdx.data.length; i++) {
              if (rawRowIdx.data[i] !== 0) rowIndices.push(i + 1);
            }
          } else {
            for (let i = 0; i < rawRowIdx.data.length; i++) {
              rowIndices.push(rawRowIdx.data[i]);
            }
          }
          const numRows = rowIndices.length;
          const resultData = new FloatXArray(numRows * cols);
          const resultImag = base.imag
            ? new FloatXArray(numRows * cols)
            : undefined;
          for (let ri = 0; ri < numRows; ri++) {
            const r = Math.round(rowIndices[ri]) - 1;
            if (r < 0 || r >= rows)
              throw new RuntimeError("Index exceeds array bounds");
            for (let c = 0; c < cols; c++) {
              const srcIdx = colMajorIndex(r, c, rows);
              const dstIdx = colMajorIndex(ri, c, numRows);
              resultData[dstIdx] = base.data[srcIdx];
              if (resultImag && base.imag) {
                resultImag[dstIdx] = base.imag[srcIdx];
              }
            }
          }
          return RTV.tensor(resultData, [numRows, cols], resultImag);
        }
        // Logical scalar row index: true → row 1, false → empty
        if (isRuntimeLogical(indices[0])) {
          if (!indices[0]) {
            return RTV.tensor(new FloatXArray(0), [0, cols]);
          }
          // true → first row (index 0)
          const rowData = new FloatXArray(cols);
          const rowImag = base.imag ? new FloatXArray(cols) : undefined;
          for (let c = 0; c < cols; c++) {
            const idx = colMajorIndex(0, c, rows);
            rowData[c] = base.data[idx];
            if (rowImag && base.imag) {
              rowImag[c] = base.imag[idx];
            }
          }
          return RTV.tensor(rowData, [1, cols], rowImag);
        }
        // Single row: base(r, :)
        const r = Math.round(toNumber(indices[0])) - 1;
        if (r < 0 || r >= rows)
          throw new RuntimeError("Index exceeds array bounds");
        const rowData = new FloatXArray(cols);
        const rowImag = base.imag ? new FloatXArray(cols) : undefined;
        for (let c = 0; c < cols; c++) {
          const idx = colMajorIndex(r, c, rows);
          rowData[c] = base.data[idx];
          if (rowImag && base.imag) {
            rowImag[c] = base.imag[idx];
          }
        }
        return RTV.tensor(rowData, [1, cols], rowImag);
      }

      // General case: row and col can each be scalar or tensor index
      const rowIdxArr = resolveIndex(indices[0], rows);
      const colIdxArr = resolveIndex(indices[1], cols);

      const numR = rowIdxArr.length;
      const numC = colIdxArr.length;

      if (numR === 1 && numC === 1) {
        // Scalar result
        const linearIdx = colMajorIndex(rowIdxArr[0], colIdxArr[0], rows);
        if (base.imag !== undefined) {
          const im = base.imag[linearIdx];
          return im === 0
            ? RTV.num(base.data[linearIdx])
            : RTV.complex(base.data[linearIdx], im);
        }
        return RTV.num(base.data[linearIdx]);
      }

      // Submatrix result
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
    } else if (indices.length >= 3) {
      // General N-dimensional indexing
      const shape = base.shape;
      const dimIndices: number[][] = indices.map((idx, dim) => {
        const dimSize = dim < shape.length ? shape[dim] : 1;
        return resolveIndex(idx, dimSize);
      });

      // Check if result is scalar
      const allScalar = dimIndices.every(d => d.length === 1);
      if (allScalar) {
        const subs = dimIndices.map(d => d[0]);
        const linearIdx = sub2ind(shape, subs);
        if (base.imag !== undefined) {
          const im = base.imag[linearIdx];
          return im === 0
            ? RTV.num(base.data[linearIdx])
            : RTV.complex(base.data[linearIdx], im);
        }
        return RTV.num(base.data[linearIdx]);
      }

      // Compute result shape and total elements
      // Squeeze trailing singleton dimensions (keep at least 2)
      const resultShape = dimIndices.map(d => d.length);
      while (
        resultShape.length > 2 &&
        resultShape[resultShape.length - 1] === 1
      ) {
        resultShape.pop();
      }
      const totalElems = resultShape.reduce((a, b) => a * b, 1);
      const resultData = new FloatXArray(totalElems);
      const resultImag = base.imag ? new FloatXArray(totalElems) : undefined;

      // Precompute source strides (column-major)
      const ndimIdx = dimIndices.length;
      const srcStrides = new Array(ndimIdx);
      srcStrides[0] = 1;
      for (let d = 1; d < ndimIdx; d++)
        srcStrides[d] =
          srcStrides[d - 1] * (d - 1 < shape.length ? shape[d - 1] : 1);

      // Precompute flat-index offsets per dimension: srcStrides[d] * dimIndices[d][k]
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
        // Increment subscripts in column-major order, updating srcLinear
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
  }

  if (isRuntimeCell(base)) {
    if (indices.length === 1) {
      const idx = indices[0];

      // Colon: X(:) → return copy of entire cell
      if (isColonIndex(idx)) {
        return RTV.cell([...base.data], [...base.shape]);
      }

      // Vector/range index: X([1 3]) or X(1:2) → return sub-cell
      if (isRuntimeTensor(idx)) {
        const selected: RuntimeValue[] = [];
        if (idx._isLogical) {
          // Logical indexing: true positions select elements
          for (let k = 0; k < idx.data.length; k++) {
            if (idx.data[k] !== 0) {
              if (k >= base.data.length) {
                throw new RuntimeError("Cell index exceeds bounds");
              }
              selected.push(base.data[k]);
            }
          }
        } else {
          for (let k = 0; k < idx.data.length; k++) {
            const i = Math.round(idx.data[k]) - 1;
            if (i < 0 || i >= base.data.length) {
              throw new RuntimeError("Cell index exceeds bounds");
            }
            selected.push(base.data[i]);
          }
        }
        // Logical indexing or vector index: preserve base orientation
        const isVector =
          idx.shape.length <= 2 &&
          (idx.shape[0] === 1 ||
            (idx.shape.length === 2 && idx.shape[1] === 1));
        if (idx._isLogical || isVector) {
          if (base.shape[1] === 1 && base.shape[0] > 1) {
            return RTV.cell(selected, [selected.length, 1]);
          }
          return RTV.cell(selected, [1, selected.length]);
        }
        // Matrix index: preserve shape of index tensor
        return RTV.cell(selected, [...idx.shape]);
      }

      // Scalar index — paren indexing on cells returns a 1x1 cell
      const i = Math.round(toNumber(idx)) - 1;
      if (i < 0 || i >= base.data.length) {
        throw new RuntimeError("Cell index exceeds bounds");
      }
      return RTV.cell([base.data[i]], [1, 1]);
    }
    if (indices.length === 2) {
      const rows = base.shape[0];
      const cols = base.shape.length >= 2 ? base.shape[1] : 1;

      // Resolve row indices to 0-based array
      const rowIsColon = isColonIndex(indices[0]);
      let rowIndices: number[];
      if (rowIsColon) {
        rowIndices = Array.from({ length: rows }, (_, i) => i);
      } else if (isRuntimeTensor(indices[0])) {
        rowIndices = Array.from(indices[0].data).map(
          (v: number) => Math.round(v) - 1
        );
      } else {
        rowIndices = [Math.round(toNumber(indices[0])) - 1];
      }

      // Resolve column indices to 0-based array
      const colIsColon = isColonIndex(indices[1]);
      let colIndices: number[];
      if (colIsColon) {
        colIndices = Array.from({ length: cols }, (_, i) => i);
      } else if (isRuntimeTensor(indices[1])) {
        colIndices = Array.from(indices[1].data).map(
          (v: number) => Math.round(v) - 1
        );
      } else {
        colIndices = [Math.round(toNumber(indices[1])) - 1];
      }

      const nSelectedRows = rowIndices.length;
      const nSelectedCols = colIndices.length;

      // When both indices are plain scalars (no colon/tensor), return a 1x1 cell.
      if (
        !rowIsColon &&
        !colIsColon &&
        !isRuntimeTensor(indices[0]) &&
        !isRuntimeTensor(indices[1])
      ) {
        const linearIdx = colIndices[0] * rows + rowIndices[0];
        if (linearIdx < 0 || linearIdx >= base.data.length) {
          throw new RuntimeError("Cell index exceeds bounds");
        }
        return RTV.cell([base.data[linearIdx]], [1, 1]);
      }

      // Colon/tensor indices: return a cell sub-array (column-major order).
      // Cell paren-indexing always returns a cell.
      const result: RuntimeValue[] = [];
      for (let cj = 0; cj < nSelectedCols; cj++) {
        for (let ri = 0; ri < nSelectedRows; ri++) {
          const linearIdx = colIndices[cj] * rows + rowIndices[ri];
          if (linearIdx < 0 || linearIdx >= base.data.length) {
            throw new RuntimeError("Cell index exceeds bounds");
          }
          result.push(base.data[linearIdx]);
        }
      }
      return RTV.cell(result, [nSelectedRows, nSelectedCols]);
    }
  }

  if (isRuntimeChar(base)) {
    const charShape =
      base.shape ?? (base.value.length === 0 ? [0, 0] : [1, base.value.length]);
    const nRows = charShape[0];
    const nCols = charShape[1] || 0;

    if (indices.length === 1) {
      const idx = indices[0];
      // Colon: return entire char array
      if (isColonIndex(idx)) return base;
      // Tensor index (range or vector)
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
      // Scalar index
      const i = Math.round(toNumber(idx)) - 1;
      if (i < 0 || i >= base.value.length)
        throw new RuntimeError("Index exceeds char array length");
      return RTV.char(base.value[i]);
    }

    // 2D indexing: s(row, col)
    if (indices.length === 2) {
      const rowIdx = indices[0];
      const colIdx = indices[1];

      // Resolve row indices
      let rows: number[];
      if (isColonIndex(rowIdx)) {
        rows = Array.from({ length: nRows }, (_, i) => i);
      } else if (isRuntimeTensor(rowIdx)) {
        rows = Array.from(rowIdx.data, (v: number) => Math.round(v) - 1);
      } else {
        rows = [Math.round(toNumber(rowIdx)) - 1];
      }

      // Resolve col indices
      let cols: number[];
      if (isColonIndex(colIdx)) {
        cols = Array.from({ length: nCols }, (_, i) => i);
      } else if (isRuntimeTensor(colIdx)) {
        cols = Array.from(colIdx.data, (v: number) => Math.round(v) - 1);
      } else {
        cols = [Math.round(toNumber(colIdx)) - 1];
      }

      // Extract characters: multi-row char array stores rows contiguously
      // row r, col c => value[r * nCols + c]
      let result = "";
      for (const r of rows) {
        for (const c of cols) {
          if (r < 0 || r >= nRows || c < 0 || c >= nCols)
            throw new RuntimeError("Index exceeds char array dimensions");
          result += base.value[r * nCols + c];
        }
      }

      if (rows.length === 1 && cols.length === 1) {
        return RTV.char(result);
      }
      if (rows.length === 1) {
        return RTV.char(result);
      }
      // Multi-row result
      const resultChar: RuntimeChar = {
        kind: "char",
        value: result,
        shape: [rows.length, cols.length],
      };
      return resultChar;
    }
  }

  if (isRuntimeString(base)) {
    if (indices.length === 1) {
      // String scalar: only valid index is 1
      const i = Math.round(toNumber(indices[0]));
      if (i !== 1) throw new RuntimeError("Index exceeds string dimensions");
      return base;
    }
  }

  // Scalar logical: treated as 1x1 logical array for indexing
  if (isRuntimeLogical(base)) {
    if (indices.length === 1) {
      const idx = indices[0];
      // Colon: return self
      if (isColonIndex(idx)) return base;
      // Empty tensor: return empty
      if (isRuntimeTensor(idx) && idx.data.length === 0) {
        return RTV.tensor(new FloatXArray(0), [0, 0]);
      }
      // Tensor index: all indices must be 1; return logical array repeated
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
      // Logical scalar index: false → empty, true → self
      if (isRuntimeLogical(idx)) {
        if (!idx) return RTV.tensor(new FloatXArray(0), [0, 0]);
        return base;
      }
      // Numeric scalar index
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

  // Scalar struct/class_instance: s(1) returns s itself (scalars are 1-element arrays)
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
      // Scalar index: return single element as a struct
      if (isRuntimeNumber(idx)) {
        const i = Math.round(idx) - 1;
        if (i < 0 || i >= base.elements.length)
          throw new RuntimeError("Index exceeds struct array bounds");
        return base.elements[i];
      }
      // Tensor index (e.g. ind(2:end)): return a sub struct_array
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
      // Logical index
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

/** Store into indexed position: base(indices) = rhs — uses copy-on-write for shared data */
export function storeIntoRTValueIndex(
  base: RuntimeValue,
  indices: RuntimeValue[],
  rhs: RuntimeValue
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
    // Element deletion: base(idx) = [] removes elements at idx
    if (isRuntimeTensor(rhs) && rhs.data.length === 0 && indices.length === 1) {
      // Collect indices to delete (0-based)
      const toDelete = new Set<number>();
      const idx = indices[0];
      if (isRuntimeNumber(idx)) {
        toDelete.add(Math.round(idx) - 1);
      } else if (isRuntimeLogical(idx)) {
        if (idx) toDelete.add(0);
      } else if (isRuntimeTensor(idx) && idx._isLogical) {
        // Logical indexing: true positions are deleted
        for (let i = 0; i < idx.data.length; i++) {
          if (idx.data[i] !== 0) toDelete.add(i);
        }
      } else if (isRuntimeTensor(idx)) {
        for (let i = 0; i < idx.data.length; i++) {
          toDelete.add(Math.round(idx.data[i]) - 1);
        }
      } else if (isColonIndex(idx)) {
        // a(:) = [] → empty array
        return RTV.tensor(new FloatXArray(0), [0, 0]);
      }
      // Build new data without deleted indices (linear indexing)
      const newData: number[] = [];
      const newIm: number[] = [];
      const hasImag = base.imag !== undefined;
      for (let i = 0; i < base.data.length; i++) {
        if (!toDelete.has(i)) {
          newData.push(base.data[i]);
          if (hasImag) newIm.push(base.imag![i]);
        }
      }
      // Preserve vector orientation: column vector stays column, row stays row
      const baseIsColVec =
        base.shape.length >= 2 && base.shape[1] === 1 && base.shape[0] !== 1;
      const outShape = baseIsColVec ? [newData.length, 1] : [1, newData.length];
      const imOut =
        hasImag && newIm.some(x => x !== 0)
          ? new FloatXArray(newIm)
          : undefined;
      return RTV.tensor(new FloatXArray(newData), outShape, imOut);
    }

    // Row/column deletion: base(idx,:) = [] or base(:,idx) = []
    if (isRuntimeTensor(rhs) && rhs.data.length === 0 && indices.length === 2) {
      const nrows = base.shape[0] ?? 1;
      const ncols = base.shape.length >= 2 ? base.shape[1] : 1;
      const hasImag = base.imag !== undefined;

      // Helper: collect a set of 0-based indices to delete.
      function collectDelIndices(
        idx: RuntimeValue,
        dimLen: number
      ): Set<number> {
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
      } else if (isColonIndex(indices[0])) {
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
    }

    // COW: if data is shared, copy before mutating
    if (base._rc > 1) {
      base._rc--;
      const cowImag = base.imag ? new FloatXArray(base.imag) : undefined;
      base = RTV.tensor(new FloatXArray(base.data), [...base.shape], cowImag);
    }

    if (indices.length === 1) {
      // Check for colon indexing: base(:) = rhs
      if (isColonIndex(indices[0])) {
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
        // Scalar RHS: fill all elements
        const { re, im } = toReIm(rhs);
        base.data.fill(re);
        if (im !== 0 || base.imag) {
          ensureImag(base);
          base.imag!.fill(im);
        }
        return base;
      }

      // Vector index: base(idx) = rhs where idx is a tensor
      if (isRuntimeTensor(indices[0])) {
        const idx = indices[0];

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

        // Auto-grow 1-index tensor assignment if any index exceeds bounds
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
          // Preserve vector orientation
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
        const { re: rhsRe2, im: rhsIm2 } = toReIm(rhs);
        if (rhsIm2 !== 0 || base.imag) ensureImag(base);
        for (let i = 0; i < idx.data.length; i++) {
          const li = Math.round(idx.data[i]) - 1;
          base.data[li] = rhsRe2;
          if (base.imag) base.imag[li] = rhsIm2;
        }
        return base;
      }

      // Scalar logical index: true selects first element, false is no-op
      if (isRuntimeLogical(indices[0])) {
        if (!indices[0]) return base; // false → nothing selected
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

      const { re: rhsRe, im: rhsIm } = toReIm(rhs);
      const i = Math.round(toNumber(indices[0])) - 1;
      if (i < 0) throw new RuntimeError("Index exceeds array bounds");
      // Auto-grow if needed (requires copy — inefficient)
      if (i >= base.data.length) {
        // console.log(
        //   "numbl: auto-growing tensor via indexed assignment is inefficient and discouraged. Pre-allocate with zeros() instead."
        // );
        const grown = new FloatXArray(i + 1);
        grown.set(base.data);
        grown[i] = rhsRe;
        let grownImag: InstanceType<typeof FloatXArray> | undefined;
        if (rhsIm !== 0 || base.imag) {
          grownImag = new FloatXArray(i + 1);
          if (base.imag) grownImag.set(base.imag);
          grownImag[i] = rhsIm;
        }
        // Adjust shape for 1D
        return RTV.tensor(grown, [1, i + 1], grownImag);
      }
      // Mutate in place (safe — rc is 1)
      base.data[i] = rhsRe;
      if (rhsIm !== 0 || base.imag) {
        ensureImag(base);
        base.imag![i] = rhsIm;
      }
      return base;
    } else if (indices.length === 2) {
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
        // Logical scalar column: false → no-op, true → col 1
        if (isRuntimeLogical(indices[1])) {
          if (!indices[1]) return base; // no columns selected
          // true → assign column 0
          if (isRuntimeTensor(rhs)) {
            if (rhs.data.length !== rows) {
              throw new RuntimeError(
                "Subscripted assignment dimension mismatch"
              );
            }
            if (rhs.imag || base.imag) ensureImag(base);
            for (let r = 0; r < rows; r++) {
              const li = colMajorIndex(r, 0, rows);
              base.data[li] = rhs.data[r];
              if (base.imag) base.imag[li] = rhs.imag ? rhs.imag[r] : 0;
            }
            return base;
          }
          const { re, im } = toReIm(rhs);
          if (im !== 0 || base.imag) ensureImag(base);
          for (let r = 0; r < rows; r++) {
            const li = colMajorIndex(r, 0, rows);
            base.data[li] = re;
            if (base.imag) base.imag[li] = im;
          }
          return base;
        }
        // base(:, c) = rhs — assign entire column (auto-grow if needed)
        const c = Math.round(toNumber(indices[1])) - 1;
        if (c < 0) throw new RuntimeError("Index exceeds array bounds");
        let curColsC = cols;
        let curRowsC = rows;
        // When base is empty (0 rows) and `:` is the row index,
        // infer the row count from the RHS (grows the matrix).
        if (curRowsC === 0 && isRuntimeTensor(rhs) && rhs.data.length > 0) {
          curRowsC = rhs.data.length;
          const newCols = c + 1;
          base = RTV.tensor(new FloatXArray(curRowsC * newCols), [
            curRowsC,
            newCols,
          ]);
          curColsC = newCols;
        } else if (
          curRowsC === 0 &&
          (isRuntimeNumber(rhs) || isRuntimeLogical(rhs))
        ) {
          curRowsC = 1;
          const newCols = c + 1;
          base = RTV.tensor(new FloatXArray(curRowsC * newCols), [
            curRowsC,
            newCols,
          ]);
          curColsC = newCols;
        }
        if (c >= curColsC) {
          const newCols = c + 1;
          const newData = new FloatXArray(curRowsC * newCols);
          const hasImag = base.imag !== undefined;
          const newIm = hasImag
            ? new FloatXArray(curRowsC * newCols)
            : undefined;
          for (let ci = 0; ci < curColsC; ci++) {
            for (let r = 0; r < curRowsC; r++) {
              newData[r + ci * curRowsC] = base.data[r + ci * curRowsC];
              if (newIm && base.imag)
                newIm[r + ci * curRowsC] = base.imag[r + ci * curRowsC];
            }
          }
          base = RTV.tensor(newData, [curRowsC, newCols], newIm);
          curColsC = newCols;
        }

        if (isRuntimeTensor(rhs)) {
          if (rhs.data.length !== curRowsC) {
            throw new RuntimeError("Subscripted assignment dimension mismatch");
          }
          if (rhs.imag || base.imag) ensureImag(base);
          for (let r = 0; r < curRowsC; r++) {
            const li = colMajorIndex(r, c, curRowsC);
            base.data[li] = rhs.data[r];
            if (base.imag) base.imag[li] = rhs.imag ? rhs.imag[r] : 0;
          }
          return base;
        }
        const { re, im } = toReIm(rhs);
        if (im !== 0 || base.imag) ensureImag(base);
        for (let r = 0; r < curRowsC; r++) {
          const li = colMajorIndex(r, c, curRowsC);
          base.data[li] = re;
          if (base.imag) base.imag[li] = im;
        }
        return base;
      }

      if (!rowIsColon && !rowIsTensor && colIsColon) {
        // Logical scalar row: false → no-op, true → row 1
        if (isRuntimeLogical(indices[0])) {
          if (!indices[0]) return base; // no rows selected
          // true → assign row 0
          if (isRuntimeTensor(rhs)) {
            if (rhs.data.length !== cols) {
              throw new RuntimeError(
                "Subscripted assignment dimension mismatch"
              );
            }
            if (rhs.imag || base.imag) ensureImag(base);
            for (let c = 0; c < cols; c++) {
              const li = colMajorIndex(0, c, rows);
              base.data[li] = rhs.data[c];
              if (base.imag) base.imag[li] = rhs.imag ? rhs.imag[c] : 0;
            }
            return base;
          }
          const { re, im } = toReIm(rhs);
          if (im !== 0 || base.imag) ensureImag(base);
          for (let c = 0; c < cols; c++) {
            const li = colMajorIndex(0, c, rows);
            base.data[li] = re;
            if (base.imag) base.imag[li] = im;
          }
          return base;
        }
        // base(r, :) = rhs — assign entire row (auto-grow if needed)
        const r = Math.round(toNumber(indices[0])) - 1;
        if (r < 0) throw new RuntimeError("Index exceeds array bounds");
        let curRowsR = rows;
        let curColsR = cols;
        // When base is empty (0 cols) and `:` is the col index,
        // infer the col count from the RHS (grows the matrix).
        if (curColsR === 0 && isRuntimeTensor(rhs) && rhs.data.length > 0) {
          curColsR = rhs.data.length;
          const newRows = r + 1;
          base = RTV.tensor(new FloatXArray(newRows * curColsR), [
            newRows,
            curColsR,
          ]);
          curRowsR = newRows;
        } else if (
          curColsR === 0 &&
          (isRuntimeNumber(rhs) || isRuntimeLogical(rhs))
        ) {
          curColsR = 1;
          const newRows = r + 1;
          base = RTV.tensor(new FloatXArray(newRows * curColsR), [
            newRows,
            curColsR,
          ]);
          curRowsR = newRows;
        }
        if (r >= curRowsR) {
          const newRows = r + 1;
          const newData = new FloatXArray(newRows * curColsR);
          const hasImag = base.imag !== undefined;
          const newIm = hasImag
            ? new FloatXArray(newRows * curColsR)
            : undefined;
          for (let c = 0; c < curColsR; c++) {
            for (let ri = 0; ri < curRowsR; ri++) {
              newData[ri + c * newRows] = base.data[ri + c * curRowsR];
              if (newIm && base.imag)
                newIm[ri + c * newRows] = base.imag[ri + c * curRowsR];
            }
          }
          base = RTV.tensor(newData, [newRows, curColsR], newIm);
          curRowsR = newRows;
        }

        if (isRuntimeTensor(rhs)) {
          if (rhs.data.length !== curColsR) {
            throw new RuntimeError("Subscripted assignment dimension mismatch");
          }
          if (rhs.imag || base.imag) ensureImag(base);
          for (let c = 0; c < curColsR; c++) {
            const li = colMajorIndex(r, c, curRowsR);
            base.data[li] = rhs.data[c];
            if (base.imag) base.imag[li] = rhs.imag ? rhs.imag[c] : 0;
          }
          return base;
        }
        const { re, im } = toReIm(rhs);
        if (im !== 0 || base.imag) ensureImag(base);
        for (let c = 0; c < curColsR; c++) {
          const li = colMajorIndex(r, c, curRowsR);
          base.data[li] = re;
          if (base.imag) base.imag[li] = im;
        }
        return base;
      }

      // Handle tensor indices (ranges): base(rowRange, colRange) = rhs
      if (rowIsTensor || colIsTensor) {
        const rowIdx = indices[0];
        const colIdx = indices[1];

        // Resolve indices (boundsLimit=0: skip upper-bound check, tensor can auto-grow)
        const rowIndices = resolveIndex(rowIdx, rows, 0);
        const colIndices = resolveIndex(colIdx, cols, 0);

        // Auto-grow: if any index exceeds current dimensions, expand the
        // tensor with zero-fill (growing via indexed assignment)
        const maxRow =
          rowIndices.length > 0 ? Math.max(...rowIndices) + 1 : rows;
        const maxCol =
          colIndices.length > 0 ? Math.max(...colIndices) + 1 : cols;
        const newRows = Math.max(rows, maxRow);
        const newCols = Math.max(cols, maxCol);
        let curRows = rows;
        let curCols = cols;
        if (newRows > curRows || newCols > curCols) {
          const newData = new FloatXArray(newRows * newCols);
          const hasImag = base.imag !== undefined;
          const newIm = hasImag
            ? new FloatXArray(newRows * newCols)
            : undefined;
          // Copy old data into new (column-major layout)
          for (let c = 0; c < curCols; c++) {
            for (let r = 0; r < curRows; r++) {
              newData[r + c * newRows] = base.data[r + c * curRows];
              if (newIm && base.imag)
                newIm[r + c * newRows] = base.imag[r + c * curRows];
            }
          }
          base = RTV.tensor(newData, [newRows, newCols], newIm);
          curRows = newRows;
          curCols = newCols;
        }

        // Assign values
        if (isRuntimeTensor(rhs)) {
          const [rhsRows, rhsCols] = tensorSize2D(rhs);
          if (rhsRows !== rowIndices.length || rhsCols !== colIndices.length) {
            throw new RuntimeError("Subscripted assignment dimension mismatch");
          }
          if (rhs.imag || base.imag) ensureImag(base);
          for (let ri = 0; ri < rowIndices.length; ri++) {
            for (let ci = 0; ci < colIndices.length; ci++) {
              const r = rowIndices[ri];
              const c = colIndices[ci];
              const dstLi = colMajorIndex(r, c, curRows);
              const srcLi = colMajorIndex(ri, ci, rhsRows);
              base.data[dstLi] = rhs.data[srcLi];
              if (base.imag) base.imag[dstLi] = rhs.imag ? rhs.imag[srcLi] : 0;
            }
          }
          return base;
        }
        // Scalar RHS
        const { re: rhsReT, im: rhsImT } = toReIm(rhs);
        if (rhsImT !== 0 || base.imag) ensureImag(base);
        for (const r of rowIndices) {
          for (const c of colIndices) {
            const li = colMajorIndex(r, c, curRows);
            base.data[li] = rhsReT;
            if (base.imag) base.imag[li] = rhsImT;
          }
        }
        return base;
      }

      // Both are scalar indices
      // Logical scalar: false → no-op (selects nothing)
      if (
        (isRuntimeLogical(indices[0]) && !indices[0]) ||
        (isRuntimeLogical(indices[1]) && !indices[1])
      ) {
        return base;
      }
      const { re: rhsRe2, im: rhsIm2 } = toReIm(rhs);
      const r = isRuntimeLogical(indices[0])
        ? 0
        : Math.round(toNumber(indices[0])) - 1;
      const c = isRuntimeLogical(indices[1])
        ? 0
        : Math.round(toNumber(indices[1])) - 1;
      if (r < 0 || c < 0) throw new RuntimeError("Index exceeds array bounds");
      // Auto-grow if needed
      let scRows = rows;
      let scCols = cols;
      if (r >= scRows || c >= scCols) {
        const newRows = Math.max(scRows, r + 1);
        const newCols = Math.max(scCols, c + 1);
        const newData = new FloatXArray(newRows * newCols);
        const hasImag = base.imag !== undefined;
        const newIm = hasImag ? new FloatXArray(newRows * newCols) : undefined;
        for (let ci = 0; ci < scCols; ci++) {
          for (let ri = 0; ri < scRows; ri++) {
            newData[ri + ci * newRows] = base.data[ri + ci * scRows];
            if (newIm && base.imag)
              newIm[ri + ci * newRows] = base.imag[ri + ci * scRows];
          }
        }
        base = RTV.tensor(newData, [newRows, newCols], newIm);
        scRows = newRows;
        scCols = newCols;
      }
      const linearIdx = colMajorIndex(r, c, scRows);
      // Mutate in place (safe — rc is 1)
      base.data[linearIdx] = rhsRe2;
      if (rhsIm2 !== 0 || base.imag) {
        ensureImag(base);
        base.imag![linearIdx] = rhsIm2;
      }
      return base;
    } else if (indices.length >= 3) {
      // General N-dimensional indexed assignment (with auto-grow support)
      let shape = [...base.shape];

      // First pass: figure out the RHS shape for colon-on-empty resolution
      const rhsTensor = isRuntimeTensor(rhs) ? rhs : null;
      const rhsShape = rhsTensor ? rhsTensor.shape : null;

      // Resolve each index to an array of 0-based indices.
      // For colon on a zero-sized dim, infer size from RHS shape.
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
        // boundsLimit=0: skip upper-bound check, tensor can auto-grow
        return resolveIndex(idx, dimSize, 0);
      });

      // Determine required shape — grow base if any index exceeds current dims
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
        // Copy old data into new array using stride arithmetic
        if (base.data.length > 0) {
          const oldShape = [...shape];
          while (oldShape.length < requiredShape.length) oldShape.push(1);
          const oldTotal = base.data.length;
          const ndimGrow = oldShape.length;
          // Precompute destination strides mapped to old shape iteration
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
        // Squeeze trailing singleton dims beyond 2nd
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

      // Precompute destination strides and per-dimension offset tables
      const ndimStore = dimIndices.length;
      const dstStrides = new Array(ndimStore);
      dstStrides[0] = 1;
      for (let d = 1; d < ndimStore; d++)
        dstStrides[d] =
          dstStrides[d - 1] * (d - 1 < shape.length ? shape[d - 1] : 1);
      const dimOffsetsStore: number[][] = dimIndices.map((indices, d) =>
        indices.map(idx => idx * dstStrides[d])
      );

      if (isRuntimeTensor(rhs)) {
        if (rhs.data.length !== totalElems) {
          throw new RuntimeError("Subscripted assignment dimension mismatch");
        }
        const hasRhsImag = rhs.imag !== undefined;
        if (hasRhsImag || base.imag) ensureImag(base);
        const subs = new Array(ndimStore).fill(0);
        let dstLinear = 0;
        for (let d = 0; d < ndimStore; d++) dstLinear += dimOffsetsStore[d][0];
        for (let i = 0; i < totalElems; i++) {
          base.data[dstLinear] = rhs.data[i];
          if (base.imag) base.imag[dstLinear] = hasRhsImag ? rhs.imag![i] : 0;
          for (let d = 0; d < ndimStore; d++) {
            const prev = subs[d];
            subs[d]++;
            if (subs[d] < dimIndices[d].length) {
              dstLinear +=
                dimOffsetsStore[d][subs[d]] - dimOffsetsStore[d][prev];
              break;
            }
            dstLinear -= dimOffsetsStore[d][prev] - dimOffsetsStore[d][0];
            subs[d] = 0;
          }
        }
        return base;
      }

      // Scalar RHS: fill all target positions
      const { re: rhsReN, im: rhsImN } = toReIm(rhs);
      if (rhsImN !== 0 || base.imag) ensureImag(base);
      const subs = new Array(ndimStore).fill(0);
      let dstLinear = 0;
      for (let d = 0; d < ndimStore; d++) dstLinear += dimOffsetsStore[d][0];
      for (let i = 0; i < totalElems; i++) {
        base.data[dstLinear] = rhsReN;
        if (base.imag) base.imag[dstLinear] = rhsImN;
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
      return base;
    }
  }

  if (isRuntimeCell(base)) {
    // COW: if data is shared, copy before mutating
    if (base._rc > 1) {
      base._rc--;
      // Deep-share each element so that value-class instances inside the cell
      // get their own COW wrappers and subsequent mutations don't leak back.
      const sharedData = base.data.map(elem => shareRuntimeValue(elem));
      base = RTV.cell(sharedData, [...base.shape]);
    }

    // Helper: update shape after linear-index growth.
    // If no growth occurred (oldLen === newLen), shape is unchanged.
    // If growth occurred, extend along vector dimension (column stays column,
    // row stays row); non-vector cells become row vectors.
    const updateShapeAfterLinearAssign = (oldLen: number) => {
      const c = base as RuntimeCell;
      const newLen = c.data.length;
      if (newLen === oldLen) return; // no growth — shape unchanged
      const isColVec = c.shape[0] > 1 && c.shape[1] === 1;
      c.shape = isColVec ? [newLen, 1] : [1, newLen];
    };

    if (indices.length === 1) {
      // Vector index: c([2 3]) = {val1, val2}
      if (isRuntimeTensor(indices[0])) {
        const idx = indices[0];

        // Convert logical index to numeric positions (0-based)
        let positions: number[];
        if (idx._isLogical) {
          positions = [];
          for (let i = 0; i < idx.data.length; i++) {
            if (idx.data[i] !== 0) positions.push(i);
          }
        } else {
          positions = Array.from(idx.data).map(
            (v: number) => Math.round(v) - 1
          );
        }

        // Single-element index with non-cell RHS: treat as scalar assignment.
        // This handles patterns like [varargout{1:nargout}] = func() where
        // nargout=1 produces a 1-element range index with a scalar result.
        if (positions.length === 1 && !isRuntimeCell(rhs)) {
          const pos = positions[0];
          if (pos < 0) throw new RuntimeError("Cell index exceeds bounds");
          const oldLen = base.data.length;
          while (base.data.length <= pos)
            base.data.push(RTV.tensor(new FloatXArray(0), [0, 0]));
          base.data[pos] = rhs;
          updateShapeAfterLinearAssign(oldLen);
          return base;
        }
        if (
          !isRuntimeCell(rhs) ||
          (rhs.data.length !== positions.length && rhs.data.length !== 1)
        ) {
          throw new RuntimeError("Subscripted assignment dimension mismatch");
        }
        const scalarExpand = rhs.data.length === 1;
        // Auto-grow if needed
        const oldLen = base.data.length;
        let maxIdx = -1;
        for (let j = 0; j < positions.length; j++) {
          const pos = positions[j];
          if (pos < 0) throw new RuntimeError("Cell index exceeds bounds");
          if (pos > maxIdx) maxIdx = pos;
        }
        while (base.data.length <= maxIdx)
          base.data.push(RTV.tensor(new FloatXArray(0), [0, 0]));
        // Assign each element
        for (let j = 0; j < positions.length; j++) {
          const pos = positions[j];
          base.data[pos] = scalarExpand ? rhs.data[0] : rhs.data[j];
        }
        updateShapeAfterLinearAssign(oldLen);
        return base;
      }

      const i = Math.round(toNumber(indices[0])) - 1;
      if (i < 0) throw new RuntimeError("Cell index exceeds bounds");
      // Auto-grow if needed
      const oldLen = base.data.length;
      if (i >= base.data.length) {
        while (base.data.length <= i)
          base.data.push(RTV.tensor(new FloatXArray(0), [0, 0]));
      }
      // Mutate in place (safe — rc is 1)
      // Unwrap 1x1 cell RHS: c(i) = {val} stores val, not the cell
      base.data[i] =
        isRuntimeCell(rhs) && rhs.data.length === 1 ? rhs.data[0] : rhs;
      updateShapeAfterLinearAssign(oldLen);
      return base;
    }
    if (indices.length === 2) {
      let rows = base.shape[0];
      let cols = base.shape.length >= 2 ? base.shape[1] : 1;

      // Resolve row indices to 0-based array
      const rowIsColon = isColonIndex(indices[0]);
      let rowIndices: number[];
      if (rowIsColon) {
        rowIndices = Array.from({ length: rows }, (_, i) => i);
      } else if (isRuntimeTensor(indices[0])) {
        if (indices[0]._isLogical) {
          rowIndices = [];
          for (let i = 0; i < indices[0].data.length; i++) {
            if (indices[0].data[i] !== 0) rowIndices.push(i);
          }
        } else {
          rowIndices = Array.from(indices[0].data).map(
            (v: number) => Math.round(v) - 1
          );
        }
      } else if (isRuntimeLogical(indices[0])) {
        rowIndices = indices[0] ? [0] : [];
      } else {
        rowIndices = [Math.round(toNumber(indices[0])) - 1];
      }

      // Resolve column indices to 0-based array
      const colIsColon = isColonIndex(indices[1]);
      let colIndices: number[];
      if (colIsColon) {
        colIndices = Array.from({ length: cols }, (_, i) => i);
      } else if (isRuntimeTensor(indices[1])) {
        if (indices[1]._isLogical) {
          colIndices = [];
          for (let i = 0; i < indices[1].data.length; i++) {
            if (indices[1].data[i] !== 0) colIndices.push(i);
          }
        } else {
          colIndices = Array.from(indices[1].data).map(
            (v: number) => Math.round(v) - 1
          );
        }
      } else if (isRuntimeLogical(indices[1])) {
        colIndices = indices[1] ? [0] : [];
      } else {
        colIndices = [Math.round(toNumber(indices[1])) - 1];
      }

      // Auto-grow cell if any index exceeds current dimensions
      const maxRow = Math.max(...rowIndices) + 1;
      const maxCol = Math.max(...colIndices) + 1;
      const newRows = Math.max(rows, maxRow);
      const newCols = Math.max(cols, maxCol);
      if (newRows > rows || newCols > cols) {
        const emptyVal = () => RTV.tensor(new FloatXArray(0), [0, 0]);
        const newData: RuntimeValue[] = new Array(newRows * newCols);
        for (let k = 0; k < newData.length; k++) newData[k] = emptyVal();
        // Copy old data (column-major layout)
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

      if (isRuntimeCell(rhs)) {
        // Cell RHS: spread elements into selected positions (column-major)
        // Scalar expansion: {val} expands to fill all selected positions
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
        // Single value RHS: assign to single position
        const linearIdx = colIndices[0] * rows + rowIndices[0];
        base.data[linearIdx] = rhs;
      }
      return base;
    }
  }

  throw new RuntimeError(`Cannot index-assign into ${kstr(base)}`);
}

// ── Internal helpers ─────────────────────────────────────────────────────

function indexIntoTensorWithTensor(
  base: RuntimeTensor,
  idx: RuntimeTensor
): RuntimeValue {
  // Logical indexing: extract elements where mask is true (nonzero)
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
    // Preserve orientation: row vector base → row result, otherwise column
    const isRow = base.shape.length === 2 && base.shape[0] === 1;
    const outShape: number[] = isRow
      ? [1, selected.length]
      : [selected.length, 1];
    return RTV.tensor(new FloatXArray(selected), outShape, imOut);
  }
  // Numeric indexing: use values as 1-based indices
  const indices: number[] = [];
  const hasImag = base.imag !== undefined;
  const imIndices: number[] = [];
  for (let i = 0; i < idx.data.length; i++) {
    const k = Math.round(idx.data[i]) - 1;
    indices.push(base.data[k]);
    if (hasImag) imIndices.push(base.imag![k]);
  }
  // When the index is 0x0 ([]), result is always 0x0.
  // When base is a vector and index is a shaped empty (e.g. ones(1,0)),
  // the output orientation matches base, not the index.
  // When base is a matrix, output matches the index shape.
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
        ? [1, indices.length] // base is row → result is row
        : [indices.length, 1] // base is column → result is column
      : idx.shape;
  const imOut =
    hasImag && imIndices.some(x => x !== 0)
      ? new FloatXArray(imIndices)
      : undefined;
  return RTV.tensor(new FloatXArray(indices), outShape, imOut);
}
