/**
 * Range creation and tensor concatenation.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  type RuntimeCell,
  type RuntimeStruct,
  isRuntimeCell,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  isRuntimeChar,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeSparseMatrix,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeStringArray,
  stringArrayValue,
  type RuntimeSparseMatrix,
  RuntimeChar,
  kstr,
} from "./types.js";
import { RuntimeError } from "./error.js";
import { RTV } from "./constructors.js";
import { numel, matlabNumToString } from "./utils.js";
import { allocFloat64Array } from "./alloc.js";

/** Create a range start:end or start:step:end */
export function makeRangeTensor(
  start: number,
  step: number,
  end: number
): RuntimeTensor {
  // Compute number of elements
  let n: number;
  if (step > 0) {
    n = Math.max(0, Math.floor((end - start) / step + 1 + 1e-10));
  } else if (step < 0) {
    n = Math.max(0, Math.floor((end - start) / step + 1 + 1e-10));
  } else {
    n = 0;
  }
  if (n === 0) {
    return RTV.tensor(allocFloat64Array(0), [1, 0]) as RuntimeTensor;
  }
  // Compute each element as start + i*step to avoid accumulating error
  const values = allocFloat64Array(n);
  for (let i = 0; i < n; i++) {
    values[i] = start + i * step;
  }
  // Force last element to be exactly 'end' when it should be
  if (n > 1) {
    const lastComputed = start + (n - 1) * step;
    const absStep = Math.abs(step);
    if (Math.abs(lastComputed - end) < absStep * 1e-10) {
      values[n - 1] = end;
    }
  }
  return RTV.tensor(values, [1, n]) as RuntimeTensor;
}

/** Horizontal concatenation [A, B] */
export function horzcat(...values: RuntimeValue[]): RuntimeValue {
  if (values.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
  if (values.length === 1) return values[0];

  // If any element is sparse, concatenate as sparse
  if (values.some(v => isRuntimeSparseMatrix(v))) {
    return sparseCatAlongDim(values, 1);
  }

  // Cell concatenation: [cellA, cellB]. MATLAB: if any operand is a cell,
  // the result is a cell and non-cell operands are wrapped — so this must
  // win over the char/string paths below.
  if (values.some(v => isRuntimeCell(v))) {
    return cellCatAlongDim(values, 1);
  }

  // If any element is a string, the result is a string ARRAY (MATLAB rule:
  // string wins over char; each char operand becomes one string element).
  if (values.some(v => isRuntimeString(v) || isRuntimeStringArray(v))) {
    return stringCat(values, 1);
  }

  // If any element is a char, concatenate as char array
  if (values.some(v => isRuntimeChar(v))) {
    let result = "";
    for (const v of values) {
      if (isRuntimeChar(v)) result += v.value;
      else if (isRuntimeNumber(v)) result += String.fromCharCode(Math.round(v));
      else throw new RuntimeError(`Cannot concatenate ${kstr(v)} into char`);
    }
    return RTV.char(result);
  }

  // Struct / struct-array concatenation: [sA, sB]
  if (values.some(v => isRuntimeStruct(v) || isRuntimeStructArray(v))) {
    return structCat(values);
  }

  return catAlongDim(values, 1); // dim 2 = 0-based index 1
}

/** Vertical concatenation [A; B] */
export function vertcat(...values: RuntimeValue[]): RuntimeValue {
  if (values.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
  if (values.length === 1) return values[0];

  // If any element is sparse, concatenate as sparse
  if (values.some(v => isRuntimeSparseMatrix(v))) {
    return sparseCatAlongDim(values, 0);
  }

  // Cell concatenation: [cellA; cellB]. As in horzcat, a cell operand makes
  // the whole result a cell (must win over the char path below).
  if (values.some(v => isRuntimeCell(v))) {
    return cellCatAlongDim(values, 0);
  }

  // String operands make the result a string array (see horzcat).
  if (values.some(v => isRuntimeString(v) || isRuntimeStringArray(v))) {
    return stringCat(values, 0);
  }

  // If any element is a char, build a 2-D char array (rows stacked).
  if (values.some(v => isRuntimeChar(v))) {
    return vertcatChars(values);
  }

  // Struct / struct-array concatenation: [sA; sB]
  if (values.some(v => isRuntimeStruct(v) || isRuntimeStructArray(v))) {
    return structCat(values);
  }

  return catAlongDim(values, 0); // dim 1 = 0-based index 0
}

// ── String-array concatenation ──────────────────────────────────────────

interface StrBlock {
  /** Elements in column-major order. */
  data: string[];
  rows: number;
  cols: number;
}

/** Convert a concat operand to a string block, or null for empties (which
 *  MATLAB drops from concatenations). */
function toStringBlock(v: RuntimeValue): StrBlock | null {
  if (isRuntimeString(v)) return { data: [v], rows: 1, cols: 1 };
  if (isRuntimeStringArray(v)) {
    if (v.data.length === 0) return null;
    return { data: v.data.slice(), rows: v.shape[0], cols: v.shape[1] };
  }
  if (isRuntimeChar(v)) {
    const rows = v.shape ? v.shape[0] : 1;
    const width = v.shape ? v.shape[1] : v.value.length;
    if (rows <= 1) return { data: [v.value], rows: 1, cols: 1 };
    // A multi-row char matrix contributes one string element per row.
    const out: string[] = [];
    for (let r = 0; r < rows; r++) {
      out.push(v.value.slice(r * width, (r + 1) * width));
    }
    return { data: out, rows, cols: 1 };
  }
  if (isRuntimeNumber(v)) {
    return { data: [matlabNumToString(v)], rows: 1, cols: 1 };
  }
  if (isRuntimeLogical(v)) {
    return { data: [v ? "true" : "false"], rows: 1, cols: 1 };
  }
  if (isRuntimeTensor(v)) {
    if (v.data.length === 0) return null;
    const rows = v.shape.length >= 2 ? v.shape[0] : 1;
    const cols = v.data.length / (rows || 1);
    const out: string[] = [];
    for (let i = 0; i < v.data.length; i++) {
      out.push(
        v._isLogical
          ? v.data[i]
            ? "true"
            : "false"
          : matlabNumToString(v.data[i])
      );
    }
    return { data: out, rows, cols };
  }
  throw new RuntimeError(`Cannot concatenate ${kstr(v)} into string array`);
}

/** Concatenate operands into a string array along dim (0 = rows, 1 = cols).
 *  1x1 results collapse to a primitive string. */
function stringCat(values: RuntimeValue[], dim: 0 | 1): RuntimeValue {
  const blocks: StrBlock[] = [];
  for (const v of values) {
    const b = toStringBlock(v);
    if (b) blocks.push(b);
  }
  if (blocks.length === 0) return RTV.stringArray([], [0, 0]);
  if (dim === 1) {
    const rows = blocks[0].rows;
    let cols = 0;
    for (const b of blocks) {
      if (b.rows !== rows) {
        throw new RuntimeError(
          "Dimensions of arrays being concatenated are not consistent"
        );
      }
      cols += b.cols;
    }
    // Column-major blocks placed side by side are just concatenated data.
    const data: string[] = [];
    for (const b of blocks) data.push(...b.data);
    return stringArrayValue(data, [rows, cols]);
  }
  const cols = blocks[0].cols;
  let rows = 0;
  for (const b of blocks) {
    if (b.cols !== cols) {
      throw new RuntimeError(
        "Dimensions of arrays being concatenated are not consistent"
      );
    }
    rows += b.rows;
  }
  const data: string[] = new Array(rows * cols);
  let rOff = 0;
  for (const b of blocks) {
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < b.rows; r++) {
        data[c * rows + rOff + r] = b.data[c * b.rows + r];
      }
    }
    rOff += b.rows;
  }
  return stringArrayValue(data, [rows, cols]);
}

/** Vertical concatenation when at least one operand is a char array.
 *  Char arrays store rows row-major (each row is `shape[1]` chars), so
 *  stacking rows is just concatenating the row-major values. All operands
 *  must share a common column width; numeric/string operands are coerced
 *  to char rows (numeric → code points). */
function vertcatChars(values: RuntimeValue[]): RuntimeValue {
  let cols: number | null = null;
  let totalRows = 0;
  let out = "";
  for (const v of values) {
    let rows: number;
    let width: number;
    let rowMajor: string;
    if (isRuntimeChar(v)) {
      rows = v.shape ? v.shape[0] : 1;
      width = v.shape ? v.shape[1] : v.value.length;
      rowMajor = v.value;
    } else if (isRuntimeString(v)) {
      rows = 1;
      width = (v as string).length;
      rowMajor = v as string;
    } else if (isRuntimeNumber(v)) {
      rows = 1;
      width = 1;
      rowMajor = String.fromCharCode(Math.round(v as number));
    } else if (isRuntimeTensor(v)) {
      const t = v as RuntimeTensor;
      const m = t.shape[0] ?? 0;
      const n = t.shape[1] ?? 0;
      rows = m;
      width = n;
      let s = "";
      // Tensor data is column-major; emit row-major code points.
      for (let r = 0; r < m; r++) {
        for (let c = 0; c < n; c++) {
          s += String.fromCharCode(Math.round(t.data[c * m + r]));
        }
      }
      rowMajor = s;
    } else {
      throw new RuntimeError(`Cannot concatenate ${kstr(v)} into char`);
    }
    if (cols === null) cols = width;
    else if (width !== cols) {
      throw new RuntimeError(
        "Dimensions of arrays being concatenated are not consistent"
      );
    }
    totalRows += rows;
    out += rowMajor;
  }
  if (totalRows <= 1) return RTV.char(out);
  return new RuntimeChar(out, [totalRows, cols ?? 0]);
}

/** Concatenate struct scalars and struct arrays into a flat struct array.
 *  numbl's RuntimeStructArray doesn't carry a shape, so both horzcat and
 *  vertcat produce a 1-D concatenated list.  All operands must have the
 *  same set of field names. */
function structCat(values: RuntimeValue[]): RuntimeValue {
  const elements: RuntimeStruct[] = [];
  let fieldNames: string[] | null = null;
  for (const v of values) {
    if (isRuntimeStruct(v)) {
      const keys = Array.from(v.fields.keys());
      if (fieldNames === null) fieldNames = keys;
      else if (!arraysEqual(fieldNames, keys)) {
        throw new RuntimeError(
          "Cannot concatenate structs with different field names"
        );
      }
      elements.push(v);
    } else if (isRuntimeStructArray(v)) {
      if (fieldNames === null) fieldNames = [...v.fieldNames];
      else if (!arraysEqual(fieldNames, v.fieldNames)) {
        throw new RuntimeError(
          "Cannot concatenate struct arrays with different field names"
        );
      }
      for (const e of v.elements) elements.push(e);
    } else {
      // Empty numeric [] is MATLAB's empty and is allowed in struct cats.
      if (
        isRuntimeTensor(v) &&
        v.data.length === 0 &&
        v.shape.every(d => d === 0)
      ) {
        continue;
      }
      throw new RuntimeError(`Cannot concatenate ${kstr(v)} into struct`);
    }
  }
  if (fieldNames === null) fieldNames = [];
  return RTV.structArray(fieldNames, elements);
}

function arraysEqual<T>(a: T[], b: T[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Convert a value to a sparse matrix for concatenation */
function toSparseForCat(v: RuntimeValue): RuntimeSparseMatrix {
  if (isRuntimeSparseMatrix(v)) return v;
  if (isRuntimeNumber(v)) {
    if (v === 0)
      return RTV.sparseMatrix(
        1,
        1,
        new Int32Array(0),
        new Int32Array([0, 0]),
        allocFloat64Array(0)
      );
    return RTV.sparseMatrix(
      1,
      1,
      new Int32Array([0]),
      new Int32Array([0, 1]),
      allocFloat64Array([v])
    );
  }
  if (isRuntimeLogical(v)) {
    const val = v ? 1 : 0;
    if (val === 0)
      return RTV.sparseMatrix(
        1,
        1,
        new Int32Array(0),
        new Int32Array([0, 0]),
        allocFloat64Array(0)
      );
    return RTV.sparseMatrix(
      1,
      1,
      new Int32Array([0]),
      new Int32Array([0, 1]),
      allocFloat64Array([val])
    );
  }
  if (isRuntimeComplexNumber(v)) {
    if (v.re === 0 && v.im === 0)
      return RTV.sparseMatrix(
        1,
        1,
        new Int32Array(0),
        new Int32Array([0, 0]),
        allocFloat64Array(0)
      );
    return RTV.sparseMatrix(
      1,
      1,
      new Int32Array([0]),
      new Int32Array([0, 1]),
      allocFloat64Array([v.re]),
      v.im !== 0 ? allocFloat64Array([v.im]) : undefined
    );
  }
  if (isRuntimeTensor(v)) {
    // Dense tensor → sparse
    const m = v.shape[0] ?? 1;
    const n = v.shape.length >= 2 ? v.shape[1] : 1;
    const isComplex = v.imag !== undefined;
    const irArr: number[] = [];
    const prArr: number[] = [];
    const piArr: number[] = [];
    const jc = new Int32Array(n + 1);
    for (let c = 0; c < n; c++) {
      jc[c] = irArr.length;
      for (let r = 0; r < m; r++) {
        const idx = c * m + r;
        const re = v.data[idx];
        const im = isComplex ? v.imag![idx] : 0;
        if (re !== 0 || im !== 0) {
          irArr.push(r);
          prArr.push(re);
          if (isComplex) piArr.push(im);
        }
      }
    }
    jc[n] = irArr.length;
    return RTV.sparseMatrix(
      m,
      n,
      new Int32Array(irArr),
      jc,
      allocFloat64Array(prArr),
      isComplex ? allocFloat64Array(piArr) : undefined
    );
  }
  throw new RuntimeError(`Cannot concatenate ${kstr(v)} into sparse matrix`);
}

/** Sparse concatenation along a 0-based dimension index (0=vertical, 1=horizontal) */
function sparseCatAlongDim(
  values: RuntimeValue[],
  dimIdx: number
): RuntimeValue {
  // Convert all to sparse, filtering empty [0,0]
  const parts = values.map(toSparseForCat).filter(s => s.m > 0 || s.n > 0);
  if (parts.length === 0)
    return RTV.sparseMatrix(
      0,
      0,
      new Int32Array(0),
      new Int32Array([0]),
      allocFloat64Array(0)
    );
  if (parts.length === 1) return parts[0];

  // Verify dimensions match on non-cat axis
  if (dimIdx === 1) {
    // Horizontal: all must have same number of rows
    const m = parts[0].m;
    for (let i = 1; i < parts.length; i++) {
      if (parts[i].m !== m && parts[i].m > 0)
        throw new RuntimeError(
          "Dimensions of arrays being concatenated are not consistent"
        );
    }
    // Build result: total cols = sum of cols
    const totalN = parts.reduce((s, p) => s + p.n, 0);
    const isComplex = parts.some(p => p.pi !== undefined);
    const totalNnz = parts.reduce((s, p) => s + p.jc[p.n], 0);
    const ir = new Int32Array(totalNnz);
    const pr = allocFloat64Array(totalNnz);
    const pi = isComplex ? allocFloat64Array(totalNnz) : undefined;
    const jc = new Int32Array(totalN + 1);
    let nnzOff = 0;
    let colOff = 0;
    for (const p of parts) {
      const pNnz = p.jc[p.n];
      for (let k = 0; k < pNnz; k++) {
        ir[nnzOff + k] = p.ir[k];
        pr[nnzOff + k] = p.pr[k];
        if (pi) pi[nnzOff + k] = p.pi ? p.pi[k] : 0;
      }
      for (let c = 0; c < p.n; c++) {
        jc[colOff + c] = p.jc[c] + nnzOff;
      }
      nnzOff += pNnz;
      colOff += p.n;
    }
    jc[totalN] = nnzOff;
    return RTV.sparseMatrix(m, totalN, ir, jc, pr, pi);
  } else {
    // Vertical: all must have same number of columns
    const n = parts[0].n;
    for (let i = 1; i < parts.length; i++) {
      if (parts[i].n !== n && parts[i].n > 0)
        throw new RuntimeError(
          "Dimensions of arrays being concatenated are not consistent"
        );
    }
    const totalM = parts.reduce((s, p) => s + p.m, 0);
    const isComplex = parts.some(p => p.pi !== undefined);
    const totalNnz = parts.reduce((s, p) => s + p.jc[p.n], 0);
    const ir = new Int32Array(totalNnz);
    const pr = allocFloat64Array(totalNnz);
    const pi = isComplex ? allocFloat64Array(totalNnz) : undefined;
    const jc = new Int32Array(n + 1);
    let nnzOff = 0;
    // For each column, concatenate all parts' entries for that column
    for (let c = 0; c < n; c++) {
      jc[c] = nnzOff;
      let rowOff = 0;
      for (const p of parts) {
        for (let k = p.jc[c]; k < p.jc[c + 1]; k++) {
          ir[nnzOff] = p.ir[k] + rowOff;
          pr[nnzOff] = p.pr[k];
          if (pi) pi[nnzOff] = p.pi ? p.pi[k] : 0;
          nnzOff++;
        }
        rowOff += p.m;
      }
    }
    jc[n] = nnzOff;
    return RTV.sparseMatrix(totalM, n, ir, jc, pr, pi);
  }
}

/** N-D concatenation along a 0-based dimension index */
function catAlongDim(values: RuntimeValue[], dimIdx: number): RuntimeValue {
  // Track whether all inputs are logical so we can propagate _isLogical.
  const allLogical = values.every(
    v => isRuntimeLogical(v) || (isRuntimeTensor(v) && v._isLogical === true)
  );
  let tensors: RuntimeTensor[] = values.map(v => {
    if (isRuntimeNumber(v))
      return RTV.tensor(allocFloat64Array([v]), [1, 1]) as RuntimeTensor;
    if (isRuntimeLogical(v)) {
      const t = RTV.tensor(
        allocFloat64Array([v ? 1 : 0]),
        [1, 1]
      ) as RuntimeTensor;
      t._isLogical = true;
      return t;
    }
    if (isRuntimeComplexNumber(v))
      return RTV.tensor(
        allocFloat64Array([v.re]),
        [1, 1],
        allocFloat64Array([v.im])
      ) as RuntimeTensor;
    if (isRuntimeTensor(v)) return v;
    throw new RuntimeError(`Cannot concatenate ${kstr(v)} into matrix`);
  });

  // Filter out [0,0] tensors (always safe to drop).
  tensors = tensors.filter(t => t.shape.some(d => d > 0));
  if (tensors.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
  if (tensors.length === 1) return tensors[0];

  // Determine max ndim across all tensors (at least 2)
  const ndim = Math.max(2, ...tensors.map(t => t.shape.length));

  // Pad all shapes to ndim
  let shapes = tensors.map(t => {
    const s = [...t.shape];
    while (s.length < ndim) s.push(1);
    return s;
  });

  // Find reference shape from first tensor with elements (non-zero-element).
  // Drop zero-element tensors whose non-cat dimensions are incompatible with it.
  // MATLAB allows e.g. horzcat(zeros(0,1), [1 2 3]) → [1 2 3].
  const refIdx = tensors.findIndex(t => t.data.length > 0);
  if (refIdx >= 0) {
    const ref = shapes[refIdx];
    const keep: boolean[] = tensors.map((t, i) => {
      if (t.data.length > 0) return true;
      for (let d = 0; d < ndim; d++) {
        if (d === dimIdx) continue;
        if (shapes[i][d] !== ref[d]) return false;
      }
      return true;
    });
    tensors = tensors.filter((_, i) => keep[i]);
    shapes = shapes.filter((_, i) => keep[i]);
    if (tensors.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
    if (tensors.length === 1) return tensors[0];
  }

  // Verify all non-cat dimensions match
  const refShape = shapes[0];
  for (let i = 1; i < shapes.length; i++) {
    for (let d = 0; d < ndim; d++) {
      if (d === dimIdx) continue;
      if (shapes[i][d] !== refShape[d])
        throw new RuntimeError(
          "Dimensions of arrays being concatenated are not consistent"
        );
    }
  }

  // Compute result shape
  const resultShape = [...refShape];
  resultShape[dimIdx] = shapes.reduce((s, sh) => s + sh[dimIdx], 0);
  const totalElems = numel(resultShape);

  const isComplex = tensors.some(t => t.imag !== undefined);
  const resultRe = allocFloat64Array(totalElems);
  const resultIm = isComplex ? allocFloat64Array(totalElems) : undefined;

  // Copy blocks using stride-based arithmetic (column-major layout).
  // strideDim = product of dims below dimIdx (size of one contiguous "column" block)
  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= resultShape[d];

  // numOuter = product of dims above dimIdx
  let numOuter = 1;
  for (let d = dimIdx + 1; d < ndim; d++) numOuter *= resultShape[d];

  for (let outer = 0; outer < numOuter; outer++) {
    let dstOff = outer * strideDim * resultShape[dimIdx];
    for (let t = 0; t < tensors.length; t++) {
      const srcDimSize = shapes[t][dimIdx];
      const blockSize = strideDim * srcDimSize;
      // Source offset for this outer slab in tensor t
      const srcOff = outer * blockSize;
      resultRe.set(
        tensors[t].data.subarray(srcOff, srcOff + blockSize),
        dstOff
      );
      if (resultIm) {
        const srcImag = tensors[t].imag;
        if (srcImag) {
          resultIm.set(srcImag.subarray(srcOff, srcOff + blockSize), dstOff);
        }
        // else: already zero-initialized
      }
      dstOff += blockSize;
    }
  }

  const result = RTV.tensor(resultRe, resultShape, resultIm) as RuntimeTensor;
  if (allLogical) result._isLogical = true;
  return result;
}

/** Concatenate cell arrays along a 0-based dimension index */
/** True for empty non-cell operands ('' or []) that MATLAB drops when
 *  concatenating into a cell. */
function isEmptyNonCellOperand(v: RuntimeValue): boolean {
  if (isRuntimeChar(v)) return v.value.length === 0;
  if (isRuntimeTensor(v)) return v.data.length === 0;
  return false;
}

function cellCatAlongDim(values: RuntimeValue[], dimIdx: number): RuntimeCell {
  // Convert inputs to cells: a non-cell operand is wrapped as a single 1x1
  // cell ([5, {'x'}] -> {5,'x'} — the value is preserved, not char-coerced).
  // MATLAB drops empty operands ('' / []) during concatenation, so skip them.
  let cells: RuntimeCell[] = [];
  for (const v of values) {
    if (isRuntimeCell(v)) {
      cells.push(v);
    } else if (isEmptyNonCellOperand(v)) {
      continue;
    } else {
      cells.push(RTV.cell([v], [1, 1]));
    }
  }

  // Filter out empty cells ({})
  cells = cells.filter(c => c.data.length > 0);
  if (cells.length === 0) return RTV.cell([], [0, 0]);
  if (cells.length === 1) return cells[0];

  const ndim = Math.max(2, ...cells.map(c => c.shape.length));

  // Pad all shapes to ndim
  const shapes = cells.map(c => {
    const s = [...c.shape];
    while (s.length < ndim) s.push(1);
    return s;
  });

  // Verify all non-cat dimensions match
  const refShape = shapes[0];
  for (let i = 1; i < shapes.length; i++) {
    for (let d = 0; d < ndim; d++) {
      if (d === dimIdx) continue;
      if (shapes[i][d] !== refShape[d])
        throw new RuntimeError(
          "Dimensions of arrays being concatenated are not consistent"
        );
    }
  }

  // Compute result shape
  const resultShape = [...refShape];
  resultShape[dimIdx] = shapes.reduce((s, sh) => s + sh[dimIdx], 0);
  const totalElems = numel(resultShape);

  const resultData: RuntimeValue[] = new Array(totalElems);

  // Stride-based slab copies for cell concatenation
  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= resultShape[d];
  let numOuter = 1;
  for (let d = dimIdx + 1; d < ndim; d++) numOuter *= resultShape[d];

  for (let outer = 0; outer < numOuter; outer++) {
    let dstOff = outer * strideDim * resultShape[dimIdx];
    for (let c = 0; c < cells.length; c++) {
      const srcDimSize = shapes[c][dimIdx];
      const blockSize = strideDim * srcDimSize;
      const srcOff = outer * blockSize;
      for (let j = 0; j < blockSize; j++) {
        resultData[dstOff + j] = cells[c].data[srcOff + j];
      }
      dstOff += blockSize;
    }
  }

  return RTV.cell(resultData, resultShape);
}
