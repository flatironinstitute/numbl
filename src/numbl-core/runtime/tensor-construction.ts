/**
 * Range creation and tensor concatenation.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  type RuntimeCell,
  FloatXArray,
  isRuntimeCell,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  isRuntimeChar,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeSparseMatrix,
  type RuntimeSparseMatrix,
  kstr,
} from "./types.js";
import { RuntimeError } from "./error.js";
import { RTV } from "./constructors.js";
import { numel } from "./utils.js";

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
    return RTV.tensor(new FloatXArray(0), [1, 0]) as RuntimeTensor;
  }
  // Compute each element as start + i*step to avoid accumulating error
  const values = new FloatXArray(n);
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
  if (values.length === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
  if (values.length === 1) return values[0];

  // If any element is sparse, concatenate as sparse
  if (values.some(v => isRuntimeSparseMatrix(v))) {
    return sparseCatAlongDim(values, 1);
  }

  // If any element is a char, concatenate as char array
  if (values.some(v => isRuntimeChar(v))) {
    let result = "";
    for (const v of values) {
      if (isRuntimeChar(v)) result += v.value;
      else if (isRuntimeString(v)) result += v;
      else if (isRuntimeNumber(v)) result += String.fromCharCode(Math.round(v));
      else throw new RuntimeError(`Cannot concatenate ${kstr(v)} into char`);
    }
    return RTV.char(result);
  }

  // If any element is a string or char, concatenate as strings
  if (values.some(v => isRuntimeString(v) || isRuntimeChar(v))) {
    let result = "";
    for (const v of values) {
      if (isRuntimeString(v)) result += v;
      else if (isRuntimeChar(v)) result += v.value;
      else if (isRuntimeNumber(v)) result += String(v);
      else throw new RuntimeError(`Cannot concatenate ${v} into string`);
    }
    return RTV.string(result);
  }

  // Cell concatenation: [cellA, cellB]
  if (values.some(v => isRuntimeCell(v))) {
    return cellCatAlongDim(values, 1);
  }

  return catAlongDim(values, 1); // dim 2 = 0-based index 1
}

/** Vertical concatenation [A; B] */
export function vertcat(...values: RuntimeValue[]): RuntimeValue {
  if (values.length === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
  if (values.length === 1) return values[0];

  // If any element is sparse, concatenate as sparse
  if (values.some(v => isRuntimeSparseMatrix(v))) {
    return sparseCatAlongDim(values, 0);
  }

  // Cell concatenation: [cellA; cellB]
  if (values.some(v => isRuntimeCell(v))) {
    return cellCatAlongDim(values, 0);
  }

  return catAlongDim(values, 0); // dim 1 = 0-based index 0
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
        new Float64Array(0)
      );
    return RTV.sparseMatrix(
      1,
      1,
      new Int32Array([0]),
      new Int32Array([0, 1]),
      new Float64Array([v])
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
        new Float64Array(0)
      );
    return RTV.sparseMatrix(
      1,
      1,
      new Int32Array([0]),
      new Int32Array([0, 1]),
      new Float64Array([val])
    );
  }
  if (isRuntimeComplexNumber(v)) {
    if (v.re === 0 && v.im === 0)
      return RTV.sparseMatrix(
        1,
        1,
        new Int32Array(0),
        new Int32Array([0, 0]),
        new Float64Array(0)
      );
    return RTV.sparseMatrix(
      1,
      1,
      new Int32Array([0]),
      new Int32Array([0, 1]),
      new Float64Array([v.re]),
      v.im !== 0 ? new Float64Array([v.im]) : undefined
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
      new Float64Array(prArr),
      isComplex ? new Float64Array(piArr) : undefined
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
      new Float64Array(0)
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
    const pr = new Float64Array(totalNnz);
    const pi = isComplex ? new Float64Array(totalNnz) : undefined;
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
    const pr = new Float64Array(totalNnz);
    const pi = isComplex ? new Float64Array(totalNnz) : undefined;
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
      return RTV.tensor(new FloatXArray([v]), [1, 1]) as RuntimeTensor;
    if (isRuntimeLogical(v)) {
      const t = RTV.tensor(
        new FloatXArray([v ? 1 : 0]),
        [1, 1]
      ) as RuntimeTensor;
      t._isLogical = true;
      return t;
    }
    if (isRuntimeComplexNumber(v))
      return RTV.tensor(
        new FloatXArray([v.re]),
        [1, 1],
        new FloatXArray([v.im])
      ) as RuntimeTensor;
    if (isRuntimeTensor(v)) return v;
    throw new RuntimeError(`Cannot concatenate ${kstr(v)} into matrix`);
  });

  // Filter out truly empty tensors
  // Only filter [0,0] shaped tensors; keep shaped empties like zeros(30,0)
  tensors = tensors.filter(t => t.shape.some(d => d > 0));
  if (tensors.length === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
  if (tensors.length === 1) return tensors[0];

  // Determine max ndim across all tensors (at least 2)
  const ndim = Math.max(2, ...tensors.map(t => t.shape.length));

  // Pad all shapes to ndim
  const shapes = tensors.map(t => {
    const s = [...t.shape];
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

  const isComplex = tensors.some(t => t.imag !== undefined);
  const resultRe = new FloatXArray(totalElems);
  const resultIm = isComplex ? new FloatXArray(totalElems) : undefined;

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
function cellCatAlongDim(values: RuntimeValue[], dimIdx: number): RuntimeCell {
  // Convert all inputs to cells: scalars/tensors become 1x1 cells
  let cells: RuntimeCell[] = values.map(v => {
    if (isRuntimeCell(v)) return v;
    return RTV.cell([v], [1, 1]);
  });

  // Filter out empty cells
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
