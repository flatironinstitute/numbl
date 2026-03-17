/**
 * Array manipulation builtin functions
 */

import {
  RTV,
  toNumber,
  RuntimeError,
  tensorSize2D,
  colMajorIndex,
  numel,
  horzcat,
  vertcat,
} from "../runtime/index.js";
import {
  FloatXArray,
  type FloatXArrayType,
  type RuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeComplexNumber,
} from "../runtime/types.js";
import { register, builtinSingle } from "./registry.js";
import { mTranspose, mConjugateTranspose } from "./arithmetic.js";
import { coerceToTensor } from "./shape-utils.js";

/** Flip a tensor along a specific dimension (0-based dimIdx). N-D safe. */
function flipAlongDim(v: RuntimeTensor, dimIdx: number): RuntimeTensor {
  const shape = v.shape;
  const totalElems = numel(shape);
  const result = new FloatXArray(totalElems);
  const resultImag = v.imag ? new FloatXArray(totalElems) : undefined;
  const dimSize = dimIdx < shape.length ? shape[dimIdx] : 1;

  // Stride-based reversal: blocks of size strideDim are swapped along dim.
  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= shape[d];
  const slabSize = strideDim * dimSize;
  const numOuter = totalElems / slabSize;

  for (let outer = 0; outer < numOuter; outer++) {
    const base = outer * slabSize;
    for (let k = 0; k < dimSize; k++) {
      const srcOff = base + k * strideDim;
      const dstOff = base + (dimSize - 1 - k) * strideDim;
      result.set(v.data.subarray(srcOff, srcOff + strideDim), dstOff);
      if (resultImag) {
        resultImag.set(v.imag!.subarray(srcOff, srcOff + strideDim), dstOff);
      }
    }
  }

  return RTV.tensor(result, [...shape], resultImag) as RuntimeTensor;
}

export function registerArrayManipulationFunctions(): void {
  register(
    "reshape",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("reshape requires at least 2 arguments");
      const v = args[0];
      if (
        !isRuntimeTensor(v) &&
        !isRuntimeNumber(v) &&
        !isRuntimeComplexNumber(v)
      )
        throw new RuntimeError("reshape: first argument must be numeric");
      const data = isRuntimeTensor(v)
        ? v.data
        : isRuntimeComplexNumber(v)
          ? new FloatXArray([v.re])
          : new FloatXArray([v]);
      const imag = isRuntimeTensor(v)
        ? v.imag
        : isRuntimeComplexNumber(v)
          ? new FloatXArray([v.im])
          : undefined;

      // Determine raw dimensions (null = auto/[])
      let rawDims: (number | null)[];
      if (
        args.length === 2 &&
        isRuntimeTensor(args[1]) &&
        args[1].data.length > 1
      ) {
        // reshape(A, [m n ...]) — size vector form
        rawDims = Array.from(args[1].data).map(x => Math.round(x));
      } else {
        // reshape(A, m, n, ...) — individual dimension arguments
        // A dimension of [] (empty tensor with 0 elements) means "auto"
        rawDims = args.slice(1).map(a => {
          if (isRuntimeTensor(a) && a.data.length === 0) return null;
          return Math.round(toNumber(a));
        });
      }

      const autoCount = rawDims.filter(d => d === null).length;
      if (autoCount > 1)
        throw new RuntimeError("reshape: only one dimension size can be []");

      let shape: number[];
      if (autoCount === 1) {
        const known = rawDims.filter(d => d !== null) as number[];
        const knownProduct = known.reduce((a, b) => a * b, 1);
        if (data.length % knownProduct !== 0)
          throw new RuntimeError("reshape: number of elements must not change");
        shape = rawDims.map(d => (d === null ? data.length / knownProduct : d));
      } else {
        shape = rawDims as number[];
      }

      const n = numel(shape);
      if (n !== data.length) {
        throw new RuntimeError("reshape: number of elements must not change");
      }
      return RTV.tensor(
        new FloatXArray(data),
        shape,
        imag ? new FloatXArray(imag) : undefined
      );
    })
  );

  register(
    "transpose",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("transpose requires 1 argument");
      return mTranspose(args[0]);
    })
  );

  register(
    "ctranspose",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("ctranspose requires 1 argument");
      return mConjugateTranspose(args[0]);
    })
  );

  register(
    "diag",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("diag requires at least 1 argument");
      const v = args[0];
      // Parse optional diagonal offset k (default 0)
      const k = args.length >= 2 ? Math.round(toNumber(args[1])) : 0;
      const absK = Math.abs(k);
      if (isRuntimeNumber(v)) {
        const m = 1 + absK;
        const data = new FloatXArray(m * m);
        const r = k < 0 ? -k : 0;
        const c = k > 0 ? k : 0;
        data[colMajorIndex(r, c, m)] = v;
        return RTV.tensor(data, [m, m]);
      }
      if (isRuntimeComplexNumber(v)) {
        const m = 1 + absK;
        const data = new FloatXArray(m * m);
        const imag = new FloatXArray(m * m);
        const r = k < 0 ? -k : 0;
        const c = k > 0 ? k : 0;
        data[colMajorIndex(r, c, m)] = v.re;
        imag[colMajorIndex(r, c, m)] = v.im;
        return RTV.tensor(data, [m, m], imag);
      }
      if (!isRuntimeTensor(v))
        throw new RuntimeError("diag: argument must be numeric");
      const [rows, cols] = tensorSize2D(v);
      if (rows === 1 || cols === 1) {
        // Vector → diagonal matrix with offset k
        const vecLen = Math.max(rows, cols);
        const m = vecLen + absK;
        const data = new FloatXArray(m * m);
        const imag = v.imag ? new FloatXArray(m * m) : undefined;
        for (let i = 0; i < vecLen; i++) {
          const r = k < 0 ? i - k : i;
          const c = k > 0 ? i + k : i;
          data[colMajorIndex(r, c, m)] = v.data[i];
          if (imag) imag[colMajorIndex(r, c, m)] = v.imag![i];
        }
        return RTV.tensor(data, [m, m], imag);
      }
      // Matrix → extract k-th diagonal
      // For k>=0: elements A(i, i+k) for i=0..min(rows, cols-k)-1
      // For k<0: elements A(i-k, i) for i=0..min(rows+k, cols)-1
      const diagLen =
        k >= 0
          ? Math.max(0, Math.min(rows, cols - k))
          : Math.max(0, Math.min(rows + k, cols));
      const data = new FloatXArray(diagLen);
      const imag = v.imag ? new FloatXArray(diagLen) : undefined;
      for (let i = 0; i < diagLen; i++) {
        const r = k < 0 ? i - k : i;
        const c = k > 0 ? i + k : i;
        data[i] = v.data[colMajorIndex(r, c, rows)];
        if (imag) imag[i] = v.imag![colMajorIndex(r, c, rows)];
      }
      return RTV.tensor(data, [diagLen, 1], imag);
    })
  );

  register(
    "cat",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("cat requires at least 1 argument");
      const dim = Math.round(toNumber(args[0]));
      const arrays = args.slice(1);
      if (dim === 1) return vertcat(...arrays);
      if (dim === 2) return horzcat(...arrays);
      // N-D cat for dim >= 3
      const dimIdx = dim - 1; // 0-based
      // Normalize all inputs to tensors with shapes padded to at least `dim` dimensions
      const tensors = arrays.map(a => {
        if (isRuntimeNumber(a))
          return {
            data: new FloatXArray([a]),
            imag: null as FloatXArrayType | null,
            shape: [1, 1],
          };
        if (!isRuntimeTensor(a))
          throw new RuntimeError("cat: arguments must be numeric");
        return { data: a.data, imag: a.imag ?? null, shape: [...a.shape] };
      });
      const hasComplex = tensors.some(t => t.imag !== null);
      // Pad shapes to at least `dim` dimensions
      for (const t of tensors) {
        while (t.shape.length < dim) t.shape.push(1);
      }
      // Verify all non-cat dimensions match
      const refShape = tensors[0].shape;
      for (let i = 1; i < tensors.length; i++) {
        for (let d = 0; d < refShape.length; d++) {
          if (d === dimIdx) continue;
          if (tensors[i].shape[d] !== refShape[d])
            throw new RuntimeError(
              `cat: dimension mismatch on dimension ${d + 1}`
            );
        }
      }
      // Compute result shape
      const resultShape = [...refShape];
      resultShape[dimIdx] = tensors.reduce((s, t) => s + t.shape[dimIdx], 0);
      const totalElems = numel(resultShape);
      const result = new FloatXArray(totalElems);
      const resultImag = hasComplex ? new FloatXArray(totalElems) : undefined;

      // Stride-based slab copies (same approach as catAlongDim)
      const ndim = resultShape.length;
      let strideDim = 1;
      for (let d = 0; d < dimIdx; d++) strideDim *= resultShape[d];
      let numOuter = 1;
      for (let d = dimIdx + 1; d < ndim; d++) numOuter *= resultShape[d];

      for (let outer = 0; outer < numOuter; outer++) {
        let dstOff = outer * strideDim * resultShape[dimIdx];
        for (let t = 0; t < tensors.length; t++) {
          const srcDimSize = tensors[t].shape[dimIdx];
          const blockSize = strideDim * srcDimSize;
          const srcOff = outer * blockSize;
          for (let j = 0; j < blockSize; j++) {
            result[dstOff + j] = tensors[t].data[srcOff + j];
          }
          if (resultImag) {
            const srcImag = tensors[t].imag;
            if (srcImag) {
              for (let j = 0; j < blockSize; j++) {
                resultImag[dstOff + j] = srcImag[srcOff + j];
              }
            }
          }
          dstOff += blockSize;
        }
      }
      return RTV.tensor(result, resultShape, resultImag);
    })
  );

  register(
    "horzcat",
    builtinSingle(args => {
      return horzcat(...args);
    })
  );

  register(
    "vertcat",
    builtinSingle(args => {
      return vertcat(...args);
    })
  );

  register(
    "fliplr",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("fliplr requires 1 argument");
      const v = args[0];
      if (isRuntimeNumber(v)) return v;
      if (isRuntimeChar(v))
        return { kind: "char", value: v.value.split("").reverse().join("") };
      if (!isRuntimeTensor(v))
        throw new RuntimeError("fliplr: argument must be numeric or char");
      return flipAlongDim(v, 1); // dim 2 (0-based index 1)
    })
  );

  register(
    "flipud",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("flipud requires 1 argument");
      const v = args[0];
      if (isRuntimeNumber(v)) return v;
      if (isRuntimeChar(v)) return v; // char is 1×N row vector, flipud is identity
      if (!isRuntimeTensor(v))
        throw new RuntimeError("flipud: argument must be numeric or char");
      return flipAlongDim(v, 0); // dim 1 (0-based index 0)
    })
  );

  register(
    "repmat",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("repmat requires at least 2 arguments");
      const v = args[0];
      let reps: number[];
      if (args.length === 2) {
        const arg1 = args[1];
        if (isRuntimeTensor(arg1)) {
          // repmat(A, r) where r is a vector
          reps = Array.from(arg1.data).map(x => Math.round(x));
        } else {
          // repmat(A, n) where n is a scalar — repeat n times in rows and cols
          const n = Math.round(toNumber(arg1));
          reps = [n, n];
        }
      } else {
        // repmat(A, r1, ..., rN) — individual scalar args
        reps = args.slice(1).map(a => Math.round(toNumber(a)));
      }
      if (isRuntimeNumber(v)) {
        const total = reps.reduce((a, b) => a * b, 1);
        const data = new FloatXArray(total).fill(v);
        return RTV.tensor(data, reps.length >= 2 ? reps : [reps[0], reps[0]]);
      }
      if (isRuntimeLogical(v)) {
        const total = reps.reduce((a, b) => a * b, 1);
        const data = new FloatXArray(total).fill(v ? 1 : 0);
        const shape = reps.length >= 2 ? reps : [reps[0], reps[0]];
        const t = RTV.tensor(data, shape);
        t._isLogical = true;
        return t;
      }
      if (isRuntimeChar(v)) {
        const rowReps = reps[0] ?? 1;
        const colReps = reps.length >= 2 ? reps[1] : 1;
        if (rowReps === 0 || colReps === 0) return RTV.char("");
        let row = "";
        for (let j = 0; j < colReps; j++) row += v.value;
        let result = "";
        for (let i = 0; i < rowReps; i++) result += row;
        return RTV.char(result);
      }
      if (isRuntimeComplexNumber(v)) {
        const total = reps.reduce((a, b) => a * b, 1);
        const data = new FloatXArray(total).fill(v.re);
        const imag = new FloatXArray(total).fill(v.im);
        const shape = reps.length >= 2 ? reps : [reps[0], reps[0]];
        return RTV.tensor(data, shape, imag);
      }
      if (!isRuntimeTensor(v))
        throw new RuntimeError("repmat: first argument must be numeric");
      const srcShape = v.shape.length >= 2 ? v.shape : [1, v.shape[0] || 1];
      const ndim = Math.max(srcShape.length, reps.length);
      // Pad both to ndim
      const padSrc = [...srcShape];
      while (padSrc.length < ndim) padSrc.push(1);
      const padReps = [...reps];
      while (padReps.length < ndim) padReps.push(1);
      const resultShape = padSrc.map((s, i) => s * padReps[i]);

      // Tile dimension-by-dimension using block copies.
      // Start with a copy of the source data, then for each dimension,
      // replicate the current buffer along that dimension.
      let curData: FloatXArrayType = new FloatXArray(v.data) as FloatXArrayType;
      let curImag: FloatXArrayType | undefined = v.imag
        ? (new FloatXArray(v.imag) as FloatXArrayType)
        : undefined;
      const curShape = [...padSrc];

      for (let d = 0; d < ndim; d++) {
        const r = padReps[d];
        if (r === 1) continue;
        const curTotal = curData.length;
        const newTotal = curTotal * r;
        const newData = new FloatXArray(newTotal);
        const newImag = curImag ? new FloatXArray(newTotal) : undefined;

        // blockSize = product of curShape[0..d] (the size of a slab up to and including dim d)
        let blockSize = 1;
        for (let i = 0; i <= d; i++) blockSize *= curShape[i];
        const numBlocks = curTotal / blockSize;

        // For each outer block, copy the block r times
        for (let b = 0; b < numBlocks; b++) {
          const srcOff = b * blockSize;
          const dstBase = b * blockSize * r;
          for (let rep = 0; rep < r; rep++) {
            const dstOff = dstBase + rep * blockSize;
            newData.set(curData.subarray(srcOff, srcOff + blockSize), dstOff);
            if (newImag && curImag) {
              newImag.set(curImag.subarray(srcOff, srcOff + blockSize), dstOff);
            }
          }
        }
        curData = newData;
        curImag = newImag;
        curShape[d] *= r;
      }

      const out = RTV.tensor(curData, resultShape, curImag);
      if (v._isLogical) out._isLogical = true;
      return out;
    })
  );

  // ── flip: reverse along a specified dimension ─────────────────────
  register(
    "flip",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("flip requires at least 1 argument");
      const v = args[0];
      if (isRuntimeNumber(v)) return v;
      if (!isRuntimeTensor(v))
        throw new RuntimeError("flip: argument must be numeric");
      // Default dim: first non-singleton dimension
      let dimIdx = 0; // 0-based
      if (args.length >= 2) {
        dimIdx = Math.round(toNumber(args[1])) - 1;
      } else {
        const shape = v.shape.length >= 2 ? v.shape : [1, ...v.shape];
        dimIdx = shape.findIndex(d => d > 1);
        if (dimIdx === -1) dimIdx = 0;
      }
      return flipAlongDim(v, dimIdx);
    })
  );

  // ── rot90: rotate matrix 90 degrees counter-clockwise ────────────
  register(
    "rot90",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("rot90 requires at least 1 argument");
      const v = args[0];
      if (isRuntimeNumber(v)) return v;
      if (!isRuntimeTensor(v))
        throw new RuntimeError("rot90: argument must be numeric");
      let k = args.length >= 2 ? Math.round(toNumber(args[1])) : 1;
      k = ((k % 4) + 4) % 4; // normalize to 0-3
      if (k === 0) {
        const result = RTV.tensor(
          new FloatXArray(v.data),
          [...v.shape],
          v.imag ? new FloatXArray(v.imag) : undefined
        );
        if (v._isLogical) result._isLogical = true;
        return result;
      }
      const [rows, cols] = tensorSize2D(v);
      let data = v.data;
      let imag = v.imag;
      let r = rows,
        c = cols;
      for (let iter = 0; iter < k; iter++) {
        const newData = new FloatXArray(r * c);
        const newImag = imag ? new FloatXArray(r * c) : undefined;
        // 90 CCW: result(i, j) = src(j, c-1-i), where result is c x r
        for (let i = 0; i < c; i++) {
          for (let j = 0; j < r; j++) {
            const srcIdx = (c - 1 - i) * r + j; // col-major: src[j, c-1-i]
            const dstIdx = j * c + i; // col-major: dst[i, j] in (c x r)
            newData[dstIdx] = data[srcIdx];
            if (newImag) newImag[dstIdx] = imag![srcIdx];
          }
        }
        data = newData;
        imag = newImag;
        const tmp = r;
        r = c;
        c = tmp; // swap dimensions
      }
      const result = RTV.tensor(data, [r, c], imag);
      if (v._isLogical) result._isLogical = true;
      return result;
    })
  );

  // ── circshift: circular shift along a dimension ──────────────────
  register(
    "circshift",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("circshift requires 2 arguments");
      const v = args[0];
      if (isRuntimeNumber(v)) return v;
      if (!isRuntimeTensor(v))
        throw new RuntimeError("circshift: first argument must be numeric");
      const shiftArg = args[1];
      const shape = v.shape.length >= 2 ? v.shape : [1, ...v.shape];

      // Build per-dimension shift array
      let shifts: number[];
      if (isRuntimeTensor(shiftArg)) {
        shifts = Array.from(shiftArg.data).map(s => Math.round(s));
      } else {
        const scalarShift = Math.round(toNumber(shiftArg));
        // For scalar shift: shift along first non-singleton dim
        let dimIdx = 0;
        if (shape[0] === 1 && shape.length === 2) dimIdx = 1;
        shifts = new Array(shape.length).fill(0);
        shifts[dimIdx] = scalarShift;
      }

      const totalElems = v.data.length;
      const result = new FloatXArray(totalElems);
      const resultImag = v.imag ? new FloatXArray(totalElems) : undefined;

      // Precompute strides and per-dimension lookup tables
      const ndimCS = shape.length;
      const strides = new Array(ndimCS);
      strides[0] = 1;
      for (let d = 1; d < ndimCS; d++)
        strides[d] = strides[d - 1] * shape[d - 1];

      // For each dimension, precompute shifted offset: strides[d] * ((k - shift + n) % n)
      const dimLookup: number[][] = new Array(ndimCS);
      for (let d = 0; d < ndimCS; d++) {
        const n = shape[d];
        const s = d < shifts.length ? ((-shifts[d] % n) + n) % n : 0;
        const lookup = new Array(n);
        for (let k = 0; k < n; k++) {
          lookup[k] = ((k + s) % n) * strides[d];
        }
        dimLookup[d] = lookup;
      }

      const subs = new Array(ndimCS).fill(0);
      let srcIdx = 0;
      for (let d = 0; d < ndimCS; d++) srcIdx += dimLookup[d][0];
      for (let i = 0; i < totalElems; i++) {
        result[i] = v.data[srcIdx];
        if (resultImag) resultImag[i] = v.imag![srcIdx];
        for (let d = 0; d < ndimCS; d++) {
          const prev = subs[d];
          subs[d]++;
          if (subs[d] < shape[d]) {
            srcIdx += dimLookup[d][subs[d]] - dimLookup[d][prev];
            break;
          }
          srcIdx -= dimLookup[d][prev] - dimLookup[d][0];
          subs[d] = 0;
        }
      }
      return RTV.tensor(result, [...v.shape], resultImag);
    })
  );

  // ── repelem: repeat elements of an array ─────────────────────────
  register(
    "repelem",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("repelem requires at least 2 arguments");
      const v = args[0];
      if (args.length === 2) {
        // repelem(v, n) — repeat each element n times (vector case)
        const n = Math.round(toNumber(args[1]));
        if (isRuntimeNumber(v)) {
          const data = new FloatXArray(n).fill(v);
          return RTV.tensor(data, [1, n]);
        }
        if (!isRuntimeTensor(v))
          throw new RuntimeError("repelem: first argument must be numeric");
        const len = v.data.length;
        const result = new FloatXArray(len * n);
        const resultImag = v.imag ? new FloatXArray(len * n) : undefined;
        const isCol = v.shape.length === 2 && v.shape[1] === 1;
        for (let i = 0; i < len; i++) {
          for (let j = 0; j < n; j++) {
            result[i * n + j] = v.data[i];
            if (resultImag) resultImag[i * n + j] = v.imag![i];
          }
        }
        if (isCol) return RTV.tensor(result, [len * n, 1], resultImag);
        return RTV.tensor(result, [1, len * n], resultImag);
      }
      // repelem(M, r, c) — repeat each element r times vertically, c times horizontally
      const rRep = Math.round(toNumber(args[1]));
      const cRep = Math.round(toNumber(args[2]));
      if (isRuntimeNumber(v)) {
        const data = new FloatXArray(rRep * cRep).fill(v);
        return RTV.tensor(data, [rRep, cRep]);
      }
      if (!isRuntimeTensor(v))
        throw new RuntimeError("repelem: first argument must be numeric");
      const [rows, cols] = tensorSize2D(v);
      const newRows = rows * rRep;
      const newCols = cols * cRep;
      const result = new FloatXArray(newRows * newCols);
      const resultImag = v.imag
        ? new FloatXArray(newRows * newCols)
        : undefined;
      for (let c = 0; c < cols; c++) {
        for (let r = 0; r < rows; r++) {
          const srcIdx = c * rows + r; // col-major
          const val = v.data[srcIdx];
          const valIm = v.imag ? v.imag[srcIdx] : 0;
          for (let dc = 0; dc < cRep; dc++) {
            for (let dr = 0; dr < rRep; dr++) {
              const dstRow = r * rRep + dr;
              const dstCol = c * cRep + dc;
              const dstIdx = dstCol * newRows + dstRow;
              result[dstIdx] = val;
              if (resultImag) resultImag[dstIdx] = valIm;
            }
          }
        }
      }
      return RTV.tensor(result, [newRows, newCols], resultImag);
    })
  );

  // ── squeeze: remove singleton dimensions ──────────────────────────
  register(
    "squeeze",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("squeeze requires 1 argument");
      const v = args[0];
      if (isRuntimeNumber(v) || isRuntimeLogical(v)) return v;
      if (isRuntimeTensor(v)) {
        // Strips trailing singleton dims (keeping at least 2) before
        // squeeze, so e.g. shape [1,3,1] is effectively [1,3] (2-D, no-op).
        const effectiveShape = [...v.shape];
        while (
          effectiveShape.length > 2 &&
          effectiveShape[effectiveShape.length - 1] === 1
        ) {
          effectiveShape.pop();
        }
        // For 2-D arrays squeeze is a no-op
        if (effectiveShape.length <= 2) {
          return RTV.tensor(v.data, effectiveShape, v.imag);
        }
        const newShape = effectiveShape.filter(d => d !== 1);
        if (newShape.length === 0) {
          if (v.imag && v.imag[0] !== 0)
            return RTV.complex(v.data[0], v.imag[0]);
          return RTV.num(v.data[0]);
        }
        if (newShape.length === 1) {
          // Column vector when squeezing to 1-D
          return RTV.tensor(v.data, [newShape[0], 1], v.imag);
        }
        return RTV.tensor(v.data, newShape, v.imag);
      }
      throw new RuntimeError("squeeze: argument must be numeric");
    })
  );

  // ── ndgrid: N-D rectangular grid ──────────────────────────────────
  register(
    "ndgrid",
    builtinSingle((args, nargout) => {
      if (args.length < 1)
        throw new RuntimeError("ndgrid requires at least 1 argument");

      // Helper: extract a 1-D array of values from a scalar or tensor
      const extractVec = (v: (typeof args)[0]): number[] => {
        if (isRuntimeNumber(v)) return [v];
        if (isRuntimeTensor(v)) return Array.from(v.data);
        throw new RuntimeError("ndgrid: arguments must be numeric vectors");
      };

      // Build the list of grid vectors (one per output dimension)
      const vecs: number[][] = [];
      if (args.length === 1) {
        const single = extractVec(args[0]);
        for (let k = 0; k < nargout; k++) vecs.push(single);
      } else {
        for (const a of args) vecs.push(extractVec(a));
        if (vecs.length < nargout)
          throw new RuntimeError(
            "ndgrid: not enough input vectors for requested outputs"
          );
      }

      const n = nargout;
      const shape = vecs.slice(0, n).map(v => v.length);
      const totalElems = shape.reduce((acc, s) => acc * s, 1);

      // Build one output tensor per dimension; k-th output varies along dim k.
      // For dim k: stride = product(shape[0..k-1]), period = stride * shape[k]
      const outputs = [];
      for (let k = 0; k < n; k++) {
        const data = new FloatXArray(totalElems);
        let stride = 1;
        for (let d = 0; d < k; d++) stride *= shape[d];
        const dimLen = shape[k];
        const period = stride * dimLen;
        for (let i = 0; i < totalElems; i++) {
          data[i] = vecs[k][Math.floor((i % period) / stride)];
        }
        outputs.push(RTV.tensor(data, [...shape]));
      }
      return outputs;
    })
  );

  // ── meshgrid: 2-D and 3-D grids ───────────────────────────────────
  // meshgrid(x,y) → shape [len(y), len(x)]; X rows = x, Y cols = y
  // Equivalent to ndgrid but with first two inputs and outputs swapped.
  register(
    "meshgrid",
    builtinSingle((args, nargout) => {
      if (args.length < 1)
        throw new RuntimeError("meshgrid requires at least 1 argument");

      const extractVec = (v: (typeof args)[0]): number[] => {
        if (isRuntimeNumber(v)) return [v];
        if (isRuntimeTensor(v)) return Array.from(v.data);
        throw new RuntimeError("meshgrid: arguments must be numeric vectors");
      };

      const n = Math.max(nargout, 2);

      // Build input vectors reordered as (y, x, z, ...) for ndgrid-style logic
      let reordered: number[][];
      if (args.length === 1) {
        const single = extractVec(args[0]);
        reordered = Array(n).fill(single);
      } else {
        const vecs = args.map(extractVec);
        // Swap x (index 0) and y (index 1)
        reordered = [vecs[1] ?? vecs[0], vecs[0], ...vecs.slice(2)];
        while (reordered.length < n)
          reordered.push(reordered[reordered.length - 1]);
      }

      const shape = reordered.slice(0, n).map(v => v.length);
      const totalElems = shape.reduce((acc, s) => acc * s, 1);

      // Build outputs using stride-based ndgrid logic on reordered vecs
      const ndgridOuts = [];
      for (let k = 0; k < n; k++) {
        const data = new FloatXArray(totalElems);
        let stride = 1;
        for (let d = 0; d < k; d++) stride *= shape[d];
        const dimLen = shape[k];
        const period = stride * dimLen;
        for (let idx = 0; idx < totalElems; idx++) {
          data[idx] = reordered[k][Math.floor((idx % period) / stride)];
        }
        ndgridOuts.push(RTV.tensor(data, [...shape]));
      }

      // Swap first two outputs back: ndgrid gives [Y,X,Z,...], meshgrid wants [X,Y,Z,...]
      const outputs = [...ndgridOuts];
      if (outputs.length >= 2)
        [outputs[0], outputs[1]] = [outputs[1], outputs[0]];

      return outputs;
    })
  );

  // ── permute: rearrange dimensions ─────────────────────────────────
  register(
    "permute",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("permute requires 2 arguments");
      const v = coerceToTensor(args[0], "permute");
      const orderArg = args[1];
      let order: number[];
      if (isRuntimeTensor(orderArg)) {
        order = Array.from(orderArg.data).map(x => Math.round(x));
      } else if (isRuntimeNumber(orderArg)) {
        order = [Math.round(orderArg)];
      } else {
        throw new RuntimeError("permute: second argument must be numeric");
      }
      // Convert from 1-based to 0-based
      const perm = order.map(x => x - 1);
      const srcShape = v.shape;
      // Pad source shape if perm references higher dims
      const maxDim = Math.max(...perm) + 1;
      const padShape = [...srcShape];
      while (padShape.length < maxDim) padShape.push(1);
      const newShape = perm.map(d => padShape[d]);
      const totalElems = v.data.length;
      const result = new FloatXArray(totalElems);
      const resultImag = v.imag ? new FloatXArray(totalElems) : undefined;

      // Precompute source strides for each dimension
      const ndim = perm.length;
      const srcStrides = new Array(padShape.length);
      srcStrides[0] = 1;
      for (let d = 1; d < padShape.length; d++)
        srcStrides[d] = srcStrides[d - 1] * padShape[d - 1];

      // For each destination dimension d, stepping by 1 in dst dim d
      // moves by srcStrides[perm[d]] in the source
      const mappedStrides = new Array(ndim);
      for (let d = 0; d < ndim; d++) mappedStrides[d] = srcStrides[perm[d]];

      // Iterate in column-major order over destination, tracking source index
      const subs = new Array(ndim).fill(0);
      let srcIdx = 0;
      for (let i = 0; i < totalElems; i++) {
        result[i] = v.data[srcIdx];
        if (resultImag) resultImag[i] = v.imag![srcIdx];
        // Increment subscripts in column-major order
        for (let d = 0; d < ndim; d++) {
          subs[d]++;
          srcIdx += mappedStrides[d];
          if (subs[d] < newShape[d]) break;
          srcIdx -= subs[d] * mappedStrides[d];
          subs[d] = 0;
        }
      }
      return RTV.tensor(result, newShape, resultImag);
    })
  );

  // ── ipermute: inverse permute array dimensions ──────────────────────
  register(
    "ipermute",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("ipermute requires 2 arguments");
      const v = coerceToTensor(args[0], "ipermute");
      const orderArg = args[1];
      let order: number[];
      if (isRuntimeTensor(orderArg)) {
        order = Array.from(orderArg.data).map(x => Math.round(x));
      } else if (isRuntimeNumber(orderArg)) {
        order = [Math.round(orderArg)];
      } else {
        throw new RuntimeError("ipermute: second argument must be numeric");
      }
      // Compute inverse permutation: if dimorder = [3 1 2], then
      // invPerm[dimorder[i]-1] = i, so invPerm = [1 2 0] (0-based)
      const invPerm = new Array(order.length);
      for (let i = 0; i < order.length; i++) {
        invPerm[order[i] - 1] = i;
      }
      const srcShape = v.shape;
      const maxDim = Math.max(...invPerm) + 1;
      const padShape = [...srcShape];
      while (padShape.length < maxDim) padShape.push(1);
      const newShape = invPerm.map(d => padShape[d]);
      const totalElems = v.data.length;
      const result = new FloatXArray(totalElems);
      const resultImag = v.imag ? new FloatXArray(totalElems) : undefined;

      const ndim2 = invPerm.length;
      const srcStrides2 = new Array(padShape.length);
      srcStrides2[0] = 1;
      for (let d = 1; d < padShape.length; d++)
        srcStrides2[d] = srcStrides2[d - 1] * padShape[d - 1];
      const mappedStrides2 = new Array(ndim2);
      for (let d = 0; d < ndim2; d++)
        mappedStrides2[d] = srcStrides2[invPerm[d]];

      const subs2 = new Array(ndim2).fill(0);
      let srcIdx2 = 0;
      for (let i = 0; i < totalElems; i++) {
        result[i] = v.data[srcIdx2];
        if (resultImag) resultImag[i] = v.imag![srcIdx2];
        for (let d = 0; d < ndim2; d++) {
          subs2[d]++;
          srcIdx2 += mappedStrides2[d];
          if (subs2[d] < newShape[d]) break;
          srcIdx2 -= subs2[d] * mappedStrides2[d];
          subs2[d] = 0;
        }
      }
      return RTV.tensor(result, newShape, resultImag);
    })
  );

  // ── sub2ind: convert subscripts to linear indices ────────────────────
  register(
    "sub2ind",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("sub2ind requires at least 2 arguments");
      const szArg = args[0];
      if (!isRuntimeTensor(szArg) && !isRuntimeNumber(szArg))
        throw new RuntimeError("sub2ind: first argument must be a size vector");
      const shape = isRuntimeNumber(szArg) ? [szArg] : Array.from(szArg.data);

      const subscriptArgs = args.slice(1);

      const getValues = (v: (typeof args)[0]): number[] => {
        if (isRuntimeNumber(v)) return [v];
        if (isRuntimeLogical(v)) return [v ? 1 : 0];
        if (isRuntimeTensor(v)) return Array.from(v.data);
        throw new RuntimeError("sub2ind: subscript arguments must be numeric");
      };

      const allSubs = subscriptArgs.map(getValues);
      const n = allSubs[0].length;

      // Compute strides
      const strides: number[] = [1];
      for (let d = 1; d < shape.length; d++) {
        strides[d] = strides[d - 1] * shape[d - 1];
      }

      const result = new FloatXArray(n);
      for (let i = 0; i < n; i++) {
        let idx = 0;
        for (let d = 0; d < subscriptArgs.length; d++) {
          const s = allSubs[d][i];
          const stride = d < strides.length ? strides[d] : 0;
          idx += (s - 1) * stride;
        }
        result[i] = idx + 1; // 1-based output
      }

      if (n === 1) return RTV.num(result[0]);
      const firstArg = subscriptArgs[0];
      const outShape = isRuntimeTensor(firstArg) ? [...firstArg.shape] : [1, n];
      return RTV.tensor(result, outShape);
    })
  );

  // ── ind2sub: convert linear indices to subscripts ────────────────────
  register(
    "ind2sub",
    builtinSingle((args, nargout) => {
      if (args.length !== 2)
        throw new RuntimeError("ind2sub requires 2 arguments");
      const szArg = args[0];
      if (!isRuntimeTensor(szArg) && !isRuntimeNumber(szArg))
        throw new RuntimeError("ind2sub: first argument must be a size vector");
      const shape = isRuntimeNumber(szArg) ? [szArg] : Array.from(szArg.data);

      const indArg = args[1];
      let indices: number[];
      let indShape: number[];
      if (isRuntimeNumber(indArg)) {
        indices = [indArg];
        indShape = [1, 1];
      } else if (isRuntimeLogical(indArg)) {
        indices = [indArg ? 1 : 0];
        indShape = [1, 1];
      } else if (isRuntimeTensor(indArg)) {
        indices = Array.from(indArg.data);
        indShape = [...indArg.shape];
      } else {
        throw new RuntimeError("ind2sub: second argument must be numeric");
      }

      const n = indices.length;
      const ndims = Math.max(nargout, 2);

      // Compute strides (column-major)
      const strides: number[] = [1];
      for (let d = 1; d < shape.length; d++) {
        strides[d] = strides[d - 1] * shape[d - 1];
      }

      // Extend strides for dimensions beyond shape (all trailing dims have size 1)
      while (strides.length < ndims) {
        strides.push(
          strides[strides.length - 1] * (shape[strides.length - 1] || 1)
        );
      }

      const outputs: InstanceType<typeof FloatXArray>[] = [];
      for (let d = 0; d < ndims; d++) outputs.push(new FloatXArray(n));

      for (let i = 0; i < n; i++) {
        let rem = indices[i] - 1; // convert to 0-based
        for (let d = ndims - 1; d >= 0; d--) {
          if (d === 0) {
            outputs[d][i] = rem + 1; // back to 1-based
          } else {
            const q = Math.floor(rem / strides[d]);
            outputs[d][i] = q + 1; // 1-based
            rem = rem - q * strides[d];
          }
        }
      }

      if (nargout <= 1) {
        // Single output: return row subscripts
        if (n === 1) return RTV.num(outputs[0][0]);
        return RTV.tensor(outputs[0], indShape);
      }

      const result = [];
      for (let d = 0; d < ndims; d++) {
        if (n === 1) {
          result.push(RTV.num(outputs[d][0]));
        } else {
          result.push(RTV.tensor(outputs[d], [...indShape]));
        }
      }
      return result;
    })
  );
}
