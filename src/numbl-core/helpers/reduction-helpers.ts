/**
 * Shared helpers for reduction operations: dimension iteration, type checks,
 * reduction factories, sparse helpers, and logical/complex scan utilities.
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  toString,
  RuntimeError,
} from "../runtime/index.js";
import { ItemType } from "../lowering/itemTypes.js";
import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeTensor,
  type RuntimeTensor,
  type RuntimeSparseMatrix,
} from "../runtime/types.js";

// ── Dimension iteration helpers ─────────────────────────────────────────

/** Squeeze trailing singleton dimensions, keeping at least 2. Mutates in place. */
export function squeezeTrailing(shape: number[]): void {
  while (shape.length > 2 && shape[shape.length - 1] === 1) {
    shape.pop();
  }
}

/** Iterate over all 1-D fibers along `dim` (1-based).
 * For each fiber, calls `callback(outIndex, srcIndices)` where
 * `srcIndices` is an array of flat indices into the source data.
 * Returns `{ resultShape, totalElems }` (resultShape is squeezed).
 * Returns null if dim exceeds the tensor's rank. */
export function forEachSlice(
  shape: number[],
  dim: number,
  callback: (outIdx: number, srcIndices: number[]) => void
): { resultShape: number[]; totalElems: number } | null {
  const dimIdx = dim - 1;
  if (dimIdx >= shape.length) return null;

  const reduceDimSize = shape[dimIdx];
  const resultShape = [...shape];
  resultShape[dimIdx] = 1;
  const totalElems = resultShape.reduce((a, b) => a * b, 1);

  let strideDim = 1;
  for (let d = 0; d < dimIdx; d++) strideDim *= shape[d];

  const innerCount = strideDim;
  const slabSize = strideDim * reduceDimSize;

  const srcIndices = new Array(reduceDimSize);
  let outIdx = 0;
  for (let outer = 0; outer < totalElems; outer += innerCount) {
    const slabBase = (outer / innerCount) * slabSize;
    for (let inner = 0; inner < innerCount; inner++) {
      const base = slabBase + inner;
      for (let k = 0; k < reduceDimSize; k++) {
        srcIndices[k] = base + k * strideDim;
      }
      callback(outIdx++, srcIndices);
    }
  }

  squeezeTrailing(resultShape);
  return { resultShape, totalElems };
}

/** Return 1-based dim to reduce along (first non-singleton), or 0 for "reduce to scalar". */
export function firstReduceDim(shape: number[]): number {
  const numNonSingleton = shape.filter(d => d > 1).length;
  if (numNonSingleton <= 1) return 0;
  return shape.findIndex(d => d > 1) + 1;
}

/** Return a deep copy of a tensor (data + shape + optional imag). */
export function copyTensor(v: RuntimeTensor): RuntimeValue {
  return RTV.tensor(
    new FloatXArray(v.data),
    [...v.shape],
    v.imag ? new FloatXArray(v.imag) : undefined
  );
}

// ── Generic reduction along a dimension ────────────────────────────────

/**
 * Reduce a tensor along a dimension using an accumulator function.
 * @param v The tensor to reduce
 * @param dim The dimension along which to reduce (1-based)
 * @param reduceFn Function that takes (accumulator, value) and returns new accumulator
 * @param initialValue Initial accumulator value
 * @param finalizeFn Optional function to transform final result (e.g., divide by count for mean)
 */
export function dimReduce(
  v: RuntimeValue,
  dim: number,
  reduceFn: (acc: number, val: number) => number,
  initialValue: number,
  finalizeFn?: (acc: number, count: number) => number
): RuntimeValue {
  if (!isRuntimeTensor(v))
    throw new RuntimeError("dimReduce: argument must be a tensor");

  const info = forEachSlice(v.shape, dim, () => {});
  if (!info) return copyTensor(v);

  const result = new FloatXArray(info.totalElems);
  const resultImag = v.imag ? new FloatXArray(info.totalElems) : undefined;

  forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
    let acc = initialValue;
    let accIm = resultImag ? initialValue : 0;
    for (let k = 0; k < srcIndices.length; k++) {
      acc = reduceFn(acc, v.data[srcIndices[k]]);
      if (resultImag) accIm = reduceFn(accIm, v.imag![srcIndices[k]]);
    }
    result[outIdx] = finalizeFn ? finalizeFn(acc, srcIndices.length) : acc;
    if (resultImag)
      resultImag[outIdx] = finalizeFn
        ? finalizeFn(accIm, srcIndices.length)
        : accIm;
  });

  const imOut =
    resultImag && resultImag.some(x => x !== 0) ? resultImag : undefined;
  return RTV.tensor(result, info.resultShape, imOut);
}

/** Like dimReduce but skips NaN values (for 'omitnan' flag). */
export function dimReduceOmitNaN(
  v: RuntimeValue,
  dim: number,
  reduceFn: (acc: number, val: number) => number,
  initialValue: number,
  finalizeFn?: (acc: number, count: number) => number
): RuntimeValue {
  if (!isRuntimeTensor(v))
    throw new RuntimeError("dimReduceOmitNaN: argument must be a tensor");

  const info = forEachSlice(v.shape, dim, () => {});
  if (!info) return copyTensor(v);

  const result = new FloatXArray(info.totalElems);

  forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
    let acc = initialValue;
    let count = 0;
    for (let k = 0; k < srcIndices.length; k++) {
      const val = v.data[srcIndices[k]];
      if (!isNaN(val)) {
        acc = reduceFn(acc, val);
        count++;
      }
    }
    result[outIdx] = finalizeFn ? finalizeFn(acc, count) : acc;
  });

  return RTV.tensor(result, info.resultShape);
}

/** Reduce a tensor along a dimension using a whole-slice function (for median, mode, etc.) */
export function sliceDimReduce(
  v: RuntimeTensor,
  dim: number,
  sliceFn: (slice: ArrayLike<number>) => number
): RuntimeValue {
  const info = forEachSlice(v.shape, dim, () => {});
  if (!info) return RTV.tensor(new FloatXArray(v.data), [...v.shape]);

  const result = new FloatXArray(info.totalElems);
  forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
    const slice = new FloatXArray(srcIndices.length);
    for (let k = 0; k < srcIndices.length; k++) {
      slice[k] = v.data[srcIndices[k]];
    }
    result[outIdx] = sliceFn(slice);
  });
  return RTV.tensor(result, info.resultShape);
}

// ── Complex product reduction ──────────────────────────────────────────

/**
 * Complex product reduction. Can't use the independent real/imag approach
 * because (a+bi)(c+di) = (ac-bd)+(ad+bc)i mixes both parts.
 */
export function complexProd(v: RuntimeTensor, dim?: number): RuntimeValue {
  const re = v.data;
  const im = v.imag!;

  if (dim !== undefined) {
    const info = forEachSlice(v.shape, dim, () => {});
    if (!info) return copyTensor(v);

    const resultRe = new FloatXArray(info.totalElems);
    const resultIm = new FloatXArray(info.totalElems);
    forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
      let accRe = 1,
        accIm = 0;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const newRe = accRe * re[idx] - accIm * im[idx];
        const newIm = accRe * im[idx] + accIm * re[idx];
        accRe = newRe;
        accIm = newIm;
      }
      resultRe[outIdx] = accRe;
      resultIm[outIdx] = accIm;
    });
    const imOut = resultIm.some(x => x !== 0) ? resultIm : undefined;
    return RTV.tensor(resultRe, info.resultShape, imOut);
  }

  // No dim: reduce along first non-singleton dimension or to scalar
  const d = firstReduceDim(v.shape);
  if (d === 0) {
    let accRe = re[0],
      accIm = im[0];
    for (let k = 1; k < re.length; k++) {
      const newRe = accRe * re[k] - accIm * im[k];
      const newIm = accRe * im[k] + accIm * re[k];
      accRe = newRe;
      accIm = newIm;
    }
    return accIm !== 0 ? RTV.complex(accRe, accIm) : RTV.num(accRe);
  }
  return complexProd(v, d);
}

// ── Logical scan/reduction helpers ─────────────────────────────────────

/** Scan tensor elements for logical reduction (any/all).
 * mode 'any': returns true if any element is nonzero.
 * mode 'all': returns true if all elements are nonzero (true for empty). */
export function scanLogical(
  data: ArrayLike<number>,
  imag: ArrayLike<number> | undefined,
  mode: "any" | "all"
): boolean {
  const defaultResult = mode === "all";
  for (let i = 0; i < data.length; i++) {
    const isNonZero = data[i] !== 0 || (imag !== undefined && imag[i] !== 0);
    if (isNonZero !== defaultResult) return !defaultResult;
  }
  return defaultResult;
}

/** Reduce a tensor along a dimension using a logical test (any/all). */
export function logicalAlongDim(
  v: RuntimeTensor,
  dim: number,
  mode: "any" | "all"
): RuntimeValue {
  const info = forEachSlice(v.shape, dim, () => {});
  if (!info) {
    // dim exceeds rank: element-wise cast to logical
    const result = new FloatXArray(v.data.length);
    for (let i = 0; i < v.data.length; i++)
      result[i] = v.data[i] !== 0 || (v.imag && v.imag[i] !== 0) ? 1 : 0;
    const t = RTV.tensor(result, [...v.shape]);
    t._isLogical = true;
    return t;
  }

  const result = new FloatXArray(info.totalElems);
  forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
    let val: boolean = mode === "all";
    for (let k = 0; k < srcIndices.length; k++) {
      const srcIdx = srcIndices[k];
      const isNonZero =
        v.data[srcIdx] !== 0 || (v.imag !== undefined && v.imag[srcIdx] !== 0);
      if (mode === "any" && isNonZero) {
        val = true;
        break;
      }
      if (mode === "all" && !isNonZero) {
        val = false;
        break;
      }
    }
    result[outIdx] = val ? 1 : 0;
  });

  const t = RTV.tensor(result, info.resultShape);
  t._isLogical = true;
  return t;
}

// ── Sparse helpers ─────────────────────────────────────────────────────

/** Sum a sparse matrix along dim 1 (columns) or dim 2 (rows).
 *  Returns sparse for dim=1, dense tensor for dim=2. */
export function sparseSum(v: RuntimeSparseMatrix, dim: number): RuntimeValue {
  const isComplex = v.pi !== undefined;
  if (dim === 1) {
    const irArr: number[] = [];
    const prArr: number[] = [];
    const piArr: number[] = [];
    const jc = new Int32Array(v.n + 1);
    for (let c = 0; c < v.n; c++) {
      jc[c] = irArr.length;
      let sumRe = 0;
      let sumIm = 0;
      for (let k = v.jc[c]; k < v.jc[c + 1]; k++) {
        sumRe += v.pr[k];
        if (isComplex) sumIm += v.pi![k];
      }
      if (sumRe !== 0 || sumIm !== 0) {
        irArr.push(0);
        prArr.push(sumRe);
        if (isComplex) piArr.push(sumIm);
      }
    }
    jc[v.n] = irArr.length;
    return RTV.sparseMatrix(
      1,
      v.n,
      new Int32Array(irArr),
      jc,
      new Float64Array(prArr),
      isComplex ? new Float64Array(piArr) : undefined
    );
  }
  // dim === 2: Sum along rows → m × 1 dense column
  const result = new FloatXArray(v.m);
  const resultIm = isComplex ? new FloatXArray(v.m) : undefined;
  for (let c = 0; c < v.n; c++) {
    for (let k = v.jc[c]; k < v.jc[c + 1]; k++) {
      result[v.ir[k]] += v.pr[k];
      if (resultIm && v.pi) resultIm[v.ir[k]] += v.pi[k];
    }
  }
  return RTV.tensor(result, [v.m, 1], resultIm);
}

/** any/all on sparse matrix along a dimension.
 *  Returns sparse for dim=1, dense for dim=2. */
export function sparseAnyAll(
  v: RuntimeSparseMatrix,
  dim: number,
  mode: "any" | "all"
): RuntimeValue {
  if (dim === 1) {
    const irArr: number[] = [];
    const prArr: number[] = [];
    const jc = new Int32Array(v.n + 1);
    for (let c = 0; c < v.n; c++) {
      jc[c] = irArr.length;
      const nnzCol = v.jc[c + 1] - v.jc[c];
      const val =
        mode === "any" ? (nnzCol > 0 ? 1 : 0) : nnzCol === v.m ? 1 : 0;
      if (val !== 0) {
        irArr.push(0);
        prArr.push(1);
      }
    }
    jc[v.n] = irArr.length;
    return RTV.sparseMatrix(
      1,
      v.n,
      new Int32Array(irArr),
      jc,
      new Float64Array(prArr)
    );
  }
  // dim === 2: Along rows
  const hasNonzero = new Uint8Array(v.m);
  let nnzPerRow: Int32Array | undefined;
  if (mode === "all") nnzPerRow = new Int32Array(v.m);
  for (let c = 0; c < v.n; c++) {
    for (let k = v.jc[c]; k < v.jc[c + 1]; k++) {
      hasNonzero[v.ir[k]] = 1;
      if (nnzPerRow) nnzPerRow[v.ir[k]]++;
    }
  }
  const result = new FloatXArray(v.m);
  for (let r = 0; r < v.m; r++) {
    if (mode === "any") result[r] = hasNonzero[r] ? 1 : 0;
    else result[r] = nnzPerRow![r] === v.n ? 1 : 0;
  }
  const t = RTV.tensor(result, [v.m, 1]) as RuntimeTensor;
  t._isLogical = true;
  return t;
}

// ── Type check helpers ─────────────────────────────────────────────────

/** Type check for reductions that return Num without dim arg, Tensor with dim arg. */
export function reductionCheck(
  argTypes: ItemType[],
  nargout: number
): { outputTypes: ItemType[] } | null {
  if (nargout !== 1) return null;
  const inputIsComplex =
    argTypes.length >= 1 &&
    argTypes[0].kind === "Tensor" &&
    argTypes[0].isComplex;
  if (argTypes.length >= 2) {
    return {
      outputTypes: [
        {
          kind: "Tensor",
          isComplex: inputIsComplex || undefined,
        },
      ],
    };
  }
  if (argTypes.length >= 1 && argTypes[0].kind === "Tensor") {
    return { outputTypes: [{ kind: "Unknown" }] };
  }
  if (inputIsComplex) {
    return { outputTypes: [{ kind: "ComplexNumber" }] };
  }
  return { outputTypes: [{ kind: "Number" }] };
}

/** Type check for min/max: nargout=1 → Num, nargout=2 → [Num, Num].
 *  With 3 args (e.g. max(A,[],dim)), output is Tensor. */
export function minMaxCheck(
  argTypes: ItemType[],
  nargout: number
): { outputTypes: ItemType[] } | null {
  const inputIsComplex =
    argTypes.length >= 1 &&
    argTypes[0].kind === "Tensor" &&
    argTypes[0].isComplex;
  if (argTypes.length === 3) {
    const t: ItemType = {
      kind: "Tensor",
      isComplex: inputIsComplex || undefined,
    };
    if (nargout === 1) return { outputTypes: [t] };
    if (nargout === 2) return { outputTypes: [t, t] };
    return null;
  }
  const inputIsTensor = argTypes.length >= 1 && argTypes[0].kind === "Tensor";
  if (inputIsTensor) {
    const u: ItemType = { kind: "Unknown" };
    if (nargout === 1) return { outputTypes: [u] };
    if (nargout === 2) return { outputTypes: [u, u] };
    return null;
  }
  const scalarType: ItemType = inputIsComplex
    ? { kind: "ComplexNumber" }
    : { kind: "Number" };
  if (nargout === 1) return { outputTypes: [scalarType] };
  if (nargout === 2) return { outputTypes: [scalarType, { kind: "Number" }] };
  return null;
}

/** Type check for functions that preserve the input type (sort, cumsum). */
export function preserveTypeCheck(
  argTypes: ItemType[],
  nargout: number
): { outputTypes: ItemType[] } | null {
  if (argTypes.length < 1) return null;
  const t = argTypes[0];
  const outType =
    t.kind === "Number"
      ? { kind: "Number" as const }
      : t.kind === "Tensor"
        ? t
        : { kind: "Unknown" as const };
  if (nargout === 1) return { outputTypes: [outType] };
  if (nargout === 2) return { outputTypes: [outType, outType] };
  return null;
}

// ── NaN flag parsing ────────────────────────────────────────────────────

/** Strip trailing 'omitnan'/'includenan' from args. Returns cleaned args and flag. */
export function parseNanFlag(args: RuntimeValue[]): {
  args: RuntimeValue[];
  omitNaN: boolean;
} {
  if (args.length >= 2) {
    const last = args[args.length - 1];
    if (isRuntimeChar(last)) {
      const s = toString(last).toLowerCase();
      if (s === "omitnan") return { args: args.slice(0, -1), omitNaN: true };
      if (s === "includenan")
        return { args: args.slice(0, -1), omitNaN: false };
    }
  }
  return { args, omitNaN: false };
}

/** Filter NaN values from a Float64Array, returning a new (possibly shorter) array. */
export function filterNaN(arr: ArrayLike<number>): Float64Array {
  const out: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (!isNaN(arr[i] as number)) out.push(arr[i] as number);
  }
  return new Float64Array(out);
}

// ── Reduction factories ────────────────────────────────────────────────

export type ReductionKernel = {
  reduceAll: (v: RuntimeTensor) => RuntimeValue;
  reduceDim: (v: RuntimeTensor, dim: number) => RuntimeValue;
};

/** Unified factory for reductions (sum, mean, median, mode, etc.).
 *  Handles arg parsing: scalar passthrough, 'all' flag, dim arg, default dim.
 *  Supports 'omitnan'/'includenan' nanflag as last string argument. */
export function makeReduction(
  name: string,
  kernel: ReductionKernel,
  omitNaNKernel?: ReductionKernel
): {
  check: typeof reductionCheck;
  apply: (args: RuntimeValue[]) => RuntimeValue;
} {
  return {
    check: reductionCheck,
    apply: rawArgs => {
      if (rawArgs.length < 1)
        throw new RuntimeError(`${name} requires at least 1 argument`);
      const { args, omitNaN } = parseNanFlag(rawArgs);
      const k = omitNaN && omitNaNKernel ? omitNaNKernel : kernel;
      const v = args[0];
      if (isRuntimeNumber(v)) return v;
      if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
      if (isRuntimeTensor(v)) {
        if (args.length >= 2) {
          if (isRuntimeChar(args[1]) && toString(args[1]) === "all")
            return k.reduceAll(v);
          return k.reduceDim(v, Math.round(toNumber(args[1])));
        }
        const d = firstReduceDim(v.shape);
        return d === 0 ? k.reduceAll(v) : k.reduceDim(v, d);
      }
      throw new RuntimeError(`${name}: argument must be numeric`);
    },
  };
}

/** Create an accumulator-based reduction kernel (sum, mean, etc.) */
export function accumKernel(
  reduceFn: (acc: number, val: number) => number,
  initial: number,
  finalizeFn?: (acc: number, count: number) => number
): ReductionKernel {
  return {
    reduceAll: v => {
      let acc = initial;
      for (let i = 0; i < v.data.length; i++) acc = reduceFn(acc, v.data[i]);
      const re = finalizeFn ? finalizeFn(acc, v.data.length) : acc;
      if (v.imag) {
        let accIm = initial;
        for (let i = 0; i < v.imag.length; i++)
          accIm = reduceFn(accIm, v.imag[i]);
        const im = finalizeFn ? finalizeFn(accIm, v.data.length) : accIm;
        if (im !== 0) return RTV.complex(re, im);
      }
      return RTV.num(re);
    },
    reduceDim: (v, dim) => dimReduce(v, dim, reduceFn, initial, finalizeFn),
  };
}

/** Create a slice-based reduction kernel (median, mode, etc.) */
export function sliceKernel(
  sliceFn: (slice: ArrayLike<number>) => number
): ReductionKernel {
  return {
    reduceAll: v => RTV.num(sliceFn(v.data)),
    reduceDim: (v, dim) => sliceDimReduce(v, dim, sliceFn),
  };
}

/** Create an accumulator kernel that skips NaN values. */
export function accumKernelOmitNaN(
  reduceFn: (acc: number, val: number) => number,
  initial: number,
  finalizeFn?: (acc: number, count: number) => number
): ReductionKernel {
  return {
    reduceAll: v => {
      let acc = initial;
      let count = 0;
      for (let i = 0; i < v.data.length; i++) {
        if (!isNaN(v.data[i])) {
          acc = reduceFn(acc, v.data[i]);
          count++;
        }
      }
      const re = finalizeFn ? finalizeFn(acc, count) : acc;
      if (v.imag) {
        let accIm = initial;
        let countIm = 0;
        for (let i = 0; i < v.imag.length; i++) {
          if (!isNaN(v.imag[i])) {
            accIm = reduceFn(accIm, v.imag[i]);
            countIm++;
          }
        }
        const im = finalizeFn ? finalizeFn(accIm, countIm) : accIm;
        if (im !== 0) return RTV.complex(re, im);
      }
      return RTV.num(re);
    },
    reduceDim: (v, dim) =>
      dimReduceOmitNaN(v, dim, reduceFn, initial, finalizeFn),
  };
}

/** Create a slice kernel that filters NaN before applying the function. */
export function sliceKernelOmitNaN(
  sliceFn: (slice: ArrayLike<number>) => number
): ReductionKernel {
  const filteredFn = (slice: ArrayLike<number>) => sliceFn(filterNaN(slice));
  return {
    reduceAll: v => RTV.num(filteredFn(v.data)),
    reduceDim: (v, dim) => sliceDimReduce(v, dim, filteredFn),
  };
}

/** Helper: extract all numeric values from a RuntimeValue as a plain array. */
export function toNumArray(v: RuntimeValue, name: string): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  throw new RuntimeError(`${name}: arguments must be numeric arrays`);
}
