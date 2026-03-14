/**
 * Reduction operations builtin functions
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  toString,
  RuntimeError,
  sub2ind,
  tensorSize2D,
} from "../runtime/index.js";
import { ItemType } from "../lowering/itemTypes.js";
import { register, builtinSingle } from "./registry.js";
import {
  FloatXArray,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
  type RuntimeTensor,
} from "../runtime/types.js";
import { getBroadcastShape, broadcastIterate } from "./arithmetic.js";
import { rstr } from "../runtime/runtime.js";

/** Advance subscripts in column-major order, optionally skipping a dimension.
 * Returns true if there are more elements to visit. */
function nextSubscripts(
  subs: number[],
  shape: number[],
  skipDim?: number
): boolean {
  for (let d = 0; d < subs.length; d++) {
    if (skipDim !== undefined && d === skipDim) continue;
    subs[d]++;
    if (subs[d] < shape[d]) return true;
    subs[d] = 0;
  }
  return false;
}

/** Squeeze trailing singleton dimensions, keeping at least 2. Mutates in place. */
function squeezeTrailing(shape: number[]): void {
  while (shape.length > 2 && shape[shape.length - 1] === 1) {
    shape.pop();
  }
}

/** Scan tensor elements for logical reduction (any/all).
 * mode 'any': returns true if any element is nonzero.
 * mode 'all': returns true if all elements are nonzero (true for empty). */
function scanLogical(
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
function logicalAlongDim(
  v: RuntimeTensor,
  dim: number,
  mode: "any" | "all"
): RuntimeValue {
  const shape = v.shape;
  const dimIdx = dim - 1;

  if (dimIdx >= shape.length) {
    const result = new FloatXArray(v.data.length);
    for (let i = 0; i < v.data.length; i++)
      result[i] = v.data[i] !== 0 || (v.imag && v.imag[i] !== 0) ? 1 : 0;
    const t = RTV.tensor(result, [...shape]);
    t._isLogical = true;
    return t;
  }

  const reduceDimSize = shape[dimIdx];
  const resultShape = [...shape];
  resultShape[dimIdx] = 1;
  const totalElems = resultShape.reduce((a, b) => a * b, 1);
  const result = new FloatXArray(totalElems);

  const outSubs = new Array(shape.length).fill(0);
  for (let i = 0; i < totalElems; i++) {
    let val: boolean = mode === "all";
    for (let k = 0; k < reduceDimSize; k++) {
      const srcSubs = [...outSubs];
      srcSubs[dimIdx] = k;
      const srcIdx = sub2ind(shape, srcSubs);
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
    result[i] = val ? 1 : 0;
    nextSubscripts(outSubs, resultShape, dimIdx);
  }

  squeezeTrailing(resultShape);
  const t = RTV.tensor(result, resultShape);
  t._isLogical = true;
  return t;
}

/**
 * Helper for N-dimensional reduction operations (sum, prod, mean, etc.)
 * @param v The tensor to reduce
 * @param dim The dimension along which to reduce (1-based)
 * @param reduceFn Function that takes (accumulator, value) and returns new accumulator
 * @param initialValue Initial accumulator value
 * @param finalizeFn Optional function to transform final result (e.g., divide by count for mean)
 */
function dimReduce(
  v: RuntimeValue,
  dim: number,
  reduceFn: (acc: number, val: number) => number,
  initialValue: number,
  finalizeFn?: (acc: number, count: number) => number
): RuntimeValue {
  if (!isRuntimeTensor(v))
    throw new RuntimeError("dimReduce: argument must be a tensor");

  const shape = v.shape;
  const dimIdx = dim - 1; // Convert to 0-based

  // If dim is beyond the tensor's dimensions, nothing to reduce — return copy
  if (dimIdx >= shape.length) {
    return RTV.tensor(
      new FloatXArray(v.data),
      [...shape],
      v.imag ? new FloatXArray(v.imag) : undefined
    );
  }

  const reduceDimSize = shape[dimIdx];

  // Result shape: same as input but with reduced dimension set to 1
  const resultShape = [...shape];
  resultShape[dimIdx] = 1;
  const totalElems = resultShape.reduce((a, b) => a * b, 1);
  const result = new FloatXArray(totalElems);
  const resultImag = v.imag ? new FloatXArray(totalElems) : undefined;

  // For each output element, accumulate along the reduction dimension
  const outSubs = new Array(shape.length).fill(0);
  for (let i = 0; i < totalElems; i++) {
    let acc = initialValue;
    let accIm = resultImag ? initialValue : 0;
    for (let k = 0; k < reduceDimSize; k++) {
      const srcSubs = [...outSubs];
      srcSubs[dimIdx] = k;
      const srcIdx = sub2ind(shape, srcSubs);
      acc = reduceFn(acc, v.data[srcIdx]);
      if (resultImag) accIm = reduceFn(accIm, v.imag![srcIdx]);
    }
    result[i] = finalizeFn ? finalizeFn(acc, reduceDimSize) : acc;
    if (resultImag)
      resultImag[i] = finalizeFn ? finalizeFn(accIm, reduceDimSize) : accIm;

    nextSubscripts(outSubs, resultShape, dimIdx);
  }

  squeezeTrailing(resultShape);

  // Drop imaginary part if all zeros
  const imOut =
    resultImag && resultImag.some(x => x !== 0) ? resultImag : undefined;
  return RTV.tensor(result, resultShape, imOut);
}

/**
 * Complex product reduction. Can't use the independent real/imag approach
 * because (a+bi)(c+di) = (ac-bd)+(ad+bc)i mixes both parts.
 */
function complexProd(v: RuntimeTensor, dim?: number): RuntimeValue {
  const shape = v.shape;
  const re = v.data;
  const im = v.imag!;

  // Helper: multiply along a contiguous slice of the flat array
  const mulSlice = (
    start: number,
    count: number
  ): { re: number; im: number } => {
    let accRe = re[start],
      accIm = im[start];
    for (let k = 1; k < count; k++) {
      const idx = start + k;
      const newRe = accRe * re[idx] - accIm * im[idx];
      const newIm = accRe * im[idx] + accIm * re[idx];
      accRe = newRe;
      accIm = newIm;
    }
    return { re: accRe, im: accIm };
  };

  if (dim !== undefined) {
    // Reduce along specified dimension using sub2ind
    const dimIdx = dim - 1;
    if (dimIdx >= shape.length) {
      return RTV.tensor(new FloatXArray(re), [...shape], new FloatXArray(im));
    }
    const reduceDimSize = shape[dimIdx];
    const resultShape = [...shape];
    resultShape[dimIdx] = 1;
    const totalElems = resultShape.reduce((a, b) => a * b, 1);
    const resultRe = new FloatXArray(totalElems);
    const resultIm = new FloatXArray(totalElems);
    const outSubs = new Array(shape.length).fill(0);
    for (let i = 0; i < totalElems; i++) {
      let accRe = 1,
        accIm = 0;
      for (let k = 0; k < reduceDimSize; k++) {
        const srcSubs = [...outSubs];
        srcSubs[dimIdx] = k;
        const srcIdx = sub2ind(shape, srcSubs);
        const newRe = accRe * re[srcIdx] - accIm * im[srcIdx];
        const newIm = accRe * im[srcIdx] + accIm * re[srcIdx];
        accRe = newRe;
        accIm = newIm;
      }
      resultRe[i] = accRe;
      resultIm[i] = accIm;
      nextSubscripts(outSubs, resultShape, dimIdx);
    }
    squeezeTrailing(resultShape);
    const imOut = resultIm.some(x => x !== 0) ? resultIm : undefined;
    return RTV.tensor(resultRe, resultShape, imOut);
  }

  // No dim: reduce along first non-singleton dimension
  const numNonSingleton = shape.filter(d => d > 1).length;
  if (numNonSingleton <= 1) {
    // Vector or scalar: full linear product → scalar
    const r = mulSlice(0, re.length);
    return r.im !== 0 ? RTV.complex(r.re, r.im) : RTV.num(r.re);
  }
  // Multiple non-singleton dims: reduce along first non-singleton dimension
  const firstNonSingleton = shape.findIndex(d => d > 1);
  return complexProd(v, firstNonSingleton + 1);
}

// Type check for reductions that return Num without dim arg, Tensor with dim arg
function reductionCheck(
  argTypes: ItemType[],
  nargout: number
): { outputTypes: ItemType[] } | null {
  if (nargout !== 1) return null;
  const inputIsComplex =
    argTypes.length >= 1 &&
    argTypes[0].kind === "Tensor" &&
    argTypes[0].isComplex;
  if (argTypes.length >= 2) {
    // With dim argument → result is a Tensor
    return {
      outputTypes: [
        {
          kind: "Tensor",
          isComplex: inputIsComplex || undefined,
        },
      ],
    };
  }
  // No dim arg: result is scalar for vectors but tensor for matrices.
  // We can't distinguish at compile time, so return Unknown for tensor inputs.
  if (argTypes.length >= 1 && argTypes[0].kind === "Tensor") {
    return { outputTypes: [{ kind: "Unknown" }] };
  }
  if (inputIsComplex) {
    return { outputTypes: [{ kind: "ComplexNumber" }] };
  }
  return { outputTypes: [{ kind: "Number" }] };
}

// Type check for min/max: nargout=1 → Num, nargout=2 → [Num, Num]
// With 3 args (e.g. max(A,[],dim)), output is Tensor
function minMaxCheck(
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
  // For tensor inputs, result could be scalar (vector input) or tensor (matrix
  // input) — we can't distinguish at compile time, so return Unknown.
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
  // Second output (index) is always a real number
  if (nargout === 1) return { outputTypes: [scalarType] };
  if (nargout === 2) return { outputTypes: [scalarType, { kind: "Number" }] };
  return null;
}

// Type check for functions that preserve the input type (sort, cumsum)
function preserveTypeCheck(
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

export function registerReductionFunctions(): void {
  /** Default dim for reductions: reduce along first non-singleton dimension.
   *  Vectors (at most one non-singleton dim) → scalar. Otherwise → dimReduce. */
  const defaultDimOrScalar = (
    v: RuntimeTensor,
    reduceFn: (acc: number, val: number) => number,
    initial: number,
    finalizeFn?: (acc: number, count: number) => number
  ): RuntimeValue => {
    const shape = v.shape;
    const numNonSingleton = shape.filter(d => d > 1).length;
    // Vector or scalar: full linear reduction → scalar result
    if (numNonSingleton <= 1) {
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
    }
    // Multiple non-singleton dims: reduce along first non-singleton dimension
    const firstNonSingleton = shape.findIndex(d => d > 1);
    return dimReduce(v, firstNonSingleton + 1, reduceFn, initial, finalizeFn);
  };

  /** Factory for accumulator-based reductions (sum, mean, etc.) */
  const makeAccumReduction = (
    name: string,
    reduceFn: (acc: number, val: number) => number,
    initial: number,
    finalizeFn?: (acc: number, count: number) => number
  ): {
    check: typeof reductionCheck;
    apply: (args: RuntimeValue[]) => RuntimeValue;
  } => ({
    check: reductionCheck,
    apply: args => {
      if (args.length < 1)
        throw new RuntimeError(`${name} requires at least 1 argument`);
      const v = args[0];
      if (isRuntimeNumber(v)) return v;
      if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
      if (isRuntimeTensor(v)) {
        if (args.length >= 2) {
          if (isRuntimeChar(args[1]) && toString(args[1]) === "all") {
            let acc = initial;
            for (let i = 0; i < v.data.length; i++)
              acc = reduceFn(acc, v.data[i]);
            return RTV.num(finalizeFn ? finalizeFn(acc, v.data.length) : acc);
          }
          const dim = Math.round(toNumber(args[1]));
          return dimReduce(v, dim, reduceFn, initial, finalizeFn);
        }
        return defaultDimOrScalar(v, reduceFn, initial, finalizeFn);
      }
      throw new RuntimeError(`${name}: argument must be numeric`);
    },
  });

  /** Reduce a tensor along a dimension using a whole-slice function (for median, mode, etc.) */
  const sliceDimReduce = (
    v: RuntimeTensor,
    dim: number,
    sliceFn: (slice: ArrayLike<number>) => number
  ): RuntimeValue => {
    const shape = v.shape;
    const dimIdx = dim - 1;
    if (dimIdx >= shape.length) {
      return RTV.tensor(new FloatXArray(v.data), [...shape]);
    }
    const reduceDimSize = shape[dimIdx];
    const resultShape = [...shape];
    resultShape[dimIdx] = 1;
    const totalElems = resultShape.reduce((a, b) => a * b, 1);
    const result = new FloatXArray(totalElems);
    const outSubs = new Array(shape.length).fill(0);
    for (let i = 0; i < totalElems; i++) {
      const slice = new FloatXArray(reduceDimSize);
      for (let k = 0; k < reduceDimSize; k++) {
        const srcSubs = [...outSubs];
        srcSubs[dimIdx] = k;
        slice[k] = v.data[sub2ind(shape, srcSubs)];
      }
      result[i] = sliceFn(slice);
      nextSubscripts(outSubs, resultShape, dimIdx);
    }
    squeezeTrailing(resultShape);
    return RTV.tensor(result, resultShape);
  };

  /** Factory for slice-based reductions (median, mode, etc.) */
  const makeSliceReduction = (
    name: string,
    sliceFn: (slice: ArrayLike<number>) => number
  ): {
    check: typeof reductionCheck;
    apply: (args: RuntimeValue[]) => RuntimeValue;
  } => ({
    check: reductionCheck,
    apply: args => {
      if (args.length < 1)
        throw new RuntimeError(`${name} requires at least 1 argument`);
      const v = args[0];
      if (isRuntimeNumber(v)) return v;
      if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
      if (isRuntimeTensor(v)) {
        if (args.length >= 2) {
          if (isRuntimeChar(args[1]) && toString(args[1]) === "all") {
            return RTV.num(sliceFn(v.data));
          }
          const dim = Math.round(toNumber(args[1]));
          return sliceDimReduce(v, dim, sliceFn);
        }
        const shape = v.shape;
        const numNonSingleton = shape.filter(d => d > 1).length;
        // Vector or scalar: scalar result
        if (numNonSingleton <= 1) {
          return RTV.num(sliceFn(v.data));
        }
        // Multiple non-singleton dims: reduce along first non-singleton dimension
        const firstNonSingleton = shape.findIndex(d => d > 1);
        return sliceDimReduce(v, firstNonSingleton + 1, sliceFn);
      }
      throw new RuntimeError(`${name}: argument must be numeric`);
    },
  });

  register("sum", [makeAccumReduction("sum", (acc, val) => acc + val, 0)]);

  register("prod", [
    {
      check: reductionCheck,
      apply: args => {
        if (args.length < 1)
          throw new RuntimeError("prod requires at least 1 argument");
        const v = args[0];
        if (isRuntimeNumber(v)) return v;
        if (isRuntimeTensor(v)) {
          // Complex prod needs special handling (can't reduce real/imag independently)
          if (v.imag) {
            return complexProd(
              v,
              args.length >= 2 ? Math.round(toNumber(args[1])) : undefined
            );
          }
          if (args.length >= 2) {
            if (isRuntimeChar(args[1]) && toString(args[1]) === "all") {
              let acc = 1;
              for (let i = 0; i < v.data.length; i++) acc *= v.data[i];
              return RTV.num(acc);
            }
            const dim = Math.round(toNumber(args[1]));
            return dimReduce(v, dim, (acc, val) => acc * val, 1);
          }
          return defaultDimOrScalar(v, (acc, val) => acc * val, 1);
        }
        throw new RuntimeError("prod: argument must be numeric");
      },
    },
  ]);

  /** Shared implementation for min/max: finds the extreme value along dim 1. */
  const minMaxImpl = (
    name: string,
    args: RuntimeValue[],
    nargout: number,
    initial: number,
    isBetter: (candidate: number, current: number) => boolean,
    twoArgFn: (a: number, b: number) => number
  ): RuntimeValue | RuntimeValue[] => {
    // Complex comparison: by magnitude, ties broken by angle
    const complexIsBetter = (
      reA: number,
      imA: number,
      reB: number,
      imB: number
    ): boolean => {
      const absA = Math.sqrt(reA * reA + imA * imA);
      const absB = Math.sqrt(reB * reB + imB * imB);
      if (absA !== absB) return isBetter(absA, absB);
      return isBetter(Math.atan2(imA, reA), Math.atan2(imB, reB));
    };

    if (args.length === 1) {
      const v = args[0];
      if (isRuntimeNumber(v)) {
        if (nargout > 1) return [v, RTV.num(1)];
        return v;
      }
      if (isRuntimeLogical(v)) {
        if (nargout > 1) return [v, RTV.num(1)];
        return v;
      }
      if (isRuntimeComplexNumber(v)) {
        if (nargout > 1) return [v, RTV.num(1)];
        return v;
      }
      if (isRuntimeTensor(v)) {
        // Empty tensor: return empty array (Returns [] for max([]), min([]))
        if (v.data.length === 0) {
          const empty = RTV.tensor(new FloatXArray(0), [0, 0]);
          if (nargout > 1) return [empty, empty];
          return empty;
        }
        const shape = v.shape;
        const numNonSingleton = shape.filter(d => d > 1).length;
        if (v.imag) {
          // Complex tensor: compare by real part, ties by imaginary part
          const im = v.imag;
          if (numNonSingleton <= 1) {
            let mRe = initial,
              mIm = 0,
              mIdx = 0;
            let foundNonNaN = false;
            for (let i = 0; i < v.data.length; i++) {
              if (v.data[i] !== v.data[i] || im[i] !== im[i]) continue; // skip NaN
              if (!foundNonNaN || complexIsBetter(v.data[i], im[i], mRe, mIm)) {
                mRe = v.data[i];
                mIm = im[i];
                mIdx = i;
                foundNonNaN = true;
              }
            }
            if (!foundNonNaN) {
              mRe = NaN;
              mIm = 0;
            }
            const result = mIm === 0 ? RTV.num(mRe) : RTV.complex(mRe, mIm);
            if (nargout > 1) return [result, RTV.num(mIdx + 1)];
            return result;
          }
          // Multiple non-singleton dims: reduce along first non-singleton
          const dim = shape.findIndex(d => d > 1) + 1;
          return minMaxImpl(
            name,
            [v, RTV.num(0), RTV.num(dim)],
            nargout,
            initial,
            isBetter,
            twoArgFn
          );
        }
        // Real tensor (existing behavior)
        if (numNonSingleton <= 1) {
          let m = initial,
            mIdx = 0;
          let foundNonNaN = false;
          for (let i = 0; i < v.data.length; i++) {
            const val = v.data[i];
            if (val !== val) continue; // skip NaN
            if (!foundNonNaN || isBetter(val, m)) {
              m = val;
              mIdx = i;
              foundNonNaN = true;
            }
          }
          if (!foundNonNaN) m = NaN;
          if (nargout > 1)
            return [
              v._isLogical ? RTV.logical(m !== 0) : RTV.num(m),
              RTV.num(mIdx + 1),
            ];
          return v._isLogical ? RTV.logical(m !== 0) : RTV.num(m);
        }
        // Multiple non-singleton dims: reduce along first non-singleton dimension
        const dim = shape.findIndex(d => d > 1) + 1; // 1-based
        // Delegate to the 3-arg dim-reduction path
        return minMaxImpl(
          name,
          [v, RTV.num(0), RTV.num(dim)],
          nargout,
          initial,
          isBetter,
          twoArgFn
        );
      }
    }
    if (args.length === 2) {
      const a = args[0];
      const b = args[1];
      // Handle complex scalar inputs
      if (isRuntimeComplexNumber(a) || isRuntimeComplexNumber(b)) {
        const aRe = isRuntimeNumber(a)
          ? a
          : isRuntimeComplexNumber(a)
            ? a.re
            : NaN;
        const aIm = isRuntimeComplexNumber(a) ? a.im : 0;
        const bRe = isRuntimeNumber(b)
          ? b
          : isRuntimeComplexNumber(b)
            ? b.re
            : NaN;
        const bIm = isRuntimeComplexNumber(b) ? b.im : 0;
        const pickA = complexIsBetter(aRe, aIm, bRe, bIm);
        const re = pickA ? aRe : bRe;
        const im = pickA ? aIm : bIm;
        return im === 0 ? RTV.num(re) : RTV.complex(re, im);
      }
      const aIsScalar = isRuntimeNumber(a) || isRuntimeLogical(a);
      const bIsScalar = isRuntimeNumber(b) || isRuntimeLogical(b);
      if (aIsScalar && bIsScalar) {
        const aVal = toNumber(a),
          bVal = toNumber(b);
        // NaN-aware — if one is NaN, return the other
        const r = isNaN(aVal)
          ? bVal
          : isNaN(bVal)
            ? aVal
            : twoArgFn(aVal, bVal);
        return RTV.num(r);
      }
      // Element-wise with broadcasting
      const aT: RuntimeTensor = aIsScalar
        ? (RTV.tensor(new FloatXArray([toNumber(a)]), [1, 1]) as RuntimeTensor)
        : (a as RuntimeTensor);
      const bT: RuntimeTensor = bIsScalar
        ? (RTV.tensor(new FloatXArray([toNumber(b)]), [1, 1]) as RuntimeTensor)
        : (b as RuntimeTensor);
      const outShape = getBroadcastShape(aT.shape, bT.shape);
      if (!outShape) {
        throw new RuntimeError(`${name}: non-singleton dimensions must match`);
      }
      const result = new FloatXArray(outShape.reduce((acc, d) => acc * d, 1));
      broadcastIterate(aT.shape, bT.shape, outShape, (aIdx, bIdx, i) => {
        const aVal = aT.data[aIdx],
          bVal = bT.data[bIdx];
        result[i] = isNaN(aVal)
          ? bVal
          : isNaN(bVal)
            ? aVal
            : twoArgFn(aVal, bVal);
      });
      return RTV.tensor(result, outShape);
    }
    // max(A, [], dim) or max(A, [], 'all') — reduce along specified dimension
    if (args.length === 3) {
      const v = args[0];
      // Handle 'all' flag: reduce across all elements
      if (
        (isRuntimeString(args[2]) || isRuntimeChar(args[2])) &&
        rstr(args[2]) === "all"
      ) {
        return minMaxImpl(
          name,
          [
            isRuntimeTensor(v)
              ? RTV.tensor(v.data, [1, v.data.length], v.imag)
              : v,
          ],
          nargout,
          initial,
          isBetter,
          twoArgFn
        );
      }
      // Handle vector of dimensions: max(A, [], [d1, d2, ...])
      if (isRuntimeTensor(args[2])) {
        const dims = Array.from(args[2].data).map(d => Math.round(d));
        // Sort descending so reducing higher dims first doesn't shift lower dim indices
        const sortedDims = [...dims].sort((a, b) => b - a);
        let result: RuntimeValue = v;
        for (const d of sortedDims) {
          const r = minMaxImpl(
            name,
            [result, RTV.num(0), RTV.num(d)],
            1, // nargout=1 for intermediate reductions
            initial,
            isBetter,
            twoArgFn
          );
          result = Array.isArray(r) ? r[0] : r;
        }
        return result;
      }
      const dim = Math.round(toNumber(args[2]));
      if (isRuntimeNumber(v)) return v;
      if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
      if (isRuntimeTensor(v)) {
        const shape = v.shape;
        const dimIdx = dim - 1; // 0-based

        // If dim is beyond tensor's dimensions, return a copy
        if (dimIdx >= shape.length) {
          return RTV.tensor(
            new FloatXArray(v.data),
            [...shape],
            v.imag ? new FloatXArray(v.imag) : undefined
          );
        }

        const reduceDimSize = shape[dimIdx];
        const resultShape = [...shape];
        resultShape[dimIdx] = 1;
        const totalElems = resultShape.reduce((a, b) => a * b, 1);

        if (v.imag) {
          // Complex tensor: compare by real part, ties by imaginary part
          const im = v.imag;
          const resultRe = new FloatXArray(totalElems);
          const resultIm = new FloatXArray(totalElems);
          const indices = nargout > 1 ? new FloatXArray(totalElems) : undefined;

          const outSubsC = new Array(shape.length).fill(0);
          for (let i = 0; i < totalElems; i++) {
            let mRe = initial,
              mIm = 0,
              mIdx = 0;
            let foundNonNaN = false;
            for (let k = 0; k < reduceDimSize; k++) {
              const srcSubs = [...outSubsC];
              srcSubs[dimIdx] = k;
              const srcIdx = sub2ind(shape, srcSubs);
              if (
                v.data[srcIdx] !== v.data[srcIdx] ||
                im[srcIdx] !== im[srcIdx]
              )
                continue; // skip NaN
              if (
                !foundNonNaN ||
                complexIsBetter(v.data[srcIdx], im[srcIdx], mRe, mIm)
              ) {
                mRe = v.data[srcIdx];
                mIm = im[srcIdx];
                mIdx = k;
                foundNonNaN = true;
              }
            }
            if (!foundNonNaN) {
              mRe = NaN;
              mIm = 0;
            }
            resultRe[i] = mRe;
            resultIm[i] = mIm;
            if (indices) indices[i] = mIdx + 1;

            nextSubscripts(outSubsC, resultShape, dimIdx);
          }

          squeezeTrailing(resultShape);

          const hasImag = resultIm.some(x => x !== 0);
          const outTensorC = RTV.tensor(
            resultRe,
            resultShape,
            hasImag ? resultIm : undefined
          );
          if (nargout > 1)
            return [outTensorC, RTV.tensor(indices!, resultShape)];
          return outTensorC;
        }

        const result = new FloatXArray(totalElems);
        const indices = nargout > 1 ? new FloatXArray(totalElems) : undefined;

        const outSubs = new Array(shape.length).fill(0);
        for (let i = 0; i < totalElems; i++) {
          let m = initial,
            mIdx = 0;
          let foundNonNaN = false;
          for (let k = 0; k < reduceDimSize; k++) {
            const srcSubs = [...outSubs];
            srcSubs[dimIdx] = k;
            const srcIdx = sub2ind(shape, srcSubs);
            const val = v.data[srcIdx];
            if (val !== val) continue; // skip NaN
            if (!foundNonNaN || isBetter(val, m)) {
              m = val;
              mIdx = k;
              foundNonNaN = true;
            }
          }
          if (!foundNonNaN) m = NaN;
          result[i] = m;
          if (indices) indices[i] = mIdx + 1;

          nextSubscripts(outSubs, resultShape, dimIdx);
        }

        squeezeTrailing(resultShape);

        const outTensor3 = RTV.tensor(result, resultShape);
        if (v._isLogical) outTensor3._isLogical = true;
        if (nargout > 1) return [outTensor3, RTV.tensor(indices!, resultShape)];
        return outTensor3;
      }
    }
    throw new RuntimeError(`${name}: invalid arguments`);
  };

  register("min", [
    {
      check: minMaxCheck,
      apply: (args, nargout) =>
        minMaxImpl("min", args, nargout, Infinity, (a, b) => a < b, Math.min),
    },
  ]);

  register("max", [
    {
      check: minMaxCheck,
      apply: (args, nargout) =>
        minMaxImpl("max", args, nargout, -Infinity, (a, b) => a > b, Math.max),
    },
  ]);

  register("mean", [
    makeAccumReduction(
      "mean",
      (acc, val) => acc + val,
      0,
      (sum, count) => sum / count
    ),
  ]);

  // Helper: compute variance of a slice
  const varianceOf = (slice: ArrayLike<number>, w: number): number => {
    const n = slice.length;
    if (n <= 1 && w === 0) return 0;
    let s = 0;
    for (let i = 0; i < n; i++) s += slice[i];
    const m = s / n;
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (slice[i] - m) ** 2;
    const denom = w === 1 ? n : n - 1;
    return ss / denom;
  };

  // Helper: std/var apply with dimension support
  const stdVarApply = (
    name: string,
    transform: (variance: number) => number
  ) => {
    return (args: RuntimeValue[]): RuntimeValue => {
      if (args.length < 1)
        throw new RuntimeError(`${name} requires at least 1 argument`);
      const v = args[0];
      // w=0 (default): normalize by N-1; w=1: normalize by N
      const w = args.length >= 2 ? toNumber(args[1]) : 0;
      // dim argument: std(X,w,dim)
      const dimArg = args.length >= 3 ? Math.round(toNumber(args[2])) : 0;
      if (isRuntimeNumber(v)) return RTV.num(0);
      if (isRuntimeTensor(v)) {
        const sliceFn = (slice: ArrayLike<number>) =>
          transform(varianceOf(slice, w));
        if (dimArg > 0) {
          return sliceDimReduce(v, dimArg, sliceFn);
        }
        // Default: vector → scalar, matrix → reduce along first non-singleton dim
        const shape = v.shape;
        const numNonSingleton = shape.filter(d => d > 1).length;
        if (numNonSingleton <= 1) {
          return RTV.num(sliceFn(v.data));
        }
        const firstNonSingleton = shape.findIndex(d => d > 1);
        return sliceDimReduce(v, firstNonSingleton + 1, sliceFn);
      }
      throw new RuntimeError(`${name}: argument must be numeric`);
    };
  };

  register("std", [
    {
      check: reductionCheck,
      apply: stdVarApply("std", v => Math.sqrt(v)),
    },
  ]);

  register("var", [
    {
      check: reductionCheck,
      apply: stdVarApply("var", v => v),
    },
  ]);

  /** Factory for any/all logical reductions. */
  const makeAnyAll = (name: string, mode: "any" | "all") => {
    const anyAllCheck = (
      argTypes: ItemType[],
      nargout: number
    ): { outputTypes: ItemType[] } | null => {
      if (nargout !== 1) return null;
      if (argTypes.length === 1) {
        return { outputTypes: [{ kind: "Boolean" }] };
      }
      if (argTypes.length === 2) {
        const arg2 = argTypes[1];
        if (arg2.kind === "Char" || arg2.kind === "String") {
          return { outputTypes: [{ kind: "Boolean" }] };
        }
        return {
          outputTypes: [{ kind: "Tensor" }],
        };
      }
      return null;
    };

    const scalarLogical = (v: RuntimeValue): RuntimeValue | null => {
      if (isRuntimeNumber(v)) return RTV.logical(v !== 0);
      if (isRuntimeLogical(v)) return RTV.logical(v);
      if (isRuntimeComplexNumber(v))
        return RTV.logical(v.re !== 0 || v.im !== 0);
      return null;
    };

    return {
      check: anyAllCheck,
      apply: (args: RuntimeValue[]) => {
        if (args.length < 1)
          throw new RuntimeError(`${name} requires at least 1 argument`);
        const v = args[0];

        if (args.length === 1) {
          const scalar = scalarLogical(v);
          if (scalar !== null) return scalar;
          if (isRuntimeTensor(v)) {
            if (v.data.length === 0) return RTV.logical(mode === "all");
            if (v.shape[0] === 1)
              return RTV.logical(scanLogical(v.data, v.imag, mode));
            return logicalAlongDim(v, 1, mode);
          }
          throw new RuntimeError(
            `${name}: argument must be numeric or logical`
          );
        }

        const arg2 = args[1];

        // any/all(A, 'all') — reduce over all elements to a scalar
        if (
          (isRuntimeString(arg2) || isRuntimeChar(arg2)) &&
          rstr(arg2).toLowerCase() === "all"
        ) {
          const scalar = scalarLogical(v);
          if (scalar !== null) return scalar;
          if (isRuntimeTensor(v))
            return RTV.logical(scanLogical(v.data, v.imag, mode));
          throw new RuntimeError(
            `${name}: argument must be numeric or logical`
          );
        }

        // any/all(A, dim) or any/all(A, vecdim)
        const scalar = scalarLogical(v);
        if (scalar !== null) return scalar;
        if (isRuntimeTensor(v)) {
          if (isRuntimeNumber(arg2)) {
            return logicalAlongDim(v, Math.round(arg2), mode);
          }
          if (isRuntimeTensor(arg2)) {
            const dims = Array.from(arg2.data).map(d => Math.round(d));
            let result: RuntimeValue = v;
            for (const dim of dims) {
              if (isRuntimeTensor(result)) {
                result = logicalAlongDim(result, dim, mode);
              }
            }
            return result;
          }
        }
        throw new RuntimeError(`${name}: invalid arguments`);
      },
    };
  };

  register("any", [makeAnyAll("any", "any")]);
  register("all", [makeAnyAll("all", "all")]);

  register(
    "xor",
    builtinSingle(
      args => {
        if (args.length !== 2)
          throw new RuntimeError("xor requires 2 arguments");
        const a = args[0];
        const b = args[1];
        const aIsT = isRuntimeTensor(a);
        const bIsT = isRuntimeTensor(b);
        if (!aIsT && !bIsT) {
          const aVal = isRuntimeLogical(a) ? a : toNumber(a) !== 0;
          const bVal = isRuntimeLogical(b) ? b : toNumber(b) !== 0;
          return RTV.logical(aVal !== bVal);
        }
        // Element-wise xor for tensors
        const aScalar = !aIsT ? (toNumber(a) !== 0 ? 1 : 0) : 0;
        const bScalar = !bIsT ? (toNumber(b) !== 0 ? 1 : 0) : 0;
        if (aIsT && bIsT) {
          if (a.data.length !== b.data.length) {
            const outShape = getBroadcastShape(a.shape, b.shape);
            if (!outShape)
              throw new RuntimeError("xor: incompatible array sizes");
            const n = outShape.reduce((p, c) => p * c, 1);
            const out = new FloatXArray(n);
            broadcastIterate(a.shape, b.shape, outShape, (ai, bi, oi) => {
              out[oi] = (a.data[ai] !== 0) !== (b.data[bi] !== 0) ? 1 : 0;
            });
            const result = RTV.tensor(out, outShape);
            result._isLogical = true;
            return result;
          }
          const n = a.data.length;
          const out = new FloatXArray(n);
          for (let i = 0; i < n; i++) {
            out[i] = (a.data[i] !== 0) !== (b.data[i] !== 0) ? 1 : 0;
          }
          const result = RTV.tensor(out, a.shape);
          result._isLogical = true;
          return result;
        }
        // One tensor, one scalar
        const t = aIsT ? a : (b as RuntimeTensor);
        const s = aIsT ? bScalar : aScalar;
        const n = t.data.length;
        const out = new FloatXArray(n);
        for (let i = 0; i < n; i++) {
          out[i] = (t.data[i] !== 0) !== (s !== 0) ? 1 : 0;
        }
        const result = RTV.tensor(out, t.shape);
        result._isLogical = true;
        return result;
      },
      { outputType: { kind: "Boolean" } }
    )
  );

  register("find", [
    {
      check: (_argTypes, nargout) => ({
        outputTypes: Array(Math.max(nargout, 1)).fill({ kind: "Unknown" }),
      }),
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("find requires at least 1 argument");
        const v = args[0];

        // Parse optional n (count limit) and direction ('first'/'last')
        let countLimit = Infinity;
        let direction: "first" | "last" = "first";
        if (args.length >= 2) {
          countLimit = toNumber(args[1]);
        }
        if (args.length >= 3) {
          const dirArg = args[2];
          if (
            (isRuntimeString(dirArg) || isRuntimeChar(dirArg)) &&
            rstr(dirArg).toLowerCase() === "last"
          ) {
            direction = "last";
          }
        }

        // Collect (linearIdx, rowIdx, colIdx, value) for each nonzero.
        // Linear index is column-major (1-based). Row/col are 1-based.
        let rows: number[] = [],
          cols: number[] = [],
          vals: number[] = [];

        let linIndices: number[] = [];

        if (isRuntimeNumber(v) || isRuntimeLogical(v)) {
          const val = isRuntimeNumber(v) ? v : v ? 1 : 0;
          if (val !== 0) {
            rows.push(1);
            cols.push(1);
            vals.push(val);
            linIndices.push(1);
          }
        } else if (isRuntimeComplexNumber(v)) {
          if (v.re !== 0 || v.im !== 0) {
            rows.push(1);
            cols.push(1);
            vals.push(v.re);
            linIndices.push(1);
          }
        } else if (isRuntimeTensor(v)) {
          // Iterate over ALL elements (supports N-D arrays).
          // Row/col treat the array as 2D (higher dims fold into col).
          const nrows = v.shape[0] ?? 1;
          for (let k = 0; k < v.data.length; k++) {
            const val = v.data[k];
            if (val !== 0 || (v.imag && v.imag[k] !== 0)) {
              rows.push((k % nrows) + 1); // 1-based row
              cols.push(Math.floor(k / nrows) + 1); // 1-based col (2D sense)
              vals.push(val);
              linIndices.push(k + 1); // 1-based linear index
            }
          }
        } else {
          throw new RuntimeError("find: argument must be numeric");
        }

        // Apply count limit and direction
        if (countLimit < rows.length) {
          if (direction === "last") {
            const start = rows.length - countLimit;
            rows = rows.slice(start);
            cols = cols.slice(start);
            vals = vals.slice(start);
            linIndices = linIndices.slice(start);
          } else {
            rows = rows.slice(0, countLimit);
            cols = cols.slice(0, countLimit);
            vals = vals.slice(0, countLimit);
            linIndices = linIndices.slice(0, countLimit);
          }
        }

        const n = rows.length;
        const makeVec = (arr: number[]) =>
          n === 0
            ? RTV.tensor(new FloatXArray(0), [0, 1])
            : RTV.tensor(new FloatXArray(arr), [n, 1]);

        if (nargout <= 1) {
          // If X is a row vector, find returns a row vector.
          // Otherwise (column vector, matrix, N-D), find returns a column vector.
          const isRowVec = isRuntimeTensor(v) && v.shape[0] === 1;
          if (n === 0) {
            // Preserve orientation: row input → [1,0], else → [0,1]
            return RTV.tensor(new FloatXArray(0), isRowVec ? [1, 0] : [0, 1]);
          }
          if (isRowVec) {
            return RTV.tensor(new FloatXArray(linIndices), [1, n]);
          }
          return RTV.tensor(new FloatXArray(linIndices), [n, 1]);
        }
        if (nargout === 2) return [makeVec(rows), makeVec(cols)];
        return [makeVec(rows), makeVec(cols), makeVec(vals)];
      },
    },
  ]);

  register("sort", [
    {
      check: preserveTypeCheck,
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("sort requires at least 1 argument");
        const v = args[0];

        // Parse arguments: sort(A), sort(A,dim), sort(A,direction),
        //                  sort(A,dim,direction)
        let dim: number | undefined;
        let descend = false;
        if (args.length >= 2) {
          if (isRuntimeString(args[1]) || isRuntimeChar(args[1])) {
            descend = rstr(args[1]).toLowerCase() === "descend";
          } else {
            dim = Math.round(toNumber(args[1]));
          }
        }
        if (
          args.length >= 3 &&
          (isRuntimeString(args[2]) || isRuntimeChar(args[2]))
        ) {
          descend = rstr(args[2]).toLowerCase() === "descend";
        }

        if (isRuntimeNumber(v)) {
          if (nargout > 1) return [v, RTV.num(1)];
          return v;
        }
        if (isRuntimeComplexNumber(v)) {
          if (nargout > 1) return [v, RTV.num(1)];
          return v;
        }
        if (isRuntimeTensor(v)) {
          const shape = v.shape;
          const re = v.data;
          const im = v.imag;

          // Determine dimension (1-based): default is first non-singleton
          if (dim === undefined) {
            const idx = shape.findIndex(d => d > 1);
            dim = idx >= 0 ? idx + 1 : 1;
          }
          const dimIdx = dim - 1; // 0-based

          // If dim exceeds dimensions, return a copy
          if (dimIdx >= shape.length) {
            const cp = RTV.tensor(
              new FloatXArray(re),
              [...shape],
              im ? new FloatXArray(im) : undefined
            );
            if (nargout > 1) {
              const ones = new FloatXArray(re.length).fill(1);
              return [cp, RTV.tensor(ones, [...shape])];
            }
            return cp;
          }

          const dimSize = shape[dimIdx];

          // Comparison function: operates on flat indices into the data array
          let cmpFlatIdx: (a: number, b: number) => number;
          if (im && !im.every(x => x === 0)) {
            // tricky: check if any imaginary part is nonzero
            const mag = (i: number) => Math.sqrt(re[i] * re[i] + im[i] * im[i]);
            const phase = (i: number) => Math.atan2(im[i], re[i]);
            cmpFlatIdx = (a, b) => {
              const diff = mag(a) - mag(b);
              if (diff !== 0) return descend ? -diff : diff;
              // Tie-break by phase angle
              const pDiff = phase(a) - phase(b);
              return descend ? -pDiff : pDiff;
            };
          } else {
            // NaN-safe: puts NaN at end (ascending) or beginning (descending)
            cmpFlatIdx = descend
              ? (a, b) => {
                  if (re[a] !== re[a]) return -1; // NaN before non-NaN in descending
                  if (re[b] !== re[b]) return 1;
                  return re[b] - re[a];
                }
              : (a, b) => {
                  if (re[a] !== re[a]) return 1; // NaN after non-NaN in ascending
                  if (re[b] !== re[b]) return -1;
                  return re[a] - re[b];
                };
          }

          const resultRe = new FloatXArray(re.length);
          const resultIm = im ? new FloatXArray(re.length) : undefined;
          const resultIdx =
            nargout > 1 ? new FloatXArray(re.length) : undefined;

          if (dimIdx === 0) {
            // Fast path: dim 1 is contiguous in column-major layout
            const numSlices = re.length / dimSize;
            for (let slice = 0; slice < numSlices; slice++) {
              const offset = slice * dimSize;
              const order = Array.from(
                { length: dimSize },
                (_, r) => offset + r
              );
              order.sort((a, b) => cmpFlatIdx(a, b));
              for (let r = 0; r < dimSize; r++) {
                resultRe[offset + r] = re[order[r]];
                if (resultIm) resultIm[offset + r] = im![order[r]];
                if (resultIdx) resultIdx[offset + r] = order[r] - offset + 1;
              }
            }
          } else {
            // General case: iterate over fibers along dimIdx using sub2ind
            const numFibers = re.length / dimSize;
            const fiberShape = shape.filter((_, i) => i !== dimIdx);
            const fiberSubs = new Array(fiberShape.length).fill(0);
            for (let fiber = 0; fiber < numFibers; fiber++) {
              const fullSubs = new Array(shape.length);
              let fi = 0;
              for (let d = 0; d < shape.length; d++) {
                if (d === dimIdx) continue;
                fullSubs[d] = fiberSubs[fi++];
              }
              // Collect the flat indices for this fiber
              const fiberFlatIdx = new Array(dimSize);
              for (let k = 0; k < dimSize; k++) {
                fullSubs[dimIdx] = k;
                fiberFlatIdx[k] = sub2ind(shape, fullSubs);
              }
              // Sort fiber positions by value
              const order = Array.from({ length: dimSize }, (_, k) => k);
              order.sort((a, b) =>
                cmpFlatIdx(fiberFlatIdx[a], fiberFlatIdx[b])
              );
              // Write sorted values to fiber positions
              for (let r = 0; r < dimSize; r++) {
                resultRe[fiberFlatIdx[r]] = re[fiberFlatIdx[order[r]]];
                if (resultIm)
                  resultIm[fiberFlatIdx[r]] = im![fiberFlatIdx[order[r]]];
                if (resultIdx) resultIdx[fiberFlatIdx[r]] = order[r] + 1;
              }
              nextSubscripts(fiberSubs, fiberShape);
            }
          }

          const imOut =
            resultIm && resultIm.some(x => x !== 0) ? resultIm : undefined;
          const sorted = RTV.tensor(resultRe, [...shape], imOut);
          if (nargout > 1) return [sorted, RTV.tensor(resultIdx!, [...shape])];
          return sorted;
        }
        throw new RuntimeError("sort: argument must be numeric");
      },
    },
  ]);

  register(
    "unique",
    builtinSingle((args, nargout) => {
      if (args.length < 1)
        throw new RuntimeError("unique requires at least 1 argument");
      const v = args[0];

      // Parse options
      let byRows = false;
      let stable = false;
      for (let i = 1; i < args.length; i++) {
        const a = args[i];
        if (isRuntimeString(a) || isRuntimeChar(a)) {
          const s = rstr(a).toLowerCase();
          if (s === "rows") byRows = true;
          else if (s === "stable") stable = true;
          else if (s === "sorted") stable = false;
        }
      }

      if (isRuntimeNumber(v)) {
        if (nargout <= 1) return v;
        if (nargout === 2) return [v, RTV.num(1)];
        return [v, RTV.num(1), RTV.num(1)];
      }
      if (isRuntimeLogical(v)) {
        const r = RTV.num(v ? 1 : 0);
        if (nargout <= 1) return r;
        if (nargout === 2) return [r, RTV.num(1)];
        return [r, RTV.num(1), RTV.num(1)];
      }

      if (!isRuntimeTensor(v))
        throw new RuntimeError("unique: argument must be numeric");

      if (byRows) {
        const [rows, cols] = tensorSize2D(v);
        // Compare rows
        const rowKey = (r: number): string => {
          const parts: number[] = [];
          for (let c = 0; c < cols; c++) parts.push(v.data[c * rows + r]);
          return parts.join(",");
        };
        const seen = new Map<string, number>();
        const uniqueRowOrder: number[] = [];
        const ic = new FloatXArray(rows);

        for (let r = 0; r < rows; r++) {
          const key = rowKey(r);
          if (seen.has(key)) {
            ic[r] = seen.get(key)! + 1;
          } else {
            const idx = uniqueRowOrder.length;
            seen.set(key, idx);
            uniqueRowOrder.push(r);
            ic[r] = idx + 1;
          }
        }

        if (!stable) {
          // Sort unique rows lexicographically
          uniqueRowOrder.sort((a, b) => {
            for (let c = 0; c < cols; c++) {
              const va = v.data[c * rows + a];
              const vb = v.data[c * rows + b];
              if (va !== vb) return va - vb;
            }
            return 0;
          });
        }

        const nUnique = uniqueRowOrder.length;
        const resultData = new FloatXArray(nUnique * cols);
        for (let c = 0; c < cols; c++) {
          for (let u = 0; u < nUnique; u++) {
            resultData[c * nUnique + u] = v.data[c * rows + uniqueRowOrder[u]];
          }
        }

        const C = RTV.tensor(resultData, [nUnique, cols]);
        if (nargout <= 1) return C;

        const ia = RTV.tensor(new FloatXArray(uniqueRowOrder.map(r => r + 1)), [
          nUnique,
          1,
        ]);

        // Rebuild ic for sorted case
        if (!stable) {
          const sortedKeyOrder = uniqueRowOrder.map(r => rowKey(r));
          for (let r = 0; r < rows; r++) {
            const key = rowKey(r);
            ic[r] = sortedKeyOrder.indexOf(key) + 1;
          }
        }

        const icTensor = RTV.tensor(ic, [rows, 1]);
        if (nargout === 2) return [C, ia];
        return [C, ia, icTensor];
      }

      // Non-rows: unique elements (handle complex via string keys)
      const hasImag = !!v.imag;
      const valKey = (i: number): string =>
        hasImag ? `${v.data[i]},${v.imag![i]}` : `${v.data[i]}`;
      const seen = new Map<string, number>();
      const uniqueOrder: number[] = [];
      const icArr = new FloatXArray(v.data.length);

      for (let i = 0; i < v.data.length; i++) {
        const key = valKey(i);
        if (seen.has(key)) {
          icArr[i] = seen.get(key)! + 1;
        } else {
          const idx = uniqueOrder.length;
          seen.set(key, idx);
          uniqueOrder.push(i);
          icArr[i] = idx + 1;
        }
      }

      let uniqueRe = uniqueOrder.map(i => v.data[i]);
      let uniqueIm = hasImag ? uniqueOrder.map(i => v.imag![i]) : null;
      if (!stable) {
        // Sort by real part, then imaginary part; rebuild ic
        const indices = uniqueRe.map((_, i) => i);
        indices.sort((a, b) => {
          const ra = uniqueRe[a],
            rb = uniqueRe[b];
          if (ra !== ra) return 1;
          if (rb !== rb) return -1;
          if (ra !== rb) return ra - rb;
          if (uniqueIm) {
            const ia = uniqueIm[a],
              ib = uniqueIm[b];
            if (ia !== ib) return ia - ib;
          }
          return 0;
        });
        const reindex = new Array(uniqueRe.length);
        indices.forEach((origIdx, newIdx) => {
          reindex[origIdx] = newIdx;
        });
        for (let i = 0; i < icArr.length; i++) {
          icArr[i] = reindex[icArr[i] - 1] + 1;
        }
        uniqueRe = indices.map(i => uniqueRe[i]);
        if (uniqueIm) uniqueIm = indices.map(i => uniqueIm![i]);
        const sortedOrder = indices.map(i => uniqueOrder[i]);
        uniqueOrder.length = 0;
        uniqueOrder.push(...sortedOrder);
      }

      const isRow = v.shape.length === 2 && v.shape[0] === 1;
      const outShape: number[] = isRow
        ? [1, uniqueRe.length]
        : [uniqueRe.length, 1];
      const C = RTV.tensor(
        new FloatXArray(uniqueRe),
        outShape,
        uniqueIm ? new FloatXArray(uniqueIm) : undefined
      );

      if (nargout <= 1) return C;

      const ia = RTV.tensor(new FloatXArray(uniqueOrder.map(i => i + 1)), [
        uniqueRe.length,
        1,
      ]);
      const icTensor = RTV.tensor(icArr, [v.data.length, 1]);
      if (nargout === 2) return [C, ia];
      return [C, ia, icTensor];
    })
  );

  // ── uniquetol: unique within tolerance ────────────────────────────────
  register(
    "uniquetol",
    builtinSingle((args, nargout) => {
      if (args.length < 1)
        throw new RuntimeError("uniquetol requires at least 1 argument");
      const v = args[0];
      if (!isRuntimeTensor(v) && !isRuntimeNumber(v))
        throw new RuntimeError("uniquetol: first argument must be numeric");

      // Parse options
      let tol = 1e-6;
      let byRows = false;
      let startIdx = 1;

      // Second arg can be tolerance (number) or name-value pair start
      if (
        args.length >= 2 &&
        (isRuntimeNumber(args[1]) ||
          (isRuntimeTensor(args[1]) && args[1].data.length === 1))
      ) {
        tol = toNumber(args[1]);
        startIdx = 2;
      }

      // Parse name-value pairs
      for (let i = startIdx; i < args.length; i += 2) {
        const name = args[i];
        if (
          (isRuntimeString(name) || isRuntimeChar(name)) &&
          rstr(name).toLowerCase() === "byrows"
        ) {
          byRows = i + 1 < args.length && toNumber(args[i + 1]) !== 0;
        }
      }

      if (isRuntimeNumber(v)) {
        if (nargout > 1) {
          const result: RuntimeValue[] = [v];
          result.push(RTV.num(1)); // ia
          result.push(RTV.num(1)); // ic
          return result;
        }
        return v;
      }

      const data = v.data;
      const shape = v.shape;

      if (byRows) {
        const [rows, cols] = tensorSize2D(v);
        // For each row, check if it's within tolerance of a previously seen row
        const uniqueRowIndices: number[] = [];
        const ic = new FloatXArray(rows); // mapping from input row to unique row

        for (let r = 0; r < rows; r++) {
          let matchIdx = -1;
          for (let u = 0; u < uniqueRowIndices.length; u++) {
            const ur = uniqueRowIndices[u];
            let withinTol = true;
            for (let c = 0; c < cols; c++) {
              if (Math.abs(data[c * rows + r] - data[c * rows + ur]) > tol) {
                withinTol = false;
                break;
              }
            }
            if (withinTol) {
              matchIdx = u;
              break;
            }
          }
          if (matchIdx === -1) {
            ic[r] = uniqueRowIndices.length + 1; // 1-based
            uniqueRowIndices.push(r);
          } else {
            ic[r] = matchIdx + 1; // 1-based
          }
        }

        // Build unique rows matrix
        const nUnique = uniqueRowIndices.length;
        const resultData = new FloatXArray(nUnique * cols);
        for (let c = 0; c < cols; c++) {
          for (let u = 0; u < nUnique; u++) {
            resultData[c * nUnique + u] = data[c * rows + uniqueRowIndices[u]];
          }
        }

        const C = RTV.tensor(resultData, [nUnique, cols]);
        if (nargout <= 1) return C;

        const ia = RTV.tensor(
          new FloatXArray(uniqueRowIndices.map(r => r + 1)),
          [nUnique, 1]
        );
        const icTensor = RTV.tensor(ic, [rows, 1]);

        if (nargout === 2) return [C, ia];
        return [C, ia, icTensor];
      }

      // Non-byRows: treat as vector
      const vals = Array.from(data);
      const uniqueIndices: number[] = [];
      const icArr = new FloatXArray(vals.length);

      for (let i = 0; i < vals.length; i++) {
        let matchIdx = -1;
        for (let u = 0; u < uniqueIndices.length; u++) {
          if (Math.abs(vals[i] - vals[uniqueIndices[u]]) <= tol) {
            matchIdx = u;
            break;
          }
        }
        if (matchIdx === -1) {
          icArr[i] = uniqueIndices.length + 1;
          uniqueIndices.push(i);
        } else {
          icArr[i] = matchIdx + 1;
        }
      }

      const nUnique = uniqueIndices.length;
      const resultData = new FloatXArray(uniqueIndices.map(i => vals[i]));
      const isRow = shape.length === 2 && shape[0] === 1;
      const outShape: number[] = isRow ? [1, nUnique] : [nUnique, 1];
      const C = RTV.tensor(resultData, outShape);

      if (nargout <= 1) return C;
      const ia = RTV.tensor(new FloatXArray(uniqueIndices.map(i => i + 1)), [
        nUnique,
        1,
      ]);
      const icTensor = RTV.tensor(icArr, [vals.length, 1]);
      if (nargout === 2) return [C, ia];
      return [C, ia, icTensor];
    })
  );

  /** Generic cumulative operation along a specified or default dimension.
   *  Supports cumsum(A), cumsum(A, dim), etc. for arbitrary N-D arrays. */
  const cumOp = (
    name: string,
    args: RuntimeValue[],
    accumFn: (acc: number, val: number) => number,
    initial?: number,
    complexAccumFn?: (
      accRe: number,
      accIm: number,
      valRe: number,
      valIm: number
    ) => [number, number]
  ): RuntimeValue => {
    if (args.length < 1)
      throw new RuntimeError(`${name} requires at least 1 argument`);
    const v = args[0];
    if (isRuntimeNumber(v)) return v;
    if (isRuntimeTensor(v)) {
      const shape = v.shape;
      const hasImag = v.imag !== undefined;

      // Determine dimension to accumulate along (1-based)
      let dim: number;
      if (args.length >= 2) {
        dim = Math.round(toNumber(args[1]));
      } else {
        // Default: first non-singleton dimension
        const idx = shape.findIndex(d => d > 1);
        dim = idx >= 0 ? idx + 1 : 1;
      }
      const dimIdx = dim - 1; // 0-based

      // If dim exceeds dimensions, return a copy (dim has size 1)
      if (dimIdx >= shape.length) {
        return RTV.tensor(
          new FloatXArray(v.data),
          [...shape],
          hasImag ? new FloatXArray(v.imag!) : undefined
        );
      }

      const dimSize = shape[dimIdx];
      const result = new FloatXArray(v.data.length);
      const resultImag = hasImag ? new FloatXArray(v.data.length) : undefined;

      // Helper to accumulate one element (handles complex when needed)
      const accumOne = (
        acc: number,
        accIm: number,
        valRe: number,
        valIm: number
      ): [number, number] => {
        if (hasImag && complexAccumFn) {
          return complexAccumFn(acc, accIm, valRe, valIm);
        }
        return [accumFn(acc, valRe), hasImag ? accumFn(accIm, valIm) : 0];
      };

      if (dimIdx === 0) {
        // Fast path: dim 1 is contiguous in column-major layout
        const numSlices = v.data.length / dimSize;
        for (let slice = 0; slice < numSlices; slice++) {
          const offset = slice * dimSize;
          if (initial !== undefined) {
            let acc = initial;
            let accIm = 0;
            for (let k = 0; k < dimSize; k++) {
              [acc, accIm] = accumOne(
                acc,
                accIm,
                v.data[offset + k],
                v.imag ? v.imag[offset + k] : 0
              );
              result[offset + k] = acc;
              if (resultImag) resultImag[offset + k] = accIm;
            }
          } else {
            result[offset] = v.data[offset];
            if (resultImag) resultImag[offset] = v.imag![offset];
            let acc = v.data[offset];
            let accIm = hasImag ? v.imag![offset] : 0;
            for (let k = 1; k < dimSize; k++) {
              [acc, accIm] = accumOne(
                acc,
                accIm,
                v.data[offset + k],
                v.imag ? v.imag[offset + k] : 0
              );
              result[offset + k] = acc;
              if (resultImag) resultImag[offset + k] = accIm;
            }
          }
        }
      } else {
        // General case: accumulate along non-contiguous dimension using sub2ind
        const numFibers = v.data.length / dimSize;
        const fiberShape = shape.filter((_, i) => i !== dimIdx);
        const fiberSubs = new Array(fiberShape.length).fill(0);
        for (let fiber = 0; fiber < numFibers; fiber++) {
          // Build full subscripts by inserting dimIdx back
          const fullSubs = new Array(shape.length);
          let fi = 0;
          for (let d = 0; d < shape.length; d++) {
            if (d === dimIdx) continue;
            fullSubs[d] = fiberSubs[fi++];
          }

          if (initial !== undefined) {
            let acc = initial;
            let accIm = 0;
            for (let k = 0; k < dimSize; k++) {
              fullSubs[dimIdx] = k;
              const idx = sub2ind(shape, fullSubs);
              [acc, accIm] = accumOne(
                acc,
                accIm,
                v.data[idx],
                v.imag ? v.imag[idx] : 0
              );
              result[idx] = acc;
              if (resultImag) resultImag[idx] = accIm;
            }
          } else {
            fullSubs[dimIdx] = 0;
            const startIdx = sub2ind(shape, fullSubs);
            result[startIdx] = v.data[startIdx];
            if (resultImag) resultImag[startIdx] = v.imag![startIdx];
            let acc = v.data[startIdx];
            let accIm = hasImag ? v.imag![startIdx] : 0;
            for (let k = 1; k < dimSize; k++) {
              fullSubs[dimIdx] = k;
              const idx = sub2ind(shape, fullSubs);
              [acc, accIm] = accumOne(
                acc,
                accIm,
                v.data[idx],
                v.imag ? v.imag[idx] : 0
              );
              result[idx] = acc;
              if (resultImag) resultImag[idx] = accIm;
            }
          }

          nextSubscripts(fiberSubs, fiberShape);
        }
      }

      const imOut =
        resultImag && resultImag.some(x => x !== 0) ? resultImag : undefined;
      return RTV.tensor(result, shape, imOut);
    }
    throw new RuntimeError(`${name}: argument must be numeric`);
  };

  register("cumsum", [
    {
      check: preserveTypeCheck,
      apply: args => cumOp("cumsum", args, (acc, val) => acc + val, 0),
    },
  ]);

  register("cumprod", [
    {
      check: preserveTypeCheck,
      apply: args =>
        cumOp(
          "cumprod",
          args,
          (acc, val) => acc * val,
          1,
          (aRe, aIm, bRe, bIm) => [aRe * bRe - aIm * bIm, aRe * bIm + aIm * bRe]
        ),
    },
  ]);

  register("cummax", [
    {
      check: preserveTypeCheck,
      apply: args => cumOp("cummax", args, Math.max),
    },
  ]);

  register("cummin", [
    {
      check: preserveTypeCheck,
      apply: args => cumOp("cummin", args, Math.min),
    },
  ]);

  register("diff", [
    {
      check: (_argTypes: ItemType[], nargout: number) => {
        if (nargout !== 1) return null;
        return {
          outputTypes: [
            { kind: "Tensor" as const, ndim: 2, shape: "unknown" as const },
          ],
        };
      },
      apply: args => {
        if (args.length < 1)
          throw new RuntimeError("diff requires at least 1 argument");
        const n = args.length >= 2 ? Math.round(toNumber(args[1])) : 1;
        const dimArg =
          args.length >= 3 ? Math.round(toNumber(args[2])) : undefined;

        const diffOnce = (v: RuntimeValue, dim?: number): RuntimeValue => {
          if (isRuntimeNumber(v)) {
            return RTV.tensor(new FloatXArray(0), [0, 0]);
          }
          if (isRuntimeTensor(v)) {
            const shape = v.shape;

            // Determine operating dimension (0-based)
            let opDim: number;
            if (dim !== undefined) {
              opDim = dim - 1; // Convert 1-based to 0-based
            } else if (
              shape.length <= 1 ||
              (shape.length === 2 && shape[0] === 1)
            ) {
              // Row vector or 1D: operate along the non-singleton dim
              opDim = shape.length === 2 && shape[0] === 1 ? 1 : 0;
            } else {
              // Default: first non-singleton dimension (dim 0 for matrices)
              opDim = 0;
            }

            const dimSize = opDim < shape.length ? shape[opDim] : 1;
            if (dimSize <= 1) {
              const newShape = [...shape];
              if (opDim < newShape.length) newShape[opDim] = 0;
              return RTV.tensor(new FloatXArray(0), newShape);
            }

            const newShape = [...shape];
            newShape[opDim] = dimSize - 1;
            const totalOut = newShape.reduce((a, b) => a * b, 1);
            const result = new FloatXArray(totalOut);
            const resultImag = v.imag ? new FloatXArray(totalOut) : undefined;

            // Stride along the operating dimension (product of dims before it)
            let innerCount = 1;
            for (let d = 0; d < opDim; d++) {
              innerCount *= shape[d];
            }
            // Number of slices after the operating dimension
            let outerCount = 1;
            for (let d = opDim + 1; d < shape.length; d++) {
              outerCount *= shape[d];
            }

            let outIdx = 0;
            for (let outer = 0; outer < outerCount; outer++) {
              for (let k = 0; k < dimSize - 1; k++) {
                for (let inner = 0; inner < innerCount; inner++) {
                  const base =
                    outer * (dimSize * innerCount) + k * innerCount + inner;
                  result[outIdx] = v.data[base + innerCount] - v.data[base];
                  if (resultImag && v.imag) {
                    resultImag[outIdx] =
                      v.imag[base + innerCount] - v.imag[base];
                  }
                  outIdx++;
                }
              }
            }

            return RTV.tensor(result, newShape, resultImag);
          }
          throw new RuntimeError("diff: argument must be numeric");
        };

        let result = args[0];
        for (let i = 0; i < n; i++) {
          result = diffOnce(result, dimArg);
        }
        return result;
      },
    },
  ]);

  /** Helper: compute median of a numeric array slice */
  const medianOf = (arr: ArrayLike<number>): number => {
    const sorted = Array.from(arr).sort((a, b) => a - b);
    const n = sorted.length;
    if (n === 0) return NaN;
    if (n % 2 === 1) return sorted[(n - 1) / 2];
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  };

  register("median", [makeSliceReduction("median", medianOf)]);

  /** Helper: find the mode (most frequent value; ties → smallest) */
  const modeOf = (arr: ArrayLike<number>): number => {
    const counts = new Map<number, number>();
    for (let i = 0; i < arr.length; i++) {
      counts.set(arr[i], (counts.get(arr[i]) || 0) + 1);
    }
    let bestVal = arr[0],
      bestCount = 0;
    for (const [val, count] of counts) {
      if (count > bestCount || (count === bestCount && val < bestVal)) {
        bestVal = val;
        bestCount = count;
      }
    }
    return bestVal;
  };

  register("mode", [makeSliceReduction("mode", modeOf)]);

  register(
    "nnz",
    builtinSingle(
      args => {
        if (args.length !== 1)
          throw new RuntimeError("nnz requires 1 argument");
        const v = args[0];
        if (isRuntimeNumber(v)) return RTV.num(v !== 0 ? 1 : 0);
        if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
        if (isRuntimeTensor(v)) {
          let count = 0;
          for (let i = 0; i < v.data.length; i++) {
            if (v.data[i] !== 0) count++;
          }
          return RTV.num(count);
        }
        throw new RuntimeError("nnz: argument must be numeric");
      },
      { outputType: { kind: "Number" } }
    )
  );

  /** Helper: extract all numeric values from a RuntimeValue as a plain array */
  const toNumArray = (v: RuntimeValue, name: string): number[] => {
    if (isRuntimeNumber(v)) return [v];
    if (isRuntimeTensor(v)) return Array.from(v.data);
    throw new RuntimeError(`${name}: arguments must be numeric arrays`);
  };

  register(
    "intersect",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("intersect requires 2 arguments");
      const a = toNumArray(args[0], "intersect");
      const bSet = new Set(toNumArray(args[1], "intersect"));
      const result = [...new Set(a.filter(x => bSet.has(x)))].sort((x, y) => {
        if (x !== x) return 1;
        if (y !== y) return -1;
        return x - y;
      });
      return RTV.tensor(new FloatXArray(result), [1, result.length]);
    })
  );

  register(
    "union",
    builtinSingle(args => {
      if (args.length < 2) throw new RuntimeError("union requires 2 arguments");
      const a = toNumArray(args[0], "union");
      const b = toNumArray(args[1], "union");
      const result = [...new Set([...a, ...b])].sort((x, y) => {
        if (x !== x) return 1;
        if (y !== y) return -1;
        return x - y;
      });
      return RTV.tensor(new FloatXArray(result), [1, result.length]);
    })
  );

  register("setdiff", [
    {
      check: (_argTypes: unknown[], nargout: number) => ({
        outputTypes: Array(Math.max(nargout, 1)).fill({ kind: "Unknown" }),
      }),
      apply: (args: RuntimeValue[], nargout: number) => {
        if (args.length < 2)
          throw new RuntimeError("setdiff requires 2 arguments");
        const a = toNumArray(args[0], "setdiff");
        const bSet = new Set(toNumArray(args[1], "setdiff"));
        // Collect unique values not in B, with their original 1-based indices
        const seen = new Set<number>();
        const pairs: { val: number; idx: number }[] = [];
        for (let i = 0; i < a.length; i++) {
          if (!bSet.has(a[i]) && !seen.has(a[i])) {
            seen.add(a[i]);
            pairs.push({ val: a[i], idx: i + 1 });
          }
        }
        // Sort by value (NaN to end)
        pairs.sort((x, y) => {
          if (x.val !== x.val) return 1;
          if (y.val !== y.val) return -1;
          return x.val - y.val;
        });
        const result = new FloatXArray(pairs.map(p => p.val));
        const c = RTV.tensor(result, [1, result.length]);
        if (nargout > 1) {
          const ia = new FloatXArray(pairs.map(p => p.idx));
          return [c, RTV.tensor(ia, [ia.length, 1])];
        }
        return c;
      },
    },
  ]);

  register("ismember", [
    {
      check: (_argTypes: unknown[], nargout: number) => ({
        outputTypes: Array(Math.max(nargout, 1)).fill({ kind: "Unknown" }),
      }),
      apply: (args: RuntimeValue[], nargout: number) => {
        if (args.length < 2)
          throw new RuntimeError("ismember requires 2 arguments");
        const v = args[0];
        const b = args[1];

        // --- String/cell-of-strings path ---
        const isStringLike = (x: RuntimeValue) =>
          isRuntimeString(x) || isRuntimeChar(x);
        const isCellOfStrings = (x: RuntimeValue) =>
          isRuntimeCell(x) &&
          x.data.every(
            (e: RuntimeValue) => isRuntimeString(e) || isRuntimeChar(e)
          );

        if (
          isStringLike(v) ||
          isCellOfStrings(v) ||
          isStringLike(b) ||
          isCellOfStrings(b)
        ) {
          // Build set of strings from B
          const bStrings: string[] = [];
          if (isStringLike(b)) {
            bStrings.push(toString(b));
          } else if (isCellOfStrings(b)) {
            if (!isRuntimeCell(b)) throw new RuntimeError("unexpected type");
            for (const e of b.data as RuntimeValue[])
              bStrings.push(toString(e));
          } else {
            throw new RuntimeError("ismember: incompatible argument types");
          }
          const bSet = new Set(bStrings);

          if (isStringLike(v)) {
            const found = bSet.has(toString(v));
            const lia = RTV.logical(found);
            if (nargout > 1) {
              const idx = found ? bStrings.indexOf(toString(v)) + 1 : 0;
              return [lia, RTV.num(idx)];
            }
            return lia;
          }
          if (isRuntimeCell(v) && isCellOfStrings(v)) {
            const vData = v.data;
            const tfData = new FloatXArray(vData.length);
            const locData =
              nargout > 1 ? new FloatXArray(vData.length) : undefined;
            for (let i = 0; i < vData.length; i++) {
              const s = toString(vData[i]);
              const found = bSet.has(s);
              tfData[i] = found ? 1 : 0;
              if (locData) locData[i] = found ? bStrings.indexOf(s) + 1 : 0;
            }
            const t = RTV.tensor(tfData, [...v.shape]);
            t._isLogical = true;
            if (nargout > 1) {
              return [t, RTV.tensor(locData!, [...v.shape])];
            }
            return t;
          }
          throw new RuntimeError("ismember: incompatible argument types");
        }

        // --- Numeric path ---
        const bArr = toNumArray(b, "ismember");

        // Build a map from value → first (lowest) 1-based index in B
        const bMap = new Map<number, number>();
        for (let i = 0; i < bArr.length; i++) {
          if (!bMap.has(bArr[i])) bMap.set(bArr[i], i + 1);
        }

        if (isRuntimeNumber(v)) {
          const found = bMap.has(v);
          const lia = RTV.logical(found);
          if (nargout > 1) {
            const locb = RTV.num(found ? bMap.get(v)! : 0);
            return [lia, locb];
          }
          return lia;
        }
        if (isRuntimeTensor(v)) {
          const tfData = new FloatXArray(v.data.length);
          const locData =
            nargout > 1 ? new FloatXArray(v.data.length) : undefined;
          for (let i = 0; i < v.data.length; i++) {
            const idx = bMap.get(v.data[i]);
            tfData[i] = idx !== undefined ? 1 : 0;
            if (locData) locData[i] = idx !== undefined ? idx : 0;
          }
          const t = RTV.tensor(tfData, [...v.shape]);
          t._isLogical = true;
          if (nargout > 1) {
            return [t, RTV.tensor(locData!, [...v.shape])];
          }
          return t;
        }
        throw new RuntimeError("ismember: first argument must be numeric");
      },
    },
  ]);

  register(
    "sortrows",
    builtinSingle((args, nargout) => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("sortrows requires 1 or 2 arguments");
      const A = args[0];
      if (!isRuntimeTensor(A))
        throw new RuntimeError("sortrows: input must be a matrix");

      const m = A.shape[0];
      const n = A.shape.length >= 2 ? A.shape[1] : 1;
      const data = A.data;

      // Parse column argument: scalar or vector of signed column indices
      // Positive = ascending, negative = descending
      let cols: number[] = [];
      if (args.length >= 2) {
        const colArg = args[1];
        if (isRuntimeNumber(colArg)) {
          cols = [Math.round(colArg as number)];
        } else if (isRuntimeTensor(colArg)) {
          for (let i = 0; i < colArg.data.length; i++)
            cols.push(Math.round(colArg.data[i]));
        } else {
          throw new RuntimeError("sortrows: column argument must be numeric");
        }
      }
      // Default: sort by all columns in ascending order
      if (cols.length === 0) {
        for (let j = 1; j <= n; j++) cols.push(j);
      }

      // Create row index array and sort
      const rowIdx = Array.from({ length: m }, (_, i) => i);
      rowIdx.sort((a, b) => {
        for (const c of cols) {
          const colIdx = Math.abs(c) - 1; // 0-based
          const descend = c < 0;
          const va = data[a + colIdx * m]; // column-major
          const vb = data[b + colIdx * m];
          if (va !== vb) {
            const diff = va - vb;
            return descend ? -diff : diff;
          }
        }
        return 0;
      });

      // Build result matrix
      const resultData = new FloatXArray(m * n);
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          resultData[i + j * m] = data[rowIdx[i] + j * m];
        }
      }
      const result = RTV.tensor(resultData, [m, n]);

      if (nargout > 1) {
        const idxData = new FloatXArray(m);
        for (let i = 0; i < m; i++) idxData[i] = rowIdx[i] + 1; // 1-based
        return [result, RTV.tensor(idxData, [m, 1])];
      }
      return result;
    })
  );
}
