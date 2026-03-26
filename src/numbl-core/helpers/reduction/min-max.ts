/**
 * min/max builtin functions.
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
} from "../../runtime/index.js";
import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
  type RuntimeTensor,
} from "../../runtime/types.js";
import { getBroadcastShape, broadcastIterate } from "../arithmetic.js";
import { sparseToDense } from "../sparse-arithmetic.js";
import { rstr } from "../../runtime/runtime.js";
import {
  forEachSlice,
  firstReduceDim,
  copyTensor,
} from "../reduction-helpers.js";

// ── Scan helpers ───────────────────────────────────────────────────────

/** Scan positions in data/imag arrays for extreme value+index.
 *  Returns { mRe, mIm, mIdx } where mIdx is 0-based position within indices. */
function minMaxScan(
  data: ArrayLike<number>,
  imag: ArrayLike<number> | undefined,
  indices: number[],
  initial: number,
  isBetter: (candidate: number, current: number) => boolean,
  complexIsBetter: (
    reA: number,
    imA: number,
    reB: number,
    imB: number
  ) => boolean
): { mRe: number; mIm: number; mIdx: number } {
  let mRe = initial,
    mIm = 0,
    mIdx = 0;
  let foundNonNaN = false;
  if (imag) {
    for (let k = 0; k < indices.length; k++) {
      const idx = indices[k];
      if (data[idx] !== data[idx] || imag[idx] !== imag[idx]) continue;
      if (!foundNonNaN || complexIsBetter(data[idx], imag[idx], mRe, mIm)) {
        mRe = data[idx];
        mIm = imag[idx];
        mIdx = k;
        foundNonNaN = true;
      }
    }
  } else {
    for (let k = 0; k < indices.length; k++) {
      const val = data[indices[k]];
      if (val !== val) continue;
      if (!foundNonNaN || isBetter(val, mRe)) {
        mRe = val;
        mIdx = k;
        foundNonNaN = true;
      }
    }
  }
  if (!foundNonNaN) {
    mRe = NaN;
    mIm = 0;
  }
  return { mRe, mIm, mIdx };
}

/** Fast scan over a contiguous range [start, start+count) — avoids index array allocation. */
function minMaxScanDirect(
  data: ArrayLike<number>,
  imag: ArrayLike<number> | undefined,
  start: number,
  count: number,
  initial: number,
  isBetter: (candidate: number, current: number) => boolean,
  complexIsBetter: (
    reA: number,
    imA: number,
    reB: number,
    imB: number
  ) => boolean
): { mRe: number; mIm: number; mIdx: number } {
  let mRe = initial,
    mIm = 0,
    mIdx = 0;
  let foundNonNaN = false;
  if (imag) {
    for (let k = 0; k < count; k++) {
      const idx = start + k;
      if (data[idx] !== data[idx] || imag[idx] !== imag[idx]) continue;
      if (!foundNonNaN || complexIsBetter(data[idx], imag[idx], mRe, mIm)) {
        mRe = data[idx];
        mIm = imag[idx];
        mIdx = k;
        foundNonNaN = true;
      }
    }
  } else {
    for (let k = 0; k < count; k++) {
      const val = data[start + k];
      if (val !== val) continue;
      if (!foundNonNaN || isBetter(val, mRe)) {
        mRe = val;
        mIdx = k;
        foundNonNaN = true;
      }
    }
  }
  if (!foundNonNaN) {
    mRe = NaN;
    mIm = 0;
  }
  return { mRe, mIm, mIdx };
}

// ── Along-dimension reduction ──────────────────────────────────────────

function minMaxAlongDim(
  v: RuntimeTensor,
  dim: number,
  nargout: number,
  initial: number,
  isBetter: (candidate: number, current: number) => boolean,
  complexIsBetter: (
    reA: number,
    imA: number,
    reB: number,
    imB: number
  ) => boolean
): RuntimeValue | RuntimeValue[] {
  const info = forEachSlice(v.shape, dim, () => {});
  if (!info) return copyTensor(v);

  if (v.imag) {
    const resultRe = new FloatXArray(info.totalElems);
    const resultIm = new FloatXArray(info.totalElems);
    const idxArr = nargout > 1 ? new FloatXArray(info.totalElems) : undefined;
    forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
      const { mRe, mIm, mIdx } = minMaxScan(
        v.data,
        v.imag,
        srcIndices,
        initial,
        isBetter,
        complexIsBetter
      );
      resultRe[outIdx] = mRe;
      resultIm[outIdx] = mIm;
      if (idxArr) idxArr[outIdx] = mIdx + 1;
    });
    const hasImag = resultIm.some(x => x !== 0);
    const out = RTV.tensor(
      resultRe,
      info.resultShape,
      hasImag ? resultIm : undefined
    );
    if (nargout > 1) return [out, RTV.tensor(idxArr!, info.resultShape)];
    return out;
  }

  const result = new FloatXArray(info.totalElems);
  const idxArr = nargout > 1 ? new FloatXArray(info.totalElems) : undefined;
  forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
    const { mRe, mIdx } = minMaxScan(
      v.data,
      undefined,
      srcIndices,
      initial,
      isBetter,
      complexIsBetter
    );
    result[outIdx] = mRe;
    if (idxArr) idxArr[outIdx] = mIdx + 1;
  });
  const out = RTV.tensor(result, info.resultShape);
  if (v._isLogical) out._isLogical = true;
  if (nargout > 1) return [out, RTV.tensor(idxArr!, info.resultShape)];
  return out;
}

// ── Element-wise min/max of two arguments ──────────────────────────────

function scalarToTensor(v: RuntimeValue): RuntimeTensor {
  if (isRuntimeTensor(v)) return v;
  if (isRuntimeComplexNumber(v)) {
    const t = RTV.tensor(new FloatXArray([v.re]), [1, 1]) as RuntimeTensor;
    t.imag = new FloatXArray([v.im]);
    return t;
  }
  return RTV.tensor(new FloatXArray([toNumber(v)]), [1, 1]) as RuntimeTensor;
}

function minMaxElementwise(
  a: RuntimeValue,
  b: RuntimeValue,
  name: string,
  twoArgFn: (a: number, b: number) => number,
  complexIsBetter: (
    reA: number,
    imA: number,
    reB: number,
    imB: number
  ) => boolean
): RuntimeValue {
  const aIsScalar =
    isRuntimeNumber(a) || isRuntimeLogical(a) || isRuntimeComplexNumber(a);
  const bIsScalar =
    isRuntimeNumber(b) || isRuntimeLogical(b) || isRuntimeComplexNumber(b);
  const anyComplex =
    isRuntimeComplexNumber(a) ||
    isRuntimeComplexNumber(b) ||
    (isRuntimeTensor(a) && a.imag != null) ||
    (isRuntimeTensor(b) && b.imag != null);

  // Both scalars, no complex
  if (aIsScalar && bIsScalar && !anyComplex) {
    const aVal = toNumber(a),
      bVal = toNumber(b);
    const r = isNaN(aVal) ? bVal : isNaN(bVal) ? aVal : twoArgFn(aVal, bVal);
    return RTV.num(r);
  }

  // Both scalars, at least one complex
  if (aIsScalar && bIsScalar) {
    const aRe = isRuntimeComplexNumber(a) ? a.re : toNumber(a);
    const aIm = isRuntimeComplexNumber(a) ? a.im : 0;
    const bRe = isRuntimeComplexNumber(b) ? b.re : toNumber(b);
    const bIm = isRuntimeComplexNumber(b) ? b.im : 0;
    const pickA = complexIsBetter(aRe, aIm, bRe, bIm);
    const re = pickA ? aRe : bRe;
    const im = pickA ? aIm : bIm;
    return im === 0 ? RTV.num(re) : RTV.complex(re, im);
  }

  // At least one tensor — promote scalars (including complex) to 1x1 tensors
  const aT = scalarToTensor(a);
  const bT = scalarToTensor(b);
  const outShape = getBroadcastShape(aT.shape, bT.shape);
  if (!outShape)
    throw new RuntimeError(`${name}: non-singleton dimensions must match`);
  const n = outShape.reduce((acc, d) => acc * d, 1);
  const result = new FloatXArray(n);

  if (!anyComplex) {
    // Real-only fast path
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

  // Complex element-wise path
  const aIm = aT.imag;
  const bIm = bT.imag;
  const resultImag = new FloatXArray(n);
  let hasImag = false;
  broadcastIterate(aT.shape, bT.shape, outShape, (aIdx, bIdx, i) => {
    const aRe = aT.data[aIdx],
      aI = aIm ? aIm[aIdx] : 0;
    const bRe = bT.data[bIdx],
      bI = bIm ? bIm[bIdx] : 0;
    const pickA = complexIsBetter(aRe, aI, bRe, bI);
    result[i] = pickA ? aRe : bRe;
    const im = pickA ? aI : bI;
    resultImag[i] = im;
    if (im !== 0) hasImag = true;
  });
  const t = RTV.tensor(result, outShape) as RuntimeTensor;
  if (hasImag) t.imag = resultImag;
  return t;
}

// ── Main min/max dispatch ──────────────────────────────────────────────

export function minMaxImpl(
  name: string,
  args: RuntimeValue[],
  nargout: number,
  initial: number,
  isBetter: (candidate: number, current: number) => boolean,
  twoArgFn: (a: number, b: number) => number
): RuntimeValue | RuntimeValue[] {
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

  // Densify sparse arguments
  args = args.map(a => (isRuntimeSparseMatrix(a) ? sparseToDense(a) : a));

  // --- 1-arg: reduce to scalar or along default dim ---
  if (args.length === 1) {
    const v = args[0];
    if (
      isRuntimeNumber(v) ||
      isRuntimeLogical(v) ||
      isRuntimeComplexNumber(v)
    ) {
      if (nargout > 1) return [v, RTV.num(1)];
      return v;
    }
    if (isRuntimeTensor(v)) {
      if (v.data.length === 0) {
        const empty = RTV.tensor(new FloatXArray(0), [0, 0]);
        if (nargout > 1) return [empty, empty];
        return empty;
      }
      const d = firstReduceDim(v.shape);
      if (d === 0) {
        // Vector: full scan (direct, no index array needed)
        const { mRe, mIm, mIdx } = minMaxScanDirect(
          v.data,
          v.imag,
          0,
          v.data.length,
          initial,
          isBetter,
          complexIsBetter
        );
        if (v.imag) {
          const result = mIm === 0 ? RTV.num(mRe) : RTV.complex(mRe, mIm);
          if (nargout > 1) return [result, RTV.num(mIdx + 1)];
          return result;
        }
        if (nargout > 1)
          return [
            v._isLogical ? RTV.logical(mRe !== 0) : RTV.num(mRe),
            RTV.num(mIdx + 1),
          ];
        return v._isLogical ? RTV.logical(mRe !== 0) : RTV.num(mRe);
      }
      return minMaxAlongDim(v, d, nargout, initial, isBetter, complexIsBetter);
    }
  }

  // --- 2-arg: element-wise ---
  if (args.length === 2) {
    return minMaxElementwise(args[0], args[1], name, twoArgFn, complexIsBetter);
  }

  // --- 3-arg: reduce along specified dim ---
  if (args.length === 3) {
    const v = args[0];
    // Handle 'all' flag
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
    // Handle vector of dimensions
    if (isRuntimeTensor(args[2])) {
      const dims = Array.from(args[2].data).map(d => Math.round(d));
      const sortedDims = [...dims].sort((a, b) => b - a);
      let result: RuntimeValue = v;
      for (const d of sortedDims) {
        const r = minMaxImpl(
          name,
          [result, RTV.num(0), RTV.num(d)],
          1,
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
      return minMaxAlongDim(
        v,
        dim,
        nargout,
        initial,
        isBetter,
        complexIsBetter
      );
    }
  }
  throw new RuntimeError(`${name}: invalid arguments`);
}
