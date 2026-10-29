/**
 * min/max builtin functions.
 */

import { RTV } from "../../runtime/constructors.js";
import { toNumber } from "../../runtime/convert.js";
import { RuntimeError } from "../../runtime/error.js";
import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
  type RuntimeTensor,
  type RuntimeValue,
} from "../../runtime/types.js";
import { sparseToDense } from "../sparse-arithmetic.js";
import { rstr } from "../../runtime/runtime.js";
import {
  forEachSlice,
  firstReduceDim,
  copyTensor,
} from "../reduction-helpers.js";
import {
  tensorOps,
  OpReduce,
  OpBin,
  binaryScalar,
  elementwiseBinary,
  type Operand,
} from "../../ops/index.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { stripZeroImagValue } from "../effectively-real.js";

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
    const resultRe = allocFloat64Array(info.totalElems);
    const resultIm = allocFloat64Array(info.totalElems);
    const idxArr = nargout > 1 ? allocFloat64Array(info.totalElems) : undefined;
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

  const result = allocFloat64Array(info.totalElems);
  const idxArr = nargout > 1 ? allocFloat64Array(info.totalElems) : undefined;
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
  const out = RTV.tensor(result, info.resultShape, undefined, v.cls);
  if (nargout > 1) return [out, RTV.tensor(idxArr!, info.resultShape)];
  return out;
}

// ── Element-wise min/max of two arguments ──────────────────────────────

function minMaxOperand(v: RuntimeValue): Operand {
  if (isRuntimeTensor(v)) return v;
  if (isRuntimeComplexNumber(v))
    return {
      data: Float64Array.of(v.re),
      imag: Float64Array.of(v.im),
      shape: [1, 1],
    };
  return toNumber(v);
}

/** min(a, b) / max(a, b): NaN-skipping; complex values compare by
 *  magnitude, then angle. Two logical operands give a logical result. */
function minMaxElementwise(
  a: RuntimeValue,
  b: RuntimeValue,
  name: string
): RuntimeValue {
  const op = name === "min" ? OpBin.MIN : OpBin.MAX;
  const isLogical = (v: RuntimeValue) =>
    isRuntimeLogical(v) || (isRuntimeTensor(v) && v.cls === "logical");
  const bothLogical = isLogical(a) && isLogical(b);
  const isScalar = (v: RuntimeValue) =>
    isRuntimeNumber(v) || isRuntimeLogical(v) || isRuntimeComplexNumber(v);
  if (isScalar(a) && isScalar(b)) {
    const ac = isRuntimeComplexNumber(a);
    const bc = isRuntimeComplexNumber(b);
    const r = binaryScalar(
      op,
      ac ? a.re : toNumber(a),
      ac ? a.im : 0,
      ac,
      bc ? b.re : toNumber(b),
      bc ? b.im : 0,
      bc
    );
    if (ac || bc) return r.im === 0 ? RTV.num(r.re) : RTV.complex(r.re, r.im);
    return bothLogical ? RTV.logical(r.re !== 0) : RTV.num(r.re);
  }
  const r = elementwiseBinary(op, minMaxOperand(a), minMaxOperand(b));
  return RTV.tensor(
    r.data,
    r.shape,
    r.imag,
    bothLogical ? "logical" : "double"
  );
}

// ── Main min/max dispatch ──────────────────────────────────────────────

export function minMaxImpl(
  name: string,
  args: RuntimeValue[],
  nargout: number,
  initial: number,
  isBetter: (candidate: number, current: number) => boolean
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

  // Drop an all-zero imaginary lane so complex-but-real tensors (e.g. a
  // JIT `sqrt` that lifted to complex without leaving the real domain)
  // order by value rather than by magnitude — matching `--opt 0` and
  // MATLAB on real data. Genuinely complex tensors are untouched.
  args = args.map(stripZeroImagValue);

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
        const empty = RTV.tensor(allocFloat64Array(0), [0, 0]);
        if (nargout > 1) return [empty, empty];
        return empty;
      }
      const d = firstReduceDim(v.shape);
      if (d === 0) {
        // Vector: full scan (direct, no index array needed).
        // Fast path: single-output, real → tensorOps.realFlatReduce.
        if (nargout <= 1 && !v.imag && (name === "min" || name === "max")) {
          const out = allocFloat64Array(1);
          const op = name === "min" ? OpReduce.MIN : OpReduce.MAX;
          tensorOps.realFlatReduce(op, v.data.length, v.data, out);
          const val = out[0];
          return v.cls === "logical" ? RTV.logical(val !== 0) : RTV.num(val);
        }
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
            v.cls === "logical" ? RTV.logical(mRe !== 0) : RTV.num(mRe),
            RTV.num(mIdx + 1),
          ];
        return v.cls === "logical" ? RTV.logical(mRe !== 0) : RTV.num(mRe);
      }
      return minMaxAlongDim(v, d, nargout, initial, isBetter, complexIsBetter);
    }
  }

  // --- 2-arg: element-wise ---
  if (args.length === 2) {
    return minMaxElementwise(args[0], args[1], name);
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
            ? RTV.tensor(
                allocFloat64Array(v.data),
                [1, v.data.length],
                v.imag ? allocFloat64Array(v.imag) : undefined
              )
            : v,
        ],
        nargout,
        initial,
        isBetter
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
          isBetter
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
