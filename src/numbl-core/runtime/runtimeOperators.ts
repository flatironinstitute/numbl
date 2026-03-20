/**
 * Runtime operator and construction functions.
 *
 * Standalone functions for operators, range/iteration, and data construction.
 * None of these depend on the Runtime class instance.
 */

import { BinaryOperation } from "../lowering/index.js";
import {
  type RuntimeValue,
  type RuntimeLogical,
  type RuntimeTensor,
  type RuntimeStruct,
  type RuntimeCell,
  RTV,
  toNumber,
  toBool,
  mAdd,
  mSub,
  mMul,
  mDiv,
  mPow,
  mElemMul,
  mElemDiv,
  mElemPow,
  mNeg,
  mTranspose,
  mConjugateTranspose,
  mEqual,
  mNotEqual,
  mLess,
  mLessEqual,
  mGreater,
  mGreaterEqual,
  mRange,
  horzcat as mHorzcat,
  vertcat as mVertcat,
  RuntimeError,
  tensorSize2D,
} from "../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeSparseMatrix,
  FloatXArray,
} from "../runtime/types.js";
import { asNumber } from "../executor/helpers.js";
import { END_SENTINEL } from "../executor/types.js";
import {
  linsolveLapack,
  linsolveComplexLapack,
} from "../../numbl-core/builtins/linear-algebra/linsolve.js";
import { sparseToDense as sparseToDenseFn } from "../../numbl-core/builtins/sparse-arithmetic.js";
import {
  type DeferredRange,
  type DeferredHorzcat,
  isDeferredRange,
  ensureRuntimeValue,
  isNumericKind,
  elementWiseLogicalOp,
  switchValuesMatch,
} from "./runtimeHelpers.js";

// ── Unary operators ─────────────────────────────────────────────────────

export function uplus(v: unknown): RuntimeValue {
  if (typeof v === "number") return v;
  return ensureRuntimeValue(v);
}

export function uminus(v: unknown): unknown {
  if (typeof v === "number") return -v;
  return mNeg(ensureRuntimeValue(v));
}

export function transpose(v: unknown): unknown {
  if (typeof v === "number") return v;
  return mTranspose(ensureRuntimeValue(v));
}

export function ctranspose(v: unknown): unknown {
  if (typeof v === "number") return v;
  return mConjugateTranspose(ensureRuntimeValue(v));
}

export function not(v: unknown): RuntimeLogical | RuntimeTensor {
  if (typeof v === "number") return RTV.logical(v === 0);
  if (typeof v === "boolean") return RTV.logical(!v);
  const mv = ensureRuntimeValue(v);
  if (isRuntimeNumber(mv)) return RTV.logical(mv === 0);
  if (isRuntimeLogical(mv)) return RTV.logical(!mv);
  if (isRuntimeSparseMatrix(mv)) return not(sparseToDenseFn(mv));
  if (isRuntimeTensor(mv)) {
    const result = new FloatXArray(mv.data.length);
    for (let i = 0; i < mv.data.length; i++)
      result[i] = mv.data[i] === 0 && (!mv.imag || mv.imag[i] === 0) ? 1 : 0;
    const t = RTV.tensor(result, mv.shape);
    t._isLogical = true;
    return t;
  }
  return RTV.logical(false);
}

// ── Binary operators ────────────────────────────────────────────────────

export function binop(op: string, a: unknown, b: unknown): unknown {
  // Fast path: both plain JS numbers (skip asNumber overhead)
  if (typeof a === "number" && typeof b === "number") {
    switch (op) {
      case BinaryOperation.Add:
        return a + b;
      case BinaryOperation.Sub:
        return a - b;
      case BinaryOperation.Mul:
        return a * b;
      case BinaryOperation.Div:
        return a / b;
      case BinaryOperation.Pow: {
        const r = Math.pow(a, b);
        if (isNaN(r) && !isNaN(a) && !isNaN(b)) break; // fall to slow path for complex
        return r;
      }
      case BinaryOperation.ElemMul:
        return a * b;
      case BinaryOperation.ElemDiv:
        return a / b;
      case BinaryOperation.ElemPow: {
        const r = Math.pow(a, b);
        if (isNaN(r) && !isNaN(a) && !isNaN(b)) break; // fall to slow path for complex
        return r;
      }
      case BinaryOperation.LeftDiv:
        return b / a;
      case BinaryOperation.ElemLeftDiv:
        return b / a;
      case BinaryOperation.Equal:
        return RTV.logical(a === b);
      case BinaryOperation.NotEqual:
        return RTV.logical(a !== b);
      case BinaryOperation.Less:
        return RTV.logical(a < b);
      case BinaryOperation.LessEqual:
        return RTV.logical(a <= b);
      case BinaryOperation.Greater:
        return RTV.logical(a > b);
      case BinaryOperation.GreaterEqual:
        return RTV.logical(a >= b);
      case BinaryOperation.BitAnd:
        return RTV.logical(a !== 0 && b !== 0);
      case BinaryOperation.BitOr:
        return RTV.logical(a !== 0 || b !== 0);
    }
  }

  // Secondary fast path: booleans as numbers
  if (typeof a !== "object" || typeof b !== "object") {
    const an = asNumber(a);
    const bn = asNumber(b);
    if (an !== null && bn !== null) {
      return binop(op, an, bn);
    }
  }

  // Slow path: RuntimeValue operations
  const ma =
    typeof a === "object" && a !== null && "kind" in a
      ? (a as RuntimeValue)
      : ensureRuntimeValue(a);
  const mb =
    typeof b === "object" && b !== null && "kind" in b
      ? (b as RuntimeValue)
      : ensureRuntimeValue(b);

  let result: RuntimeValue;
  switch (op) {
    case BinaryOperation.Add:
      result = mAdd(ma, mb);
      break;
    case BinaryOperation.Sub:
      result = mSub(ma, mb);
      break;
    case BinaryOperation.Mul:
      result = mMul(ma, mb);
      break;
    case BinaryOperation.Div:
      result = mDiv(ma, mb);
      break;
    case BinaryOperation.Pow:
      result = mPow(ma, mb);
      break;
    case BinaryOperation.ElemMul:
      result = mElemMul(ma, mb);
      break;
    case BinaryOperation.ElemDiv:
      result = mElemDiv(ma, mb);
      break;
    case BinaryOperation.ElemPow:
      result = mElemPow(ma, mb);
      break;
    case BinaryOperation.LeftDiv: {
      if (isRuntimeNumber(ma)) {
        result = mElemDiv(mb, ma);
        break;
      }
      const tensorA = isRuntimeSparseMatrix(ma) ? sparseToDenseFn(ma) : ma;
      const tensorB = isRuntimeNumber(mb)
        ? RTV.tensor(new FloatXArray([mb]), [1, 1])
        : isRuntimeSparseMatrix(mb)
          ? sparseToDenseFn(mb)
          : mb;
      if (!isRuntimeTensor(tensorA) || !isRuntimeTensor(tensorB))
        throw new RuntimeError(
          "LeftDiv (\\): operands must be numeric matrices"
        );
      const [mA, nA] = tensorSize2D(tensorA);
      const [mB, nB] = tensorSize2D(tensorB);
      if (mB !== mA)
        throw new RuntimeError(
          `LeftDiv (\\): A is ${mA}×${nA} but B has ${mB} rows`
        );
      // Empty matrix: return empty result (nA × nB)
      if (mA === 0 || nA === 0 || nB === 0) {
        result = RTV.tensor(new FloatXArray(nA * nB), [nA, nB]);
        break;
      }
      if (tensorA.imag || tensorB.imag) {
        const ARe = tensorA.data;
        const AIm = tensorA.imag ?? new FloatXArray(tensorA.data.length);
        const BRe = tensorB.data;
        const BIm = tensorB.imag ?? new FloatXArray(tensorB.data.length);
        const X = linsolveComplexLapack(ARe, AIm, mA, nA, BRe, BIm, nB);
        result = RTV.tensor(
          new FloatXArray(X.re),
          [nA, nB],
          new FloatXArray(X.im)
        );
        break;
      }
      const X = linsolveLapack(tensorA.data, mA, nA, tensorB.data, nB);
      if (!X) throw new RuntimeError("LeftDiv (\\): LAPACK bridge unavailable");
      result = RTV.tensor(new FloatXArray(X), [nA, nB]);
      break;
    }
    case BinaryOperation.ElemLeftDiv:
      result = mElemDiv(mb, ma);
      break;
    case BinaryOperation.Equal:
      result = mEqual(ma, mb);
      break;
    case BinaryOperation.NotEqual:
      result = mNotEqual(ma, mb);
      break;
    case BinaryOperation.Less:
      result = mLess(ma, mb);
      break;
    case BinaryOperation.LessEqual:
      result = mLessEqual(ma, mb);
      break;
    case BinaryOperation.Greater:
      result = mGreater(ma, mb);
      break;
    case BinaryOperation.GreaterEqual:
      result = mGreaterEqual(ma, mb);
      break;
    case BinaryOperation.BitAnd:
      if (
        (isRuntimeTensor(ma) || isRuntimeTensor(mb)) &&
        isNumericKind(ma) &&
        isNumericKind(mb)
      ) {
        result = elementWiseLogicalOp(ma, mb, (x, y) =>
          x !== 0 && y !== 0 ? 1 : 0
        );
      } else {
        result = RTV.logical(toBool(ma) && toBool(mb));
      }
      break;
    case BinaryOperation.BitOr:
      if (
        (isRuntimeTensor(ma) || isRuntimeTensor(mb)) &&
        isNumericKind(ma) &&
        isNumericKind(mb)
      ) {
        result = elementWiseLogicalOp(ma, mb, (x, y) =>
          x !== 0 || y !== 0 ? 1 : 0
        );
      } else {
        result = RTV.logical(toBool(ma) || toBool(mb));
      }
      break;
    default:
      throw new RuntimeError(`Unknown binary operator: ${op}`);
  }

  if (isRuntimeNumber(result)) return result;
  return result;
}

// ── Range ───────────────────────────────────────────────────────────────

export function range(start: unknown, step: unknown, end: unknown): unknown {
  if (start === END_SENTINEL || step === END_SENTINEL || end === END_SENTINEL) {
    return { _deferredRange: true, start, step, end } as DeferredRange;
  }
  const startRV = typeof start === "number" ? null : ensureRuntimeValue(start);
  const endRV = typeof end === "number" ? null : ensureRuntimeValue(end);
  const isCharRange =
    (startRV !== null && isRuntimeChar(startRV)) ||
    (endRV !== null && isRuntimeChar(endRV));
  const s = startRV !== null ? toNumber(startRV) : (start as number);
  const st =
    typeof step === "number" ? step : toNumber(ensureRuntimeValue(step));
  const e = endRV !== null ? toNumber(endRV) : (end as number);
  const result = mRange(s, st, e);
  if (isCharRange && isRuntimeTensor(result)) {
    return RTV.char(
      Array.from(result.data)
        .map(c => String.fromCharCode(c))
        .join("")
    );
  }
  return result;
}

// ── For loop iteration ──────────────────────────────────────────────────

export function forIter(v: unknown): unknown[] {
  if (typeof v === "number") return [v];
  const mv = ensureRuntimeValue(v);
  if (isRuntimeNumber(mv)) return [mv];
  if (isRuntimeTensor(mv)) {
    const shape = mv.shape;
    if (shape.length <= 1 || (shape.length === 2 && shape[0] === 1)) {
      // Row vector or 1D: iterate as scalars
      if (mv.imag) {
        const result: RuntimeValue[] = [];
        for (let i = 0; i < mv.data.length; i++) {
          const im = mv.imag[i];
          result.push(im === 0 ? mv.data[i] : RTV.complex(mv.data[i], im));
        }
        return result;
      }
      const result: number[] = [];
      for (let i = 0; i < mv.data.length; i++) result.push(mv.data[i]);
      return result;
    }
    const rows = shape[0];
    const totalCols = mv.data.length / rows;
    const result: RuntimeValue[] = [];
    for (let c = 0; c < totalCols; c++) {
      const colData = new FloatXArray(rows);
      for (let r = 0; r < rows; r++) colData[r] = mv.data[c * rows + r];
      let colImag: InstanceType<typeof FloatXArray> | undefined;
      if (mv.imag) {
        colImag = new FloatXArray(rows);
        for (let r = 0; r < rows; r++) colImag[r] = mv.imag[c * rows + r];
      }
      result.push(RTV.tensor(colData, [rows, 1], colImag));
    }
    return result;
  }
  if (isRuntimeCell(mv)) {
    const shape = mv.shape;
    const rows = shape.length <= 1 ? 1 : shape[0];
    const cols =
      shape.length === 0 ? 1 : shape.length === 1 ? shape[0] : shape[1];
    const result: RuntimeValue[] = [];
    for (let c = 0; c < cols; c++) {
      const colData: RuntimeValue[] = [];
      for (let r = 0; r < rows; r++) colData.push(mv.data[c * rows + r]);
      result.push(RTV.cell(colData, rows === 1 ? [1, 1] : [rows, 1]));
    }
    return result;
  }
  if (isRuntimeChar(mv)) {
    return Array.from(mv.value).map(c => RTV.char(c));
  }
  return [v];
}

// ── Tensor/cell construction ────────────────────────────────────────────

export function emptyTensor(): RuntimeTensor {
  return RTV.tensor(new FloatXArray(0), [0, 0]);
}

export function emptyStruct(): RuntimeStruct {
  return RTV.struct(new Map());
}

export function makeCell(elems: unknown[], shape: number[]): RuntimeCell {
  const mvals = elems.map(e => ensureRuntimeValue(e));
  return RTV.cell(mvals, shape);
}

export function opHorzcat(elems: unknown[]): unknown {
  const flat: unknown[] = [];
  for (const e of elems) {
    if (Array.isArray(e)) flat.push(...e);
    else flat.push(e);
  }
  if (flat.some(e => e === END_SENTINEL || isDeferredRange(e))) {
    return { _deferredHorzcat: true, elems: flat } as DeferredHorzcat;
  }
  if (flat.length === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
  if (flat.length === 1) {
    const e = flat[0];
    if (typeof e === "number") return RTV.num(e);
    return ensureRuntimeValue(e);
  }
  const mvals = flat.map(e => ensureRuntimeValue(e));
  return mHorzcat(...mvals);
}

export function resolveHorzcat(
  dh: DeferredHorzcat,
  resolve: (v: RuntimeValue | typeof END_SENTINEL) => number
): RuntimeValue {
  const resolved = dh.elems.map(e => {
    if (e === END_SENTINEL) return resolve(e);
    if (isDeferredRange(e))
      return mRange(resolve(e.start), resolve(e.step), resolve(e.end));
    if (typeof e === "number") return e;
    return ensureRuntimeValue(e);
  });
  const result = opHorzcat(resolved);
  return ensureRuntimeValue(result);
}

export function opVertcat(rows: unknown[]): unknown {
  if (rows.length === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
  const mvals = rows.map(r => ensureRuntimeValue(r));
  return mVertcat(...mvals);
}

// ── Switch matching ─────────────────────────────────────────────────────

export function switchMatch(control: unknown, caseVal: unknown): boolean {
  const ma = ensureRuntimeValue(control);
  const mb = ensureRuntimeValue(caseVal);
  if (isRuntimeCell(mb)) {
    for (const elem of mb.data) {
      if (switchValuesMatch(ma, elem)) return true;
    }
    return false;
  }
  return switchValuesMatch(ma, mb);
}
