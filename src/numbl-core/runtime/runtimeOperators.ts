/**
 * Runtime operator and construction functions.
 *
 * Standalone functions for operators, range/iteration, and data construction.
 * None of these depend on the Runtime class instance.
 */

import { BinaryOperation, UnaryOperation } from "../parser/index.js";
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
import { invalidDataTypeError } from "./error.js";
import { charToNumericTensor } from "./indexing.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeSparseMatrix,
  isRuntimeStringArray,
  isRuntimeStructArray,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeComplexNumber,
  type RuntimeClassInstance,
  stringArrayValue,
} from "../runtime/types.js";
import { END_SENTINEL } from "./sentinels.js";

/** Extract number from value if possible, for fast-path operations */
function asNumber(v: unknown): number | null {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (v && typeof v === "object" && "kind" in v) {
    const mv = v as RuntimeValue;
    if (isRuntimeNumber(mv)) return mv;
    if (isRuntimeLogical(mv)) return mv ? 1 : 0;
  }
  return null;
}
import {
  linsolveLapack,
  linsolveComplexLapack,
} from "../../numbl-core/helpers/linsolve.js";
import { sparseToDense as sparseToDenseFn } from "../../numbl-core/helpers/sparse-arithmetic.js";
import { mLeftDiv, mLogical } from "../../numbl-core/helpers/arithmetic.js";
import { OpCmp } from "../ops/opCodes.js";
import {
  type DeferredRange,
  type DeferredHorzcat,
  isDeferredRange,
  ensureRuntimeValue,
  isNumericKind,
  switchValuesMatch,
} from "./runtimeHelpers.js";
import { allocFloat64Array } from "./alloc.js";

// ── Operator methods ────────────────────────────────────────────────────

/** The method each binary operator calls (its functional form): an object
 *  operand dispatches the operator to this method of its class. `&&` and
 *  `||` have none. */
export const BINARY_OPERATOR_METHODS: Readonly<
  Partial<Record<string, string>>
> = {
  [BinaryOperation.Add]: "plus",
  [BinaryOperation.Sub]: "minus",
  [BinaryOperation.Mul]: "mtimes",
  [BinaryOperation.ElemMul]: "times",
  [BinaryOperation.Div]: "mrdivide",
  [BinaryOperation.ElemDiv]: "rdivide",
  [BinaryOperation.LeftDiv]: "mldivide",
  [BinaryOperation.ElemLeftDiv]: "ldivide",
  [BinaryOperation.Pow]: "mpower",
  [BinaryOperation.ElemPow]: "power",
  [BinaryOperation.Equal]: "eq",
  [BinaryOperation.NotEqual]: "ne",
  [BinaryOperation.Less]: "lt",
  [BinaryOperation.LessEqual]: "le",
  [BinaryOperation.Greater]: "gt",
  [BinaryOperation.GreaterEqual]: "ge",
  [BinaryOperation.BitAnd]: "and",
  [BinaryOperation.BitOr]: "or",
};

/** The method each unary operator calls. */
export const UNARY_OPERATOR_METHODS: Readonly<Record<UnaryOperation, string>> =
  {
    [UnaryOperation.Plus]: "uplus",
    [UnaryOperation.Minus]: "uminus",
    [UnaryOperation.Not]: "not",
    [UnaryOperation.Transpose]: "ctranspose",
    [UnaryOperation.NonConjugateTranspose]: "transpose",
  };

// ── Unary operators ─────────────────────────────────────────────────────
// These are the built-in operations, reached after method dispatch (see
// `Runtime`) found no overload.

/** `+v`: numeric values unchanged; char and logical become double. */
export function uplus(v: unknown): RuntimeValue {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  const mv = ensureRuntimeValue(v);
  if (isRuntimeLogical(mv)) return mv ? 1 : 0;
  if (isRuntimeChar(mv)) return charToNumericTensor(mv);
  if (isRuntimeTensor(mv)) {
    if (mv.cls !== "logical") return mv;
    return RTV.tensor(allocFloat64Array(mv.data), mv.shape);
  }
  if (
    isRuntimeNumber(mv) ||
    isRuntimeComplexNumber(mv) ||
    isRuntimeSparseMatrix(mv) ||
    (isRuntimeClassInstance(mv) && mv._builtinData !== undefined)
  )
    return mv;
  throw invalidDataTypeError();
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

function notRealError(): RuntimeError {
  return new RuntimeError("Operands must be real.", null, {
    identifier: "MATLAB:Not:operandsNotReal",
  });
}

export function not(v: unknown): RuntimeLogical | RuntimeTensor {
  if (typeof v === "number") return RTV.logical(v === 0);
  if (typeof v === "boolean") return RTV.logical(!v);
  const mv = ensureRuntimeValue(v);
  if (isRuntimeNumber(mv)) return RTV.logical(mv === 0);
  if (isRuntimeLogical(mv)) return RTV.logical(!mv);
  if (isRuntimeSparseMatrix(mv)) return not(sparseToDenseFn(mv));
  if (isRuntimeChar(mv)) return not(charToNumericTensor(mv));
  if (isRuntimeComplexNumber(mv)) {
    if (mv.im !== 0) throw notRealError();
    return RTV.logical(mv.re === 0);
  }
  if (isRuntimeTensor(mv)) {
    const result = allocFloat64Array(mv.data.length);
    for (let i = 0; i < mv.data.length; i++) {
      if (mv.imag && mv.imag[i] !== 0) throw notRealError();
      result[i] = mv.data[i] === 0 ? 1 : 0;
    }
    const t = RTV.logicalTensor(result, mv.shape);
    return t;
  }
  // `classdef Foo < logical` (or a numeric superclass) without a `not`.
  if (isRuntimeClassInstance(mv) && mv._builtinData !== undefined)
    return not(mv._builtinData);
  throw invalidDataTypeError();
}

/** An operand of `&` / `|` as a numeric value: char becomes its codes;
 *  cells, structs, strings, function handles and objects are rejected. */
function logicalOperand(v: RuntimeValue): RuntimeValue {
  if (isRuntimeChar(v)) return charToNumericTensor(v);
  if (isRuntimeComplexNumber(v) || (isRuntimeTensor(v) && v.imag)) {
    const im = isRuntimeTensor(v) ? v.imag! : [v.im];
    for (let i = 0; i < im.length; i++)
      if (im[i] !== 0)
        throw new RuntimeError("Argument must be real.", null, {
          identifier: "MATLAB:math:mustBeReal",
        });
  }
  if (isNumericKind(v) || isRuntimeComplexNumber(v)) return v;
  throw invalidDataTypeError();
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

  // Secondary fast path: convert non-number primitives (e.g. booleans)
  // to numbers and retry. Skip when BOTH operands are already plain
  // numbers — the primary numeric fast path above already processed them,
  // and a Pow/ElemPow whose Math.pow result was NaN deliberately `break`s
  // out of that switch to reach the RuntimeValue slow path (for complex /
  // NaN handling). Re-entering here with the same two numbers would
  // recurse forever (e.g. `1 .^ Inf`, where Math.pow(1, Inf) stays NaN).
  if (
    (typeof a !== "object" || typeof b !== "object") &&
    !(typeof a === "number" && typeof b === "number")
  ) {
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
      if (isRuntimeSparseMatrix(ma)) {
        // Sparse direct solvers (with dense fallback) live in mLeftDiv.
        result = mLeftDiv(ma, mb);
        break;
      }
      const tensorA = ma;
      const tensorB = isRuntimeNumber(mb)
        ? RTV.tensor(allocFloat64Array([mb]), [1, 1])
        : isRuntimeSparseMatrix(mb)
          ? sparseToDenseFn(mb)
          : mb;
      if (!isRuntimeTensor(tensorA) || !isRuntimeTensor(tensorB)) {
        if (
          !(isNumericKind(tensorA) || isRuntimeChar(tensorA)) ||
          !(isNumericKind(tensorB) || isRuntimeChar(tensorB))
        )
          throw invalidDataTypeError();
        throw new RuntimeError(
          "LeftDiv (\\): operands must be numeric matrices"
        );
      }
      const [mA, nA] = tensorSize2D(tensorA);
      const [mB, nB] = tensorSize2D(tensorB);
      if (mB !== mA)
        throw new RuntimeError(
          `LeftDiv (\\): A is ${mA}×${nA} but B has ${mB} rows`
        );
      // Empty matrix: return empty result (nA × nB)
      if (mA === 0 || nA === 0 || nB === 0) {
        result = RTV.tensor(allocFloat64Array(nA * nB), [nA, nB]);
        break;
      }
      if (tensorA.imag || tensorB.imag) {
        const ARe = tensorA.data;
        const AIm = tensorA.imag ?? allocFloat64Array(tensorA.data.length);
        const BRe = tensorB.data;
        const BIm = tensorB.imag ?? allocFloat64Array(tensorB.data.length);
        const X = linsolveComplexLapack(ARe, AIm, mA, nA, BRe, BIm, nB);
        result = RTV.tensor(X.re, [nA, nB], X.im);
        break;
      }
      const X = linsolveLapack(tensorA.data, mA, nA, tensorB.data, nB);
      result = RTV.tensor(X, [nA, nB]);
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
    case BinaryOperation.BitAnd: {
      const la = logicalOperand(ma);
      const lb = logicalOperand(mb);
      if (
        (isRuntimeTensor(la) || isRuntimeTensor(lb)) &&
        isNumericKind(la) &&
        isNumericKind(lb)
      ) {
        result = mLogical(OpCmp.AND, la, lb);
      } else {
        result = RTV.logical(toBool(la) && toBool(lb));
      }
      break;
    }
    case BinaryOperation.BitOr: {
      const la = logicalOperand(ma);
      const lb = logicalOperand(mb);
      if (
        (isRuntimeTensor(la) || isRuntimeTensor(lb)) &&
        isNumericKind(la) &&
        isNumericKind(lb)
      ) {
        result = mLogical(OpCmp.OR, la, lb);
      } else {
        result = RTV.logical(toBool(la) || toBool(lb));
      }
      break;
    }
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
  checkColonOperand(start);
  checkColonOperand(step);
  checkColonOperand(end);
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

/** MATLAB's error for a `:` operand that is not numeric, char or logical
 *  (a cell, struct, string, object, ...). */
function checkColonOperand(v: unknown): void {
  if (typeof v === "number" || typeof v === "boolean") return;
  const rv = ensureRuntimeValue(v);
  if (
    isRuntimeTensor(rv) ||
    isRuntimeChar(rv) ||
    isRuntimeComplexNumber(rv) ||
    isRuntimeSparseMatrix(rv)
  )
    return;
  throw new RuntimeError("Inputs must be numeric, char, or logical.", null, {
    identifier: "MATLAB:colon:inputsMustBeNumericCharLogical",
  });
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
      const colData = allocFloat64Array(rows);
      for (let r = 0; r < rows; r++) colData[r] = mv.data[c * rows + r];
      let colImag: Float64Array | undefined;
      if (mv.imag) {
        colImag = allocFloat64Array(rows);
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
  if (isRuntimeStringArray(mv)) {
    // Iterate columns, like tensors: each iteration value is one column.
    const [rows, cols] = mv.shape;
    const out: RuntimeValue[] = [];
    for (let c = 0; c < cols; c++) {
      out.push(
        stringArrayValue(mv.data.slice(c * rows, (c + 1) * rows), [rows, 1])
      );
    }
    return out;
  }
  if (isRuntimeStructArray(mv) || isRuntimeClassInstanceArray(mv)) {
    // Iterate columns; a column of one element is the element itself.
    const rows = mv.shape[0];
    const n = mv.elements.length;
    const out: RuntimeValue[] = [];
    for (let c = 0; rows > 0 && c * rows < n; c++) {
      const col = mv.elements.slice(c * rows, (c + 1) * rows);
      if (rows === 1) out.push(col[0]);
      else if (isRuntimeStructArray(mv))
        out.push(
          RTV.structArray(mv.fieldNames.slice(), col as RuntimeStruct[], [
            rows,
            1,
          ])
        );
      else
        out.push(
          RTV.classInstanceArray(mv.className, col as RuntimeClassInstance[], [
            rows,
            1,
          ])
        );
    }
    return out;
  }
  return [v];
}

// ── Tensor/cell construction ────────────────────────────────────────────

export function emptyTensor(): RuntimeTensor {
  return RTV.tensor(allocFloat64Array(0), [0, 0]);
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
  if (flat.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
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
  if (rows.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
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
