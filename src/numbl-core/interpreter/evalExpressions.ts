/**
 * Expression evaluation: literals, operators, calls and member access.
 */

import type { Expr } from "../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../parser/types.js";
import { tryExtractDottedName } from "../parser/astUtils.js";
import type { RuntimeValue } from "../runtime/types.js";
import {
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeFunction,
  isRuntimeStructArray,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import {
  binop,
  range as runtimeRange,
  BINARY_OPERATOR_METHODS,
} from "../runtime/runtimeOperators.js";
import { dispatchClassOf } from "../runtime/kinds.js";
import { enumEqualityOp } from "../runtime/runtimeEnum.js";
import { mPow } from "../helpers/arithmetic.js";
import { COLON_SENTINEL, END_SENTINEL } from "../runtime/sentinels.js";
import { numel } from "../runtime/utils.js";
import { endOf } from "../runtime/runtimeIndexing.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import { resolveQualified, targetOf } from "../resolution/index.js";

import type { Environment } from "./types.js";
import type { Interpreter } from "./interpreter.js";
import {
  evalIndex,
  evalIndexCell,
  evalIndicesWithEnd,
  bypassesOverload,
} from "./indexing.js";
import { evalAnonFunc, makeFuncHandle } from "./handles.js";
import { callFunction, interpretTarget } from "./calls.js";
import {
  classConstantValue,
  instantiateClass,
  interpretEnumMember,
  evalSuperMethodCall,
} from "./oop.js";
import { metaClassFor } from "./metaClass.js";

export function evalExpr(interp: Interpreter, expr: Expr): unknown {
  return evalExprNargout(interp, expr, 1);
}

export function evalExprNargout(
  interp: Interpreter,
  expr: Expr,
  nargout: number
): unknown {
  switch (expr.type) {
    case "Number":
      return parseFloat(expr.value);

    case "Char": {
      let s = expr.value.slice(1, expr.value.length - 1);
      s = s.replaceAll("''", "'");
      return RTV.char(s);
    }

    case "String": {
      let s = expr.value.slice(1, expr.value.length - 1);
      s = s.replaceAll('""', '"');
      return RTV.string(s);
    }

    case "Ident": {
      const val = interp.frame.env.get(expr.name);
      if (val !== undefined) return val;
      // A function called without arguments, or a constant (pi, ...).
      return callFunction(interp, expr.name, [], nargout);
    }

    case "ImagUnit":
      return RTV.complex(0, 1);

    case "EndKeyword": {
      const ctx = interp.endContextStack[interp.endContextStack.length - 1];
      if (ctx) return endOf(interp.rt, ctx.base, ctx.dimIndex, ctx.numIndices);
      return END_SENTINEL;
    }

    case "Colon":
      return COLON_SENTINEL;

    case "Binary":
      return evalBinary(interp, expr);

    case "Unary":
      return evalUnary(interp, expr);

    case "Range": {
      const startVal = evalExpr(interp, expr.start);
      const endVal = evalExpr(interp, expr.end);
      const stepVal = expr.step ? evalExpr(interp, expr.step) : 1;
      return runtimeRange(startVal, stepVal, endVal);
    }

    case "FuncCall":
      return evalFuncCall(interp, expr, nargout);

    case "Index":
      return evalIndex(interp, expr, nargout);

    case "IndexCell":
      return evalIndexCell(interp, expr);

    case "Member":
      return evalMember(interp, expr, nargout);

    case "MemberDynamic": {
      const base = evalExpr(interp, expr.base);
      const nameVal = evalExpr(interp, expr.nameExpr);
      const name =
        typeof nameVal === "string"
          ? nameVal
          : isRuntimeChar(ensureRuntimeValue(nameVal))
            ? (ensureRuntimeValue(nameVal) as { value: string }).value
            : String(nameVal);
      return interp.rt.getMember(base, name);
    }

    case "MethodCall":
      return evalMethodCall(interp, expr, nargout);

    case "SuperMethodCall":
      return evalSuperMethodCall(interp, expr, nargout);

    case "AnonFunc":
      return evalAnonFunc(interp, expr);

    case "FuncHandle":
      return makeFuncHandle(interp, expr.name);

    case "Tensor":
      return evalTensorLiteral(interp, expr);

    case "Cell":
      return evalCellLiteral(interp, expr);

    case "ClassInstantiation": {
      const args = evalArgs(interp, expr.args);
      return instantiateClass(interp, expr.className, args, nargout);
    }

    case "MetaClass":
      return metaClassFor(interp, expr.name);
  }
}

/** Evaluate call arguments, expanding comma-separated lists. */
export function evalArgs(interp: Interpreter, argExprs: Expr[]): unknown[] {
  const args: unknown[] = [];
  for (const a of argExprs) {
    const val = evalExpr(interp, a);
    if (Array.isArray(val)) {
      for (const elem of val) {
        args.push(elem);
      }
    } else {
      args.push(val);
    }
  }
  return args;
}

// ── Operators ────────────────────────────────────────────────────────────

function evalBinary(
  interp: Interpreter,
  expr: Extract<Expr, { type: "Binary" }>
): unknown {
  const rt = interp.rt;
  if (expr.op === BinaryOperation.AndAnd) {
    const left = evalExpr(interp, expr.left);
    if (!rt.toBool(left)) return RTV.logical(false);
    const right = evalExpr(interp, expr.right);
    return RTV.logical(rt.toBool(right));
  }
  if (expr.op === BinaryOperation.OrOr) {
    const left = evalExpr(interp, expr.left);
    if (rt.toBool(left)) return RTV.logical(true);
    const right = evalExpr(interp, expr.right);
    return RTV.logical(rt.toBool(right));
  }

  const left = evalExpr(interp, expr.left);
  const right = evalExpr(interp, expr.right);

  const lv = ensureRuntimeValue(left);
  const rv = ensureRuntimeValue(right);
  // Enumeration ==/~= (member vs member / numeric / char). Handles both scalar
  // members and member arrays; returns null for other ops / non-enum operands.
  const enumResult = enumEqualityOp(expr.op, lv, rv);
  if (enumResult !== null) return enumResult;
  if (dispatchClassOf(lv) !== null || dispatchClassOf(rv) !== null) {
    return rt.binop(expr.op, left, right);
  }

  if (
    (expr.op === BinaryOperation.Pow || expr.op === BinaryOperation.ElemPow) &&
    typeof left === "number" &&
    typeof right === "number" &&
    left < 0
  ) {
    const r = Math.pow(left, right);
    if (!isNaN(r)) return r;
    return mPow(ensureRuntimeValue(left), ensureRuntimeValue(right));
  }

  // Profile non-scalar binary ops (tensor arithmetic)
  if (
    rt.profilingEnabled &&
    (typeof left !== "number" || typeof right !== "number")
  ) {
    const opName = BINARY_OPERATOR_METHODS[expr.op] ?? expr.op;
    rt.profileEnter("builtin:interp:" + opName);
    const result = binop(expr.op, left, right);
    rt.profileLeave();
    return result;
  }

  return binop(expr.op, left, right);
}

/** A value counts as scalar (numel === 1) for condition short-circuiting. */
function isScalarValue(v: unknown): boolean {
  if (typeof v === "number" || typeof v === "boolean") return true;
  const rv = ensureRuntimeValue(v);
  if (typeof rv === "number" || typeof rv === "boolean") return true;
  if (typeof rv === "string") return false; // treat char/string as non-scalar
  const shape = (rv as { shape?: number[] }).shape;
  if (shape) return numel(shape) === 1;
  return true; // complex_number and other scalar kinds
}

/**
 * Evaluate an `if`/`while`/`elseif` condition.
 *
 * MATLAB short-circuits the element-wise `&` and `|` operators when they
 * appear at the top of a conditional expression and the left operand is
 * scalar — e.g. `if nargin<4 | isempty(x)` must NOT evaluate `isempty(x)`
 * when `nargin<4` (otherwise `x` may be undefined). Outside this context
 * `&`/`|` evaluate both operands element-wise, so the special handling
 * lives here rather than in `evalBinary`.
 */
export function evalCondition(interp: Interpreter, expr: Expr): unknown {
  if (
    expr.type === "Binary" &&
    (expr.op === BinaryOperation.BitOr || expr.op === BinaryOperation.BitAnd)
  ) {
    const left = evalCondition(interp, expr.left);
    if (isScalarValue(left)) {
      const lb = interp.rt.toBool(left);
      if (expr.op === BinaryOperation.BitOr) {
        if (lb) return RTV.logical(true);
      } else if (!lb) {
        return RTV.logical(false);
      }
      // Left scalar didn't decide it; the result follows the right side.
      return evalCondition(interp, expr.right);
    }
    // Non-scalar left: no short-circuit, evaluate element-wise as usual.
    const right = evalExpr(interp, expr.right);
    return interp.rt.binop(expr.op, left, right);
  }
  return evalExpr(interp, expr);
}

function evalUnary(
  interp: Interpreter,
  expr: Extract<Expr, { type: "Unary" }>
): unknown {
  const operand = evalExpr(interp, expr.operand);
  switch (expr.op) {
    case UnaryOperation.Plus:
      return interp.rt.uplus(operand);
    case UnaryOperation.Minus:
      return interp.rt.uminus(operand);
    case UnaryOperation.Not: {
      if (typeof operand === "number") return RTV.logical(operand === 0);
      return interp.rt.not(operand);
    }
    case UnaryOperation.Transpose:
      // ' = conjugate transpose
      return interp.rt.ctranspose(operand);
    case UnaryOperation.NonConjugateTranspose:
      // .' = non-conjugate transpose
      return interp.rt.transpose(operand);
  }
}

// ── Calls ────────────────────────────────────────────────────────────────

/** `name(args)`: index a variable, call a function handle held in a
 *  variable, or call the function `name` resolves to. */
function evalFuncCall(
  interp: Interpreter,
  expr: Extract<Expr, { type: "FuncCall" }>,
  nargout: number
): unknown {
  const frame = interp.frame;
  const varVal = frame.env.get(expr.name);
  if (varVal !== undefined) {
    const rv = ensureRuntimeValue(varVal);
    if (isRuntimeFunction(rv)) {
      const args = evalArgs(interp, expr.args);
      return interp.rt.index(rv, args, nargout);
    }
    const args = evalIndicesWithEnd(interp, varVal, expr.args);
    return interp.rt.index(varVal, args, nargout, bypassesOverload(interp, rv));
  }
  const args = evalArgs(interp, expr.args);
  // The argument names, for the callee's inputname(), are the frame's for
  // the duration of the call. Computed after evalArgs (which may itself
  // make calls).
  const outer = frame.callSiteNames;
  frame.callSiteNames =
    expr.args.length > 0 ? computeInputNames(expr.args, frame.env) : undefined;
  try {
    return callFunction(interp, expr.name, args, nargout);
  } finally {
    frame.callSiteNames = outer;
  }
}

/** Compute the call-site variable names for `inputname`, aligned to the
 *  argument expressions. Entry i is the name of the variable passed as
 *  argument i+1, or '' if that argument is not a plain workspace variable.
 *
 *  Per MATLAB: an argument that uses cell `{}` or dot `.` indexing produces
 *  a comma-separated list, making the position of every following argument
 *  dynamic — so that argument and all subsequent ones report ''. Plain
 *  literals, expressions, and paren-indexing report '' only for themselves.
 *
 *  The returned array may be shorter than the flattened arg count (when an
 *  argument expanded to a CSL); callers treat out-of-range indices as ''.
 */
function computeInputNames(argExprs: Expr[], callerEnv: Environment): string[] {
  const names: string[] = [];
  let blanked = false;
  for (const a of argExprs) {
    if (blanked) {
      names.push("");
    } else if (a.type === "Ident" && callerEnv.has(a.name)) {
      names.push(a.name);
    } else if (
      a.type === "IndexCell" ||
      a.type === "Member" ||
      a.type === "MemberDynamic"
    ) {
      // cell / dot indexing: '' here and for everything after it
      names.push("");
      blanked = true;
    } else {
      names.push("");
    }
  }
  return names;
}

/** `base.name(args)`: `pkg.fn(args)`, `pkg.Class(args)`,
 *  `Class.method(args)`, `Class.EnumMember(idx)`, or a method call or
 *  indexed field on a value. */
function evalMethodCall(
  interp: Interpreter,
  expr: Extract<Expr, { type: "MethodCall" }>,
  nargout: number
): unknown {
  const dottedBase = tryExtractDottedName(expr.base);
  if (dottedBase) {
    const r = resolveQualified(
      [...dottedBase.split("."), expr.name],
      interp.frame
    );
    if (r.kind === "enumMember") {
      const member = interpretEnumMember(interp, r.className, r.memberName)!;
      if (expr.args.length === 0) return member;
      return interp.rt.index(
        member,
        evalIndicesWithEnd(interp, member, expr.args)
      );
    }
    if (r.kind === "classConstant") {
      const value = classConstantValue(interp, r.className, r.propName);
      return interp.rt.index(
        value,
        evalIndicesWithEnd(interp, value, expr.args),
        nargout
      );
    }
    const target = targetOf(r);
    if (target) {
      const args = evalArgs(interp, expr.args);
      return interpretTarget(interp, target, args, nargout);
    }
  }
  const base = evalExpr(interp, expr.base);
  // For struct/class fields, try to get the field value for end-context
  // resolution. For class instances, read the field directly (not via
  // getMember which could trigger subsref or call a no-arg method as a side
  // effect).
  let fieldVal: unknown = undefined;
  const baseRv = ensureRuntimeValue(base);
  if (isRuntimeClassInstance(baseRv)) {
    const fv = baseRv.fields.get(expr.name);
    if (fv !== undefined) fieldVal = fv;
  } else {
    try {
      fieldVal = interp.rt.getMember(base, expr.name);
    } catch {
      // Not a struct field
    }
  }
  const args =
    fieldVal !== undefined
      ? evalIndicesWithEnd(interp, fieldVal, expr.args)
      : evalArgs(interp, expr.args);
  return interp.rt.methodDispatch(expr.name, nargout, [base, ...args]);
}

// ── Member access ────────────────────────────────────────────────────────

function evalMember(
  interp: Interpreter,
  expr: Extract<Expr, { type: "Member" }>,
  nargout: number
): unknown {
  // `pkg.fn`, `pkg.Class`, `Class.staticMethod`, `Class.EnumMember`, ...
  // Any other `Class.name` (e.g. a constant property) evaluates `Class`.
  const dottedName = tryExtractDottedName(expr);
  if (dottedName) {
    const r = resolveQualified(dottedName.split("."), interp.frame);
    switch (r.kind) {
      case "enumMember":
        return interpretEnumMember(interp, r.className, r.memberName);
      case "classConstant":
        return classConstantValue(interp, r.className, r.propName);
      case "packageFunction":
      case "classConstructor":
      case "staticMethod":
        return interpretTarget(interp, r.target, [], nargout);
    }
  }
  const base = evalExpr(interp, expr.base);
  // Field access on a non-scalar struct array, or property access on a
  // non-scalar object array, yields a comma-separated list — represented as
  // a JS array, like c{:} — so {s.f}, [s.f], and f(s.f) expand per element.
  // (Other member access on an object array, e.g. a method called without
  // parentheses, keeps the materialized-horzcat behavior of rt.getMember.)
  const baseRv = ensureRuntimeValue(base);
  if (
    (isRuntimeStructArray(baseRv) ||
      (isRuntimeClassInstanceArray(baseRv) &&
        (baseRv.elements[0]?.fields.has(expr.name) ||
          interp.code.classes.findProperty(baseRv.className, expr.name)))) &&
    baseRv.elements.length !== 1
  ) {
    return baseRv.elements.map(el =>
      ensureRuntimeValue(interp.rt.getMember(el, expr.name))
    );
  }
  return interp.rt.getMember(base, expr.name, nargout);
}

// ── Tensor and cell literals ─────────────────────────────────────────────

/** Evaluate a literal row's elements, expanding comma-separated lists. */
function evalRow(interp: Interpreter, row: Expr[]): unknown[] {
  const vals: unknown[] = [];
  for (const e of row) {
    const v = evalExpr(interp, e);
    if (Array.isArray(v)) {
      for (const elem of v) vals.push(elem);
    } else {
      vals.push(v);
    }
  }
  return vals;
}

function evalTensorLiteral(
  interp: Interpreter,
  expr: Extract<Expr, { type: "Tensor" }>
): unknown {
  if (expr.rows.length === 0) {
    return RTV.tensor(allocFloat64Array(0), [0, 0]);
  }
  const rowValues: unknown[] = [];
  for (const row of expr.rows) {
    const vals = evalRow(interp, row);
    if (vals.length === 1) {
      rowValues.push(vals[0]);
    } else {
      rowValues.push(interp.rt.horzcat(vals));
    }
  }
  if (rowValues.length === 1) {
    return rowValues[0];
  }
  return interp.rt.vertcat(rowValues);
}

function evalCellLiteral(
  interp: Interpreter,
  expr: Extract<Expr, { type: "Cell" }>
): unknown {
  if (expr.rows.length === 0) {
    return RTV.cell([], [0, 0]);
  }
  // Evaluate each row, expanding comma-separated lists (JS arrays, e.g.
  // c{:} or structArray.field).
  const rowValues: RuntimeValue[][] = expr.rows.map(row =>
    evalRow(interp, row).map(v => ensureRuntimeValue(v))
  );
  if (rowValues.length === 1) {
    return RTV.cell(rowValues[0], [1, rowValues[0].length]);
  }
  // Multi-row: require rectangularity.
  const numRows = rowValues.length;
  const numCols = rowValues[0].length;
  for (const rv of rowValues) {
    if (rv.length !== numCols) {
      throw new RuntimeError(
        "Cell array rows must have the same number of columns"
      );
    }
  }
  const elements: RuntimeValue[] = [];
  for (let c = 0; c < numCols; c++) {
    for (let r = 0; r < numRows; r++) {
      elements.push(rowValues[r][c]);
    }
  }
  return RTV.cell(elements, [numRows, numCols]);
}
