/**
 * Shared scalar-expression emission used by both the JS-JIT and C-JIT
 * backends.
 *
 * The value-form switches for scalar Binary/Unary and the truthiness
 * walker have identical IR traversal between the two backends — only
 * the leaf syntax (operator spelling, coercion rules) differs. A
 * backend supplies a `ScalarOpTarget` describing how to spell each op,
 * and the shared functions below handle the dispatch.
 *
 * Complex-scalar and tensor ops are *not* covered here — those remain
 * backend-specific (JS uses `$h.cAdd` etc.; C-JIT bails on complex).
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { JitExpr } from "./jitTypes.js";

export interface ScalarOpTarget {
  // Binary scalar ops — value form (result is a scalar expression).
  binAdd(l: string, r: string): string;
  binSub(l: string, r: string): string;
  binMul(l: string, r: string): string; // also used for ElemMul
  binDiv(l: string, r: string): string; // also used for ElemDiv
  binPow(l: string, r: string): string; // also used for ElemPow
  binEq(l: string, r: string): string;
  binNe(l: string, r: string): string;
  binLt(l: string, r: string): string;
  binLe(l: string, r: string): string;
  binGt(l: string, r: string): string;
  binGe(l: string, r: string): string;
  binAnd(l: string, r: string): string;
  binOr(l: string, r: string): string;

  // Unary scalar ops — value form.
  unaryPlus(operand: string): string;
  unaryMinus(operand: string): string;
  unaryNot(operand: string): string;

  // Coerce a value expression to condition form (e.g. `(+(v)) !== 0` / `(v) != 0.0`).
  toTruthy(valueExpr: string): string;
  // Binary comparisons in condition context (no double-cast wrapping).
  condEq(l: string, r: string): string;
  condNe(l: string, r: string): string;
  condLt(l: string, r: string): string;
  condLe(l: string, r: string): string;
  condGt(l: string, r: string): string;
  condGe(l: string, r: string): string;
  // Logical combinators in condition context.
  condNot(truthyExpr: string): string;
  condAnd(l: string, r: string): string;
  condOr(l: string, r: string): string;
}

/** Dispatch a scalar Binary op to the target. Throws on unsupported ops. */
export function emitScalarBinaryOp(
  op: BinaryOperation,
  left: string,
  right: string,
  target: ScalarOpTarget
): string {
  switch (op) {
    case BinaryOperation.Add:
      return target.binAdd(left, right);
    case BinaryOperation.Sub:
      return target.binSub(left, right);
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return target.binMul(left, right);
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return target.binDiv(left, right);
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return target.binPow(left, right);
    case BinaryOperation.Equal:
      return target.binEq(left, right);
    case BinaryOperation.NotEqual:
      return target.binNe(left, right);
    case BinaryOperation.Less:
      return target.binLt(left, right);
    case BinaryOperation.LessEqual:
      return target.binLe(left, right);
    case BinaryOperation.Greater:
      return target.binGt(left, right);
    case BinaryOperation.GreaterEqual:
      return target.binGe(left, right);
    case BinaryOperation.AndAnd:
      return target.binAnd(left, right);
    case BinaryOperation.OrOr:
      return target.binOr(left, right);
    default:
      throw new Error(`scalar binary op ${op}: unsupported`);
  }
}

/** Dispatch a scalar Unary op to the target. Transpose is scalar-identity. */
export function emitScalarUnaryOp(
  op: UnaryOperation,
  operand: string,
  target: ScalarOpTarget
): string {
  switch (op) {
    case UnaryOperation.Plus:
      return target.unaryPlus(operand);
    case UnaryOperation.Minus:
      return target.unaryMinus(operand);
    case UnaryOperation.Not:
      return target.unaryNot(operand);
    case UnaryOperation.Transpose:
    case UnaryOperation.NonConjugateTranspose:
      return operand;
    default:
      throw new Error(`scalar unary op ${op}: unsupported`);
  }
}

/**
 * Emit a condition expression for `if` / `while` / `&&` / `||` operands.
 *
 * Recurses through nested comparison / logical operators so that the
 * whole condition emits as a native boolean/condition expression rather
 * than being wrapped in a trailing `!= 0` on every nested result.
 *
 * `emitValue` is the backend's value-form expression emitter — it is
 * called for the leaf operands of comparisons and for the fallback path.
 */
export function emitScalarTruthiness(
  expr: JitExpr,
  emitValue: (e: JitExpr) => string,
  target: ScalarOpTarget
): string {
  if (expr.tag === "Binary") {
    switch (expr.op) {
      case BinaryOperation.Equal:
        return target.condEq(emitValue(expr.left), emitValue(expr.right));
      case BinaryOperation.NotEqual:
        return target.condNe(emitValue(expr.left), emitValue(expr.right));
      case BinaryOperation.Less:
        return target.condLt(emitValue(expr.left), emitValue(expr.right));
      case BinaryOperation.LessEqual:
        return target.condLe(emitValue(expr.left), emitValue(expr.right));
      case BinaryOperation.Greater:
        return target.condGt(emitValue(expr.left), emitValue(expr.right));
      case BinaryOperation.GreaterEqual:
        return target.condGe(emitValue(expr.left), emitValue(expr.right));
      case BinaryOperation.AndAnd:
        return target.condAnd(
          emitScalarTruthiness(expr.left, emitValue, target),
          emitScalarTruthiness(expr.right, emitValue, target)
        );
      case BinaryOperation.OrOr:
        return target.condOr(
          emitScalarTruthiness(expr.left, emitValue, target),
          emitScalarTruthiness(expr.right, emitValue, target)
        );
      default:
        break;
    }
  }
  if (expr.tag === "Unary" && expr.op === UnaryOperation.Not) {
    return target.condNot(
      emitScalarTruthiness(expr.operand, emitValue, target)
    );
  }
  return target.toTruthy(emitValue(expr));
}
