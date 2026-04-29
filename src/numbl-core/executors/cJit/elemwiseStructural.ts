/**
 * Shared structural classifier for element-wise C-JIT.
 *
 * The "is this AST expression element-wise lowerable?" check, in two
 * pieces:
 *
 *   - `isElemwiseStructuralExpr(e)` — fully structural, no env access.
 *     Used by `chainPass` (runs at stmt-list-entry time, before env
 *     types are known).
 *
 *   - `isElemwiseBinaryOp(op)` / `binaryOpNeedsScalarCheck(op)` — the
 *     binary-op subset, exposed so env-aware walkers (`fuseAnalyze`)
 *     can apply runtime-scalar checks on top of the structural shape.
 *
 * Pairs with `builtins.ts` (which owns the unary builtin set). Anything
 * not accepted here is rejected at propose-time.
 */

import {
  BinaryOperation,
  type Expr,
  UnaryOperation,
} from "../../parser/types.js";
import { ELEMWISE_REAL_BUILTINS } from "./builtins.js";

/** Binary ops the element-wise codegen can emit. `Mul`/`Div` are matrix
 *  ops in MATLAB but degenerate to element-wise when at least one
 *  operand is a scalar — env-aware callers should additionally apply
 *  `binaryOpNeedsScalarCheck`. */
export function isElemwiseBinaryOp(op: BinaryOperation): boolean {
  switch (op) {
    case BinaryOperation.Add:
    case BinaryOperation.Sub:
    case BinaryOperation.Mul:
    case BinaryOperation.Div:
    case BinaryOperation.ElemMul:
    case BinaryOperation.ElemDiv:
      return true;
    default:
      return false;
  }
}

/** True iff `op` requires a runtime scalar-operand check before being
 *  emitted element-wise (i.e. plain `*` and `/`). */
export function binaryOpNeedsScalarCheck(op: BinaryOperation): boolean {
  return op === BinaryOperation.Mul || op === BinaryOperation.Div;
}

/** Unary ops the element-wise codegen can emit. */
export function isElemwiseUnaryOp(op: UnaryOperation): boolean {
  return op === UnaryOperation.Plus || op === UnaryOperation.Minus;
}

/** Structural check: is `e` element-wise lowerable as a tree shape?
 *  Does NOT consult env. Leaves (`Ident`, `Number`) are accepted
 *  unconditionally; runtime classification of identifiers (tensor vs.
 *  scalar) is the caller's responsibility. */
export function isElemwiseStructuralExpr(e: Expr): boolean {
  switch (e.type) {
    case "Number":
      return true;
    case "Ident":
      return true;
    case "Binary":
      if (!isElemwiseBinaryOp(e.op)) return false;
      return (
        isElemwiseStructuralExpr(e.left) && isElemwiseStructuralExpr(e.right)
      );
    case "Unary":
      if (!isElemwiseUnaryOp(e.op)) return false;
      return isElemwiseStructuralExpr(e.operand);
    case "FuncCall":
      if (!ELEMWISE_REAL_BUILTINS.has(e.name)) return false;
      if (e.args.length !== 1) return false;
      return isElemwiseStructuralExpr(e.args[0]);
    default:
      return false;
  }
}
