/**
 * e2 — minimal AST `Expr` → `JitExpr` lowerer.
 *
 * Only handles the whitelist that `classify.ts` accepts: Number, Ident,
 * whitelisted Binary/Unary, whitelisted FuncCall. Types are read from
 * the live runtime environment (the caller passes in a per-name
 * `JitType` lookup), so there's no cross-branch unification.
 *
 * The classifier already replaced opaque subtrees with synthetic Ident
 * nodes whose `name` is also in `envTypes`, so this lowerer doesn't need
 * to know about opacity — every Ident it sees has a known runtime type.
 */

import type { Expr } from "../../parser/types.js";
import { BinaryOperation } from "../../parser/types.js";
import type { JitExpr, JitType } from "../jitTypes.js";
import { E2_BUILTIN_WHITELIST } from "./classify.js";

export class E2LowerError extends Error {}

/** Pick the result type of a Binary op given operand types. Tensor wins
 *  over scalar; complex propagates. Booleans coerce to number. */
function unifyBinaryType(l: JitType, r: JitType): JitType {
  // Any tensor input yields a tensor output (elementwise).
  if (l.kind === "tensor" || r.kind === "tensor") {
    const lt = l.kind === "tensor" ? l : null;
    const rt = r.kind === "tensor" ? r : null;
    const isComplex = !!lt?.isComplex || !!rt?.isComplex;
    return { kind: "tensor", isComplex };
  }
  if (l.kind === "complex_or_number" || r.kind === "complex_or_number") {
    return { kind: "complex_or_number" };
  }
  return { kind: "number" };
}

/** True when both operand types are tensor-shaped. Used to reject
 *  matrix-multiplication operators (`Mul`/`Div`/`Pow` without dot)
 *  on tensor pairs — those aren't elementwise and have no per-element
 *  meaning in a fused loop. */
function bothTensor(l: JitType, r: JitType): boolean {
  return l.kind === "tensor" && r.kind === "tensor";
}

const MATRIX_OPS: ReadonlySet<BinaryOperation> = new Set([
  BinaryOperation.Mul,
  BinaryOperation.Div,
  BinaryOperation.LeftDiv,
  BinaryOperation.Pow,
]);

export function lowerAstToJitExpr(
  expr: Expr,
  envTypes: ReadonlyMap<string, JitType>
): JitExpr {
  switch (expr.type) {
    case "Number": {
      const value = parseFloat(expr.value);
      return {
        tag: "NumberLiteral",
        value,
        jitType: { kind: "number", exact: value },
      };
    }
    case "Ident": {
      const t = envTypes.get(expr.name);
      if (!t) {
        throw new E2LowerError(
          `e2: identifier '${expr.name}' has no known type`
        );
      }
      return { tag: "Var", name: expr.name, jitType: t };
    }
    case "Binary": {
      const left = lowerAstToJitExpr(expr.left, envTypes);
      const right = lowerAstToJitExpr(expr.right, envTypes);
      if (MATRIX_OPS.has(expr.op) && bothTensor(left.jitType, right.jitType)) {
        throw new E2LowerError(
          `e2: matrix op ${expr.op} on two tensors is not elementwise`
        );
      }
      return {
        tag: "Binary",
        op: expr.op,
        left,
        right,
        jitType: unifyBinaryType(left.jitType, right.jitType),
      };
    }
    case "Unary": {
      const operand = lowerAstToJitExpr(expr.operand, envTypes);
      return {
        tag: "Unary",
        op: expr.op,
        operand,
        jitType: operand.jitType,
      };
    }
    case "FuncCall": {
      if (!E2_BUILTIN_WHITELIST.has(expr.name)) {
        throw new E2LowerError(`e2: builtin '${expr.name}' not whitelisted`);
      }
      const args = expr.args.map(a => lowerAstToJitExpr(a, envTypes));
      // Result is tensor if any arg is tensor; complex if any arg is
      // complex; otherwise scalar number.
      let isTensor = false;
      let isComplex = false;
      for (const a of args) {
        if (a.jitType.kind === "tensor") {
          isTensor = true;
          if (a.jitType.isComplex) isComplex = true;
        } else if (a.jitType.kind === "complex_or_number") {
          isComplex = true;
        }
      }
      const jitType: JitType = isTensor
        ? { kind: "tensor", isComplex }
        : isComplex
          ? { kind: "complex_or_number" }
          : { kind: "number" };
      return { tag: "Call", name: expr.name, args, jitType };
    }
    default:
      throw new E2LowerError(
        `e2: AST node '${(expr as { type: string }).type}' not handled`
      );
  }
}
