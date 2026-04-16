/**
 * Feasibility prepass for the C-JIT path.
 *
 * Given the lowered JIT IR for a function and the argument types, decide
 * whether the scalar-only C codegen can handle it. On any construct that
 * isn't in the MVP whitelist, return `{ok: false, reason}` so the caller
 * falls through to the JS-JIT path.
 *
 * The whitelist intentionally mirrors what [jitCodegenC.ts](./jitCodegenC.ts)
 * can emit. Widen both together.
 */

import { BinaryOperation, UnaryOperation } from "../../../parser/types.js";
import type { JitExpr, JitStmt, JitType } from "../jitTypes.js";

export type FeasibilityResult = { ok: true } | { ok: false; reason: string };

/**
 * Scalar math builtins that map 1:1 to `<math.h>` functions in the C emitter.
 *
 * **Deliberately excluded** (domain-restricted in MATLAB, where out-of-domain
 * inputs promote to complex rather than returning NaN):
 *   asin, acos, sqrt, log, log2, log10, acosh, atanh, log1p
 * The JS-JIT gates these with `requireNonneg`; we don't track the same
 * type refinement at feasibility time, so the conservative choice is to
 * bail for all call sites, letting JS-JIT handle them.
 */
export const C_SCALAR_MATH_BUILTINS = new Set<string>([
  "sin",
  "cos",
  "tan",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "exp",
  "abs",
  "floor",
  "ceil",
  "fix",
  "round",
  "sign",
  "atan2",
  "hypot",
  "mod",
  "rem",
  "expm1",
]);

function isScalarKind(k: JitType["kind"]): boolean {
  return k === "number" || k === "boolean";
}

function checkType(t: JitType): FeasibilityResult {
  if (!isScalarKind(t.kind)) {
    return { ok: false, reason: `non-scalar type: ${t.kind}` };
  }
  return { ok: true };
}

function checkExpr(expr: JitExpr): FeasibilityResult {
  const typeCheck = checkType(expr.jitType);
  if (!typeCheck.ok) return typeCheck;

  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
      return { ok: true };

    case "Binary": {
      switch (expr.op) {
        case BinaryOperation.Add:
        case BinaryOperation.Sub:
        case BinaryOperation.Mul:
        case BinaryOperation.ElemMul:
        case BinaryOperation.Div:
        case BinaryOperation.ElemDiv:
        case BinaryOperation.Pow:
        case BinaryOperation.ElemPow:
        case BinaryOperation.Equal:
        case BinaryOperation.NotEqual:
        case BinaryOperation.Less:
        case BinaryOperation.LessEqual:
        case BinaryOperation.Greater:
        case BinaryOperation.GreaterEqual:
        case BinaryOperation.AndAnd:
        case BinaryOperation.OrOr:
          break;
        default:
          return { ok: false, reason: `unsupported binary op ${expr.op}` };
      }
      const l = checkExpr(expr.left);
      if (!l.ok) return l;
      const r = checkExpr(expr.right);
      if (!r.ok) return r;
      return { ok: true };
    }

    case "Unary": {
      switch (expr.op) {
        case UnaryOperation.Plus:
        case UnaryOperation.Minus:
        case UnaryOperation.Not:
          break;
        default:
          return { ok: false, reason: `unsupported unary op ${expr.op}` };
      }
      return checkExpr(expr.operand);
    }

    case "Call": {
      if (!C_SCALAR_MATH_BUILTINS.has(expr.name)) {
        return { ok: false, reason: `non-C-mappable builtin: ${expr.name}` };
      }
      for (const a of expr.args) {
        const r = checkExpr(a);
        if (!r.ok) return r;
      }
      return { ok: true };
    }

    // Everything else is out of MVP scope — bail to JS-JIT.
    case "ImagLiteral":
    case "StringLiteral":
    case "TensorLiteral":
    case "VConcatGrow":
    case "RangeSliceRead":
    case "MemberRead":
    case "StructArrayMemberRead":
    case "UserCall":
    case "FuncHandleCall":
    case "UserDispatchCall":
    case "Index":
      return { ok: false, reason: `unsupported expr: ${expr.tag}` };
  }
}

function checkStmts(stmts: JitStmt[]): FeasibilityResult {
  for (const s of stmts) {
    const r = checkStmt(s);
    if (!r.ok) return r;
  }
  return { ok: true };
}

function checkStmt(stmt: JitStmt): FeasibilityResult {
  switch (stmt.tag) {
    case "Assign":
      return checkExpr(stmt.expr);

    case "ExprStmt":
      return checkExpr(stmt.expr);

    case "If": {
      const c = checkExpr(stmt.cond);
      if (!c.ok) return c;
      const t = checkStmts(stmt.thenBody);
      if (!t.ok) return t;
      for (const eib of stmt.elseifBlocks) {
        const ec = checkExpr(eib.cond);
        if (!ec.ok) return ec;
        const eb = checkStmts(eib.body);
        if (!eb.ok) return eb;
      }
      if (stmt.elseBody) return checkStmts(stmt.elseBody);
      return { ok: true };
    }

    case "For": {
      const s = checkExpr(stmt.start);
      if (!s.ok) return s;
      const e = checkExpr(stmt.end);
      if (!e.ok) return e;
      if (stmt.step) {
        const stepR = checkExpr(stmt.step);
        if (!stepR.ok) return stepR;
      }
      return checkStmts(stmt.body);
    }

    case "While": {
      const c = checkExpr(stmt.cond);
      if (!c.ok) return c;
      return checkStmts(stmt.body);
    }

    case "Break":
    case "Continue":
    case "Return":
    case "SetLoc":
      return { ok: true };

    // Out of MVP scope: tensor/struct writes, multi-assign, member writes.
    case "AssignIndex":
    case "AssignIndexRange":
    case "AssignIndexCol":
    case "AssignMember":
    case "MultiAssign":
      return { ok: false, reason: `unsupported stmt: ${stmt.tag}` };
  }
}

/**
 * Check if the lowered function can be handled by the scalar-only C-JIT.
 * `outputType` is the type of the first output (the return value).
 */
export function checkCFeasibility(
  body: JitStmt[],
  argTypes: JitType[],
  outputType: JitType | null,
  nargout: number
): FeasibilityResult {
  if (nargout > 1) {
    return { ok: false, reason: "multi-output not supported in C-JIT MVP" };
  }
  for (const t of argTypes) {
    const r = checkType(t);
    if (!r.ok) return { ok: false, reason: `arg: ${r.reason}` };
  }
  if (outputType) {
    const r = checkType(outputType);
    if (!r.ok) return { ok: false, reason: `return: ${r.reason}` };
  }
  return checkStmts(body);
}
