/**
 * c-jit-loop feasibility whitelist.
 *
 * Walks a `LoopLowered` IR body and returns true iff every node is
 * something the C codegen can emit. Nodes outside the whitelist
 * (tensor ops, complex math, struct field access, etc.) cause
 * `propose()` to decline so the dispatcher falls through to the
 * interpreter.
 *
 * The whitelist starts narrow (scalar-bench-shaped loops) and grows
 * as features land.
 */

import type { JitExpr, JitStmt, JitType } from "../../jitTypes.js";

/** Builtin scalar-math functions the C codegen knows how to emit. */
const C_SCALAR_MATH_BUILTINS: ReadonlySet<string> = new Set([
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "atan2",
  "sinh",
  "cosh",
  "tanh",
  "exp",
  "log",
  "log2",
  "log10",
  "sqrt",
  "abs",
  "floor",
  "ceil",
  "round",
]);

/** Whitelisted JitType kinds for the C codegen. Currently only
 *  scalar real numbers — every variable, every literal, every expr
 *  result must be a scalar f64. */
function isCScalar(t: JitType): boolean {
  if (t.kind === "number") return true;
  if (t.kind === "boolean") return true;
  return false;
}

export function isCJitFeasible(body: readonly JitStmt[]): boolean {
  for (const s of body) {
    if (!stmtFeasible(s)) return false;
  }
  return true;
}

function stmtFeasible(s: JitStmt): boolean {
  switch (s.tag) {
    case "Assign":
      return exprFeasible(s.expr);
    case "For":
      if (!exprFeasible(s.start)) return false;
      if (s.step !== null && !exprFeasible(s.step)) return false;
      if (!exprFeasible(s.end)) return false;
      return isCJitFeasible(s.body);
    case "While":
      if (!exprFeasible(s.cond)) return false;
      return isCJitFeasible(s.body);
    case "If":
      if (!exprFeasible(s.cond)) return false;
      if (!isCJitFeasible(s.thenBody)) return false;
      for (const eb of s.elseifBlocks) {
        if (!exprFeasible(eb.cond)) return false;
        if (!isCJitFeasible(eb.body)) return false;
      }
      if (s.elseBody && !isCJitFeasible(s.elseBody)) return false;
      return true;
    case "Break":
    case "Continue":
      return true;
    case "ExprStmt":
      return exprFeasible(s.expr);
    case "SetLoc":
      // Line tracking — emitted as a no-op in C.
      return true;
    default:
      return false;
  }
}

function exprFeasible(e: JitExpr): boolean {
  if (!isCScalar(e.jitType)) return false;
  switch (e.tag) {
    case "NumberLiteral":
      return true;
    case "Var":
      return true;
    case "Binary":
      return exprFeasible(e.left) && exprFeasible(e.right);
    case "Unary":
      return exprFeasible(e.operand);
    case "Call":
      if (!C_SCALAR_MATH_BUILTINS.has(e.name)) return false;
      for (const a of e.args) if (!exprFeasible(a)) return false;
      return true;
    default:
      return false;
  }
}
