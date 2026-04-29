/**
 * c-jit-loop feasibility whitelist.
 *
 * Walks a `LoopLowered` IR body and returns true iff every node is
 * something the C codegen can emit. Nodes outside the whitelist
 * (tensor ops, struct field access, etc.) cause `propose()` to
 * decline so the dispatcher falls through to the interpreter.
 *
 * Today's scope: scalar-only loops, real or complex (pair-of-doubles
 * encoding). Math builtins on complex args are NOT supported except
 * `real`, `imag`, and `conj`.
 */

import type { JitExpr, JitStmt, JitType } from "../../jitTypes.js";
import {
  LOOP_REAL_MATH_BUILTINS,
  LOOP_COMPLEX_PROJECTION_BUILTINS,
} from "./builtins.js";

export function isCScalarType(t: JitType): boolean {
  if (t.kind === "number") return true;
  if (t.kind === "boolean") return true;
  if (t.kind === "complex_or_number") return true;
  return false;
}

function isComplexType(t: JitType): boolean {
  return t.kind === "complex_or_number";
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
      // Loop conditions must be real-typed (truthiness on complex
      // would need extra logic).
      if (isComplexType(s.cond.jitType)) return false;
      return isCJitFeasible(s.body);
    case "If":
      if (!exprFeasible(s.cond)) return false;
      if (isComplexType(s.cond.jitType)) return false;
      if (!isCJitFeasible(s.thenBody)) return false;
      for (const eb of s.elseifBlocks) {
        if (!exprFeasible(eb.cond)) return false;
        if (isComplexType(eb.cond.jitType)) return false;
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
      return true;
    default:
      return false;
  }
}

function exprFeasible(e: JitExpr): boolean {
  if (!isCScalarType(e.jitType)) return false;
  switch (e.tag) {
    case "NumberLiteral":
      return true;
    case "ImagLiteral":
      return true;
    case "Var":
      return true;
    case "Binary":
      // All current binary ops can be emitted for real OR complex
      // operand combinations. The codegen handles the real/complex
      // promotion; just check sub-feasibility.
      return exprFeasible(e.left) && exprFeasible(e.right);
    case "Unary":
      return exprFeasible(e.operand);
    case "Call":
      if (LOOP_COMPLEX_PROJECTION_BUILTINS.has(e.name)) {
        // real / imag / conj — accept any scalar arg shape.
        for (const a of e.args) if (!exprFeasible(a)) return false;
        return true;
      }
      if (LOOP_REAL_MATH_BUILTINS.has(e.name)) {
        // Real-only math: all args must be real.
        for (const a of e.args) {
          if (!exprFeasible(a)) return false;
          if (isComplexType(a.jitType)) return false;
        }
        return true;
      }
      return false;
    default:
      return false;
  }
}
