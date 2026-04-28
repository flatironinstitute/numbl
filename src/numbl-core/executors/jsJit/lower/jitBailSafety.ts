/**
 * Bail-safety gate for JIT bodies that contain I/O side effects.
 *
 * Mid-execution bails (JitBailToInterpreter / JitFuncHandleBailError)
 * cause the interpreter to re-run the body from the top. If the body
 * already emitted I/O before the bail, that output gets duplicated —
 * which the user would notice.
 *
 * Rule: a body with any I/O statement may only be JIT-compiled if we
 * can prove no bail can happen during execution. If there's no I/O,
 * the body can be JIT'd normally (a bail just restarts silently).
 *
 * Walkers are IR-level — they see the actual lowered constructs that
 * map 1:1 to runtime bail sites. The walkers recurse into the bodies
 * of `UserCall`-ed functions (via `generatedIRBodies`), since a bail
 * inside a callee still forces the caller to re-run.
 */
import type { JitExpr, JitStmt } from "../../../jitTypes.js";
import type { GeneratedFn } from "./jitLower.js";

/**
 * Names of I/O-emitting builtins that the JIT is willing to lower as
 * ExprStmt calls. Any `Call` to one of these in the lowered IR marks
 * the body as having observable I/O.
 */
export const JIT_IO_BUILTINS = new Set<string>([
  "disp",
  "fprintf",
  "printf",
  "warning",
]);

/**
 * Walk a body + its transitively called function bodies and return
 * whether the combined execution graph contains an I/O call.
 */
export function irHasIO(
  body: JitStmt[],
  generatedIRBodies: Map<string, GeneratedFn>
): boolean {
  const visited = new Set<string>();
  return stmtsHaveIO(body, generatedIRBodies, visited);
}

/**
 * Walk a body + its transitively called function bodies and return
 * whether any construct exists that can throw `JitBailToInterpreter`
 * or `JitFuncHandleBailError` at runtime.
 *
 * Bail-risky constructs:
 *   - AssignIndex / AssignIndexCol: may bail on out-of-bounds grow.
 *   - FuncHandleCall / UserDispatchCall: return-type check may bail.
 *   - UserCall: recurse into callee body.
 */
export function irHasBailRisk(
  body: JitStmt[],
  generatedIRBodies: Map<string, GeneratedFn>
): boolean {
  const visited = new Set<string>();
  return stmtsHaveBailRisk(body, generatedIRBodies, visited);
}

// ── I/O walkers ─────────────────────────────────────────────────────────

function stmtsHaveIO(
  stmts: JitStmt[],
  gen: Map<string, GeneratedFn>,
  visited: Set<string>
): boolean {
  for (const s of stmts) if (stmtHasIO(s, gen, visited)) return true;
  return false;
}

function stmtHasIO(
  s: JitStmt,
  gen: Map<string, GeneratedFn>,
  visited: Set<string>
): boolean {
  switch (s.tag) {
    case "Assign":
      return exprHasIO(s.expr, gen, visited);
    case "AssignIndex":
      for (const i of s.indices) if (exprHasIO(i, gen, visited)) return true;
      return exprHasIO(s.value, gen, visited);
    case "AssignIndexRange":
      if (exprHasIO(s.dstStart, gen, visited)) return true;
      if (exprHasIO(s.dstEnd, gen, visited)) return true;
      if (s.srcStart && exprHasIO(s.srcStart, gen, visited)) return true;
      if (s.srcEnd && exprHasIO(s.srcEnd, gen, visited)) return true;
      return false;
    case "AssignIndexCol":
      return exprHasIO(s.colIndex, gen, visited);
    case "AssignIndexPage3d":
      if (exprHasIO(s.pageIndex, gen, visited)) return true;
      return exprHasIO(s.value, gen, visited);
    case "AssignMember":
      return exprHasIO(s.value, gen, visited);
    case "If":
      if (exprHasIO(s.cond, gen, visited)) return true;
      if (stmtsHaveIO(s.thenBody, gen, visited)) return true;
      for (const eib of s.elseifBlocks) {
        if (exprHasIO(eib.cond, gen, visited)) return true;
        if (stmtsHaveIO(eib.body, gen, visited)) return true;
      }
      if (s.elseBody && stmtsHaveIO(s.elseBody, gen, visited)) return true;
      return false;
    case "For":
      if (exprHasIO(s.start, gen, visited)) return true;
      if (s.step && exprHasIO(s.step, gen, visited)) return true;
      if (exprHasIO(s.end, gen, visited)) return true;
      return stmtsHaveIO(s.body, gen, visited);
    case "While":
      if (exprHasIO(s.cond, gen, visited)) return true;
      return stmtsHaveIO(s.body, gen, visited);
    case "ExprStmt":
      return exprHasIO(s.expr, gen, visited);
    case "MultiAssign":
      for (const a of s.args) if (exprHasIO(a, gen, visited)) return true;
      return false;
    case "UserCallWriteback":
      for (const a of s.args) if (exprHasIO(a, gen, visited)) return true;
      return calleeHasIO(s.jitName, gen, visited);
    case "Break":
    case "Continue":
    case "Return":
    case "SetLoc":
    case "AssertCJit":
      return false;
  }
}

function exprHasIO(
  e: JitExpr,
  gen: Map<string, GeneratedFn>,
  visited: Set<string>
): boolean {
  switch (e.tag) {
    case "NumberLiteral":
    case "ImagLiteral":
    case "StringLiteral":
    case "Var":
    case "MemberRead":
      return false;
    case "Binary":
      return (
        exprHasIO(e.left, gen, visited) || exprHasIO(e.right, gen, visited)
      );
    case "Unary":
      return exprHasIO(e.operand, gen, visited);
    case "Call":
      if (JIT_IO_BUILTINS.has(e.name)) return true;
      for (const a of e.args) if (exprHasIO(a, gen, visited)) return true;
      return false;
    case "UserCall":
      for (const a of e.args) if (exprHasIO(a, gen, visited)) return true;
      return calleeHasIO(e.jitName, gen, visited);
    case "FuncHandleCall":
    case "UserDispatchCall":
      for (const a of e.args) if (exprHasIO(a, gen, visited)) return true;
      return false;
    case "Index":
      if (exprHasIO(e.base, gen, visited)) return true;
      for (const i of e.indices) if (exprHasIO(i, gen, visited)) return true;
      return false;
    case "RangeSliceRead":
      if (exprHasIO(e.start, gen, visited)) return true;
      if (e.end && exprHasIO(e.end, gen, visited)) return true;
      return false;
    case "TensorLiteral":
      for (const row of e.rows)
        for (const c of row) if (exprHasIO(c, gen, visited)) return true;
      return false;
    case "VConcatGrow":
      return (
        exprHasIO(e.base, gen, visited) || exprHasIO(e.value, gen, visited)
      );
    case "StructArrayMemberRead":
      return exprHasIO(e.indexExpr, gen, visited);
  }
}

function calleeHasIO(
  jitName: string,
  gen: Map<string, GeneratedFn>,
  visited: Set<string>
): boolean {
  if (visited.has(jitName)) return false;
  visited.add(jitName);
  const gf = gen.get(jitName);
  if (!gf) return false;
  return stmtsHaveIO(gf.body, gen, visited);
}

// ── Bail-risk walkers ───────────────────────────────────────────────────

function stmtsHaveBailRisk(
  stmts: JitStmt[],
  gen: Map<string, GeneratedFn>,
  visited: Set<string>
): boolean {
  for (const s of stmts) if (stmtHasBailRisk(s, gen, visited)) return true;
  return false;
}

function stmtHasBailRisk(
  s: JitStmt,
  gen: Map<string, GeneratedFn>,
  visited: Set<string>
): boolean {
  switch (s.tag) {
    case "AssignIndex":
    case "AssignIndexCol":
      // Out-of-bounds grow throws JitBailToInterpreter.
      return true;
    case "Assign":
      return exprHasBailRisk(s.expr, gen, visited);
    case "AssignIndexRange":
      if (exprHasBailRisk(s.dstStart, gen, visited)) return true;
      if (exprHasBailRisk(s.dstEnd, gen, visited)) return true;
      if (s.srcStart && exprHasBailRisk(s.srcStart, gen, visited)) return true;
      if (s.srcEnd && exprHasBailRisk(s.srcEnd, gen, visited)) return true;
      return false;
    case "AssignIndexPage3d":
      if (exprHasBailRisk(s.pageIndex, gen, visited)) return true;
      return exprHasBailRisk(s.value, gen, visited);
    case "AssignMember":
      return exprHasBailRisk(s.value, gen, visited);
    case "If":
      if (exprHasBailRisk(s.cond, gen, visited)) return true;
      if (stmtsHaveBailRisk(s.thenBody, gen, visited)) return true;
      for (const eib of s.elseifBlocks) {
        if (exprHasBailRisk(eib.cond, gen, visited)) return true;
        if (stmtsHaveBailRisk(eib.body, gen, visited)) return true;
      }
      if (s.elseBody && stmtsHaveBailRisk(s.elseBody, gen, visited))
        return true;
      return false;
    case "For":
      if (exprHasBailRisk(s.start, gen, visited)) return true;
      if (s.step && exprHasBailRisk(s.step, gen, visited)) return true;
      if (exprHasBailRisk(s.end, gen, visited)) return true;
      return stmtsHaveBailRisk(s.body, gen, visited);
    case "While":
      if (exprHasBailRisk(s.cond, gen, visited)) return true;
      return stmtsHaveBailRisk(s.body, gen, visited);
    case "ExprStmt":
      return exprHasBailRisk(s.expr, gen, visited);
    case "MultiAssign":
      for (const a of s.args) if (exprHasBailRisk(a, gen, visited)) return true;
      return false;
    case "UserCallWriteback":
      for (const a of s.args) if (exprHasBailRisk(a, gen, visited)) return true;
      return calleeHasBailRisk(s.jitName, gen, visited);
    case "Break":
    case "Continue":
    case "Return":
    case "SetLoc":
    case "AssertCJit":
      return false;
  }
}

function exprHasBailRisk(
  e: JitExpr,
  gen: Map<string, GeneratedFn>,
  visited: Set<string>
): boolean {
  switch (e.tag) {
    case "NumberLiteral":
    case "ImagLiteral":
    case "StringLiteral":
    case "Var":
    case "MemberRead":
      return false;
    case "Binary":
      return (
        exprHasBailRisk(e.left, gen, visited) ||
        exprHasBailRisk(e.right, gen, visited)
      );
    case "Unary":
      return exprHasBailRisk(e.operand, gen, visited);
    case "FuncHandleCall":
    case "UserDispatchCall":
      // Return-type check throws JitFuncHandleBailError on mismatch.
      return true;
    case "Call":
      for (const a of e.args) if (exprHasBailRisk(a, gen, visited)) return true;
      return false;
    case "UserCall":
      for (const a of e.args) if (exprHasBailRisk(a, gen, visited)) return true;
      return calleeHasBailRisk(e.jitName, gen, visited);
    case "Index":
      if (exprHasBailRisk(e.base, gen, visited)) return true;
      for (const i of e.indices)
        if (exprHasBailRisk(i, gen, visited)) return true;
      return false;
    case "RangeSliceRead":
      if (exprHasBailRisk(e.start, gen, visited)) return true;
      if (e.end && exprHasBailRisk(e.end, gen, visited)) return true;
      return false;
    case "TensorLiteral":
      for (const row of e.rows)
        for (const c of row) if (exprHasBailRisk(c, gen, visited)) return true;
      return false;
    case "VConcatGrow":
      return (
        exprHasBailRisk(e.base, gen, visited) ||
        exprHasBailRisk(e.value, gen, visited)
      );
    case "StructArrayMemberRead":
      return exprHasBailRisk(e.indexExpr, gen, visited);
  }
}

function calleeHasBailRisk(
  jitName: string,
  gen: Map<string, GeneratedFn>,
  visited: Set<string>
): boolean {
  if (visited.has(jitName)) return false;
  visited.add(jitName);
  const gf = gen.get(jitName);
  if (!gf) return false;
  return stmtsHaveBailRisk(gf.body, gen, visited);
}
