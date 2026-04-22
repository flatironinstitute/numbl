/**
 * Lightweight IR traversal helpers shared across the C-JIT subsystem.
 *
 * Several places in `jit/c/` need to walk a lowered IR body observing
 * (but not transforming) expressions and statements: the feasibility
 * fall-through paths, tensor-classification, hybrid-loop live-in/out
 * analysis, and the shape-propagation / callee-discovery / complex-
 * scalar scans in [jitCodegenC.ts](./jitCodegenC.ts). Each used to
 * reimplement the same switch-on-tag descent.
 *
 * This module centralizes the descent. Three primitives, composable:
 *
 *   - `walkExprNodes(expr, visit)` — post-order walk of every sub-node
 *     of `expr` (including `expr` itself). Every leaf calls `visit`
 *     once; nothing is skipped. Adding a new JitExpr tag means editing
 *     this one function.
 *
 *   - `walkStmts(body, visit)` — pre-order walk of every statement in
 *     `body`, recursing into If/For/While nested bodies. Does NOT
 *     traverse expressions inside the stmt — callers that need that
 *     compose with `walkStmtExprs` + `walkExprNodes`.
 *
 *   - `walkStmtExprs(stmt, visit)` — call `visit` on each top-level
 *     expression attached to `stmt` (the `expr` in an Assign, the
 *     `cond` in an If, the `start`/`end`/`step` in a For, etc.). Does
 *     NOT recurse into nested expression sub-nodes (use `walkExprNodes`
 *     for that) and does NOT walk into nested stmt bodies.
 *
 * The dispatchers in `cFeasibility.ts`, `emit/stmt.ts`, and
 * `emit/fused.ts` keep their native switches — they produce structured
 * results (feasibility verdicts, emitted C lines), so a callback-based
 * observer doesn't fit their shape.
 */
import type { JitExpr, JitStmt } from "../jitTypes.js";

/**
 * Walk every sub-node of `expr` in post-order (children first, then
 * `expr` itself), calling `visit` on each. Leaves (NumberLiteral,
 * ImagLiteral, Var, StringLiteral, MemberRead) are still visited once.
 *
 * Adding a new JitExpr tag: add a case here. Observer callers (which
 * is all of them) don't need to know about tag-specific sub-node
 * fields — this is the one place those are encoded.
 */
export function walkExprNodes(
  expr: JitExpr,
  visit: (e: JitExpr) => void
): void {
  switch (expr.tag) {
    case "Binary":
      walkExprNodes(expr.left, visit);
      walkExprNodes(expr.right, visit);
      break;
    case "Unary":
      walkExprNodes(expr.operand, visit);
      break;
    case "Call":
    case "UserCall":
    case "FuncHandleCall":
    case "UserDispatchCall":
      for (const a of expr.args) walkExprNodes(a, visit);
      break;
    case "Index":
      walkExprNodes(expr.base, visit);
      for (const i of expr.indices) walkExprNodes(i, visit);
      break;
    case "RangeSliceRead":
      walkExprNodes(expr.start, visit);
      if (expr.end) walkExprNodes(expr.end, visit);
      break;
    case "TensorLiteral":
      for (const row of expr.rows) {
        for (const cell of row) walkExprNodes(cell, visit);
      }
      break;
    case "VConcatGrow":
      walkExprNodes(expr.base, visit);
      walkExprNodes(expr.value, visit);
      break;
    case "StructArrayMemberRead":
      walkExprNodes(expr.indexExpr, visit);
      break;
    default:
      // Leaves: NumberLiteral, ImagLiteral, Var, StringLiteral, MemberRead.
      break;
  }
  visit(expr);
}

/**
 * Walk every statement in `body`, recursing into nested If / For /
 * While bodies. Pre-order: `visit` is called on each stmt before
 * descending. Does NOT traverse expressions inside the stmt.
 */
export function walkStmts(body: JitStmt[], visit: (s: JitStmt) => void): void {
  for (const s of body) {
    visit(s);
    switch (s.tag) {
      case "If":
        walkStmts(s.thenBody, visit);
        for (const eib of s.elseifBlocks) walkStmts(eib.body, visit);
        if (s.elseBody) walkStmts(s.elseBody, visit);
        break;
      case "For":
      case "While":
        walkStmts(s.body, visit);
        break;
      default:
        break;
    }
  }
}

/**
 * Call `visit` on every top-level expression attached to `stmt` — the
 * RHS of an Assign, the indices + value of an AssignIndex, the start /
 * end / step of a For, the cond of an If / While, and so on. Does
 * NOT recurse into expression sub-nodes (compose with `walkExprNodes`)
 * and does NOT descend into nested stmt bodies (compose with
 * `walkStmts`).
 *
 * For If, the `cond` of the primary branch AND each elseif is visited;
 * the bodies themselves are stmt-trees, not exprs, and are reached via
 * `walkStmts` recursion.
 */
export function walkStmtExprs(
  stmt: JitStmt,
  visit: (e: JitExpr) => void
): void {
  switch (stmt.tag) {
    case "Assign":
    case "ExprStmt":
      visit(stmt.expr);
      return;
    case "AssignIndex":
      visit(stmt.value);
      for (const i of stmt.indices) visit(i);
      return;
    case "AssignIndexRange":
      visit(stmt.dstStart);
      visit(stmt.dstEnd);
      if (stmt.srcStart) visit(stmt.srcStart);
      if (stmt.srcEnd) visit(stmt.srcEnd);
      return;
    case "AssignIndexCol":
      visit(stmt.colIndex);
      return;
    case "AssignIndexPage3d":
      visit(stmt.pageIndex);
      visit(stmt.value);
      return;
    case "AssignMember":
      visit(stmt.value);
      return;
    case "If":
      visit(stmt.cond);
      for (const eib of stmt.elseifBlocks) visit(eib.cond);
      return;
    case "For":
      visit(stmt.start);
      visit(stmt.end);
      if (stmt.step) visit(stmt.step);
      return;
    case "While":
      visit(stmt.cond);
      return;
    case "MultiAssign":
    case "UserCallWriteback":
      for (const a of stmt.args) visit(a);
      return;
    default:
      // Break / Continue / Return / SetLoc / AssertCJit: no exprs.
      return;
  }
}
