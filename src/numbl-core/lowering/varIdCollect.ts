/**
 * Helpers that collect all VarIds referenced in IR nodes.
 * Used by lowering (to compute script-level var sets) and by codegen
 * (to generate var declarations for function bodies).
 *
 * This is the single canonical implementation — codegenHelpers.ts
 * re-exports from here.
 */

import { IRExpr, IRLValue, IRStmt } from "./nodes.js";
import { walkExpr } from "./nodeUtils.js";

/** Collect VarIds from a list of statements (non-recursive into nested Functions). */
export function collectVarIds(stmts: IRStmt[], out: Set<string>): void {
  for (const s of stmts) {
    if (s.type === "Function") continue;
    collectStmtVarIdsOne(s, out);
  }
}

export function collectStmtVarIdsOne(s: IRStmt, out: Set<string>): void {
  switch (s.type) {
    case "Assign":
      out.add(s.variable.id.id);
      collectExprVarIds(s.expr, out);
      break;
    case "MultiAssign":
      for (const lv of s.lvalues) {
        if (lv) collectLValueVarIds(lv, out);
      }
      collectExprVarIds(s.expr, out);
      break;
    case "ExprStmt":
      collectExprVarIds(s.expr, out);
      break;
    case "AssignLValue":
      collectExprVarIds(s.expr, out);
      collectLValueVarIds(s.lvalue, out);
      break;
    case "If":
      collectExprVarIds(s.cond, out);
      collectVarIds(s.thenBody, out);
      for (const b of s.elseifBlocks) {
        collectExprVarIds(b.cond, out);
        collectVarIds(b.body, out);
      }
      if (s.elseBody) collectVarIds(s.elseBody, out);
      break;
    case "While":
      collectExprVarIds(s.cond, out);
      collectVarIds(s.body, out);
      break;
    case "For":
      out.add(s.variable.id.id);
      collectExprVarIds(s.expr, out);
      collectVarIds(s.body, out);
      break;
    case "Switch":
      collectExprVarIds(s.expr, out);
      for (const c of s.cases) {
        collectExprVarIds(c.value, out);
        collectVarIds(c.body, out);
      }
      if (s.otherwise) collectVarIds(s.otherwise, out);
      break;
    case "TryCatch":
      collectVarIds(s.tryBody, out);
      if (s.catchVar) out.add(s.catchVar.id.id);
      collectVarIds(s.catchBody, out);
      break;
    case "Function":
      // Don't recurse into nested function definitions
      break;
    case "Global":
    case "Persistent":
      for (const v of s.vars) out.add(v.variable.id.id);
      break;
    case "Return":
    case "Break":
    case "Continue":
      break;
  }
}

function collectExprVarIds(e: IRExpr, out: Set<string>): void {
  walkExpr(e, sub => {
    if (sub.kind.type === "Var") out.add(sub.kind.variable.id.id);
  });
}

function collectLValueVarIds(lv: IRLValue, out: Set<string>): void {
  switch (lv.type) {
    case "Var":
      out.add(lv.variable.id.id);
      break;
    case "Member":
      collectExprVarIds(lv.base, out);
      break;
    case "MemberDynamic":
      collectExprVarIds(lv.base, out);
      collectExprVarIds(lv.nameExpr, out);
      break;
    case "Index":
    case "IndexCell":
      collectExprVarIds(lv.base, out);
      for (const idx of lv.indices) collectExprVarIds(idx, out);
      break;
  }
}
