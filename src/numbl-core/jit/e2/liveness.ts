/**
 * e2 — AST liveness helpers.
 *
 * Used by the chain classifier to decide whether a chain LHS is
 * actually used outside the chain's own stmts. If not, it can be
 * compiled as a per-element stack-local instead of being materialized
 * as a tensor output buffer.
 *
 * The "scope" passed in is the innermost enclosing function body or
 * top-level script body — chosen so that for-bodies, if-bodies, etc.
 * are scanned recursively (MATLAB has flat function-level scoping for
 * locals, so a name introduced inside a for-loop is visible to other
 * stmts in the same function body).
 *
 * The walk excludes the chain's own stmts (and the trailing-reduction
 * stmt if any) from the scan, applying the exclusion at every nesting
 * level — so a chain inside a for-body whose LHS is read by another
 * stmt in the same function body counts as referenced, but the chain
 * stmts themselves don't trigger a false positive.
 */

import type { Expr, Stmt, LValue } from "../../parser/types.js";

function exprMentionsName(expr: Expr, name: string): boolean {
  switch (expr.type) {
    case "Number":
    case "Char":
    case "String":
    case "EndKeyword":
    case "ImagUnit":
    case "Colon":
    case "MetaClass":
      return false;
    case "Ident":
      return expr.name === name;
    case "FuncHandle":
      return expr.name === name;
    case "FuncCall":
      return (
        expr.name === name || expr.args.some(a => exprMentionsName(a, name))
      );
    case "Binary":
      return (
        exprMentionsName(expr.left, name) || exprMentionsName(expr.right, name)
      );
    case "Unary":
      return exprMentionsName(expr.operand, name);
    case "Range":
      return (
        exprMentionsName(expr.start, name) ||
        (expr.step ? exprMentionsName(expr.step, name) : false) ||
        exprMentionsName(expr.end, name)
      );
    case "Index":
    case "IndexCell":
      return (
        exprMentionsName(expr.base, name) ||
        expr.indices.some(i => exprMentionsName(i, name))
      );
    case "Member":
      return exprMentionsName(expr.base, name);
    case "MemberDynamic":
      return (
        exprMentionsName(expr.base, name) ||
        exprMentionsName(expr.nameExpr, name)
      );
    case "MethodCall":
      return (
        exprMentionsName(expr.base, name) ||
        expr.args.some(a => exprMentionsName(a, name))
      );
    case "SuperMethodCall":
      return expr.args.some(a => exprMentionsName(a, name));
    case "AnonFunc":
      // Conservative: treat anonymous-function bodies as referencing
      // the name iff its body mentions it. (Captures-by-reference
      // semantics in MATLAB are by-value snapshots, so a closure
      // doesn't change shared state — the body scan is enough.)
      return exprMentionsName(expr.body, name);
    case "Tensor":
      return expr.rows.some(row => row.some(c => exprMentionsName(c, name)));
    case "Cell":
      return expr.rows.some(row => row.some(c => exprMentionsName(c, name)));
    case "ClassInstantiation":
      return expr.args.some(a => exprMentionsName(a, name));
  }
}

function lvalueMentionsName(lv: LValue, name: string): boolean {
  switch (lv.type) {
    case "Var":
      return lv.name === name;
    case "Ignore":
      return false;
    case "Index":
    case "IndexCell":
      return (
        exprMentionsName(lv.base, name) ||
        lv.indices.some(i => exprMentionsName(i, name))
      );
    case "Member":
      return exprMentionsName(lv.base, name);
    case "MemberDynamic":
      return (
        exprMentionsName(lv.base, name) || exprMentionsName(lv.nameExpr, name)
      );
  }
}

function stmtRefsNameOutsideExcluded(
  stmt: Stmt,
  exclude: ReadonlySet<Stmt>,
  name: string
): boolean {
  switch (stmt.type) {
    case "ExprStmt":
      return exprMentionsName(stmt.expr, name);
    case "Assign":
      return stmt.name === name || exprMentionsName(stmt.expr, name);
    case "MultiAssign":
      return (
        stmt.lvalues.some(lv => lvalueMentionsName(lv, name)) ||
        exprMentionsName(stmt.expr, name)
      );
    case "AssignLValue":
      return (
        lvalueMentionsName(stmt.lvalue, name) ||
        exprMentionsName(stmt.expr, name)
      );
    case "If":
      if (exprMentionsName(stmt.cond, name)) return true;
      if (bodyHasRefOutsideExcluded(stmt.thenBody, exclude, name)) return true;
      for (const b of stmt.elseifBlocks) {
        if (exprMentionsName(b.cond, name)) return true;
        if (bodyHasRefOutsideExcluded(b.body, exclude, name)) return true;
      }
      if (
        stmt.elseBody &&
        bodyHasRefOutsideExcluded(stmt.elseBody, exclude, name)
      )
        return true;
      return false;
    case "While":
      return (
        exprMentionsName(stmt.cond, name) ||
        bodyHasRefOutsideExcluded(stmt.body, exclude, name)
      );
    case "For":
      if (stmt.varName === name) return true;
      if (exprMentionsName(stmt.expr, name)) return true;
      return bodyHasRefOutsideExcluded(stmt.body, exclude, name);
    case "Switch":
      if (exprMentionsName(stmt.expr, name)) return true;
      for (const c of stmt.cases) {
        if (exprMentionsName(c.value, name)) return true;
        if (bodyHasRefOutsideExcluded(c.body, exclude, name)) return true;
      }
      if (
        stmt.otherwise &&
        bodyHasRefOutsideExcluded(stmt.otherwise, exclude, name)
      )
        return true;
      return false;
    case "TryCatch":
      return (
        bodyHasRefOutsideExcluded(stmt.tryBody, exclude, name) ||
        bodyHasRefOutsideExcluded(stmt.catchBody, exclude, name) ||
        stmt.catchVar === name
      );
    case "Function":
      // Nested function: do NOT recurse into its body (separate scope).
      // Same-name reference (function handle) counts.
      return stmt.name === name;
    case "Global":
    case "Persistent":
      return stmt.names.includes(name);
    case "Break":
    case "Continue":
    case "Return":
    case "Import":
    case "ClassDef":
    case "Directive":
      return false;
  }
}

function bodyHasRefOutsideExcluded(
  body: readonly Stmt[],
  exclude: ReadonlySet<Stmt>,
  name: string
): boolean {
  for (const s of body) {
    if (exclude.has(s)) continue;
    if (stmtRefsNameOutsideExcluded(s, exclude, name)) return true;
  }
  return false;
}

/**
 * True iff `name` appears anywhere in `scopeBody` outside the stmts
 * listed in `excludeStmts`. The exclusion is by reference identity
 * and is applied at every nesting level — pass the chain stmts (and
 * the trailing-reduction stmt if any) so they don't trigger false
 * positives.
 */
export function isNameReferencedOutsideStmts(
  scopeBody: readonly Stmt[],
  excludeStmts: ReadonlySet<Stmt>,
  name: string
): boolean {
  return bodyHasRefOutsideExcluded(scopeBody, excludeStmts, name);
}
