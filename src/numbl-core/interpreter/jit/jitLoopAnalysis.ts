/**
 * Static AST analysis to determine input/output variables for loop JIT.
 *
 * Walks the loop body (and condition for while) to collect:
 * - assigned: variables written inside the loop
 * - referenced: variables read inside the loop
 * - hasReturn: whether the loop body contains a return statement
 */

import type { Stmt, Expr } from "../../parser/types.js";

export interface LoopVarInfo {
  /** Variables referenced in the loop that must come from enclosing scope */
  inputs: string[];
  /** Variables assigned in the loop body (written back after JIT execution) */
  outputs: string[];
  /** Whether the loop body contains a return statement (skip JIT if true) */
  hasReturn: boolean;
}

/** Analyze a for loop statement for JIT compilation. */
export function analyzeForLoop(stmt: Stmt & { type: "For" }): LoopVarInfo {
  const assigned = new Set<string>();
  const referenced = new Set<string>();
  let hasReturn = false;

  // The loop variable is assigned
  assigned.add(stmt.varName);

  // The range expression references variables
  walkExpr(stmt.expr, referenced);

  // Walk the loop body
  const result = walkStmts(stmt.body, assigned, referenced);
  hasReturn = result.hasReturn;

  // Inputs = referenced variables not exclusively created inside the loop
  // (The caller will filter by what actually exists in the interpreter env)
  const inputs = [...referenced];
  const outputs = [...assigned];

  return { inputs, outputs, hasReturn };
}

/**
 * Collect the names of all variables read in a sibling-tail starting at
 * `startIdx` of the given stmt list. Used by the loop JIT to filter the
 * loop's output set so that loop-internal temporaries don't get written
 * back when no later code reads them.
 */
export function collectReadsFromSiblings(
  stmts: Stmt[],
  startIdx: number,
  out: Set<string>
): void {
  // Walking via the existing `walkStmts` helper would also collect
  // *assigned* names, which is what we want here too — anything read
  // before being assigned in the tail is a true input, but anything
  // assigned in the tail might be read later in the tail. The simplest
  // safe approximation is to mark every name that appears textually as a
  // read (i.e. mix `referenced` and the lvalue base). The walkers below
  // already include this (Index/Member lvalue bases get added to both
  // assigned and referenced).
  const tailAssigned = new Set<string>();
  for (let i = startIdx; i < stmts.length; i++) {
    walkStmt(stmts[i], tailAssigned, out);
  }
}

/** Analyze a while loop statement for JIT compilation. */
export function analyzeWhileLoop(stmt: Stmt & { type: "While" }): LoopVarInfo {
  const assigned = new Set<string>();
  const referenced = new Set<string>();

  // The condition references variables
  walkExpr(stmt.cond, referenced);

  // Walk the loop body
  const { hasReturn } = walkStmts(stmt.body, assigned, referenced);

  const inputs = [...referenced];
  const outputs = [...assigned];

  return { inputs, outputs, hasReturn };
}

// ── AST walkers ──────────────────────────────────────────────────────────

function walkStmts(
  stmts: Stmt[],
  assigned: Set<string>,
  referenced: Set<string>
): { hasReturn: boolean } {
  let hasReturn = false;
  for (const stmt of stmts) {
    if (walkStmt(stmt, assigned, referenced)) hasReturn = true;
  }
  return { hasReturn };
}

/** Walk a statement, collecting assigned/referenced variables. Returns true if Return found. */
function walkStmt(
  stmt: Stmt,
  assigned: Set<string>,
  referenced: Set<string>
): boolean {
  switch (stmt.type) {
    case "Assign":
      walkExpr(stmt.expr, referenced);
      assigned.add(stmt.name);
      return false;

    case "AssignLValue":
      walkExpr(stmt.expr, referenced);
      walkLValue(stmt.lvalue, assigned, referenced);
      return false;

    case "MultiAssign":
      walkExpr(stmt.expr, referenced);
      for (const lv of stmt.lvalues) {
        if (lv.type === "Var") assigned.add(lv.name);
        else if (lv.type !== "Ignore") walkLValue(lv, assigned, referenced);
      }
      return false;

    case "ExprStmt":
      walkExpr(stmt.expr, referenced);
      return false;

    case "If": {
      walkExpr(stmt.cond, referenced);
      let ret = walkStmts(stmt.thenBody, assigned, referenced).hasReturn;
      for (const eib of stmt.elseifBlocks) {
        walkExpr(eib.cond, referenced);
        if (walkStmts(eib.body, assigned, referenced).hasReturn) ret = true;
      }
      if (stmt.elseBody) {
        if (walkStmts(stmt.elseBody, assigned, referenced).hasReturn)
          ret = true;
      }
      return ret;
    }

    case "For":
      assigned.add(stmt.varName);
      walkExpr(stmt.expr, referenced);
      return walkStmts(stmt.body, assigned, referenced).hasReturn;

    case "While":
      walkExpr(stmt.cond, referenced);
      return walkStmts(stmt.body, assigned, referenced).hasReturn;

    case "Switch":
      walkExpr(stmt.expr, referenced);
      for (const c of stmt.cases) {
        walkExpr(c.value, referenced);
        if (walkStmts(c.body, assigned, referenced).hasReturn) return true;
      }
      if (stmt.otherwise) {
        return walkStmts(stmt.otherwise, assigned, referenced).hasReturn;
      }
      return false;

    case "TryCatch":
      if (walkStmts(stmt.tryBody, assigned, referenced).hasReturn) return true;
      if (stmt.catchVar) assigned.add(stmt.catchVar);
      return walkStmts(stmt.catchBody, assigned, referenced).hasReturn;

    case "Return":
      return true;

    case "Break":
    case "Continue":
      return false;

    case "Global":
    case "Persistent":
      return false;

    default:
      return false;
  }
}

function walkLValue(
  lv: { type: string; [key: string]: unknown },
  assigned: Set<string>,
  referenced: Set<string>
): void {
  if (lv.type === "Var") {
    assigned.add(lv.name as string);
  } else if (lv.type === "Index" || lv.type === "IndexCell") {
    // base is referenced (and assigned to), indices are referenced
    const base = lv.base as Expr;
    if (base.type === "Ident") {
      assigned.add(base.name);
      referenced.add(base.name);
    } else {
      walkExpr(base, referenced);
    }
    for (const idx of lv.indices as Expr[]) {
      walkExpr(idx, referenced);
    }
  } else if (lv.type === "Member") {
    const base = lv.base as Expr;
    if (base.type === "Ident") {
      assigned.add(base.name);
      referenced.add(base.name);
    } else {
      walkExpr(base, referenced);
    }
  } else if (lv.type === "MemberDynamic") {
    // s.(fieldName) = v — base is both assigned and referenced, nameExpr is referenced
    const base = lv.base as Expr;
    if (base.type === "Ident") {
      assigned.add(base.name);
      referenced.add(base.name);
    } else {
      walkExpr(base, referenced);
    }
    walkExpr(lv.nameExpr as Expr, referenced);
  }
}

function walkExpr(expr: Expr, referenced: Set<string>): void {
  switch (expr.type) {
    case "Ident":
      referenced.add(expr.name);
      break;

    case "Number":
    case "ImagUnit":
    case "Char":
    case "String":
      break;

    case "Binary":
      walkExpr(expr.left, referenced);
      walkExpr(expr.right, referenced);
      break;

    case "Unary":
      walkExpr(expr.operand, referenced);
      break;

    case "FuncCall":
      // Don't add function name as referenced — it may be a builtin/function, not a variable.
      // The lowering will handle resolution. But if it turns out to be a variable
      // (indexing), lowerExpr will look it up in the env, and we'll have it as an input
      // because we include all env variables that appear as Ident references.
      // However, FuncCall.name could also be a variable used for indexing —
      // we add it as referenced and let the caller filter by what exists in env.
      referenced.add(expr.name);
      for (const arg of expr.args) walkExpr(arg, referenced);
      break;

    case "Index":
      walkExpr(expr.base, referenced);
      for (const idx of expr.indices) walkExpr(idx, referenced);
      break;

    case "IndexCell":
      walkExpr(expr.base, referenced);
      for (const idx of expr.indices) walkExpr(idx, referenced);
      break;

    case "Range":
      walkExpr(expr.start, referenced);
      if (expr.step) walkExpr(expr.step, referenced);
      walkExpr(expr.end, referenced);
      break;

    case "Tensor":
      for (const row of expr.rows) {
        for (const elem of row) walkExpr(elem, referenced);
      }
      break;

    case "Cell":
      for (const row of expr.rows) {
        for (const elem of row) walkExpr(elem, referenced);
      }
      break;

    case "Member":
      walkExpr(expr.base, referenced);
      break;

    case "MemberDynamic":
      // s.(nameExpr) — walk both base and the dynamic name expression
      walkExpr(expr.base, referenced);
      walkExpr((expr as { nameExpr: Expr }).nameExpr, referenced);
      break;

    case "MethodCall":
      // Stage 13: `T.nodes(i)` parses as MethodCall. We need `T` to
      // land in the referenced set so the loop JIT captures it as an
      // input. The interpreter also evaluates the base for method
      // dispatch, so this matches runtime semantics regardless of
      // whether the JIT ultimately lowers the call.
      walkExpr(expr.base, referenced);
      for (const arg of expr.args) walkExpr(arg, referenced);
      break;

    case "AnonFunc":
      // Walk the body expression — captures are references
      walkExpr(expr.body, referenced);
      break;

    case "EndKeyword":
    case "Colon":
      break;

    default:
      break;
  }
}
