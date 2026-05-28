/**
 * Classification (JS-JIT-independent).
 *
 * Two classification phases — top-level (whole-script) and call
 * (user-function-call) — produce the analysis records that codegen
 * executors consume to decide whether they can lower a stmt list /
 * call and to key their caches.
 *
 * Lives outside `jsJit/` and `cJit/` because the analysis is purely
 * AST + type inference and is consumed by the mtoc2 executors
 * (`executors/mtoc2/`) — no JS-JIT IR or C-JIT path involved.
 */

import type { Interpreter } from "../interpreter/interpreter.js";
import type { FunctionDef } from "../interpreter/types.js";
import type { Stmt, Expr } from "../parser/types.js";
import { jitTypeKey, unifyJitTypes, type JitType } from "../jitTypes.js";
import { inferJitType } from "../interpreter/builtins/types.js";

// ── Type-inference helpers ──────────────────────────────────────────────

/** Names that look like variables but never resolve to env values
 *  (constants, the special `end` slot, the imaginary unit). Skipped
 *  when collecting env inputs. */
export const KNOWN_CONSTANTS: ReadonlySet<string> = new Set([
  "pi",
  "inf",
  "Inf",
  "nan",
  "NaN",
  "eps",
  "true",
  "false",
  "end",
  "i",
  "j",
]);

/**
 * Drop `exact` from a numeric scalar `JitType`. Numeric `exact` only
 * survives unification when two consecutive specializations see the
 * *same* literal — almost never the case for variables — so stripping
 * up front means the first specialization's cacheKey already matches
 * later calls.
 */
export function pruneArgType(t: JitType): JitType {
  if (t.kind === "number" && t.exact !== undefined) {
    const pruned: JitType = { kind: "number" };
    if (t.sign !== undefined) pruned.sign = t.sign;
    if (t.isInteger) pruned.isInteger = true;
    return pruned;
  }
  return t;
}

/**
 * Progressive type widening: in-place unify each entry of `types`
 * with the corresponding entry of `prev`. No-op when shapes don't
 * match (different arity → different specialization, no widening).
 *
 * Widening that would collapse a known type to `unknown` is rejected
 * — keep this call's concrete type so a fresh, specific spec gets
 * built. Without this, a 1st call with (number, …) followed by a 2nd
 * call with (tensor, …) would unify the first arg to `unknown` and
 * poison every subsequent specialization with the same arg shape.
 */
export function widenAgainst(
  types: JitType[],
  prev: readonly JitType[] | undefined
): void {
  if (!prev || prev.length !== types.length) return;
  for (let i = 0; i < types.length; i++) {
    if (types[i].kind === "unknown" || prev[i].kind === "unknown") continue;
    const widened = unifyJitTypes(types[i], prev[i]);
    if (widened.kind === "unknown") continue;
    types[i] = widened;
  }
}

/**
 * Gather env inputs for the synthetic FunctionDef of a top-level
 * block. For each candidate name: skip known constants, skip names
 * not in env (likely fn names), infer the JIT type, prune `exact`.
 * Returns null if any candidate has an unknown type — that's a
 * structural blocker for lowering.
 */
export function gatherTypedEnvInputs(
  interp: Interpreter,
  candidates: readonly string[]
): { inputs: string[]; inputTypes: JitType[] } | null {
  const inputs: string[] = [];
  const inputTypes: JitType[] = [];
  for (const name of candidates) {
    if (KNOWN_CONSTANTS.has(name)) continue;
    const val = interp.env.get(name);
    if (val === undefined) continue;
    const t = inferJitType(val);
    if (t.kind === "unknown") return null;
    inputs.push(name);
    inputTypes.push(pruneArgType(t));
  }
  return { inputs, inputTypes };
}

// ── Top-level classification ────────────────────────────────────────────

export interface TopLevelClassification {
  readonly stmts: readonly Stmt[];
  readonly inputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly outputs: readonly string[];
  readonly currentFile: string;
  readonly hasReturn: boolean;
  readonly cacheKey: string;
}

export function classifyTopLevel(
  interp: Interpreter,
  stmts: readonly Stmt[],
  prevInputTypes: readonly JitType[] | undefined
): TopLevelClassification | null {
  if (stmts.length === 0) return null;

  const analysis = analyzeTopLevel(stmts as Stmt[]);

  // Candidate input order: referenced names first, then assigned
  // names that also exist in env (pre-script values to preserve).
  const seen = new Set<string>();
  const inputCandidates: string[] = [];
  for (const name of [...analysis.inputs, ...analysis.outputs]) {
    if (seen.has(name)) continue;
    seen.add(name);
    inputCandidates.push(name);
  }

  const gathered = gatherTypedEnvInputs(interp, inputCandidates);
  if (!gathered) return null;
  const { inputs, inputTypes } = gathered;

  // Every assigned name is live-out at top level.
  const outputs = [...new Set(analysis.outputs)];

  widenAgainst(inputTypes, prevInputTypes);

  const typeKey = inputs
    .map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`)
    .join(",");
  const cacheKey = `$top:${interp.currentFile}|${typeKey}`;

  return {
    stmts,
    inputs,
    inputTypes,
    outputs,
    currentFile: interp.currentFile,
    hasReturn: analysis.hasReturn,
    cacheKey,
  };
}

// ── Call classification ─────────────────────────────────────────────────

export interface CallClassification {
  readonly fn: FunctionDef;
  readonly nargout: number;
  readonly argTypes: readonly JitType[];
  readonly cacheKey: string;
  /**
   * Effective parameter names for this specialization. Mirrors
   * `fn.params` for non-varargin functions. For varargin functions,
   * the trailing `varargin` is replaced with one synthetic name per
   * variadic arg (`$va_0`, `$va_1`, …). argTypes is one-to-one with
   * effectiveParams.
   */
  readonly effectiveParams: readonly string[];
  /** Number of variadic args (0 when fn has no varargin). */
  readonly nVarargin: number;
}

export function classifyCall(
  fn: FunctionDef,
  args: unknown[],
  nargout: number,
  prevArgTypes: readonly JitType[] | undefined
): CallClassification | null {
  // `~` placeholder params aren't valid identifiers downstream.
  for (const p of fn.params) {
    if (p === "~") return null;
  }

  const hasVarargin =
    fn.params.length > 0 && fn.params[fn.params.length - 1] === "varargin";
  const regularParamCount = hasVarargin
    ? fn.params.length - 1
    : fn.params.length;

  if (hasVarargin) {
    if (args.length < regularParamCount) return null;
  } else {
    if (args.length !== fn.params.length) return null;
  }

  const argTypes: JitType[] = [];
  for (const arg of args) {
    const t = inferJitType(arg);
    if (t.kind === "unknown") return null;
    argTypes.push(pruneArgType(t));
  }

  widenAgainst(argTypes, prevArgTypes);

  const nVarargin = hasVarargin ? args.length - regularParamCount : 0;
  const effectiveParams: string[] = hasVarargin
    ? [
        ...fn.params.slice(0, regularParamCount),
        ...Array.from({ length: nVarargin }, (_, k) => `$va_${k}`),
      ]
    : fn.params.slice();

  const cacheKey = JSON.stringify({ nargout, argTypes });

  return { fn, nargout, argTypes, cacheKey, effectiveParams, nVarargin };
}

// ── AST analysis (input/output/return) ─────────────────────────────────

interface BlockVarInfo {
  inputs: string[];
  outputs: string[];
  hasReturn: boolean;
}

function analyzeTopLevel(stmts: Stmt[]): BlockVarInfo {
  const assigned = new Set<string>();
  const referenced = new Set<string>();
  const { hasReturn } = walkStmts(stmts, assigned, referenced);
  return {
    inputs: [...referenced],
    outputs: [...assigned],
    hasReturn,
  };
}

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
      referenced.add(expr.name);
      for (const arg of expr.args) walkExpr(arg, referenced);
      break;

    case "Index":
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
    case "Cell":
      for (const row of expr.rows) {
        for (const elem of row) walkExpr(elem, referenced);
      }
      break;

    case "Member":
      walkExpr(expr.base, referenced);
      break;

    case "MemberDynamic":
      walkExpr(expr.base, referenced);
      walkExpr((expr as { nameExpr: Expr }).nameExpr, referenced);
      break;

    case "MethodCall":
      walkExpr(expr.base, referenced);
      for (const arg of expr.args) walkExpr(arg, referenced);
      break;

    case "AnonFunc":
      walkExpr(expr.body, referenced);
      break;

    case "EndKeyword":
    case "Colon":
      break;

    default:
      break;
  }
}
