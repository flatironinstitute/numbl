/**
 * Classification (JS-JIT-independent).
 *
 * Two classification phases — top-level (whole-script) and call
 * (user-function-call) — produce the analysis records that codegen
 * executors consume to decide whether they can lower a stmt list /
 * call and to key their caches.
 *
 * The analysis is purely AST + type inference and is consumed by the
 * JIT executors (`executors/jit/`) — no backend-specific IR involved.
 */

import type { Interpreter } from "../interpreter/interpreter.js";
import type { FunctionDef } from "../interpreter/types.js";
import type { Stmt, Expr } from "../parser/types.js";
import { jitTypeKey, unifyJitTypes, type JitType } from "../jitTypes.js";
import { inferJitType } from "../interpreter/builtins/types.js";
import { inlinableHandleExpr } from "./handleInline.js";

/** A capture-free function handle the JIT inlines into the spec body as
 *  an in-scope `@...` constant instead of taking it as a runtime input.
 *  `identity` keys the spec cache (distinct handles → distinct specs). */
export interface ConstHandle {
  readonly name: string;
  readonly expr: Expr;
  readonly identity: string;
}

/** Stable identity for a handle's defining AST, for cache keying. */
function handleIdentity(expr: Expr): string {
  if (expr.type === "FuncHandle") return `@${expr.name}`;
  // Anonymous: the source span uniquely identifies this `@(...)` site.
  return `anon@${expr.span?.start ?? 0}`;
}

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
  /** Body contains a `%!numbl:assert_jit c` directive (the C-JIT
   *  variant) — see `containsAssertJitC`. Makes the JS-JIT executors
   *  decline at `--opt 2` so the unit must C-JIT (or fall to the
   *  interpreter, which raises). */
  readonly assertsCJit: boolean;
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
    assertsCJit: containsAssertJitC(stmts),
    cacheKey,
  };
}

// ── Loop classification ─────────────────────────────────────────────────

export interface LoopClassification {
  /** The single For/While stmt this classification describes. */
  readonly stmt: Stmt & { type: "For" | "While" };
  readonly inputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  /** Loop-local writes that are read by code AFTER the loop in the
   *  same sibling list. Loop-internal-only temporaries are filtered
   *  out so a JIT artifact doesn't need to write them back to env. */
  readonly outputs: readonly string[];
  readonly currentFile: string;
  readonly hasReturn: boolean;
  /** Loop body contains a `%!numbl:assert_jit c` directive (C-JIT variant). */
  readonly assertsCJit: boolean;
  /** Capture-free handle inputs to inline as in-scope `@...` constants
   *  in the synthesized spec body (instead of runtime inputs). */
  readonly constHandles: readonly ConstHandle[];
  readonly cacheKey: string;
}

/**
 * Classify a single For/While loop at `siblings[siblingIndex]`. The
 * post-loop tail of the sibling list is scanned to filter the
 * loop's assigned-set down to names that are actually live-out
 * (read after the loop) — purely loop-internal scratch never makes
 * it into the synthetic function's outputs.
 */
export function classifyLoop(
  interp: Interpreter,
  stmt: Stmt & { type: "For" | "While" },
  siblings: readonly Stmt[],
  siblingIndex: number,
  prevInputTypes: readonly JitType[] | undefined
): LoopClassification | null {
  const analysis = analyzeLoop(stmt);

  // Candidate input order: referenced names first, then assigned
  // names that also exist in env (so the loop body's first iter sees
  // the pre-loop value if it reads-then-writes).
  const seen = new Set<string>();
  const inputCandidates: string[] = [];
  for (const name of [...analysis.inputs, ...analysis.outputs]) {
    if (seen.has(name)) continue;
    seen.add(name);
    inputCandidates.push(name);
  }

  const gathered = gatherTypedEnvInputs(interp, inputCandidates);
  if (!gathered) return null;

  // Partition out capture-free handle inputs: instead of taking them as
  // runtime inputs (which the JIT can't type — `function_handle` maps to
  // null), inline their `@...` definition as an in-scope constant in the
  // spec body. A handle reassigned in the loop (`analysis.outputs`) is
  // not constant, so it stays a regular (declining) input.
  const rawOutputs = new Set(analysis.outputs);
  const constHandles: ConstHandle[] = [];
  const inputs: string[] = [];
  const inputTypes: JitType[] = [];
  for (let i = 0; i < gathered.inputs.length; i++) {
    const name = gathered.inputs[i];
    if (
      gathered.inputTypes[i].kind === "function_handle" &&
      !rawOutputs.has(name)
    ) {
      const expr = inlinableHandleExpr(
        interp.env.get(name),
        interp.currentFile
      );
      if (expr) {
        constHandles.push({ name, expr, identity: handleIdentity(expr) });
        continue;
      }
    }
    inputs.push(name);
    inputTypes.push(gathered.inputTypes[i]);
  }

  // Live-out set: inputs (every input the loop touched must round-trip
  // to env so a post-loop reference — direct or via `disp(...)` etc. —
  // sees the loop's final value), plus the For-loop variable, plus any
  // name read in the post-loop tail of this sibling list. Loop-internal
  // scratch (assigned but never read post-loop, and not in env) drops
  // out of the writeback set so the JIT artifact doesn't pay a Map.set.
  // When the loop is nested inside an enclosing block (another loop, or an
  // if/switch/try), `siblings` is that block's body — the post-loop tail
  // scan can't see reads after the enclosing block, so a name read there
  // would be wrongly dropped. In MATLAB every loop-assigned name persists
  // in the function scope, so in that case keep them all live-out. Only at
  // the outermost (function/script) level is the scan complete enough to
  // safely prune dead scratch.
  const nested = interp.loopDepth > 0 || interp.condBlockDepth > 0;
  let outputs: string[];
  if (nested) {
    outputs = [...new Set(analysis.outputs)];
  } else {
    const liveOut = new Set<string>(inputs);
    if (stmt.type === "For") liveOut.add(stmt.varName);
    collectReadsFromSiblings(siblings as Stmt[], siblingIndex + 1, liveOut);
    outputs = [...new Set(analysis.outputs)].filter(n => liveOut.has(n));
  }

  widenAgainst(inputTypes, prevInputTypes);

  const typeKey = inputs
    .map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`)
    .join(",");
  const outputKey = outputs.join(",");
  const lineLabel = stmt.span?.start ?? 0;
  // Const-handle identities salt the key: a different handle bound to the
  // same variable (or a switch from inlinable to not) must recompile.
  const handleKey = constHandles.map(h => `${h.name}=${h.identity}`).join(",");
  const cacheKey = `$loop:${interp.currentFile}@${lineLabel}|${typeKey}|out=${outputKey}|h=${handleKey}`;

  return {
    stmt,
    inputs,
    inputTypes,
    outputs,
    currentFile: interp.currentFile,
    hasReturn: analysis.hasReturn,
    assertsCJit: containsAssertJitC(stmt.body),
    constHandles,
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
  /** Function body contains a `%!numbl:assert_jit c` directive (C-JIT variant). */
  readonly assertsCJit: boolean;
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

  return {
    fn,
    nargout,
    argTypes,
    cacheKey,
    effectiveParams,
    nVarargin,
    assertsCJit: containsAssertJitC(fn.body),
  };
}

// ── assert_jit detection ────────────────────────────────────────────────

/** True if any statement in `stmts` is a `%!numbl:assert_jit c` directive
 *  (the C-JIT variant), recursing into control-flow bodies (if / for /
 *  while / switch / try) but NOT into nested function definitions — a
 *  directive inside a nested function belongs to that function's own unit.
 *
 *  Only the `c` variant matters here: it requires C-JIT at `--opt 2`, so
 *  the JS-JIT executors decline such a unit at `--opt 2` to force either
 *  C-JIT or an interpreter fallthrough (which then raises). The plain
 *  `%!numbl:assert_jit` (require JS-JIT at `--opt 1` only) needs no
 *  executor change — the interpreter's `Directive` handler raises if it
 *  reaches the directive at `--opt 1`. */
export function containsAssertJitC(stmts: readonly Stmt[]): boolean {
  for (const s of stmts) {
    switch (s.type) {
      case "Directive":
        if (s.directive === "assert_jit" && s.args.includes("c")) return true;
        break;
      case "If":
        if (containsAssertJitC(s.thenBody)) return true;
        for (const eib of s.elseifBlocks)
          if (containsAssertJitC(eib.body)) return true;
        if (s.elseBody && containsAssertJitC(s.elseBody)) return true;
        break;
      case "For":
      case "While":
        if (containsAssertJitC(s.body)) return true;
        break;
      case "Switch":
        for (const c of s.cases) if (containsAssertJitC(c.body)) return true;
        if (s.otherwise && containsAssertJitC(s.otherwise)) return true;
        break;
      case "TryCatch":
        if (containsAssertJitC(s.tryBody)) return true;
        if (containsAssertJitC(s.catchBody)) return true;
        break;
    }
  }
  return false;
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

/** Analyze a single For/While loop stmt — collects read/written
 *  variable names by walking the loop's header expr/cond + body. */
function analyzeLoop(stmt: Stmt & { type: "For" | "While" }): BlockVarInfo {
  const assigned = new Set<string>();
  const referenced = new Set<string>();
  const hasReturn = walkStmt(stmt, assigned, referenced);
  return {
    inputs: [...referenced],
    outputs: [...assigned],
    hasReturn,
  };
}

/** Mark every name read in the tail of a sibling list (from
 *  `startIdx` onward). Used by the loop classifier to filter the
 *  writeback set down to names that are live after the loop. The
 *  walker conflates lvalue bases with reads (which is fine here —
 *  any name appearing textually is conservatively "live"). */
function collectReadsFromSiblings(
  stmts: Stmt[],
  startIdx: number,
  out: Set<string>
): void {
  const sink = new Set<string>();
  for (let i = startIdx; i < stmts.length; i++) {
    walkStmt(stmts[i], sink, out);
  }
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
