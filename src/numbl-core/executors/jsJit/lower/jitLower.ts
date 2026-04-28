/**
 * AST -> JIT IR lowering with type propagation. Orchestrator that owns
 * the shared context types (LowerCtx, SliceAlias), the public result
 * types (GeneratedFn, LoweringResult), and the `lowerFunction` entry
 * point. Actual statement and expression lowering lives in
 * jitLowerStmt.ts and jitLowerExpr.ts; type-level helpers are in
 * jitLowerTypes.ts.
 *
 * Returns null if any unsupported construct is encountered, causing the
 * entire function to fall back to interpretation.
 */

import type { FunctionDef } from "../../../interpreter/types.js";
import type { Interpreter } from "../../../interpreter/interpreter.js";
import type { JitType, JitExpr, JitStmt } from "../../../jitTypes.js";
import { type TypeEnv } from "./jitLowerTypes.js";
import { buildLineTable } from "../../../runtime/error.js";
import { lowerStmts } from "./jitLowerStmt.js";

const LOG_CJIT_MISSES =
  typeof process !== "undefined" && !!process.env.NUMBL_LOG_CJIT_MISSES;

// ── Public result types ─────────────────────────────────────────────────

/**
 * Lowered IR for a user function reached during a top-level lowering.
 *
 * JS-JIT caches the callee as generated JS source in `generatedFns` and
 * is done. C-JIT needs to re-analyze the callee's IR (for recursive
 * feasibility + emitting a static C function per callee), so we also
 * cache the raw IR here. One entry per unique `jitName`; matches the
 * JS-JIT specialization key.
 */
export interface GeneratedFn {
  fn: FunctionDef;
  argTypes: JitType[];
  outputNames: string[];
  outputTypes: JitType[];
  body: JitStmt[];
  localVars: Set<string>;
  nargout: number;
}

export interface LoweringResult {
  body: JitStmt[];
  outputNames: string[];
  localVars: Set<string>;
  hasTensorOps: boolean;
  /** Generated JS code for called user functions: jitName → code */
  generatedFns: Map<string, string>;
  /** Lowered IR for called user functions: jitName → IR + metadata.
   *  Populated alongside `generatedFns` by `lowerUserFuncCall`; the
   *  C-JIT uses it for recursive feasibility and per-callee C emission. */
  generatedIRBodies: Map<string, GeneratedFn>;
  /** Type of the first output variable after lowering */
  outputType: JitType | null;
  /** Types of all output variables in outputNames order. Mirrors the
   *  shape of JS-JIT's `return [out0, out1, ...]`. */
  outputTypes: JitType[];
  /** Final typed environment after lowering the body. The hybrid
   *  loop-extraction pass uses this to look up live-in/live-out var
   *  types without re-running type inference. */
  endEnv: TypeEnv;
}

// ── Internal lowering context (shared with stmt/expr sides) ─────────────

/**
 * A "slice alias" records that a MATLAB local was bound to a colon-slice of
 * a real tensor (e.g. `pt = pts(:, i)`). Rather than materializing the slice
 * as a RuntimeTensor per iteration, we remember the base tensor and a
 * per-dim "template" of indices — scalar expressions captured at bind time
 * for the non-colon dims, and `"colon"` placeholders for the colon dims.
 * Subsequent reads like `pt(k)` substitute `k` into the colon positions and
 * emit a direct scalar read on the base tensor, which compiles cleanly
 * through the existing hoisted `idx{1,2,3}r_h` fast path.
 */
export interface SliceAlias {
  baseName: string;
  baseType: JitType;
  /**
   * One entry per index of the original bind expression, in source order.
   * A `"colon"` slot expects to be filled by the read-site's colon indices.
   * An `"expr"` slot carries a JitExpr that will be substituted as-is.
   */
  template: ({ kind: "colon" } | { kind: "expr"; expr: JitExpr })[];
  /** Sizes of the slice's colon dimensions, in source order. */
  sliceShape: number[];
  /** Indices into `template` where colon slots live, in source order. */
  colonPositions: number[];
}

export interface LowerCtx {
  env: TypeEnv;
  localVars: Set<string>;
  params: Set<string>;
  /** First bail reason encountered during lowering; lets callers surface
   *  a hint for why JS-JIT declined a function (purely diagnostic, only
   *  consulted by the `NUMBL_LOG_CJIT_MISSES` tally). Writes should use
   *  `setBailReason()` which is idempotent — only the first reason sticks. */
  bailReason?: string;
  bailLine?: number;
  /** Type of the deepest expr whose lowering was attempted — used as a
   *  last-resort bail reason hint when no more specific reason was set. */
  lastExprType?: string;
  lastExprLine?: number;
  /** Variables that are actually assigned in the function body. */
  assignedVars: Set<string>;
  /**
   * Map from a MATLAB local name to its slice alias, if any. A name is
   * present here iff the most recent assignment to it was a whole-tensor
   * colon slice. Reads of the name as a plain Ident bail; reads of
   * `name(...)` substitute through the template and emit a direct scalar
   * read of the base tensor. See `tryLowerAsSliceBind`.
   */
  sliceAliases: Map<string, SliceAlias>;
  _hasTensorOps?: boolean;
  interp?: Interpreter;
  /** The nargout this specialization was requested with — inlined wherever
   *  the body reads the `nargout` identifier, since the JIT specializes per
   *  nargout already. */
  nargout?: number;
  generatedFns: Map<string, string>;
  generatedIRBodies: Map<string, GeneratedFn>;
  loweringInProgress: Set<string>;
  /** Pre-built line break table for offset→line lookup. */
  lineTable?: number[];
}

/** Idempotent bail-reason setter. Only the first reason per function sticks. */
export function setBailReason(
  ctx: LowerCtx,
  reason: string,
  line?: number
): void {
  if (!ctx.bailReason && LOG_CJIT_MISSES) {
    ctx.bailReason = reason;
    ctx.bailLine = line;
  }
}

// ── Lowering entry point ────────────────────────────────────────────────

export function lowerFunction(
  fn: FunctionDef,
  argTypes: JitType[],
  nargout: number,
  interp?: Interpreter,
  generatedFns?: Map<string, string>,
  loweringInProgress?: Set<string>,
  generatedIRBodies?: Map<string, GeneratedFn>
): LoweringResult | null {
  if (argTypes.length !== fn.params.length) return null;

  const env: TypeEnv = new Map();
  const localVars = new Set<string>();

  // Initialize parameters
  for (let i = 0; i < fn.params.length; i++) {
    env.set(fn.params[i], argTypes[i]);
  }

  // Reserve output variables as locals without assigning a default type.
  // An earlier incarnation seeded env[name] = number=0 so every output had
  // *some* type at function exit, but that poisoned the loop-join merge
  // whenever the body's first assignment produced a non-number type
  // (e.g. `chld = T.nodes(i).chld` makes chld a tensor, which can't be
  // unified with the default number=0 that was already in envBefore).
  // Instead we rely on the "bail if output never assigned" check below
  // to catch outputs that the body doesn't initialize — which is the
  // correct failure mode, since reading an unassigned output is an error.
  const outputNames = fn.outputs.slice(0, nargout || 1);
  for (const name of outputNames) {
    if (!env.has(name)) {
      localVars.add(name);
    }
  }

  const sharedGeneratedFns = generatedFns ?? new Map<string, string>();
  const sharedInProgress = loweringInProgress ?? new Set<string>();
  const sharedGeneratedIRBodies =
    generatedIRBodies ?? new Map<string, GeneratedFn>();

  // Build line table for offset→line lookup from file sources
  let lineTable: number[] | undefined;
  if (interp && fn.body.length > 0 && fn.body[0].span) {
    const file = fn.body[0].span.file;
    lineTable = interp.lineTableCache.get(file);
    if (!lineTable) {
      const src = interp.fileSources.get(file) ?? "";
      lineTable = buildLineTable(src);
      interp.lineTableCache.set(file, lineTable);
    }
  }

  const ctx: LowerCtx = {
    env,
    localVars,
    params: new Set(fn.params),
    assignedVars: new Set(fn.params),
    sliceAliases: new Map(),
    interp,
    nargout,
    generatedFns: sharedGeneratedFns,
    generatedIRBodies: sharedGeneratedIRBodies,
    loweringInProgress: sharedInProgress,
    lineTable,
  };
  const body = lowerStmts(ctx, fn.body);
  if (!body) {
    if (LOG_CJIT_MISSES) {
      const reason =
        ctx.bailReason ??
        (ctx.lastExprType
          ? `fallthrough on ${ctx.lastExprType}`
          : "lowering returned null");
      const line = ctx.bailLine ?? ctx.lastExprLine;
      (fn as { _lastLowerBailReason?: string })._lastLowerBailReason = line
        ? `${reason} @L${line}`
        : reason;
    }
    return null;
  }

  // Bail if any required output variable was never assigned in the body.
  // The interpreter throws a RuntimeError for this case; the JIT would
  // silently return undefined/0 without this check.
  for (const name of outputNames) {
    if (!ctx.params.has(name) && !ctx.assignedVars.has(name)) return null;
  }

  const outputTypes: JitType[] = outputNames.map(
    n => ctx.env.get(n) ?? { kind: "unknown" as const }
  );
  const outputType = outputTypes.length > 0 ? outputTypes[0] : null;

  return {
    body,
    outputNames,
    localVars: ctx.localVars,
    hasTensorOps: ctx._hasTensorOps ?? false,
    generatedFns: sharedGeneratedFns,
    generatedIRBodies: sharedGeneratedIRBodies,
    outputType,
    outputTypes,
    endEnv: ctx.env,
  };
}
