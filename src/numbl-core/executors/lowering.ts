/**
 * Shared lowering pipeline.
 *
 * The dispatcher calls `tryLower` once per stmt-dispatch (and
 * `tryLowerCall` per function-call dispatch), before any executor is
 * asked to propose. The result — a `LoweredStmt` — is passed to every
 * executor's `propose()` as the first argument.
 *
 * Lowering produces an IR; it does NOT make codegen-feasibility
 * decisions. "Can this be JS-JIT'd?" lives in the JS-JIT executor's
 * propose. Lowering's only "no" is structural: type-unknown inputs,
 * lowerFunction declined.
 *
 * Shapes today:
 *   - `top-level` — script body (top-level scope, first stmt). Whole
 *     script lowered as a synthetic FunctionDef.
 *   - `loop`      — for/while loop stmt. Loop lowered as a synthetic
 *     FunctionDef that wraps just that stmt.
 *   - `call`      — user-function call. Lowered via `tryLowerCall`
 *     from `dispatchCall`.
 *   - `stmt`      — fallback for stmts with no specialized shape.
 *     AST-driven executors filter on this kind.
 *
 * Top-level and loop share the same underlying mechanics
 * (`shared.ts`); they're modeled as separate kinds because they have
 * distinct trigger conditions (script-root vs. control-flow stmt) and
 * runtime semantics (claim entire stmt list vs. consume one stmt).
 *
 * Lowerings are cached by (head Stmt or FunctionDef, classification
 * cacheKey).
 */

import type { Stmt } from "../parser/types.js";
import type { FunctionDef } from "../interpreter/types.js";
import type { Interpreter } from "../interpreter/interpreter.js";
import type { DispatchContext } from "./context.js";
import {
  classifyTopLevel,
  lowerTopLevel,
  type TopLevelClassification,
  type TopLevelLowered,
} from "./jsJit/jitTopLevel.js";
import {
  classifyLoop,
  lowerLoop,
  type LoopClassification,
  type LoopLowered,
} from "./jsJit/jitLoop.js";
import {
  classifyCall,
  lowerCall,
  type CallClassification,
  type CallLowered,
} from "./jsJit/jitCall.js";
import {
  JIT_IO_BUILTINS,
  irHasBailRisk,
  irHasIO,
} from "../jit/jitBailSafety.js";

/** What `propose()` receives. The dispatcher always produces some
 *  variant — there's no `null` LoweredStmt. Executors filter on
 *  `kind`. */
export type LoweredStmt =
  | TopLevelLoweredStmt
  | LoopLoweredStmt
  | CallLoweredStmt
  | RawStmt;

/** Top-level shape: script body lowered to JS-JIT IR. */
export interface TopLevelLoweredStmt {
  readonly kind: "top-level";
  readonly classification: TopLevelClassification;
  readonly lowered: TopLevelLowered;
  readonly flags: TopLevelFlags;
}

/** Pre-computed feasibility flags for top-level codegen executors. */
export interface TopLevelFlags {
  /** Body contains a `return` statement. JIT cannot model
   *  early-return from the synthetic top-level fn. */
  readonly hasReturn: boolean;
  /** Source body contains an unsuppressed assign / multiassign /
   *  non-void-call ExprStmt. In display-mode the JIT must bail —
   *  it has no emit for auto-display. */
  readonly hasUnsuppressedAssign: boolean;
  /** Lowered IR contains an I/O builtin (disp, fprintf, ...). */
  readonly hasIO: boolean;
  /** Lowered IR contains a possibly-bailing operation. Combined
   *  with hasIO, signals a body that mustn't be retried after a
   *  partial run (already-emitted output would duplicate). */
  readonly hasBailRisk: boolean;
}

/** Loop shape: a For/While stmt lowered to JS-JIT IR. */
export interface LoopLoweredStmt {
  readonly kind: "loop";
  readonly classification: LoopClassification;
  readonly lowered: LoopLowered;
  readonly flags: LoopFlags;
}

/** Pre-computed feasibility flags for loop codegen executors. */
export interface LoopFlags {
  readonly hasReturn: boolean;
  readonly hasIO: boolean;
  readonly hasBailRisk: boolean;
}

/** Call shape: a user-function call lowered to JS-JIT IR. Produced
 *  by `tryLowerCall`, not `tryLower` — function calls fire from
 *  expression evaluation, not from the stmt loop. */
export interface CallLoweredStmt {
  readonly kind: "call";
  readonly classification: CallClassification;
  readonly lowered: CallLowered;
  readonly flags: CallFlags;
  /** Runtime arg values. Carried alongside the classification
   *  because the executor needs them at runCall time; unlike
   *  stmt-shape executors, the call executor can't re-fetch them
   *  from env. */
  readonly args: readonly unknown[];
}

/** Pre-computed feasibility flags for call codegen executors. */
export interface CallFlags {
  readonly hasIO: boolean;
  readonly hasBailRisk: boolean;
}

/** Fallback wrapper for stmts that have no specialized lowering
 *  shape. AST-driven executors filter on this kind. */
export interface RawStmt {
  readonly kind: "stmt";
  readonly stmt: Stmt;
}

const BAILED = Symbol("LOWERING_BAILED");
type Bailed = typeof BAILED;

/** Per-owner lowering cache. Owner is either the head Stmt
 *  (stmt-shape lowerings) or the FunctionDef (call-shape lowerings).
 *  WeakMap-scoped so entries are reclaimed when the AST is dropped. */
export class LoweringCache {
  private readonly slots = new WeakMap<
    object,
    Map<string, LoweredStmt | Bailed>
  >();

  get(owner: object, cacheKey: string): LoweredStmt | Bailed | undefined {
    return this.slots.get(owner)?.get(cacheKey);
  }

  set(owner: object, cacheKey: string, value: LoweredStmt): void {
    let perOwner = this.slots.get(owner);
    if (!perOwner) {
      perOwner = new Map();
      this.slots.set(owner, perOwner);
    }
    perOwner.set(cacheKey, value);
  }

  markBailed(owner: object, cacheKey: string): void {
    let perOwner = this.slots.get(owner);
    if (!perOwner) {
      perOwner = new Map();
      this.slots.set(owner, perOwner);
    }
    perOwner.set(cacheKey, BAILED);
  }

  isBailed(value: unknown): value is Bailed {
    return value === BAILED;
  }
}

/**
 * Try to lower the stmt at `siblings[i]` based on current runtime
 * info. Always returns a LoweredStmt — at minimum, the raw stmt
 * wrapped in `{ kind: "stmt", stmt }`.
 */
export function tryLower(
  siblings: readonly Stmt[],
  i: number,
  ctx: DispatchContext,
  cache: LoweringCache
): LoweredStmt {
  const head = siblings[i];

  // Top-level shape: only at the script's root, on the first stmt.
  if (ctx.scope === "top-level" && i === 0) {
    const entry = tryBuildTopLevel(ctx.interp, siblings, head, cache);
    if (entry) return entry;
  }

  // Loop shape: For/While stmts.
  if (head.type === "For" || head.type === "While") {
    const entry = tryBuildLoop(ctx.interp, head, siblings, i, cache);
    if (entry) return entry;
  }

  return { kind: "stmt", stmt: head };
}

function tryBuildTopLevel(
  interp: Interpreter,
  siblings: readonly Stmt[],
  head: Stmt,
  cache: LoweringCache
): TopLevelLoweredStmt | null {
  const classification = classifyTopLevel(interp, siblings);
  if (!classification) return null;

  const hit = cache.get(head, classification.cacheKey);
  if (hit !== undefined) {
    return cache.isBailed(hit) ? null : (hit as TopLevelLoweredStmt);
  }

  const lowered = lowerTopLevel(interp, classification);
  if (!lowered) {
    cache.markBailed(head, classification.cacheKey);
    return null;
  }

  const entry: TopLevelLoweredStmt = {
    kind: "top-level",
    classification,
    lowered,
    flags: computeTopLevelFlags(classification, lowered),
  };
  cache.set(head, classification.cacheKey, entry);
  return entry;
}

function tryBuildLoop(
  interp: Interpreter,
  head: Stmt & { type: "For" | "While" },
  siblings: readonly Stmt[],
  i: number,
  cache: LoweringCache
): LoopLoweredStmt | null {
  const classification = classifyLoop(interp, head, siblings, i);
  if (!classification) return null;

  const hit = cache.get(head, classification.cacheKey);
  if (hit !== undefined) {
    return cache.isBailed(hit) ? null : (hit as LoopLoweredStmt);
  }

  const lowered = lowerLoop(interp, classification);
  if (!lowered) {
    cache.markBailed(head, classification.cacheKey);
    return null;
  }

  const entry: LoopLoweredStmt = {
    kind: "loop",
    classification,
    lowered,
    flags: computeLoopFlags(classification, lowered),
  };
  cache.set(head, classification.cacheKey, entry);
  return entry;
}

function computeTopLevelFlags(
  classification: TopLevelClassification,
  lowered: TopLevelLowered
): TopLevelFlags {
  let hasUnsuppressedAssign = false;
  for (const s of classification.stmts) {
    if (
      (s.type === "Assign" ||
        s.type === "AssignLValue" ||
        s.type === "MultiAssign") &&
      !s.suppressed
    ) {
      hasUnsuppressedAssign = true;
      break;
    }
    if (s.type === "ExprStmt" && !s.suppressed) {
      const e = s.expr;
      const isVoidCall =
        e.type === "FuncCall" &&
        (JIT_IO_BUILTINS.has(e.name) || e.name === "tic");
      if (!isVoidCall) {
        hasUnsuppressedAssign = true;
        break;
      }
    }
  }

  const result = lowered.result;
  return {
    hasReturn: classification.hasReturn,
    hasUnsuppressedAssign,
    hasIO: irHasIO(result.body, result.generatedIRBodies),
    hasBailRisk: irHasBailRisk(result.body, result.generatedIRBodies),
  };
}

function computeLoopFlags(
  classification: LoopClassification,
  lowered: LoopLowered
): LoopFlags {
  const result = lowered.result;
  return {
    hasReturn: classification.hasReturn,
    hasIO: irHasIO(result.body, result.generatedIRBodies),
    hasBailRisk: irHasBailRisk(result.body, result.generatedIRBodies),
  };
}

function computeCallFlags(lowered: CallLowered): CallFlags {
  const result = lowered.result;
  return {
    hasIO: irHasIO(result.body, result.generatedIRBodies),
    hasBailRisk: irHasBailRisk(result.body, result.generatedIRBodies),
  };
}

/**
 * Try to lower a user-function call. Always returns a
 * `CallLoweredStmt` when classification succeeds; null when the
 * cheap classify declines (`~` params, type-unknown args).
 */
export function tryLowerCall(
  fn: FunctionDef,
  args: unknown[],
  nargout: number,
  interp: Interpreter,
  cache: LoweringCache
): CallLoweredStmt | null {
  const classification = classifyCall(fn, args, nargout);
  if (!classification) return null;

  const hit = cache.get(fn, classification.cacheKey);
  if (hit !== undefined) {
    if (cache.isBailed(hit)) return null;
    // Cached entry's `args` is from a prior call; rebind to the
    // current args (values may differ; types are unified via
    // classification.cacheKey).
    return { ...(hit as CallLoweredStmt), args };
  }

  const lowered = lowerCall(interp, classification);
  if (!lowered) {
    cache.markBailed(fn, classification.cacheKey);
    return null;
  }

  const entry: CallLoweredStmt = {
    kind: "call",
    classification,
    lowered,
    flags: computeCallFlags(lowered),
    args,
  };
  cache.set(fn, classification.cacheKey, entry);
  return entry;
}
