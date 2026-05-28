/**
 * Shared lowering pipeline.
 *
 * The dispatcher calls `tryLower` once per stmt-dispatch and
 * `tryLowerCall` per function-call dispatch, before any executor is
 * asked to propose. The result — a `LoweredStmt` — is passed to every
 * executor's `propose()` as the first argument. `tryLower` returns
 * `null` for stmts with no specialized lowering shape; the dispatcher
 * falls through to its hardcoded `interp.execStmt` path in that case.
 *
 * Lowering here is *classification only*: it analyses inputs/outputs,
 * infers types, and synthesizes a cacheKey. It does NOT make
 * codegen-feasibility decisions — those live in the codegen executor's
 * `propose`. This keeps the dispatcher free of dependencies on any
 * specific JIT backend (mtoc2 today; the legacy JS-JIT / C-JIT
 * backends have been retired).
 *
 * Shapes today:
 *   - `top-level` — script body (top-level scope). Whole script
 *     analysed as a synthetic FunctionDef.
 *   - `call`      — user-function call. Produced by `tryLowerCall`
 *     from `dispatchCall`.
 *   - `synth`     — a `Synth` AST stmt produced by a registered AST
 *     transformer (no further analysis here).
 *
 * Lowerings are cached by (head Stmt or FunctionDef, classification
 * cacheKey).
 */

import type { Stmt } from "../parser/types.js";
import type { FunctionDef } from "../interpreter/types.js";
import type { Interpreter } from "../interpreter/interpreter.js";
import {
  classifyTopLevel,
  classifyCall,
  classifyLoop,
  type TopLevelClassification,
  type CallClassification,
  type LoopClassification,
} from "./classification.js";

/** What `propose()` receives — a discriminated union of the
 *  specialized shapes the dispatcher knows how to lower. Stmts with
 *  no specialized shape don't reach `propose()` at all; the dispatcher
 *  falls through to its hardcoded interpreter path. */
export type LoweredStmt =
  | TopLevelLoweredStmt
  | LoopLoweredStmt
  | CallLoweredStmt
  | SynthLoweredStmt;

/** Top-level shape: script body analysed for whole-scope codegen. */
export interface TopLevelLoweredStmt {
  readonly kind: "top-level";
  readonly classification: TopLevelClassification;
}

/** Loop shape: a single For/While stmt analysed for loop codegen.
 *  Produced by `tryLower` when the head stmt is a For/While. */
export interface LoopLoweredStmt {
  readonly kind: "loop";
  readonly classification: LoopClassification;
}

/** Call shape: a user-function call analysed for call codegen.
 *  Produced by `tryLowerCall`, not `tryLower` — function calls fire
 *  from expression evaluation, not from the stmt loop. */
export interface CallLoweredStmt {
  readonly kind: "call";
  readonly classification: CallClassification;
  /** Runtime arg values. Carried alongside the classification
   *  because the executor needs them at runCall time; unlike
   *  stmt-shape executors, the call executor can't re-fetch them
   *  from env. */
  readonly args: readonly unknown[];
}

/** Synth shape: a `Synth` AST stmt produced by a registered AST
 *  transformer. The matching executor reads `data` (analysis the
 *  transformer pre-computed) and the `tag` discriminates among
 *  multiple registered transformers. */
export interface SynthLoweredStmt {
  readonly kind: "synth";
  readonly tag: string;
  readonly data: unknown;
}

const BAILED = Symbol("LOWERING_BAILED");
type Bailed = typeof BAILED;

/** Per-owner lowering cache. Owner is either the head Stmt
 *  (stmt-shape lowerings) or the FunctionDef (call-shape lowerings).
 *  WeakMap-scoped so entries are reclaimed when the AST is dropped.
 *
 *  Also tracks per-owner type-widening state: the most recent input
 *  type signature seen for a given (owner, slot). Classify phases
 *  consult this so a callee invoked with shifting input types
 *  converges to a single specialization rather than thrashing the
 *  cache. The slot string lets one owner host multiple widening
 *  trackers (e.g. one per nargout for call-shape). */
export class LoweringCache {
  private readonly slots = new WeakMap<
    object,
    Map<string, LoweredStmt | Bailed>
  >();
  private readonly widening = new WeakMap<
    object,
    Map<string, import("../jitTypes.js").JitType[]>
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

  /** Most recent input-type signature recorded for (owner, slot), or
   *  undefined when nothing has been recorded yet. */
  getLastInputTypes(
    owner: object,
    slot: string
  ): import("../jitTypes.js").JitType[] | undefined {
    return this.widening.get(owner)?.get(slot);
  }

  /** Record the latest unified input-type signature for (owner, slot). */
  setLastInputTypes(
    owner: object,
    slot: string,
    types: readonly import("../jitTypes.js").JitType[]
  ): void {
    let perOwner = this.widening.get(owner);
    if (!perOwner) {
      perOwner = new Map();
      this.widening.set(owner, perOwner);
    }
    perOwner.set(slot, [...types]);
  }
}

/**
 * Try to lower the stmt at `siblings[i]`. Returns a `LoweredStmt` for
 * stmts that match a specialized shape, or `null` for stmts with no
 * shape — the dispatcher falls through to its hardcoded interpreter
 * path in that case.
 *
 * Whole-scope shapes (`top-level`) are NOT produced here — they're
 * lowered separately via `tryLowerTopLevel` and dispatched through
 * `Registry.tryRunWholeScope` before the per-stmt loop runs.
 */
export function tryLower(
  siblings: readonly Stmt[],
  i: number,
  interp: Interpreter,
  cache: LoweringCache
): LoweredStmt | null {
  const head = siblings[i];

  // Synth shape: a stmt that a registered AST transformer wrapped
  // up for a specialized executor. No further lowering needed —
  // the analysis was done at transform time.
  if (head.type === "Synth") {
    return { kind: "synth", tag: head.tag, data: head.data };
  }

  // Loop shape: a single For/While stmt. Classification needs the
  // post-loop tail of the sibling list (to filter outputs) and the
  // interpreter env (for input type inference), so it can't be done
  // by an executor's `propose()` in isolation.
  if (head.type === "For" || head.type === "While") {
    return tryBuildLoop(interp, head, siblings, i, cache);
  }

  return null;
}

function tryBuildLoop(
  interp: Interpreter,
  head: Stmt & { type: "For" | "While" },
  siblings: readonly Stmt[],
  i: number,
  cache: LoweringCache
): LoopLoweredStmt | null {
  const prev = cache.getLastInputTypes(head, "");
  const classification = classifyLoop(interp, head, siblings, i, prev);
  if (!classification) return null;
  cache.setLastInputTypes(head, "", classification.inputTypes);

  const hit = cache.get(head, classification.cacheKey);
  if (hit !== undefined) {
    return cache.isBailed(hit) ? null : (hit as LoopLoweredStmt);
  }

  const entry: LoopLoweredStmt = { kind: "loop", classification };
  cache.set(head, classification.cacheKey, entry);
  return entry;
}

/**
 * Lower a script body as a whole-scope unit. Called by the registry
 * before the per-stmt dispatch loop runs; returns a TopLevelLoweredStmt
 * for whole-scope executors to consider, or null when the classification
 * declines.
 */
export function tryLowerTopLevel(
  interp: Interpreter,
  siblings: readonly Stmt[],
  cache: LoweringCache
): TopLevelLoweredStmt | null {
  if (siblings.length === 0) return null;
  const head = siblings[0];

  const prev = cache.getLastInputTypes(head, "");
  const classification = classifyTopLevel(interp, siblings, prev);
  if (!classification) return null;
  cache.setLastInputTypes(head, "", classification.inputTypes);

  const hit = cache.get(head, classification.cacheKey);
  if (hit !== undefined) {
    return cache.isBailed(hit) ? null : (hit as TopLevelLoweredStmt);
  }

  const entry: TopLevelLoweredStmt = { kind: "top-level", classification };
  cache.set(head, classification.cacheKey, entry);
  return entry;
}

/**
 * Try to lower a user-function call. Always returns a
 * `CallLoweredStmt` when classification succeeds; null when the
 * classify declines (`~` params, type-unknown args, varargin
 * arity mismatch).
 */
export function tryLowerCall(
  fn: FunctionDef,
  args: unknown[],
  nargout: number,
  cache: LoweringCache
): CallLoweredStmt | null {
  const slot = String(nargout);
  const prev = cache.getLastInputTypes(fn, slot);
  const classification = classifyCall(fn, args, nargout, prev);
  if (!classification) return null;
  cache.setLastInputTypes(fn, slot, classification.argTypes);

  const hit = cache.get(fn, classification.cacheKey);
  if (hit !== undefined) {
    if (cache.isBailed(hit)) return null;
    // Cached entry's `args` is from a prior call; rebind to the
    // current args (values may differ; types are unified via
    // classification.cacheKey).
    return { ...(hit as CallLoweredStmt), args };
  }

  const entry: CallLoweredStmt = { kind: "call", classification, args };
  cache.set(fn, classification.cacheKey, entry);
  return entry;
}
