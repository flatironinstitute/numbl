/**
 * Shared lowering pipeline.
 *
 * The registry lowers each unit once, before any executor is asked to
 * propose, and passes the resulting `LoweredStmt` to every executor
 * registered for that shape:
 *
 *   - `top-level` — a script body (`tryLowerTopLevel`), analysed as a
 *     synthetic FunctionDef.
 *   - `loop`      — a For/While stmt (`tryLowerLoop`).
 *   - `call`      — a user-function call (`tryLowerCall`).
 *
 * Lowering here is *classification only*: it analyses inputs/outputs,
 * infers types, and synthesizes a cacheKey. Codegen-feasibility decisions
 * live in each executor's `propose`, which keeps this layer free of
 * dependencies on the JIT compiler (`src/numbl-core/jit`).
 */

import type { Stmt } from "../parser/types.js";
import type { FunctionDef } from "../interpreter/types.js";
import type { Interpreter } from "../interpreter/interpreter.js";
import type { JitType } from "../jitTypes.js";
import {
  classifyTopLevel,
  classifyCall,
  classifyLoop,
  type TopLevelClassification,
  type CallClassification,
  type LoopClassification,
} from "./classification.js";

/** What `propose()` receives — a discriminated union of the shapes the
 *  registry knows how to lower. */
export type LoweredStmt =
  | TopLevelLoweredStmt
  | LoopLoweredStmt
  | CallLoweredStmt;

export interface TopLevelLoweredStmt {
  readonly kind: "top-level";
  readonly classification: TopLevelClassification;
}

export interface LoopLoweredStmt {
  readonly kind: "loop";
  readonly classification: LoopClassification;
}

export interface CallLoweredStmt {
  readonly kind: "call";
  readonly classification: CallClassification;
  /** Runtime arg values. Unlike the env-reading shapes, a call executor
   *  cannot re-fetch its inputs from the env. */
  readonly args: readonly unknown[];
}

/** Per-owner type-widening state: the most recent input type signature
 *  seen for (owner, slot), where owner is the head Stmt (top-level, loop)
 *  or the FunctionDef (call). Classification widens against it so a unit
 *  invoked with shifting input types converges to a single specialization
 *  rather than thrashing the compile cache. The slot lets one owner host
 *  several trackers (one per nargout for calls). WeakMap-scoped so entries
 *  are reclaimed when the AST is dropped. */
export class WideningState {
  private readonly last = new WeakMap<object, Map<string, JitType[]>>();

  get(owner: object, slot: string): JitType[] | undefined {
    return this.last.get(owner)?.get(slot);
  }

  set(owner: object, slot: string, types: readonly JitType[]): void {
    let perOwner = this.last.get(owner);
    if (!perOwner) {
      perOwner = new Map();
      this.last.set(owner, perOwner);
    }
    perOwner.set(slot, [...types]);
  }
}

/** Lower a For/While stmt at `siblings[i]`. Classification needs the
 *  post-loop tail of the sibling list (to filter outputs) and the env
 *  (for input types). Null when classification declines. */
export function tryLowerLoop(
  interp: Interpreter,
  siblings: readonly Stmt[],
  i: number,
  widening: WideningState
): LoopLoweredStmt | null {
  const head = siblings[i] as Stmt & { type: "For" | "While" };
  const prev = widening.get(head, "");
  const classification = classifyLoop(interp, head, siblings, i, prev);
  if (!classification) return null;
  widening.set(head, "", classification.inputTypes);
  return { kind: "loop", classification };
}

/** Lower a script body as a whole-scope unit. Null when classification
 *  declines. */
export function tryLowerTopLevel(
  interp: Interpreter,
  siblings: readonly Stmt[],
  widening: WideningState
): TopLevelLoweredStmt | null {
  if (siblings.length === 0) return null;
  const head = siblings[0];
  const prev = widening.get(head, "");
  const classification = classifyTopLevel(interp, siblings, prev);
  if (!classification) return null;
  widening.set(head, "", classification.inputTypes);
  return { kind: "top-level", classification };
}

/** Lower a user-function call. Null when classification declines (`~`
 *  params, type-unknown args, varargin arity mismatch). */
export function tryLowerCall(
  fn: FunctionDef,
  args: unknown[],
  nargout: number,
  widening: WideningState
): CallLoweredStmt | null {
  const slot = String(nargout);
  const prev = widening.get(fn, slot);
  const classification = classifyCall(fn, args, nargout, prev);
  if (!classification) return null;
  widening.set(fn, slot, classification.argTypes);
  return { kind: "call", classification, args };
}
