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
 * types them in the JIT compiler's `Type` lattice, and synthesizes a
 * cacheKey. Codegen-feasibility decisions live in each executor's
 * `propose`.
 */

import type { Stmt } from "../parser/types.js";
import type { FunctionDef } from "../interpreter/types.js";
import type { Interpreter } from "../interpreter/interpreter.js";
import type { Type } from "../jit/index.js";
import {
  classifyTopLevel,
  classifyCall,
  classifyLoop,
  type TopLevelClassification,
  classifyStmt,
  type CallClassification,
  type LoopClassification,
  type StmtClassification,
} from "./classification.js";
import {
  stmtCandidate,
  stmtDeclined,
  type StmtCandidate,
} from "./stmtPrefilter.js";

/** What `propose()` receives — a discriminated union of the shapes the
 *  registry knows how to lower. */
export type LoweredStmt =
  | TopLevelLoweredStmt
  | LoopLoweredStmt
  | CallLoweredStmt
  | StmtLoweredStmt;

export interface TopLevelLoweredStmt {
  readonly kind: "top-level";
  readonly classification: TopLevelClassification;
}

export interface LoopLoweredStmt {
  readonly kind: "loop";
  readonly classification: LoopClassification;
}

export interface StmtLoweredStmt {
  readonly kind: "stmt";
  readonly classification: StmtClassification;
  /** The statement's prefilter record, for the back-off (`stmtDeclined`). */
  readonly candidate: StmtCandidate;
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
  private readonly last = new WeakMap<object, Map<string, Type[]>>();

  get(owner: object, slot: string): Type[] | undefined {
    return this.last.get(owner)?.get(slot);
  }

  set(owner: object, slot: string, types: readonly Type[]): void {
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

/** A statement is offered to the `stmt` shape from its `STMT_MIN_RUNS`th
 *  dispatch on, and only when one of its tensor inputs has at least
 *  `STMT_MIN_NUMEL` elements: compiling a statement, and the first runs of
 *  its freshly compiled (cold) code, cost more than one fused run saves
 *  (see `stmtExecutor.ts`). */
export const STMT_MIN_RUNS = 3;
export const STMT_MIN_NUMEL = 1 << 16;

/** Lower an interpreted statement for the `stmt` shape. Null when it is
 *  not a candidate (`stmtCandidate`, cached per AST node), is not yet hot,
 *  or has no large tensor input. */
export function tryLowerStmt(
  interp: Interpreter,
  stmt: Stmt,
  widening: WideningState
): StmtLoweredStmt | null {
  const cand = stmtCandidate(stmt);
  if (cand === null) return null;
  if (cand.skip > 0) {
    cand.skip--;
    return null;
  }
  if (++cand.runs < STMT_MIN_RUNS) return null;
  const head = stmt as Stmt & { type: "Assign" };
  const prev = widening.get(head, "");
  const classification = classifyStmt(
    interp,
    head,
    cand.reads,
    cand.calls,
    STMT_MIN_NUMEL,
    prev
  );
  if (!classification) {
    stmtDeclined(cand);
    return null;
  }
  widening.set(head, "", classification.inputTypes);
  return { kind: "stmt", classification, candidate: cand };
}

/** Lower a user-function call. Null when classification declines (`~`
 *  params, varargin arity mismatch). */
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
