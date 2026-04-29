/**
 * Dispatch context — passed to every executor call.
 *
 * Provides env access, type-info queries, sub-dispatch into the
 * registry, and runtime callbacks. Memoizes type-info lookups for the
 * lifetime of a single dispatch so repeated `typeOf(name)` calls from
 * different executors don't re-infer.
 */

import type { Interpreter } from "../interpreter/interpreter.js";
import type { Stmt } from "../parser/types.js";
import type { ControlSignal } from "../interpreter/types.js";
import type { Registry } from "./registry.js";
import { type JitType } from "../jitTypes.js";
import { inferJitType } from "../interpreter/builtins/types.js";

export class DispatchContext {
  readonly interp: Interpreter;
  readonly registry: Registry;
  /** When true, only no-bail executors are eligible. Set when the
   *  caller's compiled artifact has emitted observable side effects
   *  that mustn't repeat. */
  readonly requireNoBail: boolean;

  /** Per-dispatch memoization of typeOf(name) lookups. */
  private readonly typeCache = new Map<string, JitType>();

  /** Re-entrancy guard: which executors are currently running on
   *  which head stmts. Prevents an executor sub-dispatching back into
   *  itself on the same stmt. */
  private readonly active: Set<string>;

  /** Sibling list and head index for the current dispatch. Set by
   *  Registry.dispatch before calling executors; read by peekSibling.
   *  Most executors don't read these directly. */
  private _siblings: readonly Stmt[] = [];
  private _i: number = 0;

  constructor(
    interp: Interpreter,
    registry: Registry,
    requireNoBail: boolean,
    active?: Set<string>
  ) {
    this.interp = interp;
    this.registry = registry;
    this.requireNoBail = requireNoBail;
    this.active = active ?? new Set();
  }

  /** Look up a name's JitType from the interpreter env, memoized. */
  typeOf(name: string): JitType {
    const cached = this.typeCache.get(name);
    if (cached) return cached;
    const v = this.interp.env.get(name);
    const t = v === undefined ? { kind: "unknown" as const } : inferJitType(v);
    this.typeCache.set(name, t);
    return t;
  }

  /** Read a value from env (no inference). Returns undefined if unbound. */
  envValue(name: string): unknown {
    return this.interp.env.get(name);
  }

  /** Peek at a sibling stmt at `offset` from the current head.
   *  offset=0 is the current stmt; positive values look ahead.
   *  Returns null if the offset is past the end of the scope. Used by
   *  chain-style executors that may consume multiple consecutive
   *  stmts. */
  peekSibling(offset: number): Stmt | null {
    const k = this._i + offset;
    if (k < 0 || k >= this._siblings.length) return null;
    return this._siblings[k];
  }

  /** Sibling list of the current scope. The head stmt is at
   *  `siblings[headIndex]`. Most executors don't need this directly —
   *  use `peekSibling(offset)` for lookahead. The rare cases that
   *  forward the raw list to a legacy adapter (the chain executor) or
   *  claim the whole scope (the top-level executor) read these. */
  get siblings(): readonly Stmt[] {
    return this._siblings;
  }
  get headIndex(): number {
    return this._i;
  }

  /** @internal Set the position state for an upcoming dispatch. Only
   *  the registry calls this. */
  _setPosition(siblings: readonly Stmt[], i: number): void {
    this._siblings = siblings;
    this._i = i;
  }

  /** Reset per-dispatch state so this context can be reused for the
   *  next stmt in a sibling loop. The reentrancy guard is expected to
   *  already be empty (every pushActive is paired with a popActive in
   *  Registry.runCandidate's finally). */
  resetForNextDispatch(): void {
    if (this.typeCache.size > 0) this.typeCache.clear();
  }

  /** Construct a child context for sub-dispatch. `requireNoBail` is
   *  the OR of the parent flag and the parent executor's
   *  `requireNoBailInChildren` declaration. */
  childContext(requireNoBail: boolean): DispatchContext {
    return new DispatchContext(
      this.interp,
      this.registry,
      this.requireNoBail || requireNoBail,
      this.active
    );
  }

  /** @internal Whether the reentrancy guard has any entries. The
   *  registry uses this to skip the guard check on the hot path —
   *  when nothing is active there's nothing to detect. */
  get hasActive(): boolean {
    return this.active.size > 0;
  }

  /** @internal Used by the registry's reentrancy guard. Pre-checked
   *  with hasActive on the hot path, so the actual `has` only fires
   *  during sub-dispatch. */
  isActive(executorName: string, stmt: Stmt): boolean {
    return this.active.has(reentrancyKey(executorName, stmt));
  }

  /** @internal */
  pushActive(executorName: string, stmt: Stmt): void {
    this.active.add(reentrancyKey(executorName, stmt));
  }

  /** @internal */
  popActive(executorName: string, stmt: Stmt): void {
    this.active.delete(reentrancyKey(executorName, stmt));
  }
}

function reentrancyKey(executorName: string, stmt: Stmt): string {
  // Stmts don't have stable string IDs; use a WeakMap-backed counter.
  return `${executorName}@${stmtId(stmt)}`;
}

const STMT_IDS = new WeakMap<Stmt, number>();
let nextStmtId = 1;
function stmtId(stmt: Stmt): number {
  let id = STMT_IDS.get(stmt);
  if (id === undefined) {
    id = nextStmtId++;
    STMT_IDS.set(stmt, id);
  }
  return id;
}

/** Re-export for the runner's signal type. */
export type { ControlSignal };
