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
import { type TypeInfo, inferTypeInfo } from "./typeInfo.js";

/** Where the dispatcher is currently running.
 *
 *   - `top-level` — the script body's top-level stmt list. Only set by
 *     `Interpreter.run()`. Whole-script executors (e.g. JS-JIT
 *     top-level) match only when this is set and `i === 0`.
 *   - `nested` — a function body, loop body, or any other block.
 *     Default for everything other than the script's top-level loop.
 *
 *  Sub-dispatch within an executor's compiled artifact always sets
 *  `nested` regardless of the parent — once you're inside a compiled
 *  artifact, you're no longer at the script's top level. */
export type DispatchScope = "top-level" | "nested";

export class DispatchContext {
  readonly interp: Interpreter;
  readonly registry: Registry;
  /** When true, only no-bail executors are eligible. Set when the
   *  caller's compiled artifact has emitted observable side effects
   *  that mustn't repeat. */
  readonly requireNoBail: boolean;
  readonly scope: DispatchScope;

  /** Per-dispatch memoization of typeOf(name) lookups. */
  private readonly typeCache = new Map<string, TypeInfo>();

  /** Re-entrancy guard: which executors are currently running on
   *  which head stmts. Prevents an executor sub-dispatching back into
   *  itself on the same stmt. */
  private readonly active: Set<string>;

  constructor(
    interp: Interpreter,
    registry: Registry,
    requireNoBail: boolean,
    active?: Set<string>,
    scope: DispatchScope = "nested"
  ) {
    this.interp = interp;
    this.registry = registry;
    this.requireNoBail = requireNoBail;
    this.active = active ?? new Set();
    this.scope = scope;
  }

  /** Look up a name's TypeInfo from the interpreter env, memoized. */
  typeOf(name: string): TypeInfo {
    const cached = this.typeCache.get(name);
    if (cached) return cached;
    const v = this.interp.env.get(name);
    const t = v === undefined ? { kind: "unknown" as const } : inferTypeInfo(v);
    this.typeCache.set(name, t);
    return t;
  }

  /** Read a value from env (no inference). Returns undefined if unbound. */
  envValue(name: string): unknown {
    return this.interp.env.get(name);
  }

  /** Construct a child context for sub-dispatch. `requireNoBail` is
   *  the OR of the parent flag and the parent executor's
   *  `requireNoBailInChildren` declaration. Scope always becomes
   *  `nested` — sub-dispatches happen inside an executor's artifact
   *  and are never at script top level. */
  childContext(requireNoBail: boolean): DispatchContext {
    return new DispatchContext(
      this.interp,
      this.registry,
      this.requireNoBail || requireNoBail,
      this.active,
      "nested"
    );
  }

  /** @internal Used by the registry's reentrancy guard. */
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
