/**
 * Shared types for the interpreter: Environment, control flow signals, FunctionDef.
 */

import type { Stmt, ArgumentsBlock } from "../parser/types.js";
import type { Runtime } from "../runtime/runtime.js";
import type { RuntimeValue } from "../runtime/types.js";
import { disposeValue } from "../runtime/utils.js";

// ── Control flow signals ─────────────────────────────────────────────────

export class BreakSignal {
  readonly _tag = "break";
}
export class ContinueSignal {
  readonly _tag = "continue";
}
export class ReturnSignal {
  readonly _tag = "return";
  constructor(public values: RuntimeValue[]) {}
}

export type ControlSignal = BreakSignal | ContinueSignal | ReturnSignal;

// ── Environment (variable scope) ─────────────────────────────────────────

export class Environment {
  private vars = new Map<string, RuntimeValue>();
  /** When true, writes to variables found in parent go to the parent (nested function semantics). */
  isNested = false;
  /** Nested function definitions registered during execution. Lazy-initialized. */
  private _nestedFunctions:
    | Map<string, { fn: FunctionDef; env: Environment }>
    | undefined;
  get nestedFunctions(): Map<string, { fn: FunctionDef; env: Environment }> {
    return (this._nestedFunctions ??= new Map());
  }
  set nestedFunctions(v: Map<string, { fn: FunctionDef; env: Environment }>) {
    this._nestedFunctions = v;
  }
  /** Names declared as `global` in this scope — reads/writes go through rt.$g. Lazy-initialized. */
  private _globalNames: Set<string> | undefined;
  get globalNames(): Set<string> {
    return (this._globalNames ??= new Set());
  }
  set globalNames(v: Set<string>) {
    this._globalNames = v;
  }
  /** Names declared as `persistent` in this scope. Lazy-initialized. */
  private _persistentNames: Set<string> | undefined;
  get persistentNames(): Set<string> {
    return (this._persistentNames ??= new Set());
  }
  set persistentNames(v: Set<string>) {
    this._persistentNames = v;
  }
  /** Function ID for persistent variable storage */
  persistentFuncId: string | undefined;
  /** Back-reference to the runtime (needed for global/persistent access) */
  rt: Runtime | null = null;
  /** Set when a `@nestedFn` handle has been created that captures this env
   *  (or an ancestor). Tells the function-exit cleanup that clearing this
   *  env would strand the handle's closure, so locals must be left alive. */
  nestedHandleCreated = false;
  /** Set when this env's wrappers have been captured by a closure snapshot
   *  (anonymous function or @nestedFn). Disposing the values would corrupt
   *  the closure's view, so the function-exit dispose path must skip this
   *  env. Strictly conservative: set whenever a snapshot includes this env,
   *  even if the resulting closure never escapes. */
  envCaptured = false;

  constructor(private parent?: Environment) {}

  get(name: string): RuntimeValue | undefined {
    if (
      this._globalNames !== undefined &&
      this._globalNames.has(name) &&
      this.rt
    ) {
      const v = this.rt.$g[name];
      return v === undefined ? undefined : v;
    }
    return this.vars.get(name) ?? this.parent?.get(name);
  }

  /** Set variable — for nested scopes, writes to parent if variable exists there. */
  set(name: string, value: RuntimeValue): void {
    if (
      this._globalNames !== undefined &&
      this._globalNames.has(name) &&
      this.rt
    ) {
      this.rt.$g[name] = value;
      return;
    }
    if (this.isNested && !this.vars.has(name) && this.parent) {
      const owner = this.findOwner(name);
      if (owner) {
        owner.setLocal(name, value);
        return;
      }
    }
    this.vars.set(name, value);
  }

  /** Always writes to this scope (for parameter binding). */
  setLocal(name: string, value: RuntimeValue): void {
    this.vars.set(name, value);
  }

  /** Remove a variable from this scope's local map.
   *  Returns true if the name was present locally. Does not touch
   *  the parent scope, globals, or persistent registrations — those
   *  are removed via `clear global` / `clear functions` (not yet
   *  implemented). */
  delete(name: string): boolean {
    return this.vars.delete(name);
  }

  /** Remove all local variables from this scope. Globals,
   *  persistents, and nested function defs are preserved. */
  clearLocals(): void {
    this.vars.clear();
  }

  /** Recursively dispose every local value (returning their dense buffers
   *  to the allocator pool) except those whose object identity is in
   *  `keep`, then clear the local map. Caller must verify the env is NOT
   *  captured (see `envCaptured` / `nestedHandleCreated`) — disposing a
   *  buffer still referenced by a closure leads to use-after-free
   *  corruption when the pool hands the buffer back out. */
  disposeLocalsExcept(keep: Set<RuntimeValue>): void {
    for (const v of this.vars.values()) {
      if (!keep.has(v)) disposeValue(v);
    }
    this.vars.clear();
  }

  has(name: string): boolean {
    if (
      this._globalNames !== undefined &&
      this._globalNames.has(name) &&
      this.rt
    ) {
      return name in this.rt.$g;
    }
    return this.vars.has(name) || (this.parent?.has(name) ?? false);
  }

  /** Check if this scope directly owns a variable (not parent). */
  hasLocal(name: string): boolean {
    if (this._globalNames !== undefined && this._globalNames.has(name))
      return true;
    return this.vars.has(name);
  }

  /** Find the environment that owns a variable. */
  private findOwner(name: string): Environment | null {
    if (this.vars.has(name)) return this;
    return this.parent?.findOwner(name) ?? null;
  }

  /** Look up a nested function definition in this scope or parent scopes. */
  getNestedFunction(
    name: string
  ): { fn: FunctionDef; env: Environment } | undefined {
    return (
      this._nestedFunctions?.get(name) ?? this.parent?.getNestedFunction(name)
    );
  }

  /** Mark this env and every ancestor up to (and including) the env that
   *  defines `name` as having had a nested-function handle created. The
   *  handle's closure references this env, so any of those scopes' locals
   *  must stay alive after the function exits. Returns true if `name`
   *  was found as a nested-function definition somewhere in the chain. */
  markChainForNestedHandle(name: string): boolean {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let e: Environment | undefined = this;
    while (e) {
      e.nestedHandleCreated = true;
      e.envCaptured = true;
      if (e._nestedFunctions?.has(name)) return true;
      e = e.parent;
    }
    return false;
  }

  localNames(): string[] {
    return [...this.vars.keys()];
  }

  /** Create a snapshot of this environment.
   *  Used for anonymous functions which capture values at definition time.
   *  Captures the wrappers directly: callers/closures get value semantics
   *  via deep-clone-on-call/assignment elsewhere.
   *
   *  Each env visited is marked `envCaptured` — its wrappers are now
   *  reachable through the snapshot, so the function-exit dispose path
   *  must leave its locals alone. */
  snapshot(): Environment {
    const snap = new Environment();
    const copyVars = (env: Environment) => {
      if (env.parent) copyVars(env.parent);
      env.envCaptured = true;
      for (const [k, v] of env.vars) {
        snap.vars.set(k, v);
      }
      for (const [k, v] of env.nestedFunctions) {
        snap.nestedFunctions.set(k, v);
      }
    };
    copyVars(this);
    snap.rt = this.rt;
    snap.globalNames = new Set(this.globalNames);
    return snap;
  }

  toRecord(): Record<string, RuntimeValue> {
    const result: Record<string, RuntimeValue> = {};
    if (this.parent) {
      Object.assign(result, this.parent.toRecord());
    }
    for (const [k, v] of this.vars) {
      result[k] = v;
    }
    // Include global variables
    if (this.rt) {
      for (const name of this.globalNames) {
        if (name in this.rt.$g) {
          result[name] = this.rt.$g[name];
        }
      }
    }
    return result;
  }
}

// ── Function definition storage ──────────────────────────────────────────

export interface FunctionDef {
  name: string;
  params: string[];
  outputs: string[];
  body: Stmt[];
  argumentsBlocks?: ArgumentsBlock[];
}

/** Create a FunctionDef from an AST Function statement. */
export function funcDefFromStmt(
  stmt: Stmt & { type: "Function" }
): FunctionDef {
  return {
    name: stmt.name,
    params: stmt.params,
    outputs: stmt.outputs,
    body: stmt.body,
    argumentsBlocks: stmt.argumentsBlocks,
  };
}
