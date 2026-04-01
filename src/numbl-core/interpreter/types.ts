/**
 * Shared types for the interpreter: Environment, control flow signals, FunctionDef.
 */

import type { Stmt, ArgumentsBlock } from "../parser/types.js";
import type { Runtime } from "../runtime/runtime.js";
import type { RuntimeValue } from "../runtime/types.js";
import type { JitType } from "./jit/jitTypes.js";

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
  /** Nested function definitions registered during execution. */
  nestedFunctions = new Map<string, { fn: FunctionDef; env: Environment }>();
  /** Names declared as `global` in this scope — reads/writes go through rt.$g */
  globalNames = new Set<string>();
  /** Names declared as `persistent` in this scope */
  persistentNames = new Set<string>();
  /** Function ID for persistent variable storage */
  persistentFuncId: string | undefined;
  /** Back-reference to the runtime (needed for global/persistent access) */
  rt: Runtime | null = null;

  constructor(private parent?: Environment) {}

  get(name: string): RuntimeValue | undefined {
    if (this.globalNames.has(name) && this.rt) {
      const v = this.rt.$g[name];
      return v === undefined ? undefined : v;
    }
    return this.vars.get(name) ?? this.parent?.get(name);
  }

  /** Set variable — for nested scopes, writes to parent if variable exists there. */
  set(name: string, value: RuntimeValue): void {
    if (this.globalNames.has(name) && this.rt) {
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

  has(name: string): boolean {
    if (this.globalNames.has(name) && this.rt) {
      return name in this.rt.$g;
    }
    return this.vars.has(name) || (this.parent?.has(name) ?? false);
  }

  /** Check if this scope directly owns a variable (not parent). */
  hasLocal(name: string): boolean {
    if (this.globalNames.has(name)) return true;
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
      this.nestedFunctions.get(name) ?? this.parent?.getNestedFunction(name)
    );
  }

  localNames(): string[] {
    return [...this.vars.keys()];
  }

  /** Create a snapshot of this environment (copies all variables by value).
   *  Used for anonymous functions which capture values at definition time. */
  snapshot(): Environment {
    const snap = new Environment();
    // Copy all variables from the entire chain
    const copyVars = (env: Environment) => {
      if (env.parent) copyVars(env.parent);
      for (const [k, v] of env.vars) {
        snap.vars.set(k, v);
      }
      // Also copy nested function registrations
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
  /** JIT compilation cache: maps signature key -> compiled entry or null (failed). */
  _jitCache?: Map<
    string,
    { fn: (...args: unknown[]) => unknown; source: string } | null
  >;
  /** Progressive type widening: last unified arg types, keyed by nargout. */
  _lastJitArgTypes?: Map<number, JitType[]>;
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
