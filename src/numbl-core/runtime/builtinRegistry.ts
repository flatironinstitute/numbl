/**
 * The builtin registry: every interpreter builtin, keyed by name.
 *
 * Builtins are defined with `builtin()` (interpreter/builtins/builtin.ts),
 * which registers them here. The registry lives under runtime/ so that the
 * runtime (dispatch, function handles) can call builtins without depending
 * on the interpreter.
 */

import type { RuntimeValue } from "./types.js";

export interface BuiltinHelp {
  signatures: string[];
  description: string;
}

/** A registered builtin: explicit arity and an entry point on runtime
 *  values. */
export interface IBuiltin {
  name: string;
  help?: BuiltinHelp;
  /** Accepted input count [min, max]; max may be Infinity. */
  nargin: readonly [number, number];
  /** Accepted output count [min, max]; max 0 marks a void builtin. */
  nargout: readonly [number, number];
  /** Check arity, then run. Returns undefined for a void builtin. */
  call: (
    args: RuntimeValue[],
    nargout: number
  ) => RuntimeValue | RuntimeValue[] | undefined;
}

const registry = new Map<string, IBuiltin>();

export function getIBuiltin(name: string): IBuiltin | undefined {
  return registry.get(name);
}

export function registerIBuiltin(b: IBuiltin): void {
  if (registry.has(b.name)) {
    throw new Error(`registerIBuiltin: duplicate registration for '${b.name}'`);
  }
  registry.set(b.name, b);
}

export function isBuiltin(name: string): boolean {
  return registry.has(name);
}

export function getAllIBuiltinNames(): string[] {
  return Array.from(registry.keys());
}

const helpRegistry = new Map<string, BuiltinHelp>();

export function getIBuiltinHelp(name: string): BuiltinHelp | undefined {
  return registry.get(name)?.help ?? helpRegistry.get(name);
}

export function registerBuiltinHelp(name: string, help: BuiltinHelp): void {
  helpRegistry.set(name, help);
}

/** MATLAB's `nargin(@name)` for a builtin: the maximum input count, or -1
 *  when it is unbounded (MATLAB reports -1 for nearly all variadic
 *  builtins, e.g. zeros, numel, fprintf). */
export function getIBuiltinNargin(name: string): number | undefined {
  const ib = registry.get(name);
  if (!ib) return undefined;
  const max = ib.nargin[1];
  return max === Infinity ? -1 : max;
}

/** MATLAB's `nargout(@name)` for a builtin: the maximum output count, or -1
 *  when it is unbounded. */
export function getIBuiltinNargout(name: string): number | undefined {
  const ib = registry.get(name);
  if (!ib) return undefined;
  const max = ib.nargout[1];
  return max === Infinity ? -1 : max;
}
