/**
 * Builtin function registry and type definitions
 */

import { type ItemType } from "../lowering/itemTypes.js";
import { RuntimeValue } from "../runtime/index.js";

// ── Builtin registry ────────────────────────────────────────────────────

export type BuiltinFn = BuiltinFnBranch[];

export type BuiltinFnBranch = {
  check: (
    argTypes: ItemType[],
    nargout: number
  ) => {
    outputTypes: ItemType[];
  } | null;
  apply: (
    args: RuntimeValue[],
    nargout: number
  ) => RuntimeValue | RuntimeValue[] | undefined; // important that apply is always provided as a fallback, even when we have nativeJsFn
  nativeJsFn?: string; // Optional JavaScript function name for native implementation
};

// Builtin function single branch, with optional known output type
export const builtinSingle = (
  apply: (
    args: RuntimeValue[],
    nargout: number
  ) => RuntimeValue | RuntimeValue[] | undefined,
  opts?: { outputType?: ItemType }
): BuiltinFn => [
  {
    check: (_argTypes, nargout) => {
      const t = opts?.outputType ?? {};
      return { outputTypes: Array(nargout).fill(t) };
    },
    apply,
  },
];

const builtins = new Map<string, BuiltinFn>();
const builtinNarginMap = new Map<string, number>();

export function getBuiltin(name: string): BuiltinFn | undefined {
  return builtins.get(name);
}

export function getBuiltinNargin(name: string): number | undefined {
  return builtinNarginMap.get(name);
}

export function isBuiltin(name: string): boolean {
  return builtins.has(name) || _extraBuiltinNames.has(name);
}

/** Extra builtin names (IBuiltins + special builtins) added after initial load. */
const _extraBuiltinNames = new Set<string>();

/** Register additional builtin names so isBuiltin() recognizes them. */
export function registerExtraBuiltinNames(names: Iterable<string>): void {
  for (const n of names) _extraBuiltinNames.add(n);
}

export function getAllBuiltinNames(): string[] {
  return Array.from(builtins.keys());
}

export function register(name: string, fn: BuiltinFn, nargin?: number): void {
  builtins.set(name, fn);
  if (nargin !== undefined) {
    builtinNarginMap.set(name, nargin);
  }
}
