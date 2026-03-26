/**
 * Builtin function registry and type definitions
 */

import { type ItemType, IType } from "../lowering/itemTypes.js";
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

export function realArrayConstructorCheck(
  argTypes: ItemType[],
  nargout: number
): { outputTypes: ItemType[] } | null {
  if (nargout !== 1 && nargout !== 0) return null;
  if (argTypes.length === 0) {
    return { outputTypes: [IType.num()] };
  }
  return {
    outputTypes: [IType.tensor()],
  };
}

const builtins = new Map<string, BuiltinFn>();
const builtinNarginMap = new Map<string, number>();

export function getBuiltin(name: string): BuiltinFn | undefined {
  return builtins.get(name);
}

export function getBuiltinNargin(name: string): number | undefined {
  return builtinNarginMap.get(name);
}

export function findBuiltinBranch(
  name: string,
  argTypes: ItemType[],
  nargout: number
): {
  builtin: BuiltinFn | undefined;
  branch: BuiltinFnBranch | undefined;
  branchIndex: number;
} {
  const builtin = getBuiltin(name);
  if (!builtin)
    return {
      builtin: undefined,
      branch: undefined,
      branchIndex: -1,
    };
  let branchIndex = -1;
  let branch: BuiltinFnBranch | undefined = undefined;
  for (let i = 0; i < builtin.length; i++) {
    if (builtin[i].check(argTypes, nargout)) {
      branchIndex = i;
      branch = builtin[i];
      break;
    }
  }
  return {
    builtin,
    branch,
    branchIndex,
  };
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

export function unregister(name: string): void {
  builtins.delete(name);
  builtinNarginMap.delete(name);
}
