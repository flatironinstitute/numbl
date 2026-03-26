/**
 * Builtin name registry: tracks which names are recognized as builtins.
 */

/** Extra builtin names (IBuiltins + special builtins) added after initial load. */
const _extraBuiltinNames = new Set<string>();

/** Register additional builtin names so isBuiltin() recognizes them. */
export function registerExtraBuiltinNames(names: Iterable<string>): void {
  for (const n of names) _extraBuiltinNames.add(n);
}

export function isBuiltin(name: string): boolean {
  return _extraBuiltinNames.has(name);
}

export function getAllBuiltinNames(): string[] {
  return Array.from(_extraBuiltinNames);
}
