/**
 * Shared root walker used by the COW anti-aliasing check.
 *
 * Enumerates RuntimeValues reachable from a few well-defined roots and
 * descends recursively through containers (cell, struct, struct array,
 * class instance + _builtinData, function captures, env-parent chains).
 * The visitor is called on each value; returning `true` aborts traversal
 * (used by the alias check to short-circuit on first match).
 *
 * Roots traversed (matches the original aliasing.ts root set):
 *   - the `env` argument's parent chain
 *   - every entry in `rt._envStack` (saved caller envs across function
 *     calls), each walked top-down with its own parent chain
 *   - rt.$g (globals)
 *   - rt.persistentStore
 *
 * Cycle prevention is built in via a `seen` set; the visitor itself can
 * be stateless. The walker has no visit budget — callers needing one
 * enforce it inside the visitor closure.
 */

import type { RuntimeValue } from "./types.js";
import {
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeFunction,
} from "./types.js";

/** Minimal env shape used by the sweep — avoids a circular import of
 *  the interpreter's `Environment`. The `Environment` class satisfies
 *  this structurally. */
export interface AliasEnv {
  vars: Map<string, RuntimeValue>;
  parent?: AliasEnv;
}

/** Minimal runtime shape used by the sweep. The `Runtime` class
 *  satisfies this structurally. */
export interface RootRuntime {
  $g: Record<string, RuntimeValue>;
  persistentStore: Map<string, Map<string, RuntimeValue>>;
  _envStack: AliasEnv[];
  _aliasCtx: { env: AliasEnv; bindingName: string | null } | null;
}

export interface RootProvider {
  rt: RootRuntime;
  /** Active env to walk first (typically the env that owns the LHS slot
   *  of an in-progress indexed store). */
  env: AliasEnv;
  /** When set, the slot (excludeEnv, excludeBinding) is skipped during
   *  the env walk — used by the aliasing check to exclude the LHS slot. */
  excludeEnv?: AliasEnv | null;
  excludeBinding?: string | null;
}

/**
 * Walk every RuntimeValue reachable from the provider's roots, calling
 * `visit` on each. The visitor returns `true` to abort traversal early;
 * `false` to continue. walkRoots returns `true` iff any visit returned
 * `true`.
 */
export function walkRoots(
  p: RootProvider,
  visit: (v: unknown) => boolean
): boolean {
  const seen = new Set<object>();

  const visitVal = (val: unknown): boolean => {
    if (visit(val)) return true;
    if (val === null || typeof val !== "object") return false;
    if (seen.has(val as object)) return false;
    seen.add(val as object);

    const v = val as RuntimeValue;
    if (isRuntimeCell(v)) {
      for (const e of v.data) if (visitVal(e)) return true;
    } else if (isRuntimeStruct(v)) {
      for (const fv of v.fields.values()) if (visitVal(fv)) return true;
    } else if (isRuntimeStructArray(v)) {
      for (const el of v.elements) if (visitVal(el)) return true;
    } else if (isRuntimeClassInstance(v)) {
      for (const fv of v.fields.values()) if (visitVal(fv)) return true;
      if (v._builtinData !== undefined && visitVal(v._builtinData)) return true;
    } else if (isRuntimeClassInstanceArray(v)) {
      for (const el of v.elements) if (visitVal(el)) return true;
    } else if (isRuntimeFunction(v)) {
      for (const c of v.captures) if (visitVal(c)) return true;
    }
    return false;
  };

  const visitEnv = (e: AliasEnv): boolean => {
    for (const [name, v] of e.vars) {
      if (e === p.excludeEnv && name === p.excludeBinding) continue;
      if (visitVal(v)) return true;
    }
    return e.parent ? visitEnv(e.parent) : false;
  };

  const rt = p.rt;
  if (visitEnv(p.env)) return true;
  for (let i = rt._envStack.length - 1; i >= 0; i--) {
    if (visitEnv(rt._envStack[i])) return true;
  }
  for (const k in rt.$g) if (visitVal(rt.$g[k])) return true;
  for (const fmap of rt.persistentStore.values()) {
    for (const v of fmap.values()) if (visitVal(v)) return true;
  }

  return false;
}
