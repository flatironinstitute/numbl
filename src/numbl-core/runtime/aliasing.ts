/**
 * Anti-aliasing via on-demand graph sweep from runtime roots.
 *
 * Mutation sites (`storeIntoTensor`, `storeIntoCell`) call `isAliased`
 * to decide whether to copy-on-write. No per-tensor state, no inc/dec
 * bookkeeping — truth is recomputed each time from the live root set.
 *
 * Roots traversed:
 *   - the active env chain (env.parent links)
 *   - every entry in `rt._envStack` (caller envs across function calls;
 *     function envs have no `parent` to their caller, so the interpreter
 *     pushes the saved env on call entry and pops on exit)
 *   - globals (rt.$g)
 *   - persistents (rt.persistentStore)
 * Each visited container (cell, struct, struct array, class instance,
 * function captures) is descended recursively.
 *
 * The LHS slot of the in-progress indexed store is excluded so a tensor
 * that's only reachable through that slot is treated as uniquely owned.
 *
 * Buffer sharing: zero-copy `reshape` returns a new RuntimeTensor wrapper
 * that shares the data buffer with the source. Object identity alone
 * misses this — when the target is a tensor, sweep also flags any tensor
 * whose `data` (or `imag`) buffer matches.
 *
 * Bounded traversal: a hard visit budget caps worst-case cost. On
 * exhaustion the sweep returns true (conservative — we copy).
 */

import type { RuntimeValue, RuntimeTensor } from "./types.js";
import {
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeFunction,
  isRuntimeTensor,
} from "./types.js";

/** Hard cap on objects visited per sweep. Beyond this, return true
 *  (conservative copy). Calibrated so even pathological graphs stay
 *  comparable in cost to a small data-buffer copy. */
const VISIT_BUDGET = 4096;

/** Minimal env shape used by the sweep — avoids a circular import of
 *  the interpreter's `Environment`. The `Environment` class satisfies
 *  this structurally. */
export interface AliasEnv {
  vars: Map<string, RuntimeValue>;
  parent?: AliasEnv;
}

/** Minimal runtime shape used by the sweep. The `Runtime` class
 *  satisfies this structurally. */
export interface AliasRuntime {
  $g: Record<string, RuntimeValue>;
  persistentStore: Map<string, Map<string, RuntimeValue>>;
  _envStack: AliasEnv[];
  _aliasCtx: { env: AliasEnv; bindingName: string | null } | null;
}

/**
 * Returns true iff `target` is reachable from any slot OTHER than
 * `(excludeEnv, excludeBinding)`. Conservative on budget exhaustion.
 *
 * Callers should pass `excludeEnv` as the env that owns the LHS binding
 * being overwritten (typically the same as `env`, but kept explicit so
 * the contract is clear).
 */
export function isAliased(
  rt: AliasRuntime,
  env: AliasEnv,
  target: object,
  excludeEnv: AliasEnv | null,
  excludeBinding: string | null
): boolean {
  const seen = new Set<object>();
  let budget = VISIT_BUDGET;

  // For tensor targets, also flag buffer-sharing aliases (e.g. zero-copy
  // reshape returns a new wrapper with the same `data`).
  const targetTensor = isRuntimeTensor(target as RuntimeValue)
    ? (target as RuntimeTensor)
    : null;

  const visit = (val: unknown): boolean => {
    if (--budget < 0) return true;
    if (val === target) return true;
    if (val === null || typeof val !== "object") return false;
    if (seen.has(val as object)) return false;
    seen.add(val as object);

    const v = val as RuntimeValue;

    // Buffer-identity check for tensor targets.
    if (targetTensor && isRuntimeTensor(v)) {
      if (
        v.data === targetTensor.data ||
        (v.imag !== undefined && v.imag === targetTensor.imag)
      ) {
        return true;
      }
    }

    if (isRuntimeCell(v)) {
      for (const e of v.data) if (visit(e)) return true;
      return false;
    }
    if (isRuntimeStruct(v)) {
      for (const fv of v.fields.values()) if (visit(fv)) return true;
      return false;
    }
    if (isRuntimeStructArray(v)) {
      for (const el of v.elements) if (visit(el)) return true;
      return false;
    }
    if (isRuntimeClassInstance(v)) {
      for (const fv of v.fields.values()) if (visit(fv)) return true;
      if (v._builtinData !== undefined && visit(v._builtinData)) return true;
      return false;
    }
    if (isRuntimeClassInstanceArray(v)) {
      for (const el of v.elements) if (visit(el)) return true;
      return false;
    }
    if (isRuntimeFunction(v)) {
      // Anonymous-function captures: `Environment.snapshot()` produces a
      // new env that becomes the parent of the call-time env, so its
      // tensors are reached via the parent chain when the closure is
      // active. Outside an active call, `captures` is the only visible
      // hold (currently unused, but traversed for completeness).
      for (const c of v.captures) if (visit(c)) return true;
      return false;
    }
    return false;
  };

  const visitEnv = (e: AliasEnv): boolean => {
    for (const [name, v] of e.vars) {
      if (e === excludeEnv && name === excludeBinding) continue;
      if (visit(v)) return true;
    }
    return e.parent ? visitEnv(e.parent) : false;
  };

  if (visitEnv(env)) return true;
  for (let i = rt._envStack.length - 1; i >= 0; i--) {
    if (visitEnv(rt._envStack[i])) return true;
  }
  for (const k in rt.$g) if (visit(rt.$g[k])) return true;
  for (const fmap of rt.persistentStore.values())
    for (const v of fmap.values()) if (visit(v)) return true;
  return false;
}
