/**
 * Anti-aliasing via on-demand graph sweep from runtime roots.
 *
 * Mutation sites (`storeIntoTensor`, `storeIntoCell`) call `isAliased`
 * to decide whether to copy-on-write. No per-tensor state, no inc/dec
 * bookkeeping — truth is recomputed each time from the live root set.
 *
 * Implementation delegates to the shared walker in `rootWalker.ts`.
 * Roots traversed:
 *   - the active env chain (env.parent links)
 *   - every entry in `rt._envStack` (caller envs across function calls)
 *   - globals (rt.$g)
 *   - persistents (rt.persistentStore)
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
 * exhaustion the visitor returns true (conservative — we copy).
 */

import type { RuntimeValue, RuntimeTensor } from "./types.js";
import { isRuntimeTensor } from "./types.js";
import { walkRoots, type AliasEnv, type RootRuntime } from "./rootWalker.js";

// Re-export AliasEnv so existing importers (indexing.ts) keep working.
export type { AliasEnv } from "./rootWalker.js";

/** Hard cap on objects visited per sweep. Beyond this, the visitor returns
 *  true (conservative copy). Calibrated so even pathological graphs stay
 *  comparable in cost to a small data-buffer copy. */
const VISIT_BUDGET = 4096;

/** Minimal runtime shape used by the alias sweep. */
export type AliasRuntime = RootRuntime;

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
  const targetTensor = isRuntimeTensor(target as RuntimeValue)
    ? (target as RuntimeTensor)
    : null;

  let budget = VISIT_BUDGET;

  return walkRoots(
    { rt, env, excludeEnv, excludeBinding },
    (val: unknown): boolean => {
      if (--budget < 0) return true; // conservative
      if (val === target) return true;
      if (
        targetTensor &&
        val !== null &&
        typeof val === "object" &&
        (val as { kind?: string }).kind === "tensor"
      ) {
        const t = val as RuntimeTensor;
        if (
          t.data === targetTensor.data ||
          (t.imag !== undefined && t.imag === targetTensor.imag)
        ) {
          return true;
        }
      }
      return false;
    }
  );
}
