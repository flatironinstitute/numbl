/**
 * Refcount-driven copy-on-write helpers.
 *
 * The lvalue write path uses these to make sure the chain from the env
 * root to the leaf is uniquely owned before mutation: at each level, if
 * the container is shared (`isShared`), it's cloned and the clone is
 * rebound in the parent. After the chain is unique, in-place mutation
 * at the leaf is observable only to the LHS variable.
 */

import { type RuntimeValue } from "./types.js";
import { Refcounted, type RefcountRuntime } from "./refcount.js";

/** Return a fresh wrapper that shadows `v`'s contents but has rc=0
 *  (modulo scope adoption from the constructor). Delegates to the
 *  value's `shallowClone()`, so every refcounted kind is covered. Refs
 *  to child values are shared with the original. Caller is responsible
 *  for binding the fresh wrapper into the parent slot.
 *
 *  Returns the original value unchanged for kinds that don't need to
 *  be copied (handle-class instances, function handles, primitives) —
 *  the caller can `=== v` test if the copy was a no-op. */
export function cowCopy(v: RuntimeValue): RuntimeValue {
  if (v instanceof Refcounted) return v.shallowClone() as RuntimeValue;
  return v;
}

// Re-export for convenient access from the runtime surface.
export { isShared } from "./refcount.js";
export type { RefcountRuntime };
