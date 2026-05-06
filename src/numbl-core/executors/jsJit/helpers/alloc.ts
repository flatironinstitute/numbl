import { getCurrentRuntime } from "../../../runtime/memoryPool.js";

/** Allocate a Float64Array. When a Runtime is active on the module-level
 *  stack, allocations are routed through that runtime's MemoryPool so
 *  alloc counters and the IDE Memory tab can report on them. The pool
 *  also has a free-list mechanism (`acquire` checks before allocating
 *  fresh, `release` returns a buffer to it) but nothing currently calls
 *  `release` — reclamation strategy is deferred.
 *
 *  When no runtime is active (e.g. unit tests of pure helpers), this
 *  falls back to a plain `new Float64Array` with no tracking. */
export function allocFloat64Array(
  x: number | number[] | Float64Array
): Float64Array {
  const rt = getCurrentRuntime();
  if (!rt) {
    if (typeof x === "number") return new Float64Array(x);
    return new Float64Array(x);
  }
  if (typeof x === "number") return rt.pool.acquire(x);
  return rt.pool.acquireFrom(x);
}
