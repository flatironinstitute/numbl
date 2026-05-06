import { getCurrentRuntime } from "../../../runtime/memoryPool.js";

/** Allocate a Float64Array. When a Runtime is active on the module-level
 *  stack and `rt.memPool` is on, allocations are routed through that
 *  runtime's MemoryPool — `acquire` may return a buffer from the free
 *  list (filled by `release` on refcount-zero `_destroy`) and counters
 *  feed the IDE Memory tab.
 *
 *  When `rt.memPool` is off (CLI `--no-mem-pool`, IDE toggle), the
 *  pool is bypassed completely: every allocation is a fresh
 *  `new Float64Array` with no tracking, no free-list lookup, no
 *  liveSet bookkeeping. This is the fastest possible path and is
 *  intended for isolating bugs that may stem from buffer recycling.
 *
 *  When no runtime is active (e.g. unit tests of pure helpers), this
 *  also falls back to a plain `new Float64Array`. */
export function allocFloat64Array(
  x: number | number[] | Float64Array
): Float64Array {
  const rt = getCurrentRuntime();
  if (!rt || !rt.memPool) {
    if (typeof x === "number") return new Float64Array(x);
    return new Float64Array(x);
  }
  if (typeof x === "number") return rt.pool.acquire(x);
  return rt.pool.acquireFrom(x);
}
