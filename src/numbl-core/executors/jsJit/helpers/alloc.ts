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
/** Allocations of <= 100 doubles bypass the pool unconditionally — the
 *  per-acquire bookkeeping (liveSet membership + counter increments)
 *  costs more than the alloc itself, and small buffers in the free
 *  list rarely help (most reuse opportunities come from larger
 *  intermediate tensors). Tune the threshold here if profiling shows
 *  small allocations dominating. */
const SMALL_ALLOC_THRESHOLD = 100;

/** When running under vitest, route every allocation through the pool
 *  regardless of size so the test suite exercises pool semantics for
 *  buffers of every shape. Vitest sets `process.env.VITEST = "true"`. */
const IN_TEST =
  typeof process !== "undefined" && process.env?.VITEST === "true";

export function allocFloat64Array(
  x: number | number[] | Float64Array
): Float64Array {
  // Fast path for small allocations: plain `new Float64Array` regardless
  // of whether a runtime is active or memPool is on. Skipped under
  // vitest so pool semantics get exercised at every size.
  if (!IN_TEST) {
    if (typeof x === "number") {
      if (x <= SMALL_ALLOC_THRESHOLD) return new Float64Array(x);
    } else if (x.length <= SMALL_ALLOC_THRESHOLD) {
      return new Float64Array(x);
    }
  }
  const rt = getCurrentRuntime();
  if (!rt || !rt.memPool) {
    if (typeof x === "number") return new Float64Array(x);
    return new Float64Array(x);
  }
  if (typeof x === "number") return rt.pool.acquire(x);
  return rt.pool.acquireFrom(x);
}
