/** Allocate a Float64Array. Thin wrapper around `new Float64Array(x)`
 *  kept as a single allocation site so future strategies (pooling,
 *  COW-aware allocation, etc.) can be reintroduced without touching
 *  every callsite. */
export function allocFloat64Array(
  x: number | number[] | Float64Array
): Float64Array {
  if (typeof x === "number") return new Float64Array(x);
  return new Float64Array(x);
}

/** No-op. Kept as a placeholder for future explicit-release strategies
 *  (e.g. pool release, arena reset). With pure JS GC, scratch buffers
 *  are reclaimed when the wrapper goes out of scope. */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function releaseFloat64Array(_buf: Float64Array): void {
  // intentionally empty
}

/** Pass-through. Kept as a placeholder for future scratch-arena
 *  strategies. Today it just runs `fn` and returns its result. */
export function withScratch<T>(fn: () => T): T {
  return fn();
}
