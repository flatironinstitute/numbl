/**
 * Allocate an UNINITIALIZED typed array — skips the zero-fill that
 * `new Float64Array(n)` / `new FloatXArray(n)` perform by default.
 *
 * Routes through the active `BufferPool` (see `bufferPool.ts`): if a
 * matching-length buffer was previously released, it is handed back without
 * any allocation. On a miss, the pool falls through to `Buffer.allocUnsafe`
 * on Node (un-zeroed slab — ~10× cheaper than the zero-fill at large N) and
 * to `new FloatXArray(n)` elsewhere.
 *
 * SAFETY CONTRACT (very important):
 *   The caller MUST write every element before reading it. Any element that
 *   is read before being written will contain arbitrary stale bytes from
 *   recently-freed memory or from a previously-released buffer. If you
 *   cannot guarantee full coverage, use `new Float64Array(n)` /
 *   `new FloatXArray(n)` instead.
 */

import { FloatXArray } from "./types.js";
import { acquireFloatX, getActivePool } from "./bufferPool.js";

// Type of `new FloatXArray(n)` — resolves to Float32Array<ArrayBuffer> |
// Float64Array<ArrayBuffer>.  We use this as the return type (rather than the
// exported FloatXArrayType, which uses the default <ArrayBufferLike>), so
// callers that pass the result into APIs typed with concrete ArrayBuffer
// generics keep type-checking.
type FloatXInstance = InstanceType<typeof FloatXArray>;

export function uninitFloat64(n: number): Float64Array<ArrayBuffer> {
  return getActivePool().acquireF64(n) as Float64Array<ArrayBuffer>;
}

export function uninitFloatX(n: number): FloatXInstance {
  return acquireFloatX(n) as FloatXInstance;
}
