/**
 * Allocate an UNINITIALIZED typed array — skips the zero-fill that
 * `new Float64Array(n)` / `new FloatXArray(n)` perform by default.
 *
 * On Node, `Buffer.allocUnsafe(...)` returns un-zeroed slab memory (~10×
 * cheaper than the zero-fill at large N). On other runtimes, fall back to
 * the zero-filled `new FloatXArray(n)`.
 *
 * SAFETY CONTRACT:
 *   The caller MUST write every element before reading it. Any element
 *   that is read before being written will contain arbitrary stale bytes.
 *   If you cannot guarantee full coverage, use `new Float64Array(n)` /
 *   `new FloatXArray(n)` instead.
 */

import { FloatXArray } from "./types.js";

type FloatXInstance = InstanceType<typeof FloatXArray>;

const hasBuffer = typeof Buffer !== "undefined";

export function uninitFloat64(n: number): Float64Array<ArrayBuffer> {
  if (n === 0) return new Float64Array(0);
  if (hasBuffer) {
    const b = Buffer.allocUnsafe(n * 8);
    return new Float64Array(b.buffer as ArrayBuffer, b.byteOffset, n);
  }
  return new Float64Array(n);
}

export function uninitFloatX(n: number): FloatXInstance {
  if (n === 0) return new FloatXArray(0) as FloatXInstance;
  if (hasBuffer) {
    const bytes = (FloatXArray as unknown) === Float32Array ? 4 : 8;
    const b = Buffer.allocUnsafe(n * bytes);
    return new FloatXArray(
      b.buffer as ArrayBuffer,
      b.byteOffset,
      n
    ) as FloatXInstance;
  }
  return new FloatXArray(n) as FloatXInstance;
}
