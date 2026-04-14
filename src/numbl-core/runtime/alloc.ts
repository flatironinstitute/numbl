/**
 * Allocate an UNINITIALIZED typed array — skips the zero-fill that
 * `new Float64Array(n)` / `new FloatXArray(n)` perform by default.
 *
 * On Node, `Buffer.allocUnsafe` returns un-zeroed memory; wrapping it
 * in a TypedArray view costs ~10× less than the zero-fill for a 16 MB
 * buffer (~45 µs vs ~470 µs at N=2M doubles).
 *
 * SAFETY CONTRACT (very important):
 *   The caller MUST write every element before reading it.  Any element
 *   that is read before being written will contain arbitrary stale bytes
 *   from recently-freed memory.  If you cannot guarantee full coverage,
 *   use `new Float64Array(n)` / `new FloatXArray(n)` instead.
 *
 * In non-Node environments (browser, Deno without node-compat, …) where
 * `Buffer` is unavailable, we fall back to the zero-filling constructor
 * — still correct, just slower.
 */

import { FloatXArray } from "./types.js";

// Type of `new FloatXArray(n)` — resolves to Float32Array<ArrayBuffer> |
// Float64Array<ArrayBuffer>.  We use this as the return type (rather than the
// exported FloatXArrayType, which uses the default <ArrayBufferLike>), so
// callers that pass the result into APIs typed with concrete ArrayBuffer
// generics keep type-checking.
type FloatXInstance = InstanceType<typeof FloatXArray>;

const hasBuffer = typeof Buffer !== "undefined";

export function uninitFloat64(n: number): Float64Array<ArrayBuffer> {
  if (hasBuffer) {
    const buf = Buffer.allocUnsafe(n * 8);
    // Buffer.allocUnsafe's .buffer is always a plain ArrayBuffer at runtime
    // (it comes from Node's pool), but typed as ArrayBufferLike.  Cast to
    // the narrower Float64Array<ArrayBuffer> so consumers that still use the
    // default TypedArray typing keep working.
    return new Float64Array(buf.buffer as ArrayBuffer, buf.byteOffset, n);
  }
  return new Float64Array(n);
}

export function uninitFloatX(n: number): FloatXInstance {
  if (hasBuffer) {
    const buf = Buffer.allocUnsafe(n * FloatXArray.BYTES_PER_ELEMENT);
    return new FloatXArray(
      buf.buffer as ArrayBuffer,
      buf.byteOffset,
      n
    ) as FloatXInstance;
  }
  return new FloatXArray(n) as FloatXInstance;
}
