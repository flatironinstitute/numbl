/**
 * Float64Array / FloatXArray allocator with module-level tally.
 *
 * Every fresh dense-tensor buffer in the runtime should funnel through one
 * of these helpers so the counters stay representative. A future buffer-
 * pool / reuse layer plugs in here without touching call sites.
 *
 * Three entry points, all instrumented through the same counter:
 *
 *   - `allocFloat64(n)` / `allocFloatX(n)`
 *       Uninitialized buffer. Caller MUST write every element before
 *       reading it. ~10× cheaper than zero-fill on Node thanks to
 *       `Buffer.allocUnsafe`.
 *
 *   - `zeroedFloat64(n)` / `zeroedFloatX(n)`
 *       Zero-filled buffer. Use when the caller relies on zeros (e.g.
 *       `zeros()`, padding the imag of a real tensor).
 *
 *   - `copyFloat64(src)` / `copyFloatX(src)`
 *       Buffer initialized from `src` (number[] or typed array). Use for
 *       tensor literals, conversions across precisions, defensive copies.
 *
 * Counter ignores zero-length allocations.
 */

import { FloatXArray } from "./types.js";

type FloatXInstance = InstanceType<typeof FloatXArray>;

const hasBuffer = typeof Buffer !== "undefined";
const FLOATX_BYTES = (FloatXArray as unknown) === Float32Array ? 4 : 8;

// ── Tally ─────────────────────────────────────────────────────────────

export interface AllocStats {
  /** Number of allocator calls (excluding zero-length). */
  count: number;
  /** Total bytes returned across all calls. */
  bytes: number;
}

let _count = 0;
let _bytes = 0;

/** Snapshot of current allocation totals. Cheap; safe to call often. */
export function getAllocStats(): AllocStats {
  return { count: _count, bytes: _bytes };
}

/** Reset the running tally back to zero. Useful between test runs or
 *  before a benchmark window. */
export function resetAllocStats(): void {
  _count = 0;
  _bytes = 0;
}

// ── Uninitialized ─────────────────────────────────────────────────────

/** Allocate an uninitialized Float64Array of length `n`. Caller MUST
 *  write every element before reading it. */
export function allocFloat64(n: number): Float64Array<ArrayBuffer> {
  if (n === 0) return new Float64Array(0);
  _count++;
  _bytes += n * 8;
  if (hasBuffer) {
    const b = Buffer.allocUnsafe(n * 8);
    return new Float64Array(b.buffer as ArrayBuffer, b.byteOffset, n);
  }
  return new Float64Array(n);
}

/** Allocate an uninitialized FloatXArray (Float64 by default, Float32
 *  when `NUMBL_USE_FLOAT32` is set) of length `n`. Caller MUST write
 *  every element before reading it. */
export function allocFloatX(n: number): FloatXInstance {
  if (n === 0) return new FloatXArray(0) as FloatXInstance;
  _count++;
  _bytes += n * FLOATX_BYTES;
  if (hasBuffer) {
    const b = Buffer.allocUnsafe(n * FLOATX_BYTES);
    return new FloatXArray(
      b.buffer as ArrayBuffer,
      b.byteOffset,
      n
    ) as FloatXInstance;
  }
  return new FloatXArray(n) as FloatXInstance;
}

// ── Zero-filled ───────────────────────────────────────────────────────

/** Allocate a zero-filled Float64Array of length `n`. */
export function zeroedFloat64(n: number): Float64Array<ArrayBuffer> {
  if (n === 0) return new Float64Array(0);
  _count++;
  _bytes += n * 8;
  return new Float64Array(n);
}

/** Allocate a zero-filled FloatXArray of length `n`. */
export function zeroedFloatX(n: number): FloatXInstance {
  if (n === 0) return new FloatXArray(0) as FloatXInstance;
  _count++;
  _bytes += n * FLOATX_BYTES;
  return new FloatXArray(n) as FloatXInstance;
}

// ── Copy ──────────────────────────────────────────────────────────────

/** Allocate a Float64Array initialized from `src`. */
export function copyFloat64(
  src: ArrayLike<number> | Float64Array | Float32Array
): Float64Array<ArrayBuffer> {
  const out = allocFloat64(src.length);
  out.set(src as ArrayLike<number>);
  return out;
}

/** Allocate a FloatXArray initialized from `src`. */
export function copyFloatX(
  src: ArrayLike<number> | Float64Array | Float32Array
): FloatXInstance {
  const out = allocFloatX(src.length);
  out.set(src as ArrayLike<number>);
  return out;
}
