/**
 * Float64Array / FloatXArray allocator with module-level tally and pool.
 *
 * Every fresh dense-tensor buffer in the runtime should funnel through one
 * of these helpers so the counters stay representative.
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
 *   - `disposeFloat64(buf)` / `disposeFloatX(buf)`
 *       Return a buffer to the pool for later reuse. Caller asserts the
 *       buffer has no other live references — disposing a still-aliased
 *       buffer leads to use-after-free style corruption when it is
 *       handed back out via `alloc*`.
 *
 * Pool layout: per-constructor map keyed by length-in-elements, each
 * bucket a stack of free typed-array views. Capped per-bucket and in
 * total bytes held to avoid unbounded retention.
 *
 * Counters cover (alloc, dispose, pool hits/misses, bytes held). Counter
 * ignores zero-length allocations and zero-length disposals.
 */

import { FloatXArray } from "./types.js";

type FloatXInstance = InstanceType<typeof FloatXArray>;

const hasBuffer = typeof Buffer !== "undefined";
const FLOATX_BYTES = (FloatXArray as unknown) === Float32Array ? 4 : 8;
const FLOATX_IS_FLOAT64 = (FloatXArray as unknown) === Float64Array;

// ── Pool ─────────────────────────────────────────────────────────────

/** Cap on entries per length-bucket: a single very common size cannot
 *  hoard more than this many free buffers. */
const MAX_BUCKET_SIZE = 64;

/** Cap on total pooled bytes. New disposals beyond this are dropped to
 *  GC instead of being pooled. 64 MB. */
const MAX_BYTES_HELD = 64 * 1024 * 1024;

const pool64: Map<number, Float64Array[]> = new Map();
// In float64 mode, FloatXArray IS Float64Array, so the FloatX pool aliases
// pool64 — both code paths share buffers. In float32 mode they're separate.
const poolX: Map<number, FloatXInstance[]> = FLOATX_IS_FLOAT64
  ? (pool64 as unknown as Map<number, FloatXInstance[]>)
  : new Map<number, FloatXInstance[]>();

/** Tracks buffers currently in the pool. Disposing a buffer that's
 *  already here is a hard error — it means two paths in the runtime
 *  thought they uniquely owned the buffer. WeakSet auto-cleans entries
 *  when the buffer is GC'd (e.g. after clearPool drops bucket
 *  references and no other holder remains). */
const _disposed = new WeakSet<
  Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>
>();

class DoubleDisposeError extends Error {
  constructor(n: number) {
    super(
      `Buffer (length ${n}) disposed twice — the runtime aliased a buffer it thought it uniquely owned. Investigate the most recent change to dispose sites.`
    );
    this.name = "DoubleDisposeError";
    if (
      typeof process !== "undefined" &&
      process.env?.NUMBL_TRACE_DOUBLE_DISPOSE
    ) {
      console.error(this.stack);
    }
  }
}

// ── Tally ─────────────────────────────────────────────────────────────

export interface AllocStats {
  /** Number of allocator calls that returned a fresh-or-recycled buffer
   *  (zero-length skipped). */
  allocCount: number;
  /** Total bytes returned across all allocator calls. */
  allocBytes: number;
  /** Number of dispose calls (zero-length skipped). */
  disposeCount: number;
  /** Total bytes passed to dispose. */
  disposeBytes: number;
  /** Allocator calls that drew from the pool. */
  poolHits: number;
  /** Allocator calls that fell back to a fresh allocation. */
  poolMisses: number;
  /** Number of buffers currently held in the pool. */
  poolBuffersHeld: number;
  /** Total bytes currently held in the pool. */
  poolBytesHeld: number;
}

let _allocCount = 0;
let _allocBytes = 0;
let _disposeCount = 0;
let _disposeBytes = 0;
let _poolHits = 0;
let _poolMisses = 0;
let _poolBuffersHeld = 0;
let _poolBytesHeld = 0;

/** Snapshot of current allocation totals. Cheap; safe to call often. */
export function getAllocStats(): AllocStats {
  return {
    allocCount: _allocCount,
    allocBytes: _allocBytes,
    disposeCount: _disposeCount,
    disposeBytes: _disposeBytes,
    poolHits: _poolHits,
    poolMisses: _poolMisses,
    poolBuffersHeld: _poolBuffersHeld,
    poolBytesHeld: _poolBytesHeld,
  };
}

/** Reset all counters to zero. The pool itself is left intact — buffers
 *  already held remain available for future allocs. Use `clearPool()` to
 *  also drop pooled buffers. */
export function resetAllocStats(): void {
  _allocCount = 0;
  _allocBytes = 0;
  _disposeCount = 0;
  _disposeBytes = 0;
  _poolHits = 0;
  _poolMisses = 0;
  // _poolBuffersHeld / _poolBytesHeld track the live pool; do NOT zero.
}

/** Drop every buffer currently held in the pool and zero the
 *  buffers-held counters. Used between benchmark runs to start cold. */
export function clearPool(): void {
  pool64.clear();
  if (!FLOATX_IS_FLOAT64) poolX.clear();
  _poolBuffersHeld = 0;
  _poolBytesHeld = 0;
}

// ── Pool helpers ─────────────────────────────────────────────────────

function popFromPool64(n: number): Float64Array | undefined {
  const bucket = pool64.get(n);
  if (!bucket || bucket.length === 0) return undefined;
  const buf = bucket.pop()!;
  _disposed.delete(buf);
  _poolBuffersHeld--;
  _poolBytesHeld -= n * 8;
  _poolHits++;
  return buf;
}

function popFromPoolX(n: number): FloatXInstance | undefined {
  const bucket = poolX.get(n);
  if (!bucket || bucket.length === 0) return undefined;
  const buf = bucket.pop()!;
  _disposed.delete(buf);
  _poolBuffersHeld--;
  _poolBytesHeld -= n * FLOATX_BYTES;
  _poolHits++;
  return buf;
}

// ── Uninitialized ─────────────────────────────────────────────────────

/** Allocate an uninitialized Float64Array of length `n`. Caller MUST
 *  write every element before reading it. */
export function allocFloat64(n: number): Float64Array<ArrayBuffer> {
  if (n === 0) return new Float64Array(0);
  _allocCount++;
  _allocBytes += n * 8;
  const pooled = popFromPool64(n);
  if (pooled) return pooled as Float64Array<ArrayBuffer>;
  _poolMisses++;
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
  _allocCount++;
  _allocBytes += n * FLOATX_BYTES;
  const pooled = popFromPoolX(n);
  if (pooled) return pooled;
  _poolMisses++;
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
  _allocCount++;
  _allocBytes += n * 8;
  const pooled = popFromPool64(n);
  if (pooled) {
    pooled.fill(0);
    return pooled as Float64Array<ArrayBuffer>;
  }
  _poolMisses++;
  return new Float64Array(n);
}

/** Allocate a zero-filled FloatXArray of length `n`. */
export function zeroedFloatX(n: number): FloatXInstance {
  if (n === 0) return new FloatXArray(0) as FloatXInstance;
  _allocCount++;
  _allocBytes += n * FLOATX_BYTES;
  const pooled = popFromPoolX(n);
  if (pooled) {
    pooled.fill(0);
    return pooled;
  }
  _poolMisses++;
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

// ── Dispose ───────────────────────────────────────────────────────────

/** Return a Float64Array to the pool. Caller asserts no other reference
 *  to the buffer is live — handing the same buffer twice or while it is
 *  still aliased corrupts the pool.
 *
 *  Poisons the buffer with NaN before pooling. Any caller that still
 *  holds a stale reference and reads from it will see NaN propagate
 *  into downstream computations / asserts, surfacing the use-after-
 *  dispose bug instead of silently producing wrong numbers. The next
 *  `zeroed`/`copy` alloc overwrites the poison; the uninitialized
 *  `alloc` contractually requires the caller to write before reading,
 *  so a conformant caller never observes the NaN. */
export function disposeFloat64(buf: Float64Array<ArrayBufferLike>): void {
  const n = buf.length;
  if (n === 0) return;
  if (_disposed.has(buf)) throw new DoubleDisposeError(n);
  _disposeCount++;
  _disposeBytes += n * 8;
  buf.fill(NaN);
  if (_poolBytesHeld >= MAX_BYTES_HELD) return;
  let bucket = pool64.get(n);
  if (!bucket) {
    bucket = [];
    pool64.set(n, bucket);
  }
  if (bucket.length >= MAX_BUCKET_SIZE) return;
  bucket.push(buf);
  _disposed.add(buf);
  _poolBuffersHeld++;
  _poolBytesHeld += n * 8;
}

/** Return a FloatXArray to the pool. Poisons with NaN — see
 *  `disposeFloat64` for rationale. */
export function disposeFloatX(
  buf: Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>
): void {
  if (FLOATX_IS_FLOAT64) {
    disposeFloat64(buf as Float64Array<ArrayBufferLike>);
    return;
  }
  const n = buf.length;
  if (n === 0) return;
  if (_disposed.has(buf)) throw new DoubleDisposeError(n);
  _disposeCount++;
  _disposeBytes += n * FLOATX_BYTES;
  buf.fill(NaN);
  if (_poolBytesHeld >= MAX_BYTES_HELD) return;
  let bucket = poolX.get(n);
  if (!bucket) {
    bucket = [];
    poolX.set(n, bucket);
  }
  if (bucket.length >= MAX_BUCKET_SIZE) return;
  bucket.push(buf as FloatXInstance);
  _disposed.add(buf);
  _poolBuffersHeld++;
  _poolBytesHeld += n * FLOATX_BYTES;
}
