/**
 * Float64Array memory pool — alloc-tracking only, no sweep.
 *
 * Every Float64Array handed out by `allocFloat64Array` is tagged in
 * `liveSet` (a WeakSet so V8 can still GC unreleased buffers) and a
 * `_liveCount` counter is bumped. The pool exposes a `release(buf)`
 * method that moves a buffer to `freePool` (one bucket per size, unbounded);
 * subsequent `acquire(N)` calls pop from the bucket instead of allocating
 * fresh. Nothing currently calls `release` automatically — sweep-based
 * reclamation is intentionally absent in this iteration.
 *
 * Why WeakSet for liveSet: a strong-ref Set would pin every acquired
 * buffer for the runtime's lifetime, defeating V8 GC for any buffer
 * whose wrapper is collected without `_destroy` calling `release`
 * (refcount edge cases). The WeakSet still answers `has(buf)` correctly
 * for any buffer the caller hands back to `release`, since the caller
 * holds it live across the call. The numeric `_liveCount` keeps telemetry
 * (IDE Memory tab + refcount-balance tests) honest — it tracks
 * acquires-minus-releases, so a leak (acquire without release) shows up
 * as a positive count even after V8 reclaims the buffer.
 *
 * Per-runtime: each Runtime owns its own pool. A module-level stack tracks
 * the *currently active* runtime so `allocFloat64Array` can find it
 * without threading rt through every callsite.
 */

// ── Runtime stack ────────────────────────────────────────────────────────

import type { RefcountRuntime } from "./refcount.js";

/** Minimal shape needed to find the active pool + transient scope.
 *  `Runtime` (and any test stub) must satisfy this; it intentionally
 *  matches `RefcountRuntime` so refcount-aware code can call
 *  `getCurrentRuntime()` and use the result without casting. */
type PoolHolder = RefcountRuntime;

const runtimeStack: PoolHolder[] = [];

export function pushCurrentRuntime(rt: PoolHolder): void {
  runtimeStack.push(rt);
}

export function popCurrentRuntime(rt: PoolHolder): void {
  const top = runtimeStack[runtimeStack.length - 1];
  if (top !== rt) {
    throw new Error(
      "memoryPool: popCurrentRuntime mismatch — push/pop misnested"
    );
  }
  runtimeStack.pop();
}

export function getCurrentRuntime(): PoolHolder | null {
  const top = runtimeStack[runtimeStack.length - 1];
  return top ?? null;
}

// ── Stats ────────────────────────────────────────────────────────────────

/** Per-size bucket telemetry. `count` reflects current free-list depth;
 *  the four counters are cumulative across the run. */
export interface MemoryPoolBucket {
  size: number;
  /** Currently-pooled (released, awaiting reuse) buffers at this size. */
  count: number;
  /** Total `acquire` / `acquireFrom` calls for this size. */
  attempts: number;
  /** Subset of attempts served from the free list. */
  cacheHits: number;
  /** Subset of attempts that fell through to `new Float64Array`. */
  news: number;
  /** Total `release` calls for this size. */
  releases: number;
}

export interface MemoryPoolStats {
  // counts
  attemptedAllocs: number;
  actualAllocs: number;
  cacheHits: number;
  releases: number;
  // bytes (length * 8)
  attemptedBytes: number;
  actualAllocBytes: number;
  cacheHitBytes: number;
  releaseBytes: number;
  // current state
  liveSetSize: number;
  freePoolBufferCount: number;
  freePoolBytes: number;
  freePoolBuckets: MemoryPoolBucket[];
}

/** Recursively collect every Float64Array reachable from `v` into `out`.
 *  Used by `MemoryPool.withScratch` to find which buffers should survive
 *  the scratch-release pass.
 *
 *  Walks: Float64Array (collect), Array, Map, plain objects.
 *  For runtime container classes (RuntimeTensor, RuntimeCell, etc.),
 *  walks well-known `data`/`imag`/`fields`/`captures`/`elements` slots.
 *  Tracks visited objects so cycles in the runtime graph terminate. */
function collectFloat64Arrays(
  v: unknown,
  out: Set<Float64Array>,
  visited?: Set<object>
): void {
  if (v === null || v === undefined) return;
  if (v instanceof Float64Array) {
    out.add(v);
    return;
  }
  if (typeof v !== "object") return;
  const seen = visited ?? new Set<object>();
  if (seen.has(v as object)) return;
  seen.add(v as object);
  if (Array.isArray(v)) {
    for (const item of v) collectFloat64Arrays(item, out, seen);
    return;
  }
  if (v instanceof Map) {
    for (const val of v.values()) collectFloat64Arrays(val, out, seen);
    return;
  }
  if (v instanceof Set) {
    for (const val of v) collectFloat64Arrays(val, out, seen);
    return;
  }
  // Plain object: walk own values.
  if (Object.getPrototypeOf(v) === Object.prototype) {
    for (const key of Object.keys(v)) {
      collectFloat64Arrays((v as Record<string, unknown>)[key], out, seen);
    }
    return;
  }
  // Runtime container classes — walk known buffer-bearing slots.
  const r = v as Record<string, unknown>;
  if (r.data !== undefined) collectFloat64Arrays(r.data, out, seen);
  if (r.imag !== undefined) collectFloat64Arrays(r.imag, out, seen);
  if (r.pr !== undefined) collectFloat64Arrays(r.pr, out, seen);
  if (r.pi !== undefined) collectFloat64Arrays(r.pi, out, seen);
  if (r.fields !== undefined) collectFloat64Arrays(r.fields, out, seen);
  if (r.elements !== undefined) collectFloat64Arrays(r.elements, out, seen);
  if (r.captures !== undefined) collectFloat64Arrays(r.captures, out, seen);
  if (r._builtinData !== undefined)
    collectFloat64Arrays(r._builtinData, out, seen);
}

// ── MemoryPool ───────────────────────────────────────────────────────────

/** Mutable per-size counter bag, kept in `_bucketStats`. */
interface BucketCounters {
  attempts: number;
  cacheHits: number;
  news: number;
  releases: number;
}

export class MemoryPool {
  /** Weak membership set — answers "did this pool hand out `buf`?" without
   *  pinning the buffer. Cannot be enumerated and has no `.size`; for the
   *  live-buffer count, see `_liveCount`. */
  private liveSet = new WeakSet<Float64Array>();
  /** Acquires minus releases. Reports as `liveSetSize` in stats. */
  private _liveCount = 0;
  private freePool = new Map<number, Float64Array[]>();
  /** Stack of scratch trackers. While a tracker is active, every fresh
   *  `acquire` registers in the topmost tracker. Used by `withScratch`
   *  to auto-release buffers allocated by deep internals (LAPACK
   *  workspaces, etc.) that aren't returned to the caller. */
  private scratchStack: Set<Float64Array>[] = [];
  /** Per-size cumulative counters. Survives even after a size's free
   *  bucket empties out, so the report can show "size N had X attempts"
   *  even when no buffers of that size are currently pooled. */
  private _bucketStats = new Map<number, BucketCounters>();

  // Counters (never reset within an execution).
  private _attemptedAllocs = 0;
  private _actualAllocs = 0;
  private _cacheHits = 0;
  private _releases = 0;
  private _attemptedBytes = 0;
  private _actualAllocBytes = 0;
  private _cacheHitBytes = 0;
  private _releaseBytes = 0;
  private _freePoolBufferCount = 0;
  private _freePoolBytes = 0;

  private bucketCounters(size: number): BucketCounters {
    let c = this._bucketStats.get(size);
    if (!c) {
      c = { attempts: 0, cacheHits: 0, news: 0, releases: 0 };
      this._bucketStats.set(size, c);
    }
    return c;
  }

  /** Allocate (or reuse) a zero-filled Float64Array of length `size`. */
  acquire(size: number): Float64Array {
    const bytes = 8 * size;
    this._attemptedAllocs++;
    this._attemptedBytes += bytes;
    const bc = this.bucketCounters(size);
    bc.attempts++;

    const bucket = this.freePool.get(size);
    let buf: Float64Array;
    if (bucket && bucket.length > 0) {
      buf = bucket.pop()!;
      buf.fill(0);
      this._cacheHits++;
      this._cacheHitBytes += bytes;
      this._freePoolBufferCount--;
      this._freePoolBytes -= bytes;
      bc.cacheHits++;
    } else {
      buf = new Float64Array(size);
      this._actualAllocs++;
      this._actualAllocBytes += bytes;
      bc.news++;
    }
    this.liveSet.add(buf);
    this._liveCount++;
    const top = this.scratchStack[this.scratchStack.length - 1];
    if (top) top.add(buf);
    return buf;
  }

  /** Allocate a Float64Array containing `src`'s values (copies). */
  acquireFrom(src: number[] | Float64Array): Float64Array {
    const n = src.length;
    const bytes = 8 * n;
    this._attemptedAllocs++;
    this._attemptedBytes += bytes;
    const bc = this.bucketCounters(n);
    bc.attempts++;

    const bucket = this.freePool.get(n);
    let buf: Float64Array;
    if (bucket && bucket.length > 0) {
      buf = bucket.pop()!;
      buf.set(src);
      this._cacheHits++;
      this._cacheHitBytes += bytes;
      this._freePoolBufferCount--;
      this._freePoolBytes -= bytes;
      bc.cacheHits++;
    } else {
      buf = new Float64Array(src);
      this._actualAllocs++;
      this._actualAllocBytes += bytes;
      bc.news++;
    }
    this.liveSet.add(buf);
    this._liveCount++;
    const top = this.scratchStack[this.scratchStack.length - 1];
    if (top) top.add(buf);
    return buf;
  }

  /**
   * Return a buffer to the free pool. Caller asserts no live wrapper still
   * references this buffer; misuse corrupts data.
   *
   * Released buffers are poisoned with NaN so any subsequent read via a
   * stale wrapper (use-after-free) surfaces as NaN rather than silently
   * matching the zero-fill that `acquire` performs on reuse. The poison
   * is overwritten on the next `acquire` (zero-fill) or `acquireFrom`
   * (set from src), so live consumers never see it.
   */
  release(buf: Float64Array): void {
    if (!this.liveSet.has(buf)) return;
    this.liveSet.delete(buf);
    this._liveCount--;
    buf.fill(NaN);
    let bucket = this.freePool.get(buf.length);
    if (!bucket) {
      bucket = [];
      this.freePool.set(buf.length, bucket);
    }
    bucket.push(buf);
    const bytes = 8 * buf.length;
    this._releases++;
    this._releaseBytes += bytes;
    this._freePoolBufferCount++;
    this._freePoolBytes += bytes;
    this.bucketCounters(buf.length).releases++;
  }

  /** Run `fn` with a scratch tracker active. Every `acquire` /
   *  `acquireFrom` made during `fn` registers in the tracker; on
   *  return, the result is walked for any Float64Arrays — those are
   *  the buffers the caller wants to keep — and every other tracked
   *  buffer is released back to the pool.
   *
   *  Use this around bridge calls (LAPACK etc.) whose deep internals
   *  allocate workspace buffers that don't escape. Without it, those
   *  workspaces would stay in `liveSet` forever, never reused.
   *
   *  Nested `withScratch` is safe — kept buffers transfer to the
   *  parent's tracker so the outer scope can manage them. */
  withScratch<T>(fn: () => T): T {
    const tracker = new Set<Float64Array>();
    this.scratchStack.push(tracker);
    let result: T;
    try {
      result = fn();
    } finally {
      this.scratchStack.pop();
    }
    // Walk result to collect every Float64Array that should survive.
    const keep = new Set<Float64Array>();
    collectFloat64Arrays(result, keep);
    const outer = this.scratchStack[this.scratchStack.length - 1];
    for (const buf of tracker) {
      if (keep.has(buf)) {
        // Returned to caller — hand off to the outer tracker (if any)
        // so the caller's scope can manage release.
        if (outer) outer.add(buf);
      } else {
        this.release(buf);
      }
    }
    return result;
  }

  /** Snapshot the pool's stats into a plain JSON-serializable object.
   *  Buckets are returned sorted by size descending and capped at 200
   *  (the cap is for display surfaces that paginate poorly). Every size
   *  ever touched by `acquire` / `release` shows up — even if its
   *  current free-list depth is 0 — so the counters tell the full
   *  history. */
  getStats(): MemoryPoolStats {
    const buckets: MemoryPoolBucket[] = [];
    for (const [size, c] of this._bucketStats) {
      const free = this.freePool.get(size);
      buckets.push({
        size,
        count: free ? free.length : 0,
        attempts: c.attempts,
        cacheHits: c.cacheHits,
        news: c.news,
        releases: c.releases,
      });
    }
    buckets.sort((a, b) => b.size - a.size);
    const capped = buckets.length > 200 ? buckets.slice(0, 200) : buckets;

    return {
      attemptedAllocs: this._attemptedAllocs,
      actualAllocs: this._actualAllocs,
      cacheHits: this._cacheHits,
      releases: this._releases,
      attemptedBytes: this._attemptedBytes,
      actualAllocBytes: this._actualAllocBytes,
      cacheHitBytes: this._cacheHitBytes,
      releaseBytes: this._releaseBytes,
      liveSetSize: this._liveCount,
      freePoolBufferCount: this._freePoolBufferCount,
      freePoolBytes: this._freePoolBytes,
      freePoolBuckets: capped,
    };
  }
}
