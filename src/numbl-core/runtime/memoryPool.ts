/**
 * Float64Array memory pool — alloc-tracking only, no sweep.
 *
 * Every Float64Array handed out by `allocFloat64Array` is recorded in
 * `liveSet` and counters are bumped. The pool exposes a `release(buf)`
 * method that moves a buffer to `freePool` (one bucket per size, unbounded);
 * subsequent `acquire(N)` calls pop from the bucket instead of allocating
 * fresh. Nothing currently calls `release` automatically — sweep-based
 * reclamation is intentionally absent in this iteration.
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

// ── MemoryPool ───────────────────────────────────────────────────────────

/** Mutable per-size counter bag, kept in `_bucketStats`. */
interface BucketCounters {
  attempts: number;
  cacheHits: number;
  news: number;
  releases: number;
}

export class MemoryPool {
  private liveSet = new Set<Float64Array>();
  private freePool = new Map<number, Float64Array[]>();
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
      liveSetSize: this.liveSet.size,
      freePoolBufferCount: this._freePoolBufferCount,
      freePoolBytes: this._freePoolBytes,
      freePoolBuckets: capped,
    };
  }
}
