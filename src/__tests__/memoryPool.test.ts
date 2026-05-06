import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { allocFloat64Array } from "../numbl-core/executors/jsJit/helpers/alloc.js";
import {
  getCurrentRuntime,
  MemoryPool,
} from "../numbl-core/runtime/memoryPool.js";

describe("MemoryPool", () => {
  it("falls back to plain new Float64Array when no runtime is active", () => {
    expect(getCurrentRuntime()).toBeNull();
    const a = allocFloat64Array(10);
    expect(a).toBeInstanceOf(Float64Array);
    expect(a.length).toBe(10);
    a[0] = 1.5;
    expect(a[0]).toBe(1.5);
  });

  it("populates memoryStats on a basic run", () => {
    const result = executeCode("x = zeros(100, 1);");
    expect(result.memoryStats).toBeDefined();
    const s = result.memoryStats!;
    expect(s.attemptedAllocs).toBeGreaterThan(0);
    expect(s.attemptedBytes).toBe(s.actualAllocBytes + s.cacheHitBytes);
    expect(s.attemptedAllocs).toBe(s.actualAllocs + s.cacheHits);
  });

  it("counters: attemptedBytes equals actual + cacheHit bytes", () => {
    const result = executeCode(
      "for k=1:30; x = zeros(20, 1); y = zeros(40, 1); end"
    );
    const s = result.memoryStats!;
    expect(s.attemptedBytes).toBe(s.actualAllocBytes + s.cacheHitBytes);
    expect(s.attemptedAllocs).toBe(s.actualAllocs + s.cacheHits);
  });

  it("refcount-driven reclamation: a discarded transient is released", () => {
    // The intermediate `zeros(100, 1)` is bound to env.x then immediately
    // overwritten with a different size, so its 100-length buffer drops
    // to rc=0 on the second statement and is released to the pool.
    const result = executeCode(
      "x = zeros(100, 1); x = zeros(50, 1); clear x;",
      { optimization: "0" }
    );
    const s = result.memoryStats!;
    expect(s.releases).toBeGreaterThan(0);
    // Eventually the pool's freePool holds buffers for reuse.
    expect(s.freePoolBufferCount + s.cacheHits).toBeGreaterThan(0);
  });

  it("refcount-driven reclamation: in a tight loop (interpreter), buffers are reused", () => {
    // With opt 0 (interpreter), each iteration is its own statement-level
    // scope. The previous iteration's tensor is decref'd to 0 and pooled
    // before the next iteration runs, so most allocations should hit the
    // pool's free buckets.
    const result = executeCode("for k=1:50; x = zeros(100, 1); end", {
      optimization: "0",
    });
    const s = result.memoryStats!;
    expect(s.releases).toBeGreaterThanOrEqual(40);
    expect(s.cacheHits).toBeGreaterThanOrEqual(40);
  });

  it("refcount-driven reclamation: JIT top-level releases after the synthetic fn returns", () => {
    // The JIT-compiled top-level wraps the whole script in one scope, so
    // mid-loop releases don't happen, but at scope drain every dead
    // wrapper's buffer is reclaimed.
    const result = executeCode("for k=1:50; x = zeros(100, 1); end", {
      optimization: "1",
    });
    const s = result.memoryStats!;
    expect(s.releases).toBeGreaterThanOrEqual(40);
  });

  it("manual release() puts a buffer in the free pool, next acquire reuses it", () => {
    const pool = new MemoryPool();
    const a = pool.acquire(7);
    expect(pool.getStats().liveSetSize).toBe(1);
    pool.release(a);
    expect(pool.getStats().freePoolBufferCount).toBe(1);
    const b = pool.acquire(7);
    expect(b).toBe(a); // identity reuse
    const s = pool.getStats();
    expect(s.cacheHits).toBe(1);
    expect(s.actualAllocs).toBe(1);
    expect(s.attemptedAllocs).toBe(2);
  });

  it("release() zero-fills via subsequent acquire, not on release", () => {
    const pool = new MemoryPool();
    const a = pool.acquire(4);
    a.set([1, 2, 3, 4]);
    pool.release(a);
    const b = pool.acquire(4);
    expect(b).toBe(a);
    expect(Array.from(b)).toEqual([0, 0, 0, 0]);
  });

  it("acquireFrom copies the source", () => {
    const pool = new MemoryPool();
    const buf = pool.acquireFrom([1, 2, 3]);
    expect(Array.from(buf)).toEqual([1, 2, 3]);
    expect(buf).toBeInstanceOf(Float64Array);
  });

  it("freePoolBuckets reflects bucket sizes", () => {
    const pool = new MemoryPool();
    const a = pool.acquire(3);
    const b = pool.acquire(3);
    const c = pool.acquire(5);
    pool.release(a);
    pool.release(b);
    pool.release(c);
    const s = pool.getStats();
    const bySize = new Map(s.freePoolBuckets.map(x => [x.size, x.count]));
    expect(bySize.get(3)).toBe(2);
    expect(bySize.get(5)).toBe(1);
  });

  it("pool isolation: after run completes, runtime stack is empty", () => {
    expect(getCurrentRuntime()).toBeNull();
    executeCode("x = 1;");
    expect(getCurrentRuntime()).toBeNull();
  });
});

// ── COW / aliasing equivalence ──────────────────────────────────────────
//
// Sanity check that the rootWalker-based aliasing still flags the obvious
// cell/struct sharing cases. The integration suite covers more.

describe("aliasing (post-walker rework)", () => {
  it("indexed mutation of a cell-stored tensor preserves the original", () => {
    const result = executeCode("a = [1 2 3]; c = {a}; c{1}(2) = 99; b = c{1};");
    const a = result.variableValues["a"];
    const b = result.variableValues["b"];
    expect((a as { kind: string }).kind).toBe("tensor");
    expect((b as { kind: string }).kind).toBe("tensor");
    expect((a as { data: Float64Array }).data[1]).toBe(2);
    expect((b as { data: Float64Array }).data[1]).toBe(99);
  });

  it("indexed mutation of a struct-stored tensor preserves the original", () => {
    const result = executeCode("a = [1 2 3]; s.x = a; s.x(2) = 99; b = s.x;");
    const a = result.variableValues["a"];
    const b = result.variableValues["b"];
    expect((a as { data: Float64Array }).data[1]).toBe(2);
    expect((b as { data: Float64Array }).data[1]).toBe(99);
  });
});
