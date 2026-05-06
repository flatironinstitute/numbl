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

  it("nothing is released or pooled without explicit release()", () => {
    const result = executeCode("for k=1:50; x = zeros(100, 1); end");
    const s = result.memoryStats!;
    // No automatic reclamation: every alloc goes through `new Float64Array`.
    expect(s.cacheHits).toBe(0);
    expect(s.releases).toBe(0);
    expect(s.freePoolBufferCount).toBe(0);
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
