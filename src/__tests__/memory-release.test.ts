/**
 * End-to-end tests that the new release-on-overwrite + pool wiring
 * actually returns buffers to the pool when expected.
 *
 * Strategy: run a snippet through executeCode (which constructs a Runtime
 * and installs its pool as the active pool), then read getActivePool() to
 * inspect bucket sizes.
 *
 * The interpreter's `rt.share` introduces a small over-count for
 * fresh-from-RHS bindings, which prevents some user-variable buffers
 * from ever reaching `_refs.c == 0`. The COW writer's *copies* don't
 * have that problem (they're created with `_refs: { c: 1 }`), so the
 * tests below drive pool fills primarily through COW-rebind loops.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  BufferPool,
  getActivePool,
  setActivePool,
  releaseTensor,
  releaseIfTensor,
  retainIfTensor,
  assignReleasing,
} from "../numbl-core/runtime/bufferPool.js";
import { uninitFloat64 } from "../numbl-core/runtime/alloc.js";
import { RTV } from "../numbl-core/runtime/constructors.js";
import {
  isRuntimeTensor,
  type RuntimeTensor,
} from "../numbl-core/runtime/types.js";

describe("release helpers", () => {
  let pool: BufferPool;
  beforeEach(() => {
    pool = new BufferPool();
    setActivePool(pool);
  });

  it("releaseTensor pools the data buffer when refcount hits zero", () => {
    const t = RTV.tensor(uninitFloat64(64), [8, 8]) as RuntimeTensor;
    expect(t._refs.c).toBe(1);
    releaseTensor(t);
    expect(t._refs.c).toBe(0);
    expect(pool.bucketSize("f64", 64)).toBe(1);
  });

  it("releaseTensor with refcount > 1 only decrements; does not pool", () => {
    const t = RTV.tensor(uninitFloat64(32), [4, 8]) as RuntimeTensor;
    t._refs.c = 3; // simulate two extra aliases
    releaseTensor(t);
    expect(t._refs.c).toBe(2);
    expect(pool.bucketSize("f64", 32)).toBe(0);
  });

  it("releaseTensor pools both data and imag for complex tensors", () => {
    const data = uninitFloat64(16);
    const imag = uninitFloat64(16);
    const t: RuntimeTensor = {
      kind: "tensor",
      data,
      imag,
      shape: [4, 4],
      _refs: { c: 1 },
    };
    releaseTensor(t);
    expect(pool.bucketSize("f64", 16)).toBe(2); // data + imag
  });

  it("releaseIfTensor is a no-op for non-tensors", () => {
    releaseIfTensor(42);
    releaseIfTensor("hello");
    releaseIfTensor(undefined);
    releaseIfTensor(null);
    releaseIfTensor({ kind: "cell", data: [], shape: [0, 0], _rc: 1 });
    expect(pool.totalBytes()).toBe(0);
  });

  it("retainIfTensor bumps the shared refcount", () => {
    const t = RTV.tensor(uninitFloat64(8), [1, 8]) as RuntimeTensor;
    expect(t._refs.c).toBe(1);
    retainIfTensor(t);
    expect(t._refs.c).toBe(2);
    retainIfTensor(t);
    expect(t._refs.c).toBe(3);
  });

  it("retainIfTensor is a no-op for non-tensors", () => {
    retainIfTensor(42);
    retainIfTensor(undefined);
    // No assertion — just verifying no throw.
  });

  it("assignReleasing releases prev only when prev !== next", () => {
    const a = RTV.tensor(uninitFloat64(4), [1, 4]) as RuntimeTensor;
    const b = RTV.tensor(uninitFloat64(4), [1, 4]) as RuntimeTensor;
    expect(a._refs.c).toBe(1);
    expect(b._refs.c).toBe(1);

    // Same wrapper: no release.
    const ret1 = assignReleasing(a, a);
    expect(ret1).toBe(a);
    expect(a._refs.c).toBe(1);

    // Different wrapper: prev released.
    const ret2 = assignReleasing(a, b);
    expect(ret2).toBe(b);
    expect(a._refs.c).toBe(0);
    expect(b._refs.c).toBe(1);
    expect(pool.bucketSize("f64", 4)).toBe(1);
  });

  it("assignReleasing skips release when prev is undefined", () => {
    const a = RTV.tensor(uninitFloat64(4), [1, 4]) as RuntimeTensor;
    const ret = assignReleasing(undefined, a);
    expect(ret).toBe(a);
    expect(a._refs.c).toBe(1);
  });

  it("two refs shared via shared header drop independently to zero", () => {
    const t = RTV.tensor(uninitFloat64(20), [4, 5]) as RuntimeTensor;
    // Manually create a second wrapper aliasing the same buffer & header.
    const aliased: RuntimeTensor = {
      kind: "tensor",
      data: t.data,
      shape: t.shape,
      _refs: t._refs,
    };
    aliased._refs.c++;

    expect(t._refs.c).toBe(2);
    releaseTensor(t);
    expect(aliased._refs.c).toBe(1);
    expect(pool.bucketSize("f64", 20)).toBe(0);

    releaseTensor(aliased);
    expect(pool.bucketSize("f64", 20)).toBe(1);
  });
});

describe("end-to-end pool reuse via executeCode", () => {
  it("COW-rebind loop fills the pool with copies of the COW'd length", () => {
    // y = x; y(i) = ... — the COW path allocates a new length-N buffer
    // each iteration; the previous COW copy is released by env.set when
    // y is rebound on the next iteration. The pool collects those copies.
    //
    // Run in interpreter mode: PR 3's JIT release wiring lands separately,
    // so under JIT the loop's local rebinds don't yet drop refs.
    const result = executeCode(
      `
      x = (1:1000);
      for i = 1:50
        y = x;
        y(i) = 99;
      end
    `,
      { optimization: "0" }
    );
    expect(result.variableValues["x"]).toBeDefined();
    const pool = getActivePool();
    expect(pool.bucketSize("f64", 1000)).toBeGreaterThan(0);
  });

  it("pool returns the same buffer on the next acquire of matching length", () => {
    const result = executeCode(
      `
      x = (1:500);
      y = x;
      y(1) = 99;
      y = x;
      y(1) = 99;
    `,
      { optimization: "0" }
    );
    expect(result.variableValues["y"]).toBeDefined();
    const pool = getActivePool();
    expect(pool.bucketSize("f64", 500)).toBeGreaterThan(0);

    const before = pool.bucketSize("f64", 500);
    const buf = pool.acquireF64(500);
    expect(buf.length).toBe(500);
    expect(pool.bucketSize("f64", 500)).toBe(before - 1);
  });

  it("repeated COW-rebind in a script-level loop pools all but the live copy", () => {
    // Each iteration: y = x (alias), then y(...) = .. triggers a COW copy
    // with fresh `_refs: { c: 1 }`. The next iteration's `y = x` rebind
    // sees prior=copy (c=1) and releases it via env.set → c=0 → pooled.
    // After N iterations, all but the most recent copy are in the pool.
    const N = 50;
    const result = executeCode(
      `
      x = (1:200);
      for k = 1:${N}
        y = x;
        y(k) = -k;
      end
    `,
      { optimization: "0" }
    );
    expect(result.variableValues["x"]).toBeDefined();
    const pool = getActivePool();
    expect(pool.bucketSize("f64", 200)).toBe(N - 1);
  });

  it("a fresh runtime starts with an empty pool (test isolation)", () => {
    // Build a runtime, do nothing with tensors. Pool should be empty.
    executeCode(`x = 1 + 2;`);
    const pool = getActivePool();
    expect(pool.totalBytes()).toBe(0);
  });

  it("debug-mode pool scribbles released buffers with NaN", () => {
    // Switch the active pool to a debug-enabled one so newly-released
    // buffers are filled with NaN. Any subsequent stale read by a caller
    // that retained a wrapper past its release would surface as NaN.
    const debugPool = new BufferPool({ debug: true });
    setActivePool(debugPool);
    const result = executeCode(`
      x = (1:100);
      y = x;
      y(1) = 99;
      y = x;
      y(1) = 99;
    `);
    // The script ran clean — its result is intact.
    expect(result.variableValues["x"]).toBeDefined();
    // And the pool has scribbled buffers ready to expose any future UAF.
    const buckets = debugPool.bucketSize("f64", 100);
    if (buckets > 0) {
      const buf = debugPool.acquireF64(100);
      for (let i = 0; i < buf.length; i++) expect(buf[i]).toBeNaN();
    }
  });
});

describe("snapshot / lambda capture of tensors", () => {
  it("a lambda capturing a tensor still works after the function returns", () => {
    const result = executeCode(`
      f = make_lambda();
      a = f(3);
      b = f(1);
      c = f(5);
      function fn = make_lambda()
        v = [10, 20, 30, 40, 50];
        fn = @(i) v(i);
      end
    `);
    expect(result.variableValues["a"]).toBe(30);
    expect(result.variableValues["b"]).toBe(10);
    expect(result.variableValues["c"]).toBe(50);
  });
});

describe("release on isRuntimeTensor checks", () => {
  it("a freshly allocated tensor wrapper is recognized as a tensor", () => {
    const t = RTV.tensor(uninitFloat64(4), [1, 4]);
    expect(isRuntimeTensor(t)).toBe(true);
  });
});

describe("refcount underflow detection", () => {
  it("releasing a uniquely-owned tensor twice throws", () => {
    const pool = new BufferPool();
    setActivePool(pool);
    const t = RTV.tensor(uninitFloat64(8), [1, 8]) as RuntimeTensor;
    expect(t._refs.c).toBe(1);
    releaseTensor(t); // c -> 0, buffer pooled
    expect(() => releaseTensor(t)).toThrow(/underflow/);
  });

  it("releaseIfTensor on a c=0 wrapper also throws (treated as a double-release bug)", () => {
    const pool = new BufferPool();
    setActivePool(pool);
    const t = RTV.tensor(uninitFloat64(4), [1, 4]) as RuntimeTensor;
    releaseTensor(t); // c -> 0
    expect(() => releaseIfTensor(t)).toThrow(/underflow/);
  });
});

describe("BufferPool telemetry counters", () => {
  let pool: BufferPool;
  beforeEach(() => {
    pool = new BufferPool();
    setActivePool(pool);
  });

  it("counters start at zero", () => {
    const s = pool.stats();
    expect(s.acquireCount).toBe(0);
    expect(s.acquireBytes).toBe(0);
    expect(s.acquireHits).toBe(0);
    expect(s.releaseCount).toBe(0);
    expect(s.releaseBytes).toBe(0);
    expect(s.currentBytes).toBe(0);
  });

  it("acquireF64 increments count and bytes; first call is a miss", () => {
    pool.acquireF64(100);
    const s = pool.stats();
    expect(s.acquireCount).toBe(1);
    expect(s.acquireBytes).toBe(800);
    expect(s.acquireHits).toBe(0);
  });

  it("acquireF32 tracks bytes at 4 per element", () => {
    pool.acquireF32(50);
    const s = pool.stats();
    expect(s.acquireCount).toBe(1);
    expect(s.acquireBytes).toBe(200);
  });

  it("releasing a buffer to a non-empty bucket counts as a release; subsequent acquire is a hit", () => {
    const a = pool.acquireF64(16); // miss
    pool.release(a);
    const b = pool.acquireF64(16); // hit
    expect(b).toBe(a);
    const s = pool.stats();
    expect(s.acquireCount).toBe(2);
    expect(s.acquireHits).toBe(1);
    expect(s.releaseCount).toBe(1);
    expect(s.releaseBytes).toBe(128);
  });

  it("zero-length acquires/releases are not counted", () => {
    pool.acquireF64(0);
    pool.release(new Float64Array(0));
    const s = pool.stats();
    expect(s.acquireCount).toBe(0);
    expect(s.releaseCount).toBe(0);
  });

  it("releases discarded by the bucket cap are still counted as releases (audit trail)", () => {
    const tinyPool = new BufferPool({ maxPerBucket: 1 });
    const a = tinyPool.acquireF64(8);
    const b = tinyPool.acquireF64(8);
    tinyPool.release(a); // pooled
    tinyPool.release(b); // discarded by cap
    const s = tinyPool.stats();
    expect(s.releaseCount).toBe(2);
    expect(s.releaseBytes).toBe(128);
    expect(tinyPool.bucketSize("f64", 8)).toBe(1);
  });

  it("clear() resets bucket state but preserves telemetry counters", () => {
    pool.release(pool.acquireF64(8));
    const before = pool.stats();
    pool.clear();
    const after = pool.stats();
    expect(after.acquireCount).toBe(before.acquireCount);
    expect(after.releaseCount).toBe(before.releaseCount);
    expect(after.currentBytes).toBe(0);
  });

  it("end-to-end: a tensor allocated via uninitFloat64 + releaseTensor leaves clean stats", () => {
    const t = RTV.tensor(uninitFloat64(20), [4, 5]) as RuntimeTensor;
    releaseTensor(t);
    const s = pool.stats();
    expect(s.acquireCount).toBe(1);
    expect(s.releaseCount).toBe(1);
    expect(s.acquireBytes).toBe(s.releaseBytes);
  });
});
