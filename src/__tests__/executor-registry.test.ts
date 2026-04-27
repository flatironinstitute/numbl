import { describe, it, expect } from "vitest";
import { Registry, makeRootContext } from "../numbl-core/executors/registry.js";
import { interpreterExecutor } from "../numbl-core/executors/interpreter/interpreterExecutor.js";
import { executeCode } from "../numbl-core/executeCode.js";
import type {
  Executor,
  MatchResult,
  RunResult,
} from "../numbl-core/executors/types.js";
import type { Stmt } from "../numbl-core/parser/types.js";

/**
 * Skeleton smoke tests for the executor registry. The registry is not
 * yet wired into `Interpreter.execStmt`; these tests exercise the
 * dispatcher's plumbing in isolation:
 *
 *   - cost-based candidate selection
 *   - bail → invalidate cache, try next candidate
 *   - requireNoBail filters bail-risk executors
 *   - reentrancy guard
 *
 * Per-stmt correctness against the AST interpreter lands once
 * specialized executors are ported.
 */

interface StubMatch {
  tag: string;
}

interface StubExecutorOpts {
  name: string;
  bailRisk?: boolean;
  cost: { compileMs: number; perCallNs: number; runNs: number };
  /** When set, run() returns this bail. */
  bail?: { message: string };
  /** When true, bail is reported as transient (not cached). */
  bailTransient?: boolean;
  /** Counter incremented every time run() is entered. */
  runCount?: { n: number };
  /** Counter incremented every time compile() is entered. */
  compileCount?: { n: number };
  requireNoBailInChildren?: boolean;
}

function stubExecutor(opts: StubExecutorOpts): Executor<StubMatch, StubMatch> {
  return {
    name: opts.name,
    bailRisk: !!opts.bailRisk,
    match(): MatchResult<StubMatch> {
      return {
        match: { tag: opts.name },
        cost: opts.cost,
        ...(opts.requireNoBailInChildren
          ? { requireNoBailInChildren: true }
          : {}),
      };
    },
    cacheKey(m) {
      return m.tag;
    },
    compile(m) {
      if (opts.compileCount) opts.compileCount.n++;
      return m;
    },
    run(): RunResult {
      if (opts.runCount) opts.runCount.n++;
      if (opts.bail) {
        return {
          bail: { message: opts.bail.message },
          ...(opts.bailTransient ? { transient: true } : {}),
        };
      }
      return { consumed: 1 };
    },
  };
}

/** A fake stmt is fine for these tests — the registry doesn't inspect
 *  Stmt fields, just uses identity. */
function fakeStmt(label = "stmt"): Stmt {
  return { type: label } as unknown as Stmt;
}

/** Make a context that doesn't touch the interpreter. */
function fakeCtx(registry: Registry, requireNoBail = false) {
  return makeRootContext(
    // The registry's dispatch path doesn't read `interp` for these
    // stub-only tests; a casted shim is enough.
    null as never,
    registry
  ).childContext(requireNoBail);
}

describe("Registry plumbing", () => {
  it("registers the interpreter executor under the standard plugin", () => {
    const r = new Registry();
    r.register(interpreterExecutor);
    expect(r.size).toBe(1);
  });

  it("rejects duplicate registrations", () => {
    const r = new Registry();
    r.register(interpreterExecutor);
    expect(() => r.register(interpreterExecutor)).toThrow(/already registered/);
  });

  it("picks the lowest per-call cost candidate", () => {
    const fastRuns = { n: 0 };
    const slowRuns = { n: 0 };
    const r = new Registry();
    r.register(
      stubExecutor({
        name: "slow",
        cost: { compileMs: 0, perCallNs: 1000, runNs: 1000 },
        runCount: slowRuns,
      })
    );
    r.register(
      stubExecutor({
        name: "fast",
        cost: { compileMs: 0, perCallNs: 1, runNs: 1 },
        runCount: fastRuns,
      })
    );

    const stmts = [fakeStmt()];
    r.dispatch(stmts, 0, fakeCtx(r));
    expect(fastRuns.n).toBe(1);
    expect(slowRuns.n).toBe(0);
  });

  it("on bail, invalidates cache and tries the next candidate", () => {
    const fastRuns = { n: 0 };
    const slowRuns = { n: 0 };
    const fastCompiles = { n: 0 };
    const r = new Registry();
    r.register(
      stubExecutor({
        name: "fast-bails",
        cost: { compileMs: 0, perCallNs: 1, runNs: 1 },
        bail: { message: "oops" },
        runCount: fastRuns,
        compileCount: fastCompiles,
      })
    );
    r.register(
      stubExecutor({
        name: "slow-ok",
        cost: { compileMs: 0, perCallNs: 100, runNs: 100 },
        runCount: slowRuns,
      })
    );

    const stmts = [fakeStmt()];
    r.dispatch(stmts, 0, fakeCtx(r));
    expect(fastRuns.n).toBe(1);
    expect(slowRuns.n).toBe(1);

    // Second dispatch on the same stmt should skip fast-bails (cache
    // remembers the bail) and go straight to slow-ok.
    r.dispatch(stmts, 0, fakeCtx(r));
    expect(fastRuns.n).toBe(1); // not retried
    expect(fastCompiles.n).toBe(1); // not recompiled
    expect(slowRuns.n).toBe(2);
  });

  it("requireNoBail context filters bail-risk executors", () => {
    const fastRuns = { n: 0 };
    const slowRuns = { n: 0 };
    const r = new Registry();
    r.register(
      stubExecutor({
        name: "fast-risky",
        bailRisk: true,
        cost: { compileMs: 0, perCallNs: 1, runNs: 1 },
        runCount: fastRuns,
      })
    );
    r.register(
      stubExecutor({
        name: "slow-safe",
        bailRisk: false,
        cost: { compileMs: 0, perCallNs: 100, runNs: 100 },
        runCount: slowRuns,
      })
    );

    const stmts = [fakeStmt()];
    r.dispatch(stmts, 0, fakeCtx(r, /* requireNoBail */ true));
    expect(fastRuns.n).toBe(0);
    expect(slowRuns.n).toBe(1);
  });

  it("compiles once and reuses the cached artifact", () => {
    const compiles = { n: 0 };
    const runs = { n: 0 };
    const r = new Registry();
    r.register(
      stubExecutor({
        name: "cacheable",
        cost: { compileMs: 0, perCallNs: 1, runNs: 1 },
        compileCount: compiles,
        runCount: runs,
      })
    );

    const stmts = [fakeStmt()];
    r.dispatch(stmts, 0, fakeCtx(r));
    r.dispatch(stmts, 0, fakeCtx(r));
    r.dispatch(stmts, 0, fakeCtx(r));
    expect(compiles.n).toBe(1);
    expect(runs.n).toBe(3);
  });

  it("transient bail does not poison the cache", () => {
    const fastRuns = { n: 0 };
    const slowRuns = { n: 0 };
    const r = new Registry();
    r.register(
      stubExecutor({
        name: "fast-bails-transient",
        cost: { compileMs: 0, perCallNs: 1, runNs: 1 },
        bail: { message: "wrapped layer rejected" },
        bailTransient: true,
        runCount: fastRuns,
      })
    );
    r.register(
      stubExecutor({
        name: "slow-ok",
        cost: { compileMs: 0, perCallNs: 100, runNs: 100 },
        runCount: slowRuns,
      })
    );

    const stmts = [fakeStmt()];
    r.dispatch(stmts, 0, fakeCtx(r));
    r.dispatch(stmts, 0, fakeCtx(r));
    // Both dispatches should have re-entered the transient-bailing
    // executor; the cache is not poisoned.
    expect(fastRuns.n).toBe(2);
    expect(slowRuns.n).toBe(2);
  });

  it("clearCache forces recompile on next dispatch", () => {
    const compiles = { n: 0 };
    const r = new Registry();
    r.register(
      stubExecutor({
        name: "cacheable",
        cost: { compileMs: 0, perCallNs: 1, runNs: 1 },
        compileCount: compiles,
      })
    );

    const stmts = [fakeStmt()];
    r.dispatch(stmts, 0, fakeCtx(r));
    expect(compiles.n).toBe(1);
    r.clearCache();
    r.dispatch(stmts, 0, fakeCtx(r));
    expect(compiles.n).toBe(2);
  });
});

describe("Interpreter integration", () => {
  it("interpreter exposes a registry with the interpreter plugin pre-registered", () => {
    // executeCode is the user-facing entry point. After it runs, the
    // interpreter's registry should at minimum have the interpreter
    // executor registered.
    //
    // Currently the registry isn't routed through execStmt, so this
    // test just verifies the construction path. As executors are
    // ported, more registrations will appear here.
    const result = executeCode("x = 1;");
    expect(result.variableValues["x"]).toBe(1);
  });
});
