/**
 * chain-c-kernel — first port of the e2 per-assign / chain driver.
 *
 * For now this is a thin shim around the existing `tryE2Assign` in
 * `jit/e2/assignKernel.ts`. The shim is registered under `--opt e2`
 * and wins the cost competition for tensor `Assign` heads (where
 * tryE2Assign actually does work). When the wrapped layer rejects a
 * stmt — non-classifiable RHS, scalar inputs, below the
 * NUMBL_E2_MIN_ELEMS threshold, etc. — the shim returns a transient
 * bail so the dispatcher falls through to the AST interpreter without
 * caching the rejection.
 *
 * Future work: move classification into `match`, compile/cache the
 * kernel under the registry's cache, and drop the dependency on the
 * legacy `_e2ChainAdvance` interpreter field. That refactor lands in
 * its own commit; this port is a structural step.
 */

import type { Stmt } from "../parser/types.js";
import type { Executor, MatchResult, RunResult } from "./types.js";
import type { DispatchContext } from "./context.js";
import { tryE2Assign } from "../jit/e2/assignKernel.js";

interface ChainMatch {
  readonly siblings: readonly Stmt[];
  readonly i: number;
  readonly headStmt: Stmt & { type: "Assign" };
}

export const chainCKernelExecutor: Executor<ChainMatch, true> = {
  name: "chain-c-kernel",
  // The wrapped tryE2Assign produces a fully-typed C kernel; once it
  // returns successfully it does not bail mid-execution.
  bailRisk: false,

  match(siblings, i): MatchResult<ChainMatch> | null {
    const stmt = siblings[i];
    if (stmt.type !== "Assign") return null;
    return {
      match: {
        siblings,
        i,
        headStmt: stmt as Stmt & { type: "Assign" },
      },
      // Rough estimate: per-call dispatch is a few hundred ns
      // (classify + koffi marshaling); runNs is small relative to the
      // interpreter's ~1e9 stub. The interpreter loses for tensor work
      // and wins (via this executor's bail) for everything else.
      cost: { compileMs: 50, perCallNs: 300, runNs: 100 },
    };
  },

  cacheKey(): string {
    // tryE2Assign maintains its own cache keyed by signature; the
    // registry-side cache is not used. A constant key plus transient
    // bail keeps the registry cache out of the way.
    return "chain";
  },

  compile(): true {
    return true;
  },

  run(_compiled, m, ctx: DispatchContext): RunResult {
    const interp = ctx.interp;
    // tryE2Assign reads _postSiblings/_postSiblingsIdx from the
    // interpreter. The dispatch loop sets these before calling us, but
    // re-set them defensively here in case the call path changes.
    interp._postSiblings = m.siblings as Stmt[];
    interp._postSiblingsIdx = m.i + 1;
    interp._e2ChainAdvance = 0;
    const ok = tryE2Assign(interp, m.headStmt);
    if (!ok) {
      // tryE2Assign already cleared its own state; ensure the legacy
      // signal is zeroed for the surrounding dispatch loop.
      interp._e2ChainAdvance = 0;
      return {
        bail: { message: "chain-c-kernel: not classifiable" },
        transient: true,
      };
    }
    const consumed = 1 + interp._e2ChainAdvance;
    interp._e2ChainAdvance = 0;
    return { consumed };
  },
};
