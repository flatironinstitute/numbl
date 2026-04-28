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
 * kernel under the registry's cache. That refactor lands in its own
 * commit; this port is a structural step.
 */

import type { Stmt } from "../../parser/types.js";
import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import { tryE2Assign } from "./assignKernel.js";

interface ChainData {
  readonly headStmt: Stmt & { type: "Assign" };
}

// Rough estimate: per-call dispatch is a few hundred ns (classify +
// koffi marshaling); runNs is small relative to the interpreter's
// ~1e9 stub. The interpreter loses for tensor work and wins (via this
// executor's bail) for everything else.
const CHAIN_COST = { compileMs: 50, perCallNs: 300, runNs: 100 };

export const chainCKernelExecutor: Executor<ChainData, true> = {
  name: "chain-c-kernel",
  // The wrapped tryE2Assign produces a fully-typed C kernel; once it
  // returns successfully it does not bail mid-execution.
  bailRisk: false,

  propose(stmt): Proposal<ChainData> | null {
    if (stmt.type !== "Assign") return null;
    return {
      data: { headStmt: stmt as Stmt & { type: "Assign" } },
      cost: CHAIN_COST,
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

  run(_compiled, d, ctx: DispatchContext): RunResult {
    // tryE2Assign reads `interp._postSiblings` / `_postSiblingsIdx`
    // for chain lookahead. The outer `execStmts` loop already sets
    // those before each dispatch, so we don't need to re-set them
    // here — they describe the same scope and head index that ctx
    // exposes.
    const consumed = tryE2Assign(ctx.interp, d.headStmt);
    if (consumed === null) {
      return {
        bail: { message: "chain-c-kernel: not classifiable" },
        transient: true,
      };
    }
    return { consumed };
  },
};
