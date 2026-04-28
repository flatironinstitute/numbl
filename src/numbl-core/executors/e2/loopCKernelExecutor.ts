/**
 * loop-c-kernel — port of the e2 whole-loop C JIT.
 *
 * Wraps the existing `tryE2Loop` from `jit/e2/loopKernel.ts`. Matches
 * `For` statements; on success, the entire loop runs as a single
 * compiled C kernel. The wrapped layer handles its own classification,
 * caching (per-Stmt WeakMap), and runtime dispatch — this shim just
 * routes the call.
 *
 * Bails are reported transient so the registry's outer cache stays
 * out of the way of the wrapped layer's own cache (which already
 * remembers structural hard-bails via its `"bailed"` marker).
 */

import type { Stmt } from "../../parser/types.js";
import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import { tryE2Loop } from "./loopKernel.js";

interface LoopData {
  readonly stmt: Stmt & { type: "For" };
}

// Same shape of estimate as the chain executor: low per-call so we
// beat the AST interpreter's stub runNs whenever we apply.
const LOOP_C_COST = { compileMs: 50, perCallNs: 300, runNs: 100 };

export const loopCKernelExecutor: Executor<LoopData, true> = {
  name: "loop-c-kernel",
  bailRisk: false,

  propose(stmt): Proposal<LoopData> | null {
    if (stmt.type !== "For") return null;
    return { data: { stmt }, cost: LOOP_C_COST };
  },

  cacheKey(): string {
    return "loop";
  },

  compile(): true {
    return true;
  },

  run(_compiled, d, ctx: DispatchContext): RunResult {
    const ok = tryE2Loop(ctx.interp, d.stmt);
    if (!ok) {
      return {
        bail: { message: "loop-c-kernel: not classifiable" },
        transient: true,
      };
    }
    return { consumed: 1 };
  },
};
