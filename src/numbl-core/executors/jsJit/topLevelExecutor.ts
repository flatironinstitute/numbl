/**
 * js-jit-top-level — JS codegen executor for the top-level shape.
 *
 *   - `propose()` filters on `lowered.kind === "top-level"`, applies
 *     JIT-feasibility checks (hasReturn, display mode + unsuppressed
 *     assigns, IO+bail-risk).
 *
 *   - `compile()` calls `generateTopLevelJS`. Cached by the registry
 *     under the classification's cacheKey.
 *
 *   - `run()` calls `runTopLevelCompiled`. JitBailToInterpreter →
 *     transient bail; JitFuncHandleBailError → permanent bail.
 *     `consumed` claims the entire script body.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import {
  generateTopLevelJS,
  runTopLevelCompiled,
  type TopLevelLowered,
  type TopLevelCompiled,
} from "./jitTopLevel.js";

// Whole-script compile is the heaviest match; once compiled it saves
// the per-stmt dispatch overhead for the entire script.
const TOP_LEVEL_COST = { compileMs: 100, perCallNs: 1000, runNs: 1000 };

export const jsJitTopLevelExecutor: Executor<
  TopLevelLowered,
  TopLevelCompiled | null
> = {
  name: "js-jit-top-level",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<TopLevelLowered> | null {
    if (lowered.kind !== "top-level") return null;
    const flags = lowered.flags;

    if (flags.hasReturn) return null;
    if (ctx.interp.rt.displayResults && flags.hasUnsuppressedAssign) {
      return null;
    }
    if (flags.hasIO && flags.hasBailRisk) return null;

    return {
      data: lowered.lowered,
      cost: TOP_LEVEL_COST,
      bailRisk: true,
    };
  },

  cacheKey(d): string {
    return d.classification.cacheKey;
  },

  compile(d, ctx: DispatchContext): TopLevelCompiled | null {
    return generateTopLevelJS(ctx.interp, d);
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "js-jit-top-level: codegen rejected" } };
    }
    const r = runTopLevelCompiled(ctx.interp, compiled, d.classification);
    if (r.ok) return { ok: true };
    return {
      bail: { message: "js-jit-top-level: bailed at runtime" },
      ...(r.transient ? { transient: true } : {}),
    };
  },
};
