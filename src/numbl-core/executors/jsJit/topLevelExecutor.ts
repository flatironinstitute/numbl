/**
 * js-jit-top-level — JS codegen executor for the top-level shape.
 *
 * Lowering is the dispatcher's job. This executor receives the
 * lowered IR (with pre-computed feasibility flags) as the first arg
 * to propose, decides whether to commit, and on commit produces a
 * compiled JS artifact.
 *
 *   - `propose()` filters on `lowered.kind === "top-level"`, then
 *     applies JIT-feasibility checks against the flags and the
 *     runtime display-mode setting. Returns null to decline.
 *
 *   - `compile()` calls `generateTopLevelJS` against the lowered IR
 *     to produce a JS function. Cached by the registry under the
 *     classification's cacheKey.
 *
 *   - `run()` calls `runTopLevelCompiled`. JitBailToInterpreter →
 *     transient bail (cache preserved). JitFuncHandleBailError →
 *     permanent bail (cache invalidated).
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

interface TopLevelData {
  readonly lowered: TopLevelLowered;
}

// Whole-script compile is the heaviest match; once compiled it saves
// the per-stmt dispatch overhead for the entire script. The numbers
// are illustrative — what matters is winning vs the interpreter
// executor's stub runNs.
const TOP_LEVEL_COST = { compileMs: 100, perCallNs: 1000, runNs: 1000 };

export const jsJitTopLevelExecutor: Executor<
  TopLevelData,
  TopLevelCompiled | null
> = {
  name: "js-jit-top-level",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<TopLevelData> | null {
    if (lowered.kind !== "top-level") return null;
    const flags = lowered.flags;

    // JIT can't model `return` from the synthetic top-level fn.
    if (flags.hasReturn) return null;

    // MATLAB displays unsuppressed top-level statements. The JIT has
    // no emit for auto-display, so in display mode we bail when any
    // source stmt would normally print.
    if (ctx.interp.rt.displayResults && flags.hasUnsuppressedAssign) {
      return null;
    }

    // If the body contains I/O (disp/fprintf/…) and any mid-execution
    // bail could fire, decline — re-running via the interpreter after
    // a partial run would duplicate already-emitted output.
    if (flags.hasIO && flags.hasBailRisk) return null;

    return {
      data: { lowered: lowered.lowered },
      cost: TOP_LEVEL_COST,
      // Wrapped layer absorbs JitBailToInterpreter and restores state
      // on failure; bail-risky because the artifact's correctness
      // relies on type assumptions that can fail at runtime.
      bailRisk: true,
    };
  },

  cacheKey(d): string {
    return d.lowered.classification.cacheKey;
  },

  compile(d, ctx: DispatchContext): TopLevelCompiled | null {
    return generateTopLevelJS(ctx.interp, d.lowered);
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "js-jit-top-level: codegen rejected" } };
    }
    const r = runTopLevelCompiled(
      ctx.interp,
      compiled,
      d.lowered.classification
    );
    if (r.ok) return { consumed: d.lowered.classification.stmts.length };
    return {
      bail: { message: "js-jit-top-level: bailed at runtime" },
      ...(r.transient ? { transient: true } : {}),
    };
  },
};
