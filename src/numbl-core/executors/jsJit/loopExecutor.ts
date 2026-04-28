/**
 * js-jit-loop — JS codegen executor for the loop shape.
 *
 *   - `propose()` filters on `lowered.kind === "loop"`, applies
 *     JIT-feasibility checks (hasReturn, IO+bail-risk).
 *
 *   - `compile()` calls `generateLoopJS`. Cached by the registry
 *     under the classification's cacheKey.
 *
 *   - `run()` calls `runLoopCompiled`. JitBailToInterpreter →
 *     transient bail; JitFuncHandleBailError → permanent bail.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import {
  generateLoopJS,
  runLoopCompiled,
  type LoopLowered,
  type LoopCompiled,
} from "./jitLoop.js";

interface LoopData {
  readonly lowered: LoopLowered;
}

// Per-call dispatch ~hundreds of ns once compiled. The interpreter's
// stub runNs ensures we win for any matching loop whose type analysis
// succeeds.
const JS_JIT_LOOP_COST = { compileMs: 30, perCallNs: 200, runNs: 200 };

export const jsJitLoopExecutor: Executor<LoopData, LoopCompiled | null> = {
  name: "js-jit-loop",

  propose(lowered: LoweredStmt): Proposal<LoopData> | null {
    if (lowered.kind !== "loop") return null;
    const flags = lowered.flags;

    if (flags.hasReturn) return null;
    if (flags.hasIO && flags.hasBailRisk) return null;

    return {
      data: { lowered: lowered.lowered },
      cost: JS_JIT_LOOP_COST,
      bailRisk: true,
    };
  },

  cacheKey(d): string {
    return d.lowered.classification.cacheKey;
  },

  compile(d, ctx: DispatchContext): LoopCompiled | null {
    return generateLoopJS(ctx.interp, d.lowered);
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "js-jit-loop: codegen rejected" } };
    }
    const r = runLoopCompiled(ctx.interp, compiled, d.lowered.classification);
    if (r.ok) return { consumed: 1 };
    return {
      bail: { message: "js-jit-loop: bailed at runtime" },
      ...(r.transient ? { transient: true } : {}),
    };
  },
};
