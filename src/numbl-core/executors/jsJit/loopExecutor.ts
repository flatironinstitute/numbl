/**
 * js-jit-loop — JS codegen executor for the loop shape.
 *
 * Lowering is the dispatcher's job. The loop executor receives the
 * lowered IR (with pre-computed feasibility flags) as the first arg
 * to propose, decides whether to commit, and produces a compiled JS
 * artifact on commit.
 *
 *   - `propose()` filters on `lowered.kind === "loop"`, applies
 *     JIT-feasibility checks against the flags. Returns null to
 *     decline.
 *
 *   - `compile()` calls `generateLoopJS` against the lowered IR.
 *     Cached by the registry under the classification's cacheKey.
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

    // JIT can't model `return` from the synthetic loop fn.
    if (flags.hasReturn) return null;

    // If the body contains I/O and a mid-execution bail could fire,
    // decline — re-running via the interpreter after a partial run
    // would duplicate already-emitted output.
    if (flags.hasIO && flags.hasBailRisk) return null;

    return {
      data: { lowered: lowered.lowered },
      cost: JS_JIT_LOOP_COST,
      // Compiled artifact's correctness relies on type assumptions
      // that can fail at runtime.
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
