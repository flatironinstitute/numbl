/**
 * js-jit-call — JS codegen executor for the call shape.
 *
 * Lowering is the dispatcher's job. The call executor receives the
 * lowered IR (with pre-computed feasibility flags) as the first arg
 * to propose, decides whether to commit, and produces a compiled JS
 * artifact on commit.
 *
 *   - `propose()` filters on `lowered.kind === "call"`, applies
 *     IO+bail-risk feasibility check. Returns null to decline.
 *
 *   - `compile()` calls `generateCallJS` against the lowered IR
 *     (which internally tries the e1 scalar kernel under --opt e1
 *     and falls back to the regular JS body). Cached by the
 *     registry under the classification's cacheKey.
 *
 *   - `run()` calls `runCallCompiled`. JitBailToInterpreter →
 *     transient bail.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import {
  generateCallJS,
  runCallCompiled,
  type CallLowered,
  type CallCompiled,
} from "./jitCall.js";

interface JsJitCallData {
  readonly lowered: CallLowered;
  readonly args: readonly unknown[];
}

const JS_JIT_CALL_COST = { compileMs: 50, perCallNs: 200, runNs: 200 };

export const jsJitCallExecutor: Executor<JsJitCallData, CallCompiled | null> = {
  name: "js-jit-call",

  propose(lowered: LoweredStmt): Proposal<JsJitCallData> | null {
    if (lowered.kind !== "call") return null;
    const flags = lowered.flags;

    // If the body emits I/O and any mid-execution bail could fire,
    // decline — re-running via the interpreter after a partial run
    // would duplicate already-emitted output.
    if (flags.hasIO && flags.hasBailRisk) return null;

    return {
      data: { lowered: lowered.lowered, args: lowered.args },
      cost: JS_JIT_CALL_COST,
      // The artifact's correctness depends on type assumptions that
      // can fail at runtime.
      bailRisk: true,
    };
  },

  cacheKey(d): string {
    return d.lowered.classification.cacheKey;
  },

  compile(d, ctx: DispatchContext): CallCompiled | null {
    return generateCallJS(ctx.interp, d.lowered);
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "js-jit-call: codegen rejected" } };
    }
    const r = runCallCompiled(
      ctx.interp,
      d.lowered.classification.fn,
      compiled,
      [...d.args]
    );
    if (r.ok) return { result: r.result };
    return {
      bail: { message: "js-jit-call: bailed at runtime" },
      ...(r.transient ? { transient: true } : {}),
    };
  },
};
