/**
 * js-jit-loop — port of the JS-JIT loop hooks (`tryJitFor` /
 * `tryJitWhile`).
 *
 * Matches `For` and `While` statements. The wrapped layer
 * (`jit/jitLoop.ts`) does its own type signature analysis, progressive
 * widening, compile-time IO+bail-risk gating, and writes back outputs
 * to the env. A `JitBailToInterpreter` thrown at runtime is absorbed
 * inside `executeAndWriteBack`, which restores the loop to its pre-
 * call state and returns false — so from the outside this executor
 * never observes a mid-execution bail.
 *
 * Because the wrapped layer's compile-time gate refuses bodies that
 * can both emit observable I/O and bail at runtime, the compiled
 * artifact is effectively bail-safe by construction. We still mark
 * `bailRisk: true` to be conservative: the artifact's correctness
 * relies on type assumptions that can fail, even if today's gating
 * catches the unsafe overlap.
 */

import type { Stmt } from "../../parser/types.js";
import type { Executor, MatchResult, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import { tryJitFor, tryJitWhile } from "../../jit/jitLoop.js";

interface LoopMatch {
  readonly stmt: Stmt & { type: "For" | "While" };
}

// Per-call dispatch ~hundreds of ns once compiled. The interpreter's
// stub runNs ensures we win for any matching loop whose type analysis
// succeeds.
const JS_JIT_LOOP_COST = { compileMs: 30, perCallNs: 200, runNs: 200 };

export const jsJitLoopExecutor: Executor<LoopMatch, true> = {
  name: "js-jit-loop",
  bailRisk: true,

  match(siblings, i): MatchResult<LoopMatch> | null {
    const stmt = siblings[i];
    if (stmt.type !== "For" && stmt.type !== "While") return null;
    return { match: { stmt }, cost: JS_JIT_LOOP_COST };
  },

  cacheKey(): string {
    return "loop";
  },

  compile(): true {
    return true;
  },

  run(_compiled, m, ctx: DispatchContext): RunResult {
    const ok =
      m.stmt.type === "For"
        ? tryJitFor(ctx.interp, m.stmt)
        : tryJitWhile(ctx.interp, m.stmt);
    if (!ok) {
      return {
        bail: { message: "js-jit-loop: not jittable" },
        transient: true,
      };
    }
    return { consumed: 1 };
  },
};
