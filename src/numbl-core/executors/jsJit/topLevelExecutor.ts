/**
 * js-jit-top-level — port of the top-level script JIT
 * (`tryJitTopLevel`).
 *
 * Wraps the existing whole-script JS-JIT path: when the script's
 * top-level body is JIT-feasible, the entire body compiles to a
 * synthetic JS function and runs in one go. Outputs are written back
 * to the workspace env afterwards.
 *
 * Match conditions:
 *   - `ctx.scope === "top-level"` (only the script's root stmt list,
 *     not function bodies, loop bodies, or other blocks).
 *   - `i === 0` (matches once at the start of the script body).
 *
 * On match the executor consumes the entire sibling list. On bail,
 * the dispatcher falls through to the AST interpreter, which runs
 * stmts one at a time — and the per-stmt path picks up other
 * specialized executors (loop, chain, etc.) for stmts that match.
 */

import type { Stmt } from "../../parser/types.js";
import type { Executor, MatchResult, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import { tryJitTopLevel } from "../../jit/jitTopLevel.js";

interface TopLevelMatch {
  readonly siblings: Stmt[];
}

// Whole-script compile is the heaviest match; once compiled it saves
// the per-stmt dispatch overhead for the entire script. The numbers
// are illustrative — what matters is winning vs the interpreter
// executor's stub runNs.
const TOP_LEVEL_COST = { compileMs: 100, perCallNs: 1000, runNs: 1000 };

export const jsJitTopLevelExecutor: Executor<TopLevelMatch, true> = {
  name: "js-jit-top-level",
  // Wrapped layer absorbs JitBailToInterpreter and restores state on
  // failure; the surrounding compile-time IO+bail-risk gate ensures
  // that side effects don't replay. Marked true to surface the
  // assumption-based nature of the artifact through the interface.
  bailRisk: true,

  match(siblings, i, ctx: DispatchContext): MatchResult<TopLevelMatch> | null {
    if (ctx.scope !== "top-level") return null;
    if (i !== 0) return null;
    return { match: { siblings: siblings as Stmt[] }, cost: TOP_LEVEL_COST };
  },

  cacheKey(): string {
    return "top-level";
  },

  compile(): true {
    return true;
  },

  run(_compiled, m, ctx: DispatchContext): RunResult {
    const ok = tryJitTopLevel(ctx.interp, m.siblings);
    if (!ok) {
      return {
        bail: { message: "js-jit-top-level: not jittable" },
        transient: true,
      };
    }
    return { consumed: m.siblings.length };
  },
};
