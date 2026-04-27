/**
 * AST-interpreter executor — the always-matching last-resort fallback.
 *
 * Calls back into the existing `Interpreter.execStmt` for one stmt at
 * a time. Reports a high `runNs` so any specialized executor with a
 * better cost estimate wins; reports `bailRisk: false` so it can run
 * inside `requireNoBail` contexts.
 *
 * Compile is a no-op (the AST is already there); the cache only ever
 * holds a sentinel.
 */

import type { Stmt } from "../../parser/types.js";
import type { Executor, MatchResult, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";

interface InterpMatch {
  readonly stmt: Stmt;
}

// High runNs so any specialized executor outranks us. Numbers are
// illustrative — the AST walker is a few hundred ns per stmt for
// trivial work, orders of magnitude more for tensor ops. Hoisted
// to module scope to avoid per-dispatch literal allocation on the
// hot path.
const INTERP_COST = { compileMs: 0, perCallNs: 0, runNs: 1e9 };

export const interpreterExecutor: Executor<InterpMatch, true> = {
  name: "interpreter",
  bailRisk: false,

  match(siblings, i): MatchResult<InterpMatch> {
    return { match: { stmt: siblings[i] }, cost: INTERP_COST };
  },

  cacheKey(): string {
    return "interp";
  },

  compile(): true {
    return true;
  },

  run(_compiled, m, ctx: DispatchContext): RunResult {
    const signal = ctx.interp.execStmt(m.stmt);
    return { consumed: 1, signal };
  },
};
