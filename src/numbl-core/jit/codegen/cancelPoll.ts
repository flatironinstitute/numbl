/**
 * Where emitted loops poll the host's cancel flag (see `raise.h` /
 * `raise.js`), shared by the JS and C emitters.
 *
 * Every loop polls once per iteration, except a loop whose iterations take
 * constant time: a per-iteration test costs 10% or more in a one-statement
 * scalar loop (the loop is front-end bound), so such a loop is strip-mined
 * into an outer loop that polls around chunks of `STRIP_CHUNK` iterations
 * that do not. A `break` inside it must leave both loops, so the emitters
 * track the enclosing loops (`RuntimeState.loopLabels`).
 */

import type { IRExpr, IRStmt } from "../lowering/ir.js";
import { isScalar } from "../lowering/types.js";
import { forEachSubExpr, forEachTopLevelExpr } from "../lowering/walk.js";
import { getBuiltin } from "../builtins/index.js";
import type { RuntimeState } from "./runtime.js";

/** Iterations between polls in a strip-mined loop. */
export const STRIP_CHUNK = 1024;

/** Whether every iteration of a loop with this body (and condition) does a
 *  bounded amount of work: scalar computation only (a tensor may be read or
 *  stored one element at a time), no nested loop, and no call to a user
 *  function. */
export function isConstantTimeLoop(
  stmts: ReadonlyArray<IRStmt>,
  cond?: IRExpr
): boolean {
  let ok = true;
  const scalarExpr = (e: IRExpr): void =>
    forEachSubExpr(e, sub => {
      if (sub.kind === "Var" || sub.kind === "EndRef") return;
      if (!isScalar(sub.ty)) ok = false;
      else if (sub.kind === "Call" && getBuiltin(sub.cName) === undefined) {
        ok = false;
      }
    });
  const walk = (body: ReadonlyArray<IRStmt>): void => {
    for (const s of body) {
      if (!ok) return;
      switch (s.kind) {
        case "If":
          scalarExpr(s.cond);
          walk(s.thenBody);
          walk(s.elseBody);
          break;
        case "Assign":
          if (!isScalar(s.ty)) ok = false;
          else scalarExpr(s.expr);
          break;
        case "ExprStmt":
        case "IndexStore":
          forEachTopLevelExpr(s, scalarExpr);
          break;
        case "Break":
        case "Continue":
        case "TypeComment":
          break;
        default:
          ok = false;
      }
    }
  };
  if (cond !== undefined) scalarExpr(cond);
  walk(stmts);
  return ok;
}

/** Enter a loop: returns the label a `break` inside it must use when the
 *  loop is strip-mined (`strip`), else null. Pair with `exitLoop`. */
export function enterLoop(state: RuntimeState, strip: boolean): string | null {
  const loops = (state.loopLabels ??= []);
  let label: string | null = null;
  if (strip) {
    state.stripLabelCount = (state.stripLabelCount ?? 0) + 1;
    label = `_jit_strip${state.stripLabelCount}`;
  }
  loops.push(label);
  return label;
}

export function exitLoop(state: RuntimeState): void {
  state.loopLabels?.pop();
}

/** The innermost enclosing loop's strip label, or null (a plain `break`). */
export function breakLabel(state: RuntimeState): string | null {
  const loops = state.loopLabels;
  return loops && loops.length > 0 ? loops[loops.length - 1] : null;
}
