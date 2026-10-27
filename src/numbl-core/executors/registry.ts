/**
 * Executor registry and dispatch.
 *
 * Three entry points, one per lowered shape:
 *   - `tryRunWholeScope(siblings, interp)` — whole-script attempt, called
 *     once before the per-stmt loop runs.
 *   - `dispatch(siblings, i, interp)` — per-stmt dispatch; loops are
 *     offered to the loop executors, everything else is interpreted.
 *   - `dispatchCall(fn, args, nargout, interp)` — user-function call.
 *
 * Executors are tried in registration order (at `--opt 2`: C-JIT, then
 * JS-JIT). The first that proposes and runs without bailing wins. The AST
 * interpreter is the last-resort fallback and isn't a registered executor.
 */

import type { Stmt } from "../parser/types.js";
import type { ControlSignal, FunctionDef } from "../interpreter/types.js";
import type { Interpreter } from "../interpreter/interpreter.js";
import type { Executor, LoweredOf, RunResult, ShapeKind } from "./types.js";
import { ExecutorCache } from "./cache.js";
import {
  WideningState,
  tryLowerCall,
  tryLowerLoop,
  tryLowerTopLevel,
} from "./lowering.js";

export class Registry {
  private readonly executors: { [K in ShapeKind]: Executor<K>[] } = {
    "top-level": [],
    loop: [],
    call: [],
  };
  private cache = new ExecutorCache();
  private widening = new WideningState();

  /** Register an executor for its shape. Order matters: earlier
   *  executors are tried first. */
  register<K extends ShapeKind>(executor: Executor<K>): void {
    const list = this.executors[executor.kind] as Executor<K>[];
    if (list.some(e => e.name === executor.name)) {
      throw new Error(`Executor already registered: ${executor.name}`);
    }
    list.push(executor);
  }

  /** Whether any call executor is registered. */
  get hasCallExecutors(): boolean {
    return this.executors.call.length > 0;
  }

  /** Drop all cached compiled artifacts. Called from
   *  `Interpreter.clearAllCaches()` after addpath/rmpath etc. */
  clearCache(): void {
    this.cache = new ExecutorCache();
    this.widening = new WideningState();
  }

  /** Try to handle the entire script body as a unit. Returns false when
   *  no executor took it; the caller then runs the per-stmt loop. */
  tryRunWholeScope(siblings: readonly Stmt[], interp: Interpreter): boolean {
    const executors = this.executors["top-level"];
    if (executors.length === 0) return false;
    const lowered = tryLowerTopLevel(interp, siblings, this.widening);
    if (!lowered) return false;
    return this.runFirst(executors, lowered, siblings[0], interp) !== null;
  }

  /** Dispatch the stmt at `siblings[i]`. Returns its control signal. */
  dispatch(
    siblings: readonly Stmt[],
    i: number,
    interp: Interpreter
  ): ControlSignal | null {
    const stmt = siblings[i];
    if (
      (stmt.type === "For" || stmt.type === "While") &&
      this.executors.loop.length > 0
    ) {
      const lowered = tryLowerLoop(interp, siblings, i, this.widening);
      if (lowered && this.runFirst(this.executors.loop, lowered, stmt, interp))
        return null;
    }
    return this.interpretFallback(interp, stmt);
  }

  /** Dispatch a user-function call. Returns `{ result }` when an executor
   *  handled it, or `null` so the caller runs the interpreter's call path. */
  dispatchCall(
    fn: FunctionDef,
    args: unknown[],
    nargout: number,
    interp: Interpreter
  ): RunResult {
    const lowered = tryLowerCall(fn, args, nargout, this.widening);
    if (!lowered) return null;
    return this.runFirst(this.executors.call, lowered, fn, interp);
  }

  /** Run one stmt on the AST interpreter.
   *
   *  When that stmt is a loop, the interpreter is about to walk its body,
   *  which invalidates the "an enclosing loop will capture me" assumption
   *  behind the executors' `loopDepth` gate. `jitDeclinedLoopDepth` records
   *  that for the duration of the body so nested loops and calls can bid for
   *  themselves instead of running interpreted all the way down. */
  private interpretFallback(
    interp: Interpreter,
    stmt: Stmt
  ): ControlSignal | null {
    if (stmt.type === "For" || stmt.type === "While") {
      interp.jitDeclinedLoopDepth++;
      try {
        return interp.execStmt(stmt);
      } finally {
        interp.jitDeclinedLoopDepth--;
      }
    }
    return interp.execStmt(stmt);
  }

  /** Try `executors` in order; return the first success, or null when
   *  all declined or bailed. `owner` scopes the compile cache (head Stmt
   *  or FunctionDef). */
  private runFirst<K extends ShapeKind>(
    executors: readonly Executor<K>[],
    lowered: LoweredOf<K>,
    owner: object,
    interp: Interpreter
  ): RunResult {
    for (let k = 0; k < executors.length; k++) {
      const executor = executors[k];
      const data = executor.propose(lowered, interp);
      if (data === null) continue;
      const r = this.execute(executor, owner, data, interp);
      if (r) return r;
    }
    return null;
  }

  /** Cache lookup + compile-on-miss + run. A bail (including a compile
   *  that declined, which the executor reports as a bail from `run`) is
   *  cached so the same (executor, owner, key) is not retried. */
  private execute<K extends ShapeKind>(
    executor: Executor<K>,
    owner: object,
    data: unknown,
    interp: Interpreter
  ): RunResult {
    const key = executor.cacheKey(data);
    let compiled = this.cache.get(executor.name, owner, key);
    if (this.cache.isBailed(compiled)) return null;
    if (compiled === undefined) {
      compiled = executor.compile(data, interp);
      this.cache.set(executor.name, owner, key, compiled);
    }

    // Capture any output the attempt produces so a bail leaves no
    // observable side effect: on bail we discard it and the caller
    // falls through to the next executor or the interpreter, which
    // re-runs the same code — committing here would duplicate it.
    const rt = interp.rt;
    rt.beginOutputCapture();
    let committed = false;
    try {
      const result = executor.run(compiled, data, interp);
      if (result === null) {
        rt.discardOutputCapture();
        committed = true;
        this.cache.markBailed(executor.name, owner, key);
        return null;
      }
      rt.commitOutputCapture();
      committed = true;
      return result;
    } finally {
      // A throw from run() (not a bail) lands here with committed=false.
      if (!committed) rt.discardOutputCapture();
    }
  }
}
