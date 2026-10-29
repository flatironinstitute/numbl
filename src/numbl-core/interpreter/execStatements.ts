/**
 * Statement execution: assignments, control flow and loops.
 */

import type { Stmt, Expr } from "../parser/types.js";
import { isNullAssignRhs } from "../parser/astUtils.js";
import type { RuntimeValue } from "../runtime/types.js";
import { isRuntimeTensor, isRuntimeCell } from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError, CancellationError } from "../runtime/error.js";
import { getLastJitDecline } from "../jitDeclineDiagnostics.js";
import { buildLineTable, offsetToLineFast } from "../runtime/error.js";
import { COLON_SENTINEL } from "../runtime/sentinels.js";
import {
  forIter,
  switchMatch,
  range as runtimeRange,
} from "../runtime/runtimeOperators.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import { incref } from "../runtime/refcount.js";

import {
  BreakSignal,
  ContinueSignal,
  ReturnSignal,
  funcDefFromStmt,
  type ControlSignal,
} from "./types.js";
import type { Interpreter } from "./interpreter.js";
import { evalExpr, evalExprNargout, evalCondition } from "./evalExpressions.js";
import { assignLValue, evalIndicesWithEnd } from "./indexing.js";

/** Execute one statement in its own transient refcount scope. */
export function execStmt(
  interp: Interpreter,
  stmt: Stmt
): ControlSignal | null {
  return interp.rt.withScope(() => execStmtInner(interp, stmt));
}

/** Execute a statement list through the executor registry (which runs a
 *  statement on the interpreter when no JIT executor takes it). */
export function execStmts(
  interp: Interpreter,
  stmts: Stmt[]
): ControlSignal | null {
  for (let i = 0; i < stmts.length; i++) {
    const signal = interp.registry.dispatch(stmts, i, interp);
    if (signal) return signal;
  }
  return null;
}

/** Execute a conditional-block body (`if` / `switch` / `try`), tracking
 *  `condBlockDepth` so a loop dispatched inside knows its sibling list is
 *  a nested block — the loop classifier then keeps every loop-assigned
 *  name live-out (the post-loop liveness scan can't see reads after the
 *  enclosing block). */
export function execBlockStmts(
  interp: Interpreter,
  stmts: Stmt[]
): ControlSignal | null {
  interp.condBlockDepth++;
  try {
    return execStmts(interp, stmts);
  } finally {
    interp.condBlockDepth--;
  }
}

function execStmtInner(interp: Interpreter, stmt: Stmt): ControlSignal | null {
  const rt = interp.rt;
  if (stmt.span) {
    rt.$file = stmt.span.file;
    rt.$line = lineOf(interp, stmt.span.file, stmt.span.start);
  }
  const env = interp.frame.env;

  switch (stmt.type) {
    case "ExprStmt": {
      const val = evalExprNargout(interp, stmt.expr, 0);
      const singleVal = Array.isArray(val) ? val[0] : val;
      // Calls to functions with no outputs produce no value: do not set
      // `ans` and do not display (matches MATLAB behavior).
      if (singleVal === undefined) {
        return null;
      }
      const rv = ensureRuntimeValue(singleVal);
      interp.ans = rv;
      env.set("ans", rv);
      if (!stmt.suppressed && !displaysItself(stmt.expr)) {
        rt.displayResult(rv);
      }
      return null;
    }

    case "Assign": {
      const rawVal = evalExpr(interp, stmt.expr);
      const val = Array.isArray(rawVal) ? rawVal[0] : rawVal;
      const rv = val as RuntimeValue;
      env.set(stmt.name, rv);
      interp.ans = rv;
      if (!stmt.suppressed) {
        rt.displayAssign(stmt.name, rv);
      }
      return null;
    }

    case "MultiAssign":
      return execMultiAssign(interp, stmt);

    case "AssignLValue": {
      const val = evalExpr(interp, stmt.expr);
      const rv = val as RuntimeValue;
      assignLValue(interp, stmt.lvalue, rv, isNullAssignRhs(stmt.expr));
      if (!stmt.suppressed) {
        if (stmt.lvalue.type === "Var") {
          rt.displayAssign(stmt.lvalue.name, rv);
        }
      }
      return null;
    }

    case "If": {
      const cond = evalCondition(interp, stmt.cond);
      if (rt.toBool(cond)) {
        return execBlockStmts(interp, stmt.thenBody);
      }
      for (const elseif of stmt.elseifBlocks) {
        const elseifCond = evalCondition(interp, elseif.cond);
        if (rt.toBool(elseifCond)) {
          return execBlockStmts(interp, elseif.body);
        }
      }
      if (stmt.elseBody) {
        return execBlockStmts(interp, stmt.elseBody);
      }
      return null;
    }

    case "While":
      return execWhile(interp, stmt);

    case "For":
      return execFor(interp, stmt);

    case "Switch": {
      const switchVal = evalExpr(interp, stmt.expr);
      let matched = false;
      for (const c of stmt.cases) {
        const caseVal = evalExpr(interp, c.value);
        if (switchMatch(switchVal, caseVal)) {
          matched = true;
          const signal = execBlockStmts(interp, c.body);
          if (signal) return signal;
          break;
        }
      }
      if (!matched && stmt.otherwise) {
        return execBlockStmts(interp, stmt.otherwise);
      }
      return null;
    }

    case "TryCatch": {
      try {
        interp.tryDepth++;
        let signal: ControlSignal | null;
        try {
          signal = execBlockStmts(interp, stmt.tryBody);
        } finally {
          interp.tryDepth--;
        }
        if (signal) return signal;
      } catch (e) {
        // A cancellation request (Ctrl-C / interrupt) is not catchable by
        // user try/catch — it must unwind the whole run, like MATLAB.
        if (e instanceof CancellationError) {
          throw e;
        }
        if (stmt.catchVar) {
          interp.frame.env.set(stmt.catchVar, rt.wrapError(e));
        }
        const signal = execBlockStmts(interp, stmt.catchBody);
        if (signal) return signal;
      }
      return null;
    }

    case "Function": {
      env.nestedFunctions.set(stmt.name, {
        fn: funcDefFromStmt(stmt),
        env,
      });
      return null;
    }

    case "Return":
      return new ReturnSignal([]);

    case "Break":
      return new BreakSignal();

    case "Continue":
      return new ContinueSignal();

    case "Global": {
      for (const name of stmt.names) {
        env.globalNames.add(name);
        // MATLAB: declaring a global that doesn't exist yet creates it,
        // initialized to []. This makes reads (e.g. isempty(x)) work
        // before any assignment instead of erroring as undefined.
        if (rt && !(name in rt.$g)) {
          const empty = RTV.tensor(allocFloat64Array(0), [0, 0]);
          incref(empty);
          rt.$g[name] = empty;
        }
      }
      return null;
    }

    case "Persistent": {
      const funcId = env.persistentFuncId;
      if (funcId) {
        for (const name of stmt.names) {
          env.persistentNames.add(name);
          const val = rt.getPersistent(funcId, name);
          env.setLocal(name, val);
        }
      }
      return null;
    }

    case "ClassDef":
      return null;

    case "Import":
      return null;

    case "Directive":
      checkDirective(interp, stmt);
      return null;
  }
}

function execMultiAssign(
  interp: Interpreter,
  stmt: Stmt & { type: "MultiAssign" }
): ControlSignal | null {
  const env = interp.frame.env;
  // Detect [c{idx}] = func() pattern for multi-output cell assign
  if (stmt.lvalues.length === 1 && stmt.lvalues[0].type === "IndexCell") {
    const lv = stmt.lvalues[0];
    const cellBase =
      lv.base.type === "Ident"
        ? (env.get(lv.base.name) ?? RTV.cell([], [0, 0]))
        : evalExpr(interp, lv.base);
    const indices = evalIndicesWithEnd(interp, cellBase, lv.indices);
    // Determine nargout from index count
    let expandedCount = 1;
    const idx0 = indices[0];
    if (idx0 === COLON_SENTINEL) {
      // [c{:}] — expand to all elements of the cell base
      const baseRv = ensureRuntimeValue(cellBase);
      expandedCount = isRuntimeCell(baseRv) ? baseRv.data.length : 0;
    } else {
      const idxVal = ensureRuntimeValue(idx0);
      if (isRuntimeTensor(idxVal)) {
        expandedCount = idxVal.data.length;
      } else if (typeof idxVal === "number") {
        expandedCount = 1;
      }
    }
    const val = evalExprNargout(interp, stmt.expr, expandedCount);
    const values = Array.isArray(val) ? val : [val];
    const result = interp.rt.multiOutputCellAssign(
      cellBase,
      idx0,
      values.map(v => ensureRuntimeValue(v))
    );
    if (lv.base.type === "Ident") {
      env.set(lv.base.name, ensureRuntimeValue(result));
    }
    return null;
  }

  const nargout = stmt.lvalues.length;
  const val = evalExprNargout(interp, stmt.expr, nargout);
  const values = Array.isArray(val) ? val : [val];
  for (let i = 0; i < stmt.lvalues.length; i++) {
    const lv = stmt.lvalues[i];
    if (lv.type === "Ignore") continue;
    const rv = (i < values.length ? values[i] : undefined) as RuntimeValue;
    assignLValue(interp, lv, rv);
  }
  if (!stmt.suppressed) {
    for (let i = 0; i < stmt.lvalues.length; i++) {
      const lv = stmt.lvalues[i];
      if (lv.type === "Var" && i < values.length) {
        interp.rt.displayAssign(lv.name, ensureRuntimeValue(values[i]));
      }
    }
  }
  return null;
}

// ── Loops ────────────────────────────────────────────────────────────────

function execWhile(
  interp: Interpreter,
  stmt: Stmt & { type: "While" }
): ControlSignal | null {
  const rt = interp.rt;
  const startTime = rt.profilingEnabled ? performance.now() : 0;
  let iterations = 0;
  // `loopDepth` gates per-call JIT proposals — see comment on the field in
  // `Interpreter`. Bumped around the body only, not the cond eval (the cond
  // is evaluated once per iteration but is conceptually loop-control, not a
  // hot inner call).
  interp.loopDepth++;
  try {
    while (true) {
      rt.checkCancel();
      const cond = evalCondition(interp, stmt.cond);
      if (!rt.toBool(cond)) break;
      iterations++;
      const signal = execStmts(interp, stmt.body);
      if (signal instanceof BreakSignal) break;
      if (signal instanceof ContinueSignal) continue;
      if (signal instanceof ReturnSignal) {
        recordHotLoop(interp, stmt, "while", iterations, startTime);
        return signal;
      }
    }
  } finally {
    interp.loopDepth--;
  }
  recordHotLoop(interp, stmt, "while", iterations, startTime);
  return null;
}

function execFor(
  interp: Interpreter,
  stmt: Stmt & { type: "For" }
): ControlSignal | null {
  const rt = interp.rt;
  const startTime = rt.profilingEnabled ? performance.now() : 0;
  // Lazy range iteration: a `for k = a:s:b` over a scalar-numeric range is
  // iterated WITHOUT materializing the range tensor or a JS array of every
  // index. The element count and per-element value (`start + step*i`, with
  // the last snapped to `end`) are computed by the same formula as
  // makeRangeTensor, so loop-variable values are byte-identical — this only
  // skips allocating a row vector whose elements we'd consume one at a time
  // anyway (and which a `break`-heavy loop, e.g. adaptive quadrature's
  // `for i=1:1e5`, barely touches). Char ranges ('a':'z') return null from
  // asScalarNumber and keep the eager path, preserving char elems.
  let lazyRange: {
    count: number;
    start: number;
    step: number;
    end: number;
  } | null = null;
  let iterItems: unknown[] | null = null;
  if (stmt.expr.type === "Range") {
    const sv = evalExpr(interp, stmt.expr.start);
    const stv = stmt.expr.step ? evalExpr(interp, stmt.expr.step) : 1;
    const ev = evalExpr(interp, stmt.expr.end);
    const s = asScalarNumber(sv);
    const st = asScalarNumber(stv);
    const e = asScalarNumber(ev);
    if (s !== null && st !== null && e !== null) {
      // makeRangeTensor's element-count formula, clamped to a finite
      // non-negative value (matches jit_loop_count). step===0 and
      // non-finite counts yield an empty loop, same as the eager path.
      const rawN = st === 0 ? 0 : Math.floor((e - s) / st + 1 + 1e-10);
      const count = Number.isFinite(rawN) ? Math.max(0, rawN) : 0;
      lazyRange = { count, start: s, step: st, end: e };
    } else {
      iterItems = forIter(ensureRuntimeValue(runtimeRange(sv, stv, ev)));
    }
  } else {
    iterItems = forIter(ensureRuntimeValue(evalExpr(interp, stmt.expr)));
  }

  const n = lazyRange ? lazyRange.count : iterItems!.length;
  interp.loopDepth++;
  let ran = 0;
  try {
    for (let i = 0; i < n; i++) {
      rt.checkCancel();
      let elem: unknown;
      if (lazyRange) {
        let v = lazyRange.start + lazyRange.step * i;
        // Snap the last element to exactly `end` (matches makeRangeTensor
        // / jit_range_value), multi-element only.
        if (
          lazyRange.count > 1 &&
          i === lazyRange.count - 1 &&
          Math.abs(v - lazyRange.end) < Math.abs(lazyRange.step) * 1e-10
        ) {
          v = lazyRange.end;
        }
        elem = v;
      } else {
        elem = iterItems![i];
      }
      interp.frame.env.set(stmt.varName, ensureRuntimeValue(elem));
      const signal = execStmts(interp, stmt.body);
      ran = i + 1;
      if (signal instanceof BreakSignal) break;
      if (signal instanceof ContinueSignal) continue;
      if (signal instanceof ReturnSignal) {
        recordHotLoop(interp, stmt, "for", i + 1, startTime);
        return signal;
      }
    }
  } finally {
    interp.loopDepth--;
  }
  recordHotLoop(
    interp,
    stmt,
    "for",
    lazyRange ? ran : iterItems!.length,
    startTime
  );
  return null;
}

/** Extract a JS number from a scalar (number or 1×1 tensor) value;
 *  null when the value isn't a scalar. Used to detect a plain numeric
 *  range so a huge/infinite `for k = a:s:b` can iterate lazily. */
function asScalarNumber(v: unknown): number | null {
  const rv = ensureRuntimeValue(v);
  if (typeof rv === "number") return rv;
  // A single endpoint makes a single range (the eager path).
  if (isRuntimeTensor(rv) && rv.data.length === 1 && rv.cls !== "single")
    return rv.data[0];
  return null;
}

function recordHotLoop(
  interp: Interpreter,
  stmt: Stmt & { type: "For" | "While" },
  kind: "for" | "while",
  iterations: number,
  startTime: number
): void {
  if (!interp.rt.profilingEnabled || iterations <= 1000 || !stmt.span) return;
  const durationMs = performance.now() - startTime;
  const line = lineOf(interp, stmt.span.file, stmt.span.start);
  const key = `${stmt.span.file}:${line}`;
  const prev = interp.rt.hotLoops.get(key);
  if (prev) {
    prev.callCount++;
    prev.totalTimeMs += durationMs;
    if (iterations > prev.iterations) prev.iterations = iterations;
  } else {
    interp.rt.hotLoops.set(key, {
      file: stmt.span.file,
      line,
      kind,
      iterations,
      callCount: 1,
      totalTimeMs: durationMs,
    });
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/** Line number of a character offset in `file`, from a cached line table. */
function lineOf(interp: Interpreter, file: string, offset: number): number {
  let table = interp.lineTableCache.get(file);
  if (!table) {
    table = buildLineTable(interp.fileSources.get(file) ?? "");
    interp.lineTableCache.set(file, table);
  }
  return offsetToLineFast(table, offset);
}

/** Functions whose unsuppressed call statement prints its own output, so
 *  the value it returns is not displayed as `ans`. */
const SELF_DISPLAYING_FUNCTIONS: ReadonlySet<string> = new Set([
  "disp",
  "display",
  "fprintf",
  "warning",
  "assert",
  "tic",
  "toc",
  "help",
]);

function displaysItself(expr: Expr): boolean {
  return (
    (expr.type === "FuncCall" || expr.type === "Ident") &&
    SELF_DISPLAYING_FUNCTIONS.has(expr.name)
  );
}

/** `%!numbl:assert_jit` asserts that the enclosing loop / function / script
 *  body is JIT-compiled. The lowerer treats the directive as a no-op, so a
 *  JIT'd unit compiles it away and the interpreter never reaches it. If we
 *  DO reach it here, the enclosing unit ran in the interpreter — i.e. it was
 *  not JIT'd.
 *
 *    - plain `assert_jit`   requires JS-JIT at --opt 1 only.
 *    - `assert_jit c`       additionally requires C-JIT at --opt 2.
 *
 *  --opt 0 is always a no-op. The --opt 2 case where a `c` unit JS-JITs
 *  instead of C-JITs is forced here too: the JS-JIT executors decline `c`
 *  units at --opt 2, routing them to the interpreter when C-JIT also
 *  declines. Unknown directives are no-ops. */
function checkDirective(
  interp: Interpreter,
  stmt: Stmt & { type: "Directive" }
): void {
  if (stmt.directive !== "assert_jit") return;
  const wantC = stmt.args.includes("c");
  // Surface *why* the JIT declined. The guarded unit declined just before
  // falling through to the interpreter, so the most recent recorded decline
  // is (almost always) the relevant one.
  const decline = getLastJitDecline();
  const why = decline
    ? ` Most recent JIT decline (${decline.where}, ${decline.kind}): ${decline.message}`
    : ` (no JIT decline reason was recorded — the unit may have been ` +
      `declined before lowering, e.g. an unsupported input type.)`;
  if (interp.optimization === "1") {
    throw new RuntimeError(
      `%!numbl:assert_jit: expected the enclosing loop/function/script ` +
        `to be JS-JIT-compiled at --opt 1, but it ran in the ` +
        `interpreter.${why} (Run with --opt 0 to silence.)`
    );
  }
  if (interp.optimization === "2" && wantC) {
    throw new RuntimeError(
      `%!numbl:assert_jit c: expected the enclosing loop/function/` +
        `script to be C-JIT-compiled at --opt 2, but it ran in the ` +
        `interpreter.${why} (Run with --opt 0 to silence.)`
    );
  }
}
