/**
 * The raise channel, host side.
 *
 * Compiled code reports a runtime error or a bail through one channel:
 * emitted JS throws a tagged `JitRaise` (`runtime/system/raise.js`), and
 * compiled C `longjmp`s to its host-entry guard and leaves a status the C
 * backend reads (`runtime/system/raise.h`). Both backends normalize it to
 * the `JitRaise` below and throw it from `invoke`; the shape executors hand
 * it to `handleRaise`, which decides what the registry does:
 *
 *   bail             JIT limitation (e.g. an indexed store that would grow
 *                    an array). Return null: the registry discards the run's
 *                    side effects, caches the unit as bailed, and the
 *                    interpreter re-runs it.
 *   error, exact     Throw numbl's RuntimeError with the message and
 *                    identifier the interpreter would raise, attributed to
 *                    the statement that raised it (`rt.$file` / `rt.$line`,
 *                    plus the function's call frame for the call shape).
 *   error, otherwise Return RERUN: the registry discards the run's side
 *                    effects without caching a bail, and the interpreter
 *                    re-runs the unit to raise the error itself.
 *
 * An error is raised directly only when the result is indistinguishable
 * from the interpreter's: the JIT reproduces the wording (message non-null),
 * the raising statement lies in the unit's own code (not a callee, and not
 * a recursive function, whose frames the JIT does not track), and the
 * variables the unit would have
 * assigned before the error are unobservable. The last holds for the call
 * shape (the function's workspace is discarded) and, for the loop and
 * top-level shapes, outside any `try` (an uncaught error ends the run, and
 * a session does not adopt the state of a failed run). Inside a `try` the
 * catch block could read those variables, which the compiled unit has not
 * written back, so the unit re-runs.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { Stmt } from "../../parser/index.js";
import {
  RuntimeError,
  buildLineTable,
  offsetToLineFast,
} from "../../runtime/error.js";
import { RERUN } from "../types.js";

export type JitRaiseKind = "bail" | "error";

export interface SourceLoc {
  readonly file: string;
  readonly offset: number;
}

/** A raise from compiled code, normalized across backends. */
export class JitRaise extends Error {
  constructor(
    readonly kind: JitRaiseKind,
    readonly identifier: string,
    /** The error message (a bail's reason); null when the JIT leaves the
     *  wording of an error to the interpreter. */
    readonly raiseMessage: string | null,
    /** The raising statement, when known. */
    readonly loc: SourceLoc | null
  ) {
    super(raiseMessage ?? `JIT ${kind}`);
    this.name = "JitRaise";
  }
}

/** The tag an emitted JS module's `JitRaise` carries (each module defines
 *  its own class, so `instanceof` cannot be used). */
interface TaggedJsRaise {
  jitRaise: true;
  kind: JitRaiseKind;
  identifier: string;
  raiseMessage: string | null;
}

export function isTaggedJsRaise(e: unknown): e is TaggedJsRaise {
  return (
    typeof e === "object" &&
    e !== null &&
    (e as { jitRaise?: unknown }).jitRaise === true
  );
}

/** Where a unit's own code lives: the raising statement must fall in one of
 *  `ranges` of `file` for an error to be raised directly. */
export interface UnitScope {
  readonly file: string;
  readonly ranges: ReadonlyArray<readonly [number, number]>;
  /** Call shape: the function whose frame the interpreter would push. */
  readonly callName?: string;
  /** The unit's code also runs in nested frames (a recursive function),
   *  whose call stack the JIT does not track. */
  readonly reentrant?: boolean;
}

/** The scope of a unit whose code is `stmts` (nested function definitions
 *  excluded: they are separate frames). */
export function scopeOfStmts(
  stmts: readonly Stmt[],
  fallbackFile: string,
  callName?: string
): UnitScope {
  const ranges: [number, number][] = [];
  let file = fallbackFile;
  for (const s of stmts) {
    if (s.type === "Function" || !s.span) continue;
    file = s.span.file;
    ranges.push([s.span.start, s.span.end]);
  }
  return { file, ranges, callName };
}

function inScope(loc: SourceLoc, scope: UnitScope): boolean {
  if (loc.file !== scope.file) return false;
  return scope.ranges.some(([a, b]) => loc.offset >= a && loc.offset < b);
}

function jitLog(msg: string): void {
  if (typeof process !== "undefined" && process.env?.NUMBL_JIT_LOG) {
    console.error(msg);
  }
}

/** Map a raise to the executor's run result (see the module comment).
 *  Throws the RuntimeError for an error raised directly. */
export function handleRaise(
  r: JitRaise,
  scope: UnitScope,
  interp: Interpreter,
  where: string
): null | typeof RERUN {
  if (r.kind === "bail") {
    interp.onJitBail?.(
      `${where}: ${r.raiseMessage ?? "JIT bail"}; falling back to interpreter`
    );
    return null;
  }
  const direct =
    r.raiseMessage !== null &&
    r.loc !== null &&
    inScope(r.loc, scope) &&
    scope.reentrant !== true &&
    (scope.callName !== undefined || interp.tryDepth === 0);
  if (!direct) {
    jitLog(
      `[jit-rerun] ${where}: ${r.raiseMessage ?? "(interpreter wording)"}`
    );
    return RERUN;
  }
  throw toRuntimeError(r, r.loc!, r.raiseMessage!, scope, interp);
}

/** Build the RuntimeError the interpreter would raise at `loc`, leaving the
 *  runtime's location (and, for a call, its annotated frame) as the
 *  interpreter would. */
function toRuntimeError(
  r: JitRaise,
  loc: SourceLoc,
  message: string,
  scope: UnitScope,
  interp: Interpreter
): RuntimeError {
  const rt = interp.rt;
  const line = lineOf(interp, loc);
  const err = new RuntimeError(message, null, { identifier: r.identifier });
  if (scope.callName === undefined) {
    rt.$file = loc.file;
    rt.$line = line;
    return err;
  }
  // As `runFunctionBody`: the frame is pushed with the caller's location,
  // the error annotated at the callee's, and the frame popped.
  rt.pushCallFrame(scope.callName);
  try {
    rt.$file = loc.file;
    rt.$line = line;
    rt.annotateError(err);
  } finally {
    rt.popCallFrame();
  }
  return err;
}

function lineOf(interp: Interpreter, loc: SourceLoc): number {
  let table = interp.lineTableCache.get(loc.file);
  if (!table) {
    table = buildLineTable(interp.fileSources.get(loc.file) ?? "");
    interp.lineTableCache.set(loc.file, table);
  }
  return offsetToLineFast(table, loc.offset);
}
