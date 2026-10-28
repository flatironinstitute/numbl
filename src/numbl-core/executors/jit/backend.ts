/**
 * The JIT backend strategy.
 *
 * The three shape executors (top-level, loop, call) are written once
 * against `JitBackend`; `jsBackend` (`compileSpec` + `new Function`) and
 * `cBackend` (`compileSpecC` + `cc` + koffi) supply what differs: type
 * feasibility, compile, and how a compiled spec is called.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { JitType } from "../../jitTypes.js";
import type { Stmt, Span } from "../../parser/index.js";
import type { RuntimeValue } from "../../runtime/types.js";
import {
  UnsupportedConstruct,
  JitTypeError,
  type Type as CompilerType,
} from "../../jit/index.js";
import { recordJitDecline } from "../../jitDeclineDiagnostics.js";
import type { RERUN, RunResult } from "../types.js";
import { JitRaise, handleRaise, type UnitScope } from "./raise.js";

export type FuncStmt = Extract<Stmt, { type: "Function" }>;

export interface CompileRequest {
  /** Executor name, for diagnostics. */
  readonly where: string;
  readonly funcDecl: FuncStmt;
  readonly argTypes: CompilerType[];
  readonly nargout: number;
  /** True when `funcDecl` wraps a script body or loop rather than a
   *  user function (relaxes the C entry-output check). */
  readonly synthetic: boolean;
  /** Suffix for the `onJitCompile` label, e.g. `(number) -> nargout=1`. */
  readonly describe: string;
}

export interface JitBackend<A = unknown> {
  readonly lang: "js" | "c";
  /** Executor-name prefix: `jit` or `cjit`. */
  readonly prefix: string;
  /** Whether the backend can run in this interpreter at all. */
  available(interp: Interpreter): boolean;
  /** Translate a numbl input type to a compiler type; null declines. */
  mapType(jt: JitType): CompilerType | null;
  /** Compile a spec. May throw `UnsupportedConstruct` / `JitTypeError`
   *  (see `compileUnit`); returns null for any other decline. */
  compile(req: CompileRequest, interp: Interpreter): A | null;
  /** Call a compiled spec. Returns its outputs (`undefined` for one the
   *  spec left unassigned), or null to bail to the interpreter (an argument
   *  the backend cannot marshal). A runtime error or bail raised by the
   *  compiled code is thrown as a `JitRaise` (see `raise.ts`), which the
   *  shape executor maps with `handleRaise`. */
  invoke(
    art: A,
    args: readonly (RuntimeValue | undefined)[],
    interp: Interpreter
  ): (RuntimeValue | undefined)[] | null;
  /** Whether the compiled entry can call itself, so that its code also
   *  runs in frames the JIT does not track. */
  recursive(art: A): boolean;
}

/** `backend.compile`, turning a compiler rejection into a recorded
 *  decline (null). Only the compiler's own decline errors are caught. */
export function compileUnit<A>(
  backend: JitBackend<A>,
  req: CompileRequest,
  interp: Interpreter
): A | null {
  try {
    return backend.compile(req, interp);
  } catch (e) {
    if (e instanceof UnsupportedConstruct || e instanceof JitTypeError) {
      recordJitDecline({
        message: e.message,
        kind: e.constructor.name,
        at: e.span ? { file: e.span.file, start: e.span.start } : undefined,
        where: req.where,
      });
      return null;
    }
    throw e;
  }
}

/** Map every input type through the backend; null if any declines. */
export function mapTypes(
  backend: JitBackend,
  types: readonly JitType[]
): CompilerType[] | null {
  const out: CompilerType[] = [];
  for (const jt of types) {
    const t = backend.mapType(jt);
    if (t === null) return null;
    out.push(t);
  }
  return out;
}

/** `%!numbl:assert_jit c` requires C-JIT at --opt 2, so the JS backend
 *  declines such units there: they either C-JIT or fall through to the
 *  interpreter, which raises on the directive. Plain `assert_jit` only
 *  requires JS-JIT at --opt 1 and does not gate here. */
export function declinedByAssertC(
  backend: JitBackend,
  interp: Interpreter,
  assertsCJit: boolean
): boolean {
  return assertsCJit && backend.lang === "js" && interp.optimization === "2";
}

/** The loop-depth gate shared by the loop and call shapes: decline while
 *  an enclosing loop is running, since compiling that loop captures this
 *  unit. Lifts when every enclosing loop was itself declined
 *  (`Interpreter.jitDeclinedLoopDepth`). */
export function insideCompiledLoop(interp: Interpreter): boolean {
  return interp.loopDepth > 0 && interp.jitDeclinedLoopDepth === 0;
}

/** A synthetic `FuncStmt` wrapping a script body or loop: params are the
 *  env inputs, outputs the env outputs. */
export function synthesizeFuncStmt(
  name: string,
  inputs: readonly string[],
  outputs: readonly string[],
  body: Stmt[],
  fileName: string
): FuncStmt {
  const span: Span = { file: fileName, start: 0, end: 0 };
  return {
    type: "Function",
    name,
    functionId: name,
    params: [...inputs],
    outputs: [...outputs],
    body,
    argumentsBlocks: [],
    span,
  };
}

/** `backend.invoke` with a raise mapped to the run result by `handleRaise`
 *  (which throws for an error raised directly). */
export function invokeUnit<A>(
  backend: JitBackend<A>,
  art: A,
  args: readonly (RuntimeValue | undefined)[],
  scope: UnitScope,
  interp: Interpreter,
  where: string
): (RuntimeValue | undefined)[] | null | typeof RERUN {
  try {
    return backend.invoke(art, args, interp);
  } catch (e) {
    if (e instanceof JitRaise) return handleRaise(e, scope, interp, where);
    throw e;
  }
}

/** Run a compiled script-body or loop spec against the interpreter env:
 *  read `inputs` from it, write back the `outputs` the spec assigned. */
export function runInEnv<A>(
  backend: JitBackend<A>,
  art: A,
  inputs: readonly string[],
  outputs: readonly string[],
  scope: UnitScope,
  interp: Interpreter,
  where: string
): RunResult {
  const values = inputs.map(n => interp.env.get(n));
  const outs = invokeUnit(backend, art, values, scope, interp, where);
  if (!Array.isArray(outs)) return outs;
  for (let i = 0; i < outputs.length; i++) {
    const v = outs[i];
    if (v !== undefined) interp.env.set(outputs[i], v);
  }
  return { result: undefined };
}
