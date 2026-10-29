/**
 * The JIT backend strategy.
 *
 * The three shape executors (top-level, loop, call) are written once
 * against `JitBackend`; `jsBackend` (`compileSpec` + `new Function`) and
 * `cBackend` (`compileSpecC` + `cc` + koffi) supply what differs: type
 * feasibility, compile, and how a compiled spec is called.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { Stmt, Span } from "../../parser/index.js";
import type { RuntimeValue } from "../../runtime/types.js";
import {
  UnsupportedConstruct,
  JitTypeError,
  typeToString,
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
  /** Suffix for the `onJitCompile` label, e.g. `(double[1×1]:positive) -> nargout=1`. */
  readonly describe: string;
}

export interface JitBackend<A = unknown> {
  /** Executor-name prefix: `jit` or `cjit`. */
  readonly prefix: string;
  /** Compiles to native code, which `%!numbl:assert_jit c` requires at
   *  --opt 2 (see `declinedByAssertC`). */
  readonly native: boolean;
  /** Decline scripts with unsuppressed bare expressions as well as
   *  unsuppressed assignments, unless they carry `%!numbl:assert_jit c`
   *  (see `displayGateOk` in `topLevelExecutor.ts`). */
  readonly strictDisplayGate: boolean;
  /** Whether the backend can run in this interpreter at all. */
  available(interp: Interpreter): boolean;
  /** Whether the backend can take an input of this type (classified by
   *  `typeOfValue`); a unit with any input it cannot take declines. */
  accepts(t: CompilerType): boolean;
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
  /** Whether the unit's output may stream (`Executor.streamsOutput`). */
  streamsOutput(art: A): boolean;
  /** Element-wise kernels in the compiled entry (see `stmtExecutor.ts`). */
  kernels(art: A): number;
}

/** Whether the output of a compiled module is reproduced exactly by the
 *  interpreter's re-run after a bail, so it may stream (see
 *  `Runtime.beginOutputCapture`). Every JIT builtin computes the same bits
 *  on a re-run (the registry rewinds the PRNG) except `toc`, which reads
 *  the clock. `activated` is every runtime snippet the module uses. */
export function outputIsReproducible(
  activated: ReadonlyArray<string>
): boolean {
  return !activated.includes("jit_tic_toc");
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

/** Whether the backend can take every input type. */
export function acceptsAll(
  backend: JitBackend,
  types: readonly CompilerType[]
): boolean {
  return types.every(t => backend.accepts(t));
}

/** The `onJitCompile` label suffix for a unit's input types. */
export function describeTypes(types: readonly CompilerType[]): string {
  return `(${types.map(typeToString).join(", ")})`;
}

/** `%!numbl:assert_jit c` requires C-JIT at --opt 2, so a non-native
 *  backend declines such units there: they either C-JIT or fall through to
 *  the interpreter, which raises on the directive. Plain `assert_jit` only
 *  requires JS-JIT at --opt 1 and does not gate here. */
export function declinedByAssertC(
  backend: JitBackend,
  interp: Interpreter,
  assertsCJit: boolean
): boolean {
  return assertsCJit && !backend.native && interp.optimization === "2";
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
  const values = inputs.map(n => interp.frame.env.get(n));
  const outs = invokeUnit(backend, art, values, scope, interp, where);
  if (!Array.isArray(outs)) return outs;
  for (let i = 0; i < outputs.length; i++) {
    const v = outs[i];
    if (v !== undefined) interp.frame.env.set(outputs[i], v);
  }
  return { result: undefined };
}
