/**
 * AST-walking interpreter for MATLAB code.
 *
 * Walks the parsed AST directly without lowering or codegen.
 * All dispatch decisions are made at runtime using actual values.
 *
 * Function and class lookup goes through the workspace's `CodeIndex` (and
 * the `FunctionIndex` built from it).
 *
 * The Interpreter class holds the interpreter's state (the running call
 * frame, the runtime, JIT bookkeeping) and its public entry points. The
 * evaluation code lives in modules whose functions take the interpreter
 * explicitly:
 *   - execStatements.ts   statements, control flow, loops
 *   - evalExpressions.ts  expressions, operators, literals, member access
 *   - indexing.ts         indexing reads and assignment targets
 *   - handles.ts          anonymous functions and function handles
 *   - calls.ts            calling what a name resolves to
 *   - callConvention.ts   binding, running and returning from a user function
 *   - oop.ts              classes, constructors, enumerations
 *   - argumentsBlock.ts   `arguments` block defaults and name-value structs
 *   - specialContext.ts   the special builtins' context, eval of text
 */

import type { AbstractSyntaxTree, Stmt } from "../parser/types.js";
import type { Runtime } from "../runtime/runtime.js";
import type { RuntimeValue } from "../runtime/types.js";
import type { CodeIndex } from "../workspace/codeIndex.js";
import type { FunctionIndex } from "../workspace/functionIndex.js";
import {
  resolveName,
  resolveFunction,
  resolutionArgType,
  resolveClassMethod,
  functionBodyFor,
} from "../resolution/index.js";

import { Environment, type ControlSignal } from "./types.js";
import { CallFrame } from "./callFrame.js";
import { Registry } from "../executors/registry.js";
import { execStmt } from "./execStatements.js";
import { callResolved, declaredNargout, interpretTarget } from "./calls.js";
import { callBody } from "./callConvention.js";

// ── Interpreter ──────────────────────────────────────────────────────────

export class Interpreter {
  public ans: RuntimeValue | undefined;

  /** @internal */ readonly code: CodeIndex;
  /** @internal */ readonly rt: Runtime;

  /** @internal The running call frame: its environment, file, class and
   *  method, caller and input names. Replaced while a function body runs
   *  (see `CallFrame`). */
  frame: CallFrame;

  /** @internal The base workspace's frame (`evalin('base')`). */
  readonly baseFrame: CallFrame;

  /** @internal Stack of [base, dimIndex, numIndices] for resolving `end` keyword in indexing. */
  endContextStack: Array<{
    base: unknown;
    dimIndex: number;
    numIndices: number;
  }> = [];

  /** @internal Number of enclosing `for` / `while` loop bodies the
   *  interpreter is currently inside. Bumped on body entry, decremented
   *  on exit. Read by the executor registry's `propose()` to gate JIT
   *  attempts: when `loopDepth > 0`, the interpreter is iterating a
   *  hot loop and per-call JIT lookup overhead / spec-cache thrashing
   *  is a net loss. Once the JIT successfully JITs an outer call, the
   *  nested loops execute inside the compiled artifact and the
   *  interpreter never sees them, so this gate only fires on calls
   *  that genuinely happen at the interpreter level inside a loop. */
  loopDepth: number = 0;

  /** @internal Number of those enclosing loops that the JIT *declined*, so
   *  the interpreter is walking their bodies statement by statement. The
   *  `loopDepth > 0` gate above assumes an enclosing loop will be compiled
   *  and will swallow whatever is nested inside it; when that loop was
   *  declined instead, the assumption is false and everything below it would
   *  run interpreted however expensive it is. The executors therefore gate on
   *  `loopDepth > 0 && jitDeclinedLoopDepth === 0`: still no per-iteration JIT
   *  attempts underneath a loop that is about to be compiled, but a nested
   *  loop or call under a declined one gets its own chance. Bumped by the
   *  registry around the interpreter fallback for a loop stmt. */
  jitDeclinedLoopDepth: number = 0;

  /** @internal Number of enclosing conditional blocks (`if` / `switch` /
   *  `try`) whose bodies the interpreter is currently executing. Used by
   *  the loop classifier: when a loop is dispatched with `condBlockDepth >
   *  0` (or `loopDepth > 0`), its sibling list is a nested block, so the
   *  post-loop liveness scan can't see reads after the enclosing block —
   *  the classifier must then keep every loop-assigned name live-out. */
  condBlockDepth: number = 0;

  /** @internal Number of enclosing `try` bodies being executed, across
   *  calls. The C-JIT declines while it is non-zero: a runtime error in
   *  compiled C exits the process, so it can't be caught. */
  tryDepth: number = 0;

  /** @internal Pre-built line break tables for offset→line lookup per file. */
  lineTableCache = new Map<string, number[]>();

  /** @internal file→source mapping for line number computation */
  fileSources = new Map<string, string>();

  /**
   * Optimization mode:
   *   "0" — pure AST interpreter, no JIT.
   *   "1" — JS-JIT (default): top-level + call shapes get
   *         type-specialized JS via the JIT's `compileSpec`.
   *   "2" — C-JIT: scalar/tensor kernels via `compileSpecC` + koffi.
   */
  optimization: import("../executors/plugins.js").OptLevel = "1";

  /** Callback for JIT compilation logging. `lang` distinguishes the
   *  emitted source: `"js"` for the JS-JIT backend, `"c"` for the
   *  C-JIT backend (both can fire in a single `--opt 2` run). */
  onJitCompile?: (description: string, code: string, lang: "js" | "c") => void;

  /** Called when a JIT-compiled unit bails back to the interpreter at
   *  RUNTIME (as opposed to declining at compile time). Today the sole
   *  trigger is an indexed-store array growth that's only detectable at
   *  runtime — the JIT can't model the grown shape, so it bails and the
   *  interpreter re-runs the whole scope with full MATLAB semantics.
   *  Surfaced as a warning (the CLI routes it to stderr); a compile-time
   *  decline stays silent (it's the normal "not JIT-able" path). */
  onJitBail?: (message: string) => void;

  /** Bridge for loading native shared libraries. Undefined in browser
   *  contexts. */
  nativeBridge?: import("../workspace/types.js").NativeBridge;

  /** Verbose log sink (plumbed from ExecOptions.log). */
  log?: (message: string) => void;

  /** Executor registry. Holds the strategies (JS-JIT, C-kernel, ...)
   *  the dispatcher selects among at runtime. The AST interpreter is
   *  the dispatcher's hardcoded fallback (not a registered executor).
   *  Mode-driven plugins (`--opt 1`, `--opt 2`, ...) register
   *  executors during `executeCode` setup. See
   *  docs/developer_reference/executors.md. */
  readonly registry: import("../executors/registry.js").Registry;

  constructor(
    rt: Runtime,
    code: CodeIndex,
    initialVariableValues?: Record<string, RuntimeValue>
  ) {
    this.rt = rt;
    this.code = code;
    const env = new Environment();
    env.rt = rt;
    if (initialVariableValues) {
      for (const [name, value] of Object.entries(initialVariableValues)) {
        env.set(name, value);
      }
    }
    this.baseFrame = new CallFrame(
      code,
      code.mainFileName,
      undefined,
      undefined,
      env,
      undefined,
      undefined
    );
    this.frame = this.baseFrame;
    // The AST interpreter is the dispatcher's hardcoded last-resort
    // fallback (see Registry.dispatch); no plugin registration needed
    // for it. Specialized plugins are registered by executeCode based
    // on the optimization mode.
    this.registry = new Registry();
  }

  /** @internal The current function index (rebuilt after addpath/rmpath/cd). */
  get functionIndex(): FunctionIndex {
    return this.code.functionIndex;
  }

  /** Clear all JIT and function resolution caches. Called after addpath/rmpath.
   *  (Resolution and function-body caches are per function index, which is
   *  rebuilt.) */
  clearAllCaches(): void {
    this.rt.classMethodCache.clear();
    this.registry.clearCache();
  }

  /** @internal Run `fn` with `frame` as the running frame. */
  inFrame<T>(frame: CallFrame, fn: () => T): T {
    const saved = this.frame;
    this.frame = frame;
    try {
      return fn();
    } finally {
      this.frame = saved;
    }
  }

  /** Wire up runtime callbacks so dispatch() routes through the interpreter. */
  installRuntimeCallbacks(): void {
    // A call by name from the runtime (feval, a handle without a closure,
    // operator overloads) resolves from the running frame, without nested
    // functions (as MATLAB's string-named calls).
    this.rt.resolveNamedCall = (name, args, targetClassName) => {
      if (targetClassName) {
        const target = resolveFunction(
          name,
          args.map(resolutionArgType),
          { file: this.frame.file, targetClassName },
          this.functionIndex
        );
        if (!target) return null;
        return (nargout, ...a) => interpretTarget(this, target, a, nargout);
      }
      const r = resolveName(name, this.frame, {
        args,
        variables: false,
        nested: false,
      });
      if (r.kind === "builtin" || r.kind === "notFound") return null;
      return (nargout, ...a) => callResolved(this, r, a, nargout);
    };

    this.rt.resolveClassMethod = (className: string, methodName: string) => {
      const found = resolveClassMethod(
        this.code.classes,
        className,
        methodName
      );
      if (!found) return null;
      const { target, isStatic } = found;
      return (nargout: number, ...args: unknown[]) =>
        interpretTarget(this, target, isStatic ? args.slice(1) : args, nargout);
    };

    this.rt.getClassAncestors = (className: string) =>
      this.code.classes.ancestors(className);

    this.rt.declaredNargout = (name: string) => declaredNargout(this, name);
  }

  /** Run a complete AST (main script). */
  run(ast: AbstractSyntaxTree): void {
    // Execute non-function, non-classdef statements. The main file's local
    // functions were registered with the code index by executeCode.
    const nonFuncStmts = ast.body.filter(
      s => s.type !== "Function" && s.type !== "ClassDef"
    );
    const firstFn = this.code.mainLocalFunctions.keys().next().value;
    if (nonFuncStmts.length === 0 && firstFn !== undefined) {
      // Function file: call the first function with 0 args
      const body = functionBodyFor(this.code, {
        kind: "localFunction",
        name: firstFn,
        source: { from: "main" },
      })!;
      callBody(this, body, [], 0);
    } else {
      // First, try to handle the entire script body as a unit (top-level
      // JIT). If a whole-scope executor takes it, the per-stmt loop is
      // skipped.
      if (!this.registry.tryRunWholeScope(nonFuncStmts, this)) {
        for (let i = 0; i < nonFuncStmts.length; i++) {
          if (this.registry.dispatch(nonFuncStmts, i, this)) break;
        }
      }
    }
  }

  /** Get variable values for ExecResult. */
  getVariableValues(): Record<string, RuntimeValue> {
    return this.frame.env.toRecord();
  }

  /** Execute one statement on the interpreter (the executor registry's
   *  fallback). */
  execStmt(stmt: Stmt): ControlSignal | null {
    return execStmt(this, stmt);
  }
}
