/**
 * The calling convention for user functions: check the output count, bind
 * the inputs, run the body in a new call frame, and collect the outputs.
 */

import type { RuntimeValue } from "../runtime/types.js";
import { isRuntimeCell } from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import type { FunctionBody } from "../resolution/index.js";

import { Environment, funcDefFromStmt, type FunctionDef } from "./types.js";
import { CallFrame } from "./callFrame.js";
import type { Interpreter } from "./interpreter.js";
import { execStmts } from "./execStatements.js";
import {
  nameValueFieldNames,
  nameValueTailStart,
  positionalParamCount,
  processArgumentsBlocks,
} from "./argumentsBlock.js";

/** Call a function body in its own file and class context. A JIT executor
 *  may run the call; otherwise the body runs on the interpreter in a new
 *  frame. `narginOverride` sets `nargin` (a constructor counts only the user
 *  arguments, not the instance). The callee's `inputname` reads the
 *  argument names of the call expression the calling frame is evaluating. */
export function callBody(
  interp: Interpreter,
  body: FunctionBody,
  args: unknown[],
  nargout: number,
  narginOverride?: number
): unknown {
  const fn = body.fn;
  checkOutputCount(fn, nargout);

  // Try the registered call executors in order; if all decline or bail,
  // fall through to the interpreter. Refcount-driven COW handles aliasing
  // across the call boundary: binding a parameter increfs its argument, so
  // a mutation through the parameter sees rc>1 and copies.
  if (narginOverride === undefined && interp.registry.hasCallExecutors) {
    const r = interp.registry.dispatchCall(fn, args, nargout, interp);
    if (r) return r.result;
  }

  const fnEnv = new Environment();
  fnEnv.rt = interp.rt;
  fnEnv.persistentFuncId = `${body.file}:${fn.name}`;
  bindInputs(
    fn,
    fnEnv,
    args,
    nargout,
    narginOverride ?? args.length,
    (name, value) => fnEnv.set(name, value)
  );
  const frame = new CallFrame(
    interp.code,
    body.file,
    body.className,
    body.methodName,
    fnEnv,
    interp.frame,
    interp.frame.callSiteNames
  );
  return runFunctionBody(interp, fn, frame, args, nargout);
}

/** Call a nested function. It shares its parent's workspace by reference
 *  (the JIT compiles functions as pure, parameter-only code, so nested
 *  calls always run on the interpreter) and runs in the calling frame's
 *  file and class context. */
export function callNestedFunction(
  interp: Interpreter,
  fn: FunctionDef,
  parentEnv: Environment,
  args: unknown[],
  nargout: number
): unknown {
  checkOutputCount(fn, nargout);

  const caller = interp.frame;
  const fnEnv = new Environment(parentEnv);
  fnEnv.isNested = true;
  fnEnv.rt = interp.rt;
  fnEnv.persistentFuncId = `${caller.file}:${fn.name}`;
  // A nested function's own formal in/out arguments are local to it: writes to
  // them must never be redirected to a same-named variable in the parent
  // workspace (e.g. an output `s` that shadows the parent's `s`).
  fnEnv.nestedLocalNames = new Set([...fn.params, ...fn.outputs]);
  bindInputs(fn, fnEnv, args, nargout, args.length, (name, value) =>
    fnEnv.setLocal(name, value)
  );
  const frame = new CallFrame(
    interp.code,
    caller.file,
    caller.className,
    caller.methodName,
    fnEnv,
    caller,
    caller.callSiteNames
  );
  return runFunctionBody(interp, fn, frame, args, nargout);
}

/** MATLAB: calling a function with more outputs than it declares is an
 *  error raised before the body runs. `varargout` functions can supply any
 *  number. */
function checkOutputCount(fn: FunctionDef, nargout: number): void {
  const hasVarargout =
    fn.outputs.length > 0 && fn.outputs[fn.outputs.length - 1] === "varargout";
  if (!hasVarargout && nargout > fn.outputs.length) {
    throw new RuntimeError("Too many output arguments.");
  }
}

/** Bind call arguments to `fn`'s parameters in `fnEnv` (positional
 *  parameters, `varargin`, nargin and nargout), and hoist the function's
 *  own nested functions. `set` binds a name in `fnEnv`. */
function bindInputs(
  fn: FunctionDef,
  fnEnv: Environment,
  args: unknown[],
  nargout: number,
  nargin: number,
  set: (name: string, value: RuntimeValue) => void
): void {
  // `varargin` is normally last, but name-value struct parameters declared
  // via dotted `arguments` entries may follow it (the parser enforces this).
  const vararginIdx = fn.params.indexOf("varargin");
  const hasVarargin = vararginIdx !== -1;
  const regularParams = hasVarargin
    ? fn.params.slice(0, vararginIdx)
    : fn.params;

  const numPositional = positionalParamCount(fn, regularParams);
  for (let i = 0; i < numPositional; i++) {
    if (i < args.length && args[i] !== undefined) {
      set(regularParams[i], ensureRuntimeValue(args[i]));
    }
  }
  if (hasVarargin) {
    let extras = args.slice(regularParams.length);
    // When name-value struct params follow varargin, the repeating section
    // ends at the first argument naming a declared option.
    if (vararginIdx !== fn.params.length - 1) {
      extras = extras.slice(
        0,
        nameValueTailStart(extras, nameValueFieldNames(fn))
      );
    }
    const extraArgs = extras.map(a => ensureRuntimeValue(a));
    set("varargin", RTV.cell(extraArgs, [1, extraArgs.length]));
  }
  set("$nargin", nargin);
  set("$nargout", nargout);

  // Pre-register nested function definitions (hoisted, like MATLAB): a
  // nested function is visible throughout its parent's body regardless of
  // definition order.
  for (const stmt of fn.body) {
    if (stmt.type === "Function") {
      fnEnv.nestedFunctions.set(stmt.name, {
        fn: funcDefFromStmt(stmt),
        env: fnEnv,
      });
    }
  }
}

/** Run a bound function body in `frame` and return its outputs. Shared by
 *  ordinary and nested function calls. */
function runFunctionBody(
  interp: Interpreter,
  fn: FunctionDef,
  frame: CallFrame,
  args: unknown[],
  nargout: number
): unknown {
  const rt = interp.rt;
  const fnEnv = frame.env;
  rt.pushCallFrame(fn.name);
  rt.pushCleanupScope();
  const saved = interp.frame;
  interp.frame = frame;

  try {
    // Apply `arguments`-block defaults / name-value structs now that the
    // function scope is active and provided args are bound, so defaults can
    // reference earlier arguments.
    processArgumentsBlocks(interp, fn, args);

    execStmts(interp, fn.body);

    if (fnEnv.persistentFuncId) {
      for (const name of fnEnv.persistentNames) {
        const val = fnEnv.get(name);
        if (val !== undefined) {
          rt.setPersistent(fnEnv.persistentFuncId, name, val);
        }
      }
    }

    const outputs = collectOutputs(fn, fnEnv, nargout);

    // Adopt every output into the caller's transient scope so they
    // survive `fnEnv.clearLocals()` below. Without this, an output
    // that's the only ref to a value would drop to rc=0 (and _destroy
    // fire) before the caller could bind it.
    if (rt.currentScope) {
      for (const v of outputs) rt.currentScope.adopt(v);
    }

    // Drop locals after collecting outputs. Skip when a `@nestedFn`
    // handle was created during this function's execution: that handle's
    // closure references `fnEnv`, and clearing its vars would strand the
    // handle if it escaped via an output / persistent / global. Defining
    // a nested function alone is fine — only escaping handles matter.
    //
    // Note: an escaping nested-fn handle creates a refcount cycle —
    // fnEnv holds the handle (via `vars`), the handle holds fnEnv (via
    // its jsFn closure). The handle's `releaseExtra` clears fnEnv when
    // the handle's external refs drop, but the cycle prevents that
    // path from firing. Result: 1 buffer per escaping nested handle
    // can stay in the pool's liveSet for the runtime's lifetime.
    // Acceptable trade-off: breaking the cycle (e.g. by deleting the
    // output's binding from fnEnv) would strand recursive / mutually-
    // recursive nested handles whose bodies refer to each other by
    // their parent-local names — see
    // `numbl_test_scripts/functions/recursive_nested_handles.m`.
    if (!fnEnv.nestedHandleCreated) {
      fnEnv.clearLocals();
    }

    return nargout <= 1 ? outputs[0] : outputs;
  } catch (e) {
    if (!fnEnv.nestedHandleCreated) fnEnv.clearLocals();
    rt.annotateError(e);
    throw e;
  } finally {
    rt.popAndRunCleanups(fn => {
      if (fn.jsFn) {
        if (fn.jsFnExpectsNargout) fn.jsFn(0);
        else fn.jsFn();
      } else {
        rt.dispatch(fn.name, 0, []);
      }
    });
    rt.popCallFrame();
    interp.frame = saved;
  }
}

/** Read a finished call's outputs from its environment. */
function collectOutputs(
  fn: FunctionDef,
  fnEnv: Environment,
  nargout: number
): RuntimeValue[] {
  const hasVarargout =
    fn.outputs.length > 0 && fn.outputs[fn.outputs.length - 1] === "varargout";
  const regularOutputs = hasVarargout ? fn.outputs.slice(0, -1) : fn.outputs;

  // When nargout==0 but the function defines outputs, still collect the
  // first output so it can be used as `ans`.  The function body already
  // saw $nargout==0, so guards like `if nargout > 0` behave correctly.
  const collectCount =
    nargout === 0 && regularOutputs.length > 0
      ? 1
      : Math.min(regularOutputs.length, nargout);

  const outputs: RuntimeValue[] = [];
  for (let i = 0; i < collectCount; i++) {
    const val = fnEnv.get(regularOutputs[i]);
    if (val === undefined && nargout >= i + 1) {
      throw new RuntimeError(
        `Output argument '${regularOutputs[i]}' (and maybe others) not assigned during call to '${fn.name}'`
      );
    }
    outputs.push(val ?? RTV.num(0));
  }

  if (hasVarargout && nargout > regularOutputs.length) {
    const varargoutVal = fnEnv.get("varargout");
    if (varargoutVal && isRuntimeCell(varargoutVal)) {
      const remaining = nargout - regularOutputs.length;
      for (let i = 0; i < Math.min(remaining, varargoutVal.data.length); i++) {
        outputs.push(ensureRuntimeValue(varargoutVal.data[i]));
      }
    }
  }
  return outputs;
}
