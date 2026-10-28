/**
 * Calling what a name resolves to: user functions, builtins, `.numbl.js`
 * functions, scripts, class methods and constructors.
 */

import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { invokeBuiltin, NOT_A_BUILTIN } from "../runtime/runtimeDispatch.js";
import {
  resolveName,
  targetOf,
  functionBodyFor,
  type NameResolution,
  type ResolvedTarget,
} from "../resolution/index.js";

import { ReturnSignal } from "./types.js";
import type { Interpreter } from "./interpreter.js";
import { callBody, callNestedFunction } from "./callConvention.js";
import { execStmt } from "./execStatements.js";
import { instantiateClass, interpretClassMethod } from "./oop.js";

/** Call the function `name` resolves to in the running frame (a call
 *  expression or bare name; variables are handled by the caller). */
export function callFunction(
  interp: Interpreter,
  name: string,
  args: unknown[],
  nargout: number
): unknown {
  interp.rt.checkCancel();
  const r = resolveName(name, interp.frame, {
    args,
    variables: false,
  });
  return callResolved(interp, r, args, nargout);
}

/** Call what a name resolved to. */
export function callResolved(
  interp: Interpreter,
  r: NameResolution,
  args: unknown[],
  nargout: number
): unknown {
  switch (r.kind) {
    case "nested":
      return callNestedFunction(interp, r.fn, r.env, args, nargout);
    case "constant":
      return r.value;
    case "variable":
    case "enumMember":
    case "notFound":
      break;
    default:
      return interpretTarget(interp, r.target, args, nargout);
  }
  throw new RuntimeError(`Undefined function or variable '${r.name}'`, null, {
    identifier: "MATLAB:UndefinedFunction",
  });
}

export function interpretTarget(
  interp: Interpreter,
  target: ResolvedTarget,
  args: unknown[],
  nargout: number
): unknown {
  switch (target.kind) {
    case "builtin": {
      const result = invokeBuiltin(interp.rt, target.name, nargout, args);
      if (result !== NOT_A_BUILTIN) return result;
      throw new RuntimeError(`Unknown builtin: '${target.name}'`);
    }
    case "localFunction":
    case "privateFunction": {
      const body = functionBodyFor(interp.code, target);
      if (!body) {
        const what = target.kind === "privateFunction" ? "Private" : "Local";
        throw new RuntimeError(`${what} function '${target.name}' not found`);
      }
      return callBody(interp, body, args, nargout);
    }
    case "workspaceFunction":
      return interpretWorkspaceFunction(interp, target, args, nargout);
    case "jsUserFunction":
      return interpretJsUserFunction(interp, target, args, nargout);
    case "classMethod":
      return interpretClassMethod(interp, target, args, nargout);
    case "workspaceClassConstructor":
      return instantiateClass(interp, target.className, args, nargout);
  }
}

// ── nargin of a named function ───────────────────────────────────────────

/** MATLAB `nargin`: the declared input count, negated when the last
 *  parameter is `varargin`. */
function narginFromParams(params: string[]): number {
  const hasVarargin =
    params.length > 0 && params[params.length - 1] === "varargin";
  return hasVarargin ? -params.length : params.length;
}

/** Declared input count of the user function `name` resolves to in the
 *  running frame (backs `nargin(@name)` and `nargin('name')`). Undefined
 *  for builtins, constructors, `.numbl.js` functions and unknown names. */
export function declaredNargin(
  interp: Interpreter,
  name: string
): number | undefined {
  const r = resolveName(name, interp.frame, { variables: false });
  if (r.kind === "nested") return narginFromParams(r.fn.params);
  const target = targetOf(r);
  const body = target && functionBodyFor(interp.code, target);
  return body ? narginFromParams(body.fn.params) : undefined;
}

/** Declared output count of the user function `name` resolves to in the
 *  running frame (backs `nargout(@name)` and `nargout('name')`), negated
 *  when the last output is `varargout`. Undefined as for declaredNargin. */
export function declaredNargout(
  interp: Interpreter,
  name: string
): number | undefined {
  const r = resolveName(name, interp.frame, { variables: false });
  let outputs: string[] | undefined;
  if (r.kind === "nested") outputs = r.fn.outputs;
  else {
    const target = targetOf(r);
    const body = target && functionBodyFor(interp.code, target);
    outputs = body ? body.fn.outputs : undefined;
  }
  if (outputs === undefined) return undefined;
  const hasVarargout =
    outputs.length > 0 && outputs[outputs.length - 1] === "varargout";
  return hasVarargout ? -outputs.length : outputs.length;
}

// ── JS user functions ────────────────────────────────────────────────────

function interpretJsUserFunction(
  interp: Interpreter,
  target: Extract<ResolvedTarget, { kind: "jsUserFunction" }>,
  args: unknown[],
  nargout: number
): unknown {
  const entry = interp.code.jsUserFunctionsByName.get(target.name);
  if (!entry) {
    throw new RuntimeError(`JS user function '${target.name}' not found`);
  }
  const margs = args.map(a => ensureRuntimeValue(a));
  // Calls back by name from the function resolve from its file.
  return interp.inFrame(interp.frame.at({ file: entry.fileName }), () => {
    if (!interp.rt.profilingEnabled) return entry.builtin.call(margs, nargout);
    interp.rt.profileEnter("jsUserFunction:interp:" + target.name);
    try {
      return entry.builtin.call(margs, nargout);
    } finally {
      interp.rt.profileLeave();
    }
  });
}

// ── Workspace functions and scripts ──────────────────────────────────────

function interpretWorkspaceFunction(
  interp: Interpreter,
  target: Extract<ResolvedTarget, { kind: "workspaceFunction" }>,
  args: unknown[],
  nargout: number
): unknown {
  const body = functionBodyFor(interp.code, target);
  if (body) return callBody(interp, body, args, nargout);
  const entry = interp.code.filesByFuncName.get(target.name);
  if (!entry)
    throw new RuntimeError(`Workspace function '${target.name}' not found`);
  // No function body: the file is a script. MATLAB runs a script invoked by
  // name in the *caller's* workspace (variables are shared), not in an
  // isolated scope, so execute the body in the running frame's environment.
  // A script takes no arguments and returns no value; it produces no `ans`
  // of its own in the caller.
  if (args.length > 0 || nargout > 0) {
    const err = new RuntimeError(
      `Execution of script ${target.name} as a function is not supported:\n${entry.fileName}`
    );
    err.identifier = "MATLAB:scriptNotAFunction";
    throw err;
  }
  const ast = interp.code.getCachedAST(entry.fileName);
  interp.inFrame(interp.frame.at({ file: entry.fileName }), () => {
    for (const stmt of ast.body) {
      if (stmt.type === "Function") continue;
      const signal = execStmt(interp, stmt);
      if (signal instanceof ReturnSignal) break;
    }
  });
  return undefined;
}
