/**
 * Anonymous functions (`@(x) ...`) and named function handles (`@name`).
 */

import {
  parseMFile,
  SyntaxError,
  type AbstractSyntaxTree,
  type Expr,
} from "../parser/index.js";
import type { RuntimeValue } from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { getIBuiltinNargin } from "../runtime/builtinRegistry.js";
import {
  resolveName,
  resolveQualified,
  type Scope,
} from "../resolution/index.js";

import { Environment } from "./types.js";
import { CallFrame } from "./callFrame.js";
import type { Interpreter } from "./interpreter.js";
import { evalExprNargout } from "./evalExpressions.js";
import { callResolved, declaredNargin, declaredNargout } from "./calls.js";

/** Create an anonymous function. It keeps the scope it was defined in, with
 *  a snapshot of the variables visible there (captured by value). Each call
 *  runs in a frame of its own whose environment holds the parameters on top
 *  of that snapshot. */
export function evalAnonFunc(
  interp: Interpreter,
  expr: Extract<Expr, { type: "AnonFunc" }>
): RuntimeValue {
  const defScope = interp.frame.detach(interp.frame.env.snapshot());
  const capturedEnv = defScope.env;
  const bodyExpr = expr.body;
  const paramNames = expr.params;

  const hasVarargin =
    paramNames.length > 0 && paramNames[paramNames.length - 1] === "varargin";
  const regularParams = hasVarargin ? paramNames.slice(0, -1) : paramNames;

  const fn = RTV.func("anonymous", "user");
  fn.jsFn = (nargoutArg: unknown, ...rest: unknown[]) => {
    const fnEnv = new Environment(capturedEnv);
    fnEnv.rt = interp.rt;
    const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
    for (let i = 0; i < regularParams.length; i++) {
      if (i < actualArgs.length) {
        fnEnv.set(regularParams[i], ensureRuntimeValue(actualArgs[i]));
      }
    }
    // nargin / nargout in the body are the anonymous function's own.
    fnEnv.set("$nargin", actualArgs.length);
    fnEnv.set("$nargout", typeof nargoutArg === "number" ? nargoutArg : 1);
    if (hasVarargin) {
      const extraArgs = actualArgs
        .slice(regularParams.length)
        .map(a => ensureRuntimeValue(a));
      fnEnv.set("varargin", RTV.cell(extraArgs, [1, extraArgs.length]));
    }
    const narg = typeof nargoutArg === "number" ? nargoutArg : 1;
    const saved = interp.frame;
    // The body runs in a frame of its own, called from the running one.
    interp.frame = new CallFrame(
      interp.code,
      defScope.file,
      defScope.className,
      defScope.methodName,
      fnEnv,
      saved,
      undefined
    );
    try {
      const result = evalExprNargout(interp, bodyExpr, narg);
      // MATLAB: an anonymous function whose body is an expression (not a
      // multi-output function call) only produces a single output. If more
      // are requested, throw. When the body is a call that actually returned
      // multiple outputs, evalExprNargout returns an array of values, so we
      // allow that.
      if (narg > 1 && !(Array.isArray(result) && result.length >= narg)) {
        throw new RuntimeError("Too many output arguments.");
      }
      // Adopt the result into the caller's transient scope before
      // fnEnv.clearLocals decrefs every binding — without this an
      // anonymous-function output that's the only ref to a value would drop
      // to rc=0 (and _destroy fire) before the caller can bind it.
      const scope = interp.rt.currentScope;
      if (scope) {
        if (Array.isArray(result)) {
          for (const v of result) scope.adopt(v);
        } else {
          scope.adopt(result as RuntimeValue);
        }
      }
      return result;
    } finally {
      fnEnv.clearLocals();
      interp.frame = saved;
    }
  };
  fn.jsFnExpectsNargout = true;
  // MATLAB reports a varargin parameter as a negative count.
  fn.nargin = hasVarargin ? -paramNames.length : paramNames.length;
  // An anonymous function returns whatever its body does.
  fn.nargout = -1;
  // The snapshot incref'd every captured value; balance that with a
  // matching clearLocals when the function wrapper is destroyed.
  // Without this, captured Float64Array buffers stay in the pool's
  // liveSet for the rest of the runtime's life.
  fn.releaseExtra = () => capturedEnv.clearLocals();
  // Expose the snapshot to the alias sweep so a tensor held both in the
  // snapshot and in the parent env triggers COW on parent-side mutation
  // — preserves MATLAB's by-value capture semantics.
  fn.capturedEnv = capturedEnv;
  // Retain the defining AST + file so the JIT can inline a capture-free
  // handle that later crosses a compile boundary (loop input / call arg).
  fn.handleAst = expr;
  fn.handleDefFile = defScope.file;
  return fn;
}

/** Create a handle `@name`. The handle resolves `name` when called, from
 *  the scope it was created in: that file, class and method, and
 *  environment (for nested functions). A dotted name is a package function
 *  or class, or a static method (`@Class.method`), honoring imports. */
export function makeFuncHandle(
  interp: Interpreter,
  name: string
): RuntimeValue {
  const scope = interp.frame.detach();
  const capturedEnv = scope.env;

  // If `name` resolves to a nested function defined here or in an ancestor,
  // the handle keeps that env alive via closure. Mark the chain so the
  // function-exit cleanup skips clearLocals — tearing the env down would
  // strand the handle if it escapes via an output / persistent / global.
  // Builtin/user-function handles don't need this marker; their dispatch
  // doesn't depend on `capturedEnv`'s vars.
  const isNested =
    !name.includes(".") && capturedEnv.markChainForNestedHandle(name);

  const fn = RTV.func(name, "builtin");
  fn.jsFn = (nargout: unknown, ...rest: unknown[]) => {
    const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
    const narg = typeof nargout === "number" ? nargout : 1;
    // The call runs at the handle's site (so a builtin that calls back by
    // name, or a special such as `exist`, resolves from there).
    const saved = interp.frame;
    interp.frame = saved.at(scope);
    try {
      interp.rt.checkCancel();
      const r = resolveName(name, scope, {
        args: actualArgs,
        variables: false,
      });
      return callResolved(interp, r, actualArgs, narg);
    } finally {
      interp.frame = saved;
    }
  };
  fn.jsFnExpectsNargout = true;
  // Populate nargin for the handle: the declared input count of a
  // user/nested/local function (so `nargin(@f)` works for `@f` the same as it
  // would for a call to `f`), else a builtin's (e.g. nargin(@sin) == 1).
  let narg: number | undefined;
  // Never let an introspection lookup break handle creation.
  try {
    narg = declaredNargin(interp, name);
  } catch {
    narg = undefined;
  }
  if (narg === undefined) narg = getIBuiltinNargin(name);
  if (narg !== undefined) fn.nargin = narg;
  try {
    fn.nargout = declaredNargout(interp, name);
  } catch {
    fn.nargout = undefined;
  }
  // For nested-function handles, the function-exit cleanup skipped
  // clearLocals (so the closure could keep its captured env alive).
  // When the handle dies, release those captures by clearing the env.
  if (isNested) {
    fn.releaseExtra = () => capturedEnv.clearLocals();
  } else if (!isMethodHandle(name, scope)) {
    // A plain `@name` handle to a workspace/local/package function (no
    // captured env). Retain its AST + file so the JIT can inline it as an
    // in-scope `@name` constant when it crosses a compile boundary.
    // Nested-function handles are excluded: they depend on capturedEnv,
    // which the inlined form can't reconstruct.
    fn.handleAst = {
      type: "FuncHandle",
      name,
      span: { file: scope.file, start: 0, end: 0 },
    };
    fn.handleDefFile = scope.file;
  }
  return fn;
}

/** `str2func(text)`: a handle to the function `text` names (`'name'` or
 *  `'@name'`), or an anonymous function parsed from `text` (`'@(x) ...'`).
 *  As in MATLAB, the anonymous function is created in an empty workspace
 *  with no access to local functions, and a name does not reach nested
 *  functions. */
export function strToFunc(interp: Interpreter, text: string): RuntimeValue {
  const t = text.trim();
  const env = new Environment();
  env.rt = interp.rt;
  if (/^@\s*\(/.test(t)) {
    let ast: AbstractSyntaxTree;
    try {
      ast = parseMFile(t, "str2func");
    } catch (e) {
      if (e instanceof SyntaxError) throw new RuntimeError(e.message);
      throw e;
    }
    const [stmt, ...rest] = ast.body;
    if (
      rest.length > 0 ||
      stmt?.type !== "ExprStmt" ||
      stmt.expr.type !== "AnonFunc"
    )
      throw new RuntimeError(
        `str2func: '${t}' is not a valid anonymous function`
      );
    const anon = stmt.expr;
    const frame = new CallFrame(
      interp.code,
      "",
      undefined,
      undefined,
      env,
      undefined,
      undefined
    );
    return interp.inFrame(frame, () => evalAnonFunc(interp, anon));
  }
  const name = t.startsWith("@") ? t.slice(1).trim() : t;
  return interp.inFrame(interp.frame.detach(env), () =>
    makeFuncHandle(interp, name)
  );
}

/** Whether `@name` is a handle to a class member (`@Class.staticMethod`). */
function isMethodHandle(name: string, scope: Scope): boolean {
  if (!name.includes(".")) return false;
  const kind = resolveQualified(name.split("."), scope, {
    variables: false,
  }).kind;
  return (
    kind === "staticMethod" || kind === "classMember" || kind === "enumMember"
  );
}
