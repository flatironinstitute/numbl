/**
 * Interpreter function call, resolution, and class method methods.
 * Augments the Interpreter class via prototype assignment.
 */

import type { Expr } from "../parser/types.js";
import type { RuntimeValue } from "../runtime/types.js";
import {
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  RuntimeClassInstance,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError, CancellationError } from "../runtime/error.js";
import { inferDispatchType } from "./builtins/types.js";
import { invokeBuiltin, NOT_A_BUILTIN } from "../runtime/runtimeDispatch.js";
import { toString, toNumber } from "../runtime/convert.js";
import {
  resolveName,
  targetOf,
  definingFile,
  functionBodyFor,
  constructorBodyFor,
  oldStyleConstructorBodyFor,
  type NameResolution,
  type ResolvedTarget,
  type FunctionBody,
} from "../resolution/index.js";
import { parseMFile, type AbstractSyntaxTree } from "../parser/index.js";
import { SyntaxError } from "../parser/errors.js";
import type { ClassInfo } from "../workspace/classInfo.js";
import {
  getInterpreterSpecialBuiltin,
  FALL_THROUGH,
  type InterpreterContext,
} from "./interpreterSpecialBuiltins.js";

import {
  ReturnSignal,
  Environment,
  funcDefFromStmt,
  type FunctionDef,
} from "./types.js";

import type { Interpreter } from "./interpreter.js";

// ── Function calls ───────────────────────────────────────────────────────

/** Call the function `name` resolves to in the current scope (a call
 *  expression or bare name; variables are handled by the caller). */
export function callFunction(
  this: Interpreter,
  name: string,
  args: unknown[],
  nargout: number
): unknown {
  this.rt.checkCancel();
  const r = resolveName(name, this.currentScope(), {
    args,
    variables: false,
  });
  return this.callResolved(r, args, nargout);
}

/** Call a function named by a string (`feval('f', ...)`, runtime dispatch by
 *  name). Like `callFunction`, but, as in MATLAB, nested functions are not
 *  visible. */
export function callByName(
  this: Interpreter,
  name: string,
  args: unknown[],
  nargout: number
): unknown {
  this.rt.checkCancel();
  const r = resolveName(name, this.currentScope(), {
    args,
    variables: false,
    nested: false,
  });
  return this.callResolved(r, args, nargout);
}

/** Call what a name resolved to. */
export function callResolved(
  this: Interpreter,
  r: NameResolution,
  args: unknown[],
  nargout: number
): unknown {
  switch (r.kind) {
    case "nested":
      return this.callNestedFunction(r.fn, r.env, args, nargout);
    case "constant":
      this.pendingInputNames = undefined;
      return r.value;
    case "special": {
      const result = callInterpreterSpecial.call(this, r.name, args, nargout);
      if (result !== FALL_THROUGH) return result;
      if (r.target) return this.interpretTarget(r.target, args, nargout);
      break;
    }
    case "variable":
    case "enumMember":
    case "notFound":
      break;
    default:
      return this.interpretTarget(r.target, args, nargout);
  }
  throw new RuntimeError(`Undefined function or variable '${r.name}'`);
}

/** Run the interpreter special builtin `name` (one that needs the caller's
 *  scope, such as eval or exist). Returns FALL_THROUGH if it declines, in
 *  which case the registered builtin handles the call. */
function callInterpreterSpecial(
  this: Interpreter,
  name: string,
  args: unknown[],
  nargout: number
): unknown {
  const specialHandler = getInterpreterSpecialBuiltin(name);
  if (specialHandler) {
    const ctx: InterpreterContext = {
      env: this.env,
      setEnv: e => {
        this.env = e;
        ctx.env = e;
      },
      callerEnv: this.callerEnv,
      workspaceEnv: this.workspaceEnv,
      evalInLocalScope: (codeArg, fileName, nargout) =>
        this.evalInLocalScope(codeArg, fileName, nargout),
      callByName: (n, a, no) => this.callByName(n, a, no),
      lookupName: n => {
        const scope = this.currentScope();
        const r = resolveName(n, scope);
        return { resolution: r, file: definingFile(r, scope) };
      },
      declaredNargin: n => this.declaredNargin(n),
      rt: this.rt,
      optimization: this.optimization,
      lookupWorkspaceFile: n => {
        const entry = this.code.filesByFuncName.get(n);
        if (entry)
          return {
            path: entry.fileName,
            kind: "function",
            source: entry.source,
          };
        const classInfo = this.code.classes.get(n);
        if (classInfo)
          return {
            path: classInfo.fileName,
            kind: "class",
            source: classInfo.source,
          };
        const jsEntry = this.code.jsUserFunctionsByName.get(n);
        if (jsEntry)
          return {
            path: jsEntry.fileName,
            kind: "jsfunction",
            source: jsEntry.help ?? "",
          };
        return undefined;
      },
      classPublicProperties: n => classPublicProperties(this, n),
      classPublicMethods: n => classPublicMethods(this, n),
    };
    const result = specialHandler(ctx, args, nargout);
    if (result !== FALL_THROUGH) {
      // A special builtin handled the call without reaching callUserFunction,
      // so the pendingInputNames set for this call site (and any callbacks the
      // builtin invokes, e.g. feval/cellfun) must not leak to a later call.
      this.pendingInputNames = undefined;
      return result;
    }
  }
  return FALL_THROUGH;
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
 *  current scope (backs `nargin(@name)` and `nargin('name')`). Undefined
 *  for builtins, constructors, `.numbl.js` functions and unknown names. */
export function declaredNargin(
  this: Interpreter,
  name: string
): number | undefined {
  const r = resolveName(name, this.currentScope(), { variables: false });
  if (r.kind === "nested") return narginFromParams(r.fn.params);
  const target = targetOf(r);
  const body = target && functionBodyFor(this.code, target);
  return body ? narginFromParams(body.fn.params) : undefined;
}

/** Declared output count of the user function `name` resolves to in the
 *  current scope (backs `nargout(@name)` and `nargout('name')`), negated
 *  when the last output is `varargout`. Undefined as for declaredNargin. */
export function declaredNargout(
  this: Interpreter,
  name: string
): number | undefined {
  const r = resolveName(name, this.currentScope(), { variables: false });
  const outputs =
    r.kind === "nested"
      ? r.fn.outputs
      : (() => {
          const target = targetOf(r);
          const body = target && functionBodyFor(this.code, target);
          return body ? body.fn.outputs : undefined;
        })();
  if (outputs === undefined) return undefined;
  const hasVarargout =
    outputs.length > 0 && outputs[outputs.length - 1] === "varargout";
  return hasVarargout ? -outputs.length : outputs.length;
}

// ── Target interpretation ────────────────────────────────────────────────

export function interpretTarget(
  this: Interpreter,
  target: ResolvedTarget,
  args: unknown[],
  nargout: number
): unknown {
  switch (target.kind) {
    case "builtin": {
      const result = invokeBuiltin(this.rt, target.name, nargout, args);
      if (result !== NOT_A_BUILTIN) return result;
      throw new RuntimeError(`Unknown builtin: '${target.name}'`);
    }
    case "localFunction":
    case "privateFunction": {
      const body = functionBodyFor(this.code, target);
      if (!body) {
        const what = target.kind === "privateFunction" ? "Private" : "Local";
        throw new RuntimeError(`${what} function '${target.name}' not found`);
      }
      return this.callBody(body, args, nargout);
    }
    case "workspaceFunction":
      return this.interpretWorkspaceFunction(target, args, nargout);
    case "jsUserFunction":
      return this.interpretJsUserFunction(target, args, nargout);
    case "classMethod":
      return this.interpretClassMethod(target, args, nargout);
    case "workspaceClassConstructor":
      return this.instantiateClass(target.className, args, nargout);
  }
}

/** Call a function body in its own file and class context. */
export function callBody(
  this: Interpreter,
  body: FunctionBody,
  args: unknown[],
  nargout: number,
  narginOverride?: number
): unknown {
  return this.withFileContext(body.file, body.className, body.methodName, () =>
    this.callUserFunction(body.fn, args, nargout, narginOverride)
  );
}

// ── JS user function interpretation ──────────────────────────────────────

export function interpretJsUserFunction(
  this: Interpreter,
  target: Extract<ResolvedTarget, { kind: "jsUserFunction" }>,
  args: unknown[],
  nargout: number
): unknown {
  const entry = this.code.jsUserFunctionsByName.get(target.name);
  if (!entry) {
    throw new RuntimeError(`JS user function '${target.name}' not found`);
  }
  const ib = entry.builtin;
  const margs = args.map(a => ensureRuntimeValue(a));
  const argTypes = margs.map(inferDispatchType);
  const resolution = ib.resolve(argTypes, nargout);
  if (!resolution) {
    const typeNames = argTypes.map(t => t.kind);
    throw new RuntimeError(
      `JS user function '${target.name}' does not support these argument types: (${typeNames.join(", ")})`
    );
  }
  // MATLAB: a function declared with no outputs errors if the call site
  // expects any. Detect via `outputTypes: []`.
  const isVoid = resolution.outputTypes.length === 0;
  if (isVoid && nargout > 0) {
    throw new RuntimeError("Too many output arguments.");
  }
  return this.withFileContext(entry.fileName, undefined, undefined, () => {
    if (this.rt.profilingEnabled) {
      this.rt.profileEnter("jsUserFunction:interp:" + target.name);
      const result = resolution.apply(margs, nargout);
      this.rt.profileLeave();
      return isVoid ? undefined : result;
    }
    const result = resolution.apply(margs, nargout);
    return isVoid ? undefined : result;
  });
}

// ── Workspace function interpretation ────────────────────────────────────

export function interpretWorkspaceFunction(
  this: Interpreter,
  target: Extract<ResolvedTarget, { kind: "workspaceFunction" }>,
  args: unknown[],
  nargout: number
): unknown {
  const body = functionBodyFor(this.code, target);
  if (body) return this.callBody(body, args, nargout);
  const entry = this.code.filesByFuncName.get(target.name);
  if (!entry)
    throw new RuntimeError(`Workspace function '${target.name}' not found`);
  // No function body: the file is a script. MATLAB runs a script invoked by
  // name in the *caller's* workspace (variables are shared), not in an
  // isolated scope, so execute the body in `this.env`. A script takes no
  // arguments and returns no value; it produces no `ans` of its own in the
  // caller.
  if (args.length > 0 || nargout > 0) {
    const err = new RuntimeError(
      `Execution of script ${target.name} as a function is not supported:\n${entry.fileName}`
    );
    err.identifier = "MATLAB:scriptNotAFunction";
    throw err;
  }
  const ast = this.code.getCachedAST(entry.fileName);
  this.withFileContext(entry.fileName, undefined, undefined, () => {
    for (const stmt of ast.body) {
      if (stmt.type === "Function") continue;
      const signal = this.execStmt(stmt);
      if (signal instanceof ReturnSignal) break;
    }
  });
  return undefined;
}

// ── Class method interpretation ──────────────────────────────────────────

export function interpretClassMethod(
  this: Interpreter,
  target: Extract<ResolvedTarget, { kind: "classMethod" }>,
  args: unknown[],
  nargout: number
): unknown {
  const { className, methodName } = target;
  const definingClass = this.code.classes.findDefiningClass(
    className,
    methodName
  );
  const classInfo = this.code.classes.get(definingClass);
  if (!classInfo) throw new RuntimeError(`Class '${definingClass}' not found`);

  if (methodName === classInfo.constructorName) {
    return this.interpretConstructor(classInfo, args, nargout);
  }

  const body = functionBodyFor(this.code, target);
  if (body) {
    const actualArgs =
      target.stripInstance && args.length > 0 ? args.slice(1) : args;
    return this.callBody(body, actualArgs, nargout);
  }
  // Implicit static `ClassName.empty(m,n)`: every MATLAB class inherits
  // an `empty` method producing an empty object array.
  if (methodName === "empty") {
    const dims: number[] = [];
    for (const a of args) {
      const rv = ensureRuntimeValue(a);
      if (isRuntimeTensor(rv)) for (const d of rv.data) dims.push(d);
      else dims.push(toNumber(rv));
    }
    if (dims.length === 1) dims.push(dims[0]);
    if (dims.length === 0) dims.push(0, 0);
    if (!dims.some(d => d === 0)) {
      throw new RuntimeError(
        `${className}.empty: at least one dimension must be zero`
      );
    }
    return RTV.classInstanceArray(className, [], [dims[0], dims[1]]);
  }
  throw new RuntimeError(`No method '${methodName}' for class '${className}'`);
}

// ── Class instantiation ──────────────────────────────────────────────────

// ── Enumeration classes ──────────────────────────────────────────────────

/** Numeric/logical/char superclasses whose constructor produces an
 *  enumeration member's underlying value (e.g. `patchtype < uint32`). */
const NUMERIC_ENUM_SUPERS = new Set([
  "double",
  "single",
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
  "logical",
  "char",
]);

/** Underlying (superclass) value of an enumeration member from its declared
 *  constructor args. MATLAB's default enumeration constructor passes the
 *  member's arguments to the first superclass constructor. */
function enumUnderlyingValue(
  interp: Interpreter,
  classInfo: ClassInfo,
  argExprs: Expr[]
): RuntimeValue | undefined {
  if (argExprs.length === 0) return undefined;
  const first = ensureRuntimeValue(interp.evalExpr(argExprs[0]));
  const sc = classInfo.superClasses[0];
  if (sc && NUMERIC_ENUM_SUPERS.has(sc)) {
    return ensureRuntimeValue(interp.rt.callBuiltin(sc, 1, [first]));
  }
  return first;
}

/** Resolve `ClassName.MemberName` to an enumeration-member instance. Returns
 *  null when `className` is not an enumeration or has no such member. */
export function interpretEnumMember(
  this: Interpreter,
  className: string,
  memberName: string
): RuntimeValue | null {
  const classInfo = this.code.classes.get(className);
  if (!classInfo || !classInfo.enumMembers) return null;
  const memberDef = classInfo.enumMembers.find(m => m.name === memberName);
  if (!memberDef) return null;
  const value = enumUnderlyingValue(this, classInfo, memberDef.args);
  return new RuntimeClassInstance(
    className,
    new Map(),
    false,
    value,
    memberName
  );
}

/** Enumeration constructor-as-converter: `ClassName(x)` for an enumeration
 *  with no explicit constructor. Converts x — a member, a numeric value, a
 *  char/string member name, or an array thereof — to member(s). */
export function convertToEnum(
  this: Interpreter,
  className: string,
  classInfo: ClassInfo,
  args: unknown[]
): RuntimeValue {
  if (args.length === 0) {
    throw new RuntimeError(
      `${className}: enumeration requires an argument to construct`
    );
  }
  const val = ensureRuntimeValue(args[0]);

  // Already an instance / array of this enumeration → identity (value classes
  // are immutable, so no copy is needed).
  if (isRuntimeClassInstance(val) && val._enumMember !== undefined) return val;
  if (isRuntimeClassInstanceArray(val)) return val;

  const members = classInfo.enumMembers!.map(m => ({
    name: m.name,
    value: enumUnderlyingValue(this, classInfo, m.args),
  }));
  const makeMember = (name: string, value: RuntimeValue | undefined) =>
    new RuntimeClassInstance(className, new Map(), false, value, name);
  const fromName = (text: string): RuntimeClassInstance => {
    const m = members.find(mm => mm.name === text);
    if (!m)
      throw new RuntimeError(
        `'${text}' is not a valid member of enumeration '${className}'`
      );
    return makeMember(m.name, m.value);
  };
  const fromNumber = (num: number): RuntimeClassInstance => {
    const m = members.find(
      mm => mm.value !== undefined && toNumber(mm.value) === num
    );
    if (!m)
      throw new RuntimeError(
        `${num} is not a valid underlying value for enumeration '${className}'`
      );
    return makeMember(m.name, m.value);
  };

  if (isRuntimeChar(val)) return fromName(val.value);
  if (isRuntimeString(val)) return fromName(val);
  if (isRuntimeNumber(val)) return fromNumber(val);
  if (isRuntimeLogical(val)) return fromNumber(val ? 1 : 0);
  if (isRuntimeTensor(val)) {
    const elems = Array.from(val.data, x => fromNumber(x));
    if (elems.length === 1) return elems[0];
    const shape: [number, number] = [val.shape[0] ?? 0, val.shape[1] ?? 1];
    return RTV.classInstanceArray(className, elems, shape);
  }
  throw new RuntimeError(`Cannot convert value to enumeration '${className}'`);
}

export function instantiateClass(
  this: Interpreter,
  className: string,
  args: unknown[],
  nargout: number
): unknown {
  const classInfo = this.code.classes.get(className);
  if (!classInfo) {
    return this.rt.callClassMethod(className, className, nargout, args);
  }

  // Enumeration class: `ClassName(x)` converts x to a member (or member array).
  if (classInfo.enumMembers && args.length > 0) {
    return this.convertToEnum(className, classInfo, args);
  }

  // Old-style (pre-classdef) class: the constructor is a plain @Name/Name.m
  // function that builds its own instance via `class(struct, 'Name')`. Call it
  // directly with the user args — do NOT pre-create/prepend an instance.
  if (classInfo.isOldStyle) {
    const body = oldStyleConstructorBodyFor(this.code, classInfo);
    if (!body)
      throw new RuntimeError(
        `Constructor for old-style class '${className}' not found`
      );
    return this.callBody(body, args, nargout);
  }

  const { propertyNames, propertyDefaults } =
    this.collectClassProperties(classInfo);

  const defaults = new Map<string, RuntimeValue>();
  for (const [propName, defaultExpr] of propertyDefaults) {
    defaults.set(
      propName,
      this.evalPropertyDefault(className, propName, defaultExpr)
    );
  }

  const isHandle = this.isHandleClass(classInfo);
  const instance = RTV.classInstance(
    className,
    propertyNames,
    isHandle,
    defaults
  );

  if (classInfo.constructorName) {
    return this.interpretConstructor(classInfo, [instance, ...args], nargout);
  }
  return instance;
}

export function interpretConstructor(
  this: Interpreter,
  classInfo: ClassInfo,
  args: unknown[],
  nargout: number
): unknown {
  const body = constructorBodyFor(this.code, classInfo);
  if (!body) return args[0];
  // nargin in the constructor counts the user args, not the instance.
  return this.callBody(body, args, nargout, args.length - 1);
}

// ── Call user function (core execution) ──────────────────────────────────

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
 *  parameters, `varargin`, nargin/nargout and input names), and hoist the
 *  function's own nested functions. `set` binds a name in `fnEnv`. */
function bindInputs(
  fn: FunctionDef,
  fnEnv: Environment,
  args: unknown[],
  nargout: number,
  nargin: number,
  inputNames: string[] | undefined,
  set: (name: string, value: RuntimeValue) => void
): void {
  // `varargin` is normally last, but name-value struct parameters declared
  // via dotted `arguments` entries may follow it (the parser enforces this).
  const vararginIdx = fn.params.indexOf("varargin");
  const hasVarargin = vararginIdx !== -1;
  const regularParams = hasVarargin
    ? fn.params.slice(0, vararginIdx)
    : fn.params;

  // Count leading positional parameters. Name-value struct parameters
  // declared in an `arguments` block (entries like `opts.method`) are filled
  // from trailing arguments by processArgumentsBlocks, not bound
  // positionally, so positional binding stops before them.
  let numPositional = regularParams.length;
  for (const block of fn.argumentsBlocks ?? []) {
    if (block.kind === "Output") continue;
    for (const e of block.entries) {
      const dot = e.name.indexOf(".");
      if (dot < 0) continue;
      const idx = fn.params.indexOf(e.name.slice(0, dot));
      if (idx >= 0 && idx < numPositional) numPositional = idx;
    }
  }

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
  fnEnv.inputArgNames = inputNames;

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

/** Run a bound function body in `fnEnv` and return its outputs. Shared by
 *  ordinary and nested function calls. */
function runFunctionBody(
  interp: Interpreter,
  fn: FunctionDef,
  fnEnv: Environment,
  args: unknown[],
  nargout: number
): unknown {
  const savedEnv = interp.env;
  const savedCallerEnv = interp.callerEnv;
  interp.callerEnv = savedEnv;
  interp.env = fnEnv;
  interp.rt.pushCallFrame(fn.name);
  interp.rt.pushCleanupScope();

  try {
    // Apply `arguments`-block defaults / name-value structs now that the
    // function scope is active and provided args are bound, so defaults can
    // reference earlier arguments.
    interp.processArgumentsBlocks(fn, args);

    interp.execStmts(fn.body);

    if (fnEnv.persistentFuncId) {
      for (const name of fnEnv.persistentNames) {
        const val = fnEnv.get(name);
        if (val !== undefined) {
          interp.rt.setPersistent(fnEnv.persistentFuncId, name, val);
        }
      }
    }

    const outputs = collectOutputs(fn, fnEnv, nargout);

    // Adopt every output into the caller's transient scope so they
    // survive `fnEnv.clearLocals()` below. Without this, an output
    // that's the only ref to a value would drop to rc=0 (and _destroy
    // fire) before the caller could bind it.
    if (interp.rt.currentScope) {
      for (const v of outputs) interp.rt.currentScope.adopt(v);
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
    interp.rt.annotateError(e);
    throw e;
  } finally {
    interp.rt.popAndRunCleanups(fn => {
      if (fn.jsFn) {
        if (fn.jsFnExpectsNargout) fn.jsFn(0);
        else fn.jsFn();
      } else {
        interp.rt.dispatch(fn.name, 0, []);
      }
    });
    interp.rt.popCallFrame();
    interp.env = savedEnv;
    interp.callerEnv = savedCallerEnv;
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

export function callUserFunction(
  this: Interpreter,
  fn: FunctionDef,
  args: unknown[],
  nargout: number,
  narginOverride?: number
): unknown {
  checkOutputCount(fn, nargout);

  // Consume the call-site argument names recorded by evalFuncCall (for
  // inputname). One-shot: clear immediately so it never leaks to a later
  // call that did not originate from an interpreted call expression.
  const callInputNames = this.pendingInputNames;
  this.pendingInputNames = undefined;

  // Try the registered call executors in order; if all decline or bail,
  // fall through to the regular interpreter call path below.
  // Refcount-driven COW handles aliasing across the call boundary: binding
  // a parameter increfs its argument, so a mutation through the parameter
  // sees rc>1 and copies.
  if (narginOverride === undefined && this.registry.hasCallExecutors) {
    const r = this.registry.dispatchCall(fn, args, nargout, this);
    if (r) return r.result;
  }

  const fnEnv = new Environment();
  fnEnv.rt = this.rt;
  fnEnv.persistentFuncId = `${this.currentFile}:${fn.name}`;
  bindInputs(
    fn,
    fnEnv,
    args,
    nargout,
    narginOverride ?? args.length,
    callInputNames,
    (name, value) => fnEnv.set(name, value)
  );
  return runFunctionBody(this, fn, fnEnv, args, nargout);
}

// ── Nested function call (shares parent scope) ───────────────────────────

export function callNestedFunction(
  this: Interpreter,
  fn: FunctionDef,
  parentEnv: Environment,
  args: unknown[],
  nargout: number
): unknown {
  // Nested functions share the parent workspace by reference — JIT compiles
  // functions as pure (parameter-only), so it cannot handle shared state.
  // Always use the interpreter for nested calls.
  checkOutputCount(fn, nargout);

  // Consume the call-site argument names (for inputname). Without this a
  // nested function would have no names of its own (and must not inherit
  // the parent frame's).
  const callInputNames = this.pendingInputNames;
  this.pendingInputNames = undefined;

  const fnEnv = new Environment(parentEnv);
  fnEnv.isNested = true;
  fnEnv.rt = this.rt;
  fnEnv.persistentFuncId = `${this.currentFile}:${fn.name}`;
  // A nested function's own formal in/out arguments are local to it: writes to
  // them must never be redirected to a same-named variable in the parent
  // workspace (e.g. an output `s` that shadows the parent's `s`).
  fnEnv.nestedLocalNames = new Set([...fn.params, ...fn.outputs]);
  bindInputs(
    fn,
    fnEnv,
    args,
    nargout,
    args.length,
    callInputNames,
    (name, value) => fnEnv.setLocal(name, value)
  );
  return runFunctionBody(this, fn, fnEnv, args, nargout);
}

// ── Context management ───────────────────────────────────────────────────

export function withFileContext<T>(
  this: Interpreter,
  file: string,
  className: string | undefined,
  methodName: string | undefined,
  fn: () => T
): T {
  const savedFile = this.currentFile;
  const savedClassName = this.currentClassName;
  const savedMethodName = this.currentMethodName;
  this.currentFile = file;
  this.currentClassName = className;
  this.currentMethodName = methodName;
  try {
    return fn();
  } finally {
    this.currentFile = savedFile;
    this.currentClassName = savedClassName;
    this.currentMethodName = savedMethodName;
  }
}

/** Evaluate a property's default value expression. MATLAB reports an
 *  invalid default as an error rather than leaving the property unset. */
export function evalPropertyDefault(
  this: Interpreter,
  className: string,
  propName: string,
  defaultExpr: Expr
): RuntimeValue {
  try {
    return ensureRuntimeValue(this.evalExpr(defaultExpr));
  } catch (e) {
    if (e instanceof CancellationError) throw e;
    const msg = e instanceof Error ? e.message : String(e);
    throw new RuntimeError(
      `Invalid default value for property '${propName}' in class '${className}':\n${msg}`
    );
  }
}

// ── Class property helpers ───────────────────────────────────────────────

export function collectClassProperties(
  this: Interpreter,
  classInfo: ClassInfo
): {
  propertyNames: string[];
  propertyDefaults: Map<string, Expr>;
} {
  const propertyNames: string[] = [...classInfo.propertyNames];
  const propertyDefaults = new Map(classInfo.propertyDefaults);

  for (const parentInfo of this.code.classes.lineage(classInfo).slice(1)) {
    for (const propName of parentInfo.propertyNames) {
      if (!propertyNames.includes(propName)) {
        propertyNames.push(propName);
        const defaultExpr = parentInfo.propertyDefaults.get(propName);
        if (defaultExpr && !propertyDefaults.has(propName)) {
          propertyDefaults.set(propName, defaultExpr);
        }
      }
    }
  }
  return { propertyNames, propertyDefaults };
}

/** Public (GetAccess public) property names of a class, including inherited
 *  ones, in declaration order. A Properties block whose `Access` or
 *  `GetAccess` is `private`/`protected` is excluded — matching MATLAB's
 *  `properties(obj)`, which lists only publicly-gettable properties. Returns
 *  undefined when `className` is not a known class. */
function classPublicProperties(
  interp: Interpreter,
  className: string
): string[] | undefined {
  const self = interp.code.classes.get(className);
  if (!self) return undefined;
  const out: string[] = [];
  const seen = new Set<string>();
  for (const info of interp.code.classes.lineage(self)) {
    const ast = info.ast;
    if (ast) {
      for (const member of ast.members) {
        if (member.type !== "Properties") continue;
        const hidden = member.attributes.some(a => {
          const n = a.name.toLowerCase();
          if (n !== "access" && n !== "getaccess") return false;
          const val = (a.value ?? "").toLowerCase();
          return val === "private" || val === "protected";
        });
        if (hidden) continue;
        for (const pn of member.names) {
          if (!seen.has(pn)) {
            seen.add(pn);
            out.push(pn);
          }
        }
      }
    }
  }
  return out;
}

/** Methods inherited from the built-in `handle` class. */
const HANDLE_CLASS_METHODS = [
  "addlistener",
  "delete",
  "eq",
  "findobj",
  "findprop",
  "ge",
  "gt",
  "isvalid",
  "le",
  "listener",
  "lt",
  "ne",
  "notify",
];

/** Public method names of a class (instance and static methods, the
 *  constructor, @folder method files), including inherited ones. Methods in
 *  a block with private/protected Access are excluded, matching MATLAB's
 *  `ismethod`. Returns undefined when `className` is not a known class. */
function classPublicMethods(
  interp: Interpreter,
  className: string
): Set<string> | undefined {
  const self = interp.code.classes.get(className);
  if (!self) return undefined;
  const out = new Set<string>();
  if (self.constructorName) out.add(self.constructorName);
  for (const info of interp.code.classes.lineage(self)) {
    const ast = info.ast;
    if (ast) {
      for (const member of ast.members) {
        if (member.type !== "Methods") continue;
        const hidden = member.attributes.some(a => {
          if (a.name.toLowerCase() !== "access") return false;
          const val = (a.value ?? "").toLowerCase();
          return val !== "public";
        });
        if (hidden) continue;
        const names = [
          ...member.body.flatMap(m => (m.type === "Function" ? [m.name] : [])),
          ...(member.signatures ?? []).map(sig => sig.name),
        ];
        for (const n of names) {
          // skip property accessors (get.X / set.X) and superclass ctors
          if (n.includes(".") || (n === info.name && n !== className)) continue;
          out.add(n);
        }
      }
    } else {
      for (const n of info.methodNames) out.add(n);
      for (const n of info.staticMethodNames) out.add(n);
    }
    for (const n of info.externalMethodFiles.keys())
      if (n !== info.name || info.qualifiedName === className) out.add(n);
  }
  if (interp.code.classes.isHandleClass(self))
    for (const n of HANDLE_CLASS_METHODS) out.add(n);
  return out;
}

export function isHandleClass(
  this: Interpreter,
  classInfo: ClassInfo
): boolean {
  return this.code.classes.isHandleClass(classInfo);
}

// ── eval in local scope ──────────────────────────────────────────────────

/** Run MATLAB text in the current scope (eval, evalin, run, input,
 *  str2num). With `nargout` > 0 the text must be one expression, whose
 *  value(s) are returned; otherwise its statements run (displaying
 *  unsuppressed results) and nothing is returned. `fileName` sets the file
 *  context, e.g. for `run(script)`. */
export function evalInLocalScope(
  this: Interpreter,
  codeArg: unknown,
  fileName?: string,
  nargout = 0
): unknown {
  const code = toString(ensureRuntimeValue(codeArg));
  let ast: AbstractSyntaxTree;
  try {
    ast = parseMFile(code, fileName ?? "eval");
  } catch (e) {
    if (e instanceof SyntaxError) throw new RuntimeError(e.message);
    throw e;
  }
  if (nargout > 0) {
    const [stmt, ...rest] = ast.body;
    if (!stmt || rest.length > 0 || stmt.type !== "ExprStmt") {
      throw new RuntimeError(
        "eval: text must be a single expression when outputs are requested"
      );
    }
    return this.evalExprNargout(stmt.expr, nargout);
  }
  const exec = () => {
    for (const stmt of ast.body) {
      if (stmt.type === "Function") continue;
      if (this.execStmt(stmt) instanceof ReturnSignal) break;
    }
  };
  if (fileName) this.withFileContext(fileName, undefined, undefined, exec);
  else exec();
  return undefined;
}

// ── Arguments block processing ──────────────────────────────────────────

/** Field names declared by dotted (name-value) entries across a function's
 *  input arguments blocks. */
function nameValueFieldNames(fn: FunctionDef): Set<string> {
  const names = new Set<string>();
  for (const block of fn.argumentsBlocks ?? []) {
    if (block.kind === "Output" || block.kind === "OutputRepeating") continue;
    for (const e of block.entries) {
      const dot = e.name.indexOf(".");
      if (dot >= 0) names.add(e.name.slice(dot + 1));
    }
  }
  return names;
}

/** Index into `extras` where the trailing name-value section starts: the
 *  first char/string argument matching a declared option field name.
 *  Everything before it belongs to the repeating (varargin) section. */
function nameValueTailStart(
  extras: unknown[],
  fieldNames: Set<string>
): number {
  if (fieldNames.size === 0) return extras.length;
  for (let i = 0; i < extras.length; i++) {
    if (extras[i] === undefined) continue;
    const rv = ensureRuntimeValue(extras[i]);
    if (
      (isRuntimeChar(rv) || isRuntimeString(rv)) &&
      fieldNames.has(toString(rv))
    ) {
      return i;
    }
  }
  return extras.length;
}

/**
 * Apply `arguments`-block defaults and build name-value struct parameters,
 * binding the results into the *current* function environment.
 *
 * Must be called with `this.env` already set to the function scope and the
 * caller-supplied positional arguments already bound to their parameters.
 * Default expressions are evaluated in that scope so they can reference
 * earlier arguments (e.g. `mergeIdx = surfaceop.defaultIdx(dom)`), matching
 * MATLAB. Entries are matched to parameters by name via `fn.params`, which
 * also handles the constructor case where the output variable is prepended
 * to `params` (shifting argument positions by one).
 */
export function processArgumentsBlocks(
  this: Interpreter,
  fn: FunctionDef,
  args: unknown[]
): void {
  const argBlocks = fn.argumentsBlocks;
  if (!argBlocks || argBlocks.length === 0) return;

  for (const block of argBlocks) {
    if (block.kind === "Output") continue;
    const entries = block.entries;
    if (!entries || entries.length === 0) continue;

    // Split dotted entries (name-value struct fields) from positional ones.
    const nvGroups = new Map<
      string,
      { field: string; defaultExpr: Expr | null }[]
    >();
    const regularEntries = entries.filter(e => {
      const dotIdx = e.name.indexOf(".");
      if (dotIdx >= 0) {
        const paramName = e.name.slice(0, dotIdx);
        const field = e.name.slice(dotIdx + 1);
        let group = nvGroups.get(paramName);
        if (!group) {
          group = [];
          nvGroups.set(paramName, group);
        }
        group.push({ field, defaultExpr: e.defaultValue });
        return false;
      }
      return true;
    });

    // Fill any unsupplied positional argument from its default expression, in
    // declaration order, binding each into the scope so later defaults can
    // see it.
    for (const entry of regularEntries) {
      const pIdx = fn.params.indexOf(entry.name);
      if (pIdx < 0) continue;
      const provided = pIdx < args.length && args[pIdx] !== undefined;
      if (!provided && entry.defaultValue) {
        this.env.set(
          entry.name,
          ensureRuntimeValue(this.evalExpr(entry.defaultValue))
        );
      }
    }

    // Assemble each name-value struct parameter from the trailing arguments
    // (those past the struct parameter's position) plus field defaults.
    const vararginIdx = fn.params.indexOf("varargin");
    for (const [paramName, fields] of nvGroups) {
      const pIdx = fn.params.indexOf(paramName);
      let nvArgs: unknown[];
      if (pIdx >= 0 && vararginIdx >= 0 && vararginIdx < pIdx) {
        // varargin consumes a variable number of arguments; the name-value
        // tail starts at the first declared option name after it.
        const extras = args.slice(vararginIdx);
        nvArgs = extras.slice(
          nameValueTailStart(extras, nameValueFieldNames(fn))
        );
      } else {
        nvArgs = args.slice(pIdx >= 0 ? pIdx : args.length);
      }
      const defaults: Record<string, unknown> = {};
      for (const { field, defaultExpr } of fields) {
        if (defaultExpr) defaults[field] = this.evalExpr(defaultExpr);
      }
      this.env.set(paramName, this.rt.buildNameValueStruct(nvArgs, defaults));
    }
  }
}
