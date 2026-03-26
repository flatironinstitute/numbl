/**
 * Interpreter function call, resolution, and class method methods.
 * Augments the Interpreter class via prototype assignment.
 */

import type { Expr } from "../parser/types.js";
import type { RuntimeValue } from "../runtime/types.js";
import {
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeClassInstance,
  isRuntimeFunction,
} from "../runtime/types.js";
import { RTV, getItemTypeFromRuntimeValue } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { shareRuntimeValue } from "../runtime/utils.js";
import type { CallSite } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { tryJitCall, JIT_SKIP } from "./jit/index.js";
import { getIBuiltin, inferJitType } from "./builtins/index.js";
import { toNumber, toString } from "../runtime/convert.js";
import { resolveFunction, type ResolvedTarget } from "../functionResolve.js";
import type { ClassInfo } from "../lowering/classInfo.js";

import {
  ReturnSignal,
  Environment,
  funcDefFromStmt,
  type FunctionDef,
} from "./types.js";

import type { Interpreter } from "./interpreter.js";

// ── Function calls ───────────────────────────────────────────────────────

export function callFunction(
  this: Interpreter,
  name: string,
  args: unknown[],
  nargout: number
): unknown {
  // 0. Intrinsics
  if (name === "eval" && args.length === 1) {
    return this.evalInLocalScope(args[0]);
  }
  if (name === "evalin" && args.length >= 2) {
    const scope = toString(ensureRuntimeValue(args[0]));
    const targetEnv =
      scope === "caller"
        ? this.callerEnv
        : scope === "workspace"
          ? this.workspaceEnv
          : undefined;
    if (targetEnv) {
      const code = toString(ensureRuntimeValue(args[1]));
      // Simple variable name — just look it up
      if (/^[a-zA-Z_]\w*$/.test(code)) {
        const val = targetEnv.get(code);
        if (val !== undefined) return val;
        // If a default value was provided (3rd arg), return it
        if (args.length >= 3) return ensureRuntimeValue(args[2]);
        throw new RuntimeError(
          `Variable '${code}' does not exist in ${scope} scope`
        );
      }
      // Otherwise evaluate as code in target scope
      const savedEnv = this.env;
      this.env = targetEnv;
      try {
        return this.evalInLocalScope(args[1]);
      } finally {
        this.env = savedEnv;
      }
    }
    return this.evalInLocalScope(args[1]);
  }
  if (name === "assignin" && args.length >= 3) {
    const scope = toString(ensureRuntimeValue(args[0]));
    const varName = toString(ensureRuntimeValue(args[1]));
    const val = ensureRuntimeValue(args[2]);
    const targetEnv =
      scope === "caller"
        ? this.callerEnv
        : scope === "workspace"
          ? this.workspaceEnv
          : undefined;
    if (targetEnv) {
      targetEnv.set(varName, val);
    } else {
      this.env.set(varName, val);
    }
    return undefined;
  }
  if (name === "feval" && args.length >= 1) {
    const first = ensureRuntimeValue(args[0]);
    if (isRuntimeFunction(first)) {
      return this.rt.index(first, args.slice(1), nargout);
    }
    // Class instance: feval(obj, ...) dispatches to the class's feval method
    if (isRuntimeClassInstance(first)) {
      // Fall through to normal resolution which finds feval as a class method
    } else {
      const funcName = toString(first);
      const funcArgs = args.slice(1);
      return this.callFunction(funcName, funcArgs, nargout);
    }
  }
  if (name === "exist" && args.length >= 2) {
    const nameArg = toString(ensureRuntimeValue(args[0]));
    const typeArg = toString(ensureRuntimeValue(args[1]));
    if (typeArg === "var") {
      return this.env.has(nameArg) ? 1 : 0;
    }
    if (typeArg === "builtin") {
      return this.rt.builtins[nameArg] || getIBuiltin(nameArg) ? 5 : 0;
    }
  }
  if (name === "who" || name === "whos") {
    const getters: Record<string, () => unknown> = {};
    for (const varName of this.env.localNames()) {
      if (varName.startsWith("$")) continue;
      const n = varName;
      getters[n] = () => this.env.get(n);
    }
    if (name === "who") {
      return this.rt.who(nargout, getters, args);
    } else {
      return this.rt.whos(nargout, getters, args);
    }
  }
  if (name === "isa" && args.length === 2) {
    return this.rt.isa(args[0], args[1]);
  }
  if (name === "__inferred_type_str" && args.length === 1) {
    const rv = ensureRuntimeValue(args[0]);
    if (isRuntimeNumber(rv)) return RTV.string("Number");
    if (isRuntimeTensor(rv)) return RTV.string("Tensor");
    if (isRuntimeCell(rv)) return RTV.string("Cell");
    if (isRuntimeClassInstance(rv))
      return RTV.string(`ClassInstance(${rv.className})`);
    if (isRuntimeChar(rv)) return RTV.string("Char");
    if (isRuntimeString(rv)) return RTV.string("String");
    if (isRuntimeFunction(rv)) return RTV.string("Function");
    if (typeof rv === "boolean") return RTV.string("Boolean");
    return RTV.string("Unknown");
  }
  if (name === "nargin" && args.length === 0) {
    const v = this.env.get("$nargin");
    return v !== undefined ? v : 0;
  }
  if (name === "nargout" && args.length === 0) {
    const v = this.env.get("$nargout");
    return v !== undefined ? v : 0;
  }
  if (name === "narginchk" && args.length === 2) {
    const narginVal = this.env.get("$nargin");
    const n = typeof narginVal === "number" ? narginVal : 0;
    const lo = toNumber(ensureRuntimeValue(args[0]));
    const hi = toNumber(ensureRuntimeValue(args[1]));
    if (n < lo) throw new RuntimeError("Not enough input arguments.");
    if (n > hi) throw new RuntimeError("Too many input arguments.");
    return undefined;
  }
  if (name === "nargoutchk" && args.length === 2) {
    const nargoutVal = this.env.get("$nargout");
    const n = typeof nargoutVal === "number" ? nargoutVal : 0;
    const lo = toNumber(ensureRuntimeValue(args[0]));
    const hi = toNumber(ensureRuntimeValue(args[1]));
    if (n < lo) throw new RuntimeError("Not enough output arguments.");
    if (n > hi) throw new RuntimeError("Too many output arguments.");
    return undefined;
  }

  // 1. Check nested functions (share parent scope)
  const nested = this.env.getNestedFunction(name);
  if (nested) {
    return this.callNestedFunction(nested.fn, nested.env, args, nargout);
  }

  // 2. Resolve using function index
  const argTypes = args.map(a =>
    getItemTypeFromRuntimeValue(ensureRuntimeValue(a))
  );
  const callSite: CallSite = {
    file: this.currentFile,
    ...(this.currentClassName ? { className: this.currentClassName } : {}),
    ...(this.currentMethodName ? { methodName: this.currentMethodName } : {}),
  };
  const target = resolveFunction(name, argTypes, callSite, this.functionIndex);
  if (target) {
    return this.interpretTarget(target, args, nargout);
  }

  throw new RuntimeError(`Undefined function or variable '${name}'`);
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
      // Check IBuiltin first (interpreter builtins with JIT-compatible type rules)
      const ib = getIBuiltin(target.name);
      if (ib) {
        const margs = args.map(a => ensureRuntimeValue(a));
        const argTypes = margs.map(inferJitType);
        const resolution = ib.resolve(argTypes, nargout);
        if (resolution) {
          if (this.rt.profilingEnabled) {
            this.rt.profileEnter("builtin:interp:" + target.name);
            const result = resolution.apply(margs, nargout);
            this.rt.profileLeave();
            return result;
          }
          return resolution.apply(margs, nargout);
        }
      }
      const builtin = this.rt.builtins[target.name];
      if (builtin) return builtin(nargout, args);
      throw new RuntimeError(`Unknown builtin: '${target.name}'`);
    }
    case "localFunction":
      return this.interpretLocalFunction(target, args, nargout);
    case "workspaceFunction":
      return this.interpretWorkspaceFunction(target, args, nargout);
    case "classMethod":
      return this.interpretClassMethod(target, args, nargout);
    case "workspaceClassConstructor":
      return this.instantiateClass(target.className, args, nargout);
    case "privateFunction":
      return this.interpretPrivateFunction(target, args, nargout);
  }
}

// ── Local function interpretation ────────────────────────────────────────

export function interpretLocalFunction(
  this: Interpreter,
  target: Extract<ResolvedTarget, { kind: "localFunction" }>,
  args: unknown[],
  nargout: number
): unknown {
  const { source } = target;

  if (source.from === "main") {
    const fn = this.mainLocalFunctions.get(target.name);
    if (!fn)
      throw new RuntimeError(`Local function '${target.name}' not found`);
    return this.callUserFunction(fn, args, nargout);
  }

  if (source.from === "workspaceFile") {
    const fn = this.findFunctionInWorkspaceFile(source.wsName, target.name);
    if (!fn)
      throw new RuntimeError(
        `Local function '${target.name}' not found in workspace file '${source.wsName}'`
      );
    return this.withFileContext(
      this.getWorkspaceFileName(source.wsName),
      undefined,
      undefined,
      () => this.callUserFunction(fn, args, nargout)
    );
  }

  if (source.from === "classFile") {
    const fn = this.findFunctionInClassFile(
      source.className,
      target.name,
      source.methodScope
    );
    if (!fn)
      throw new RuntimeError(
        `Local function '${target.name}' not found in class file '${source.className}'`
      );
    // Use the external method file as context if methodScope points to one
    const classInfo = this.ctx.getClassInfo(source.className);
    const extFile = classInfo?.externalMethodFiles.get(
      source.methodScope ?? ""
    );
    const fileCtx =
      extFile?.fileName ?? this.getClassFileName(source.className);
    return this.withFileContext(
      fileCtx,
      source.className,
      source.methodScope,
      () => this.callUserFunction(fn, args, nargout)
    );
  }

  if (source.from === "privateFile") {
    const ast = this.ctx.getCachedAST(source.callerFile);
    for (const stmt of ast.body) {
      if (stmt.type === "Function" && stmt.name === target.name) {
        const fn = funcDefFromStmt(stmt);
        return this.withFileContext(
          source.callerFile,
          undefined,
          undefined,
          () => this.callUserFunction(fn, args, nargout)
        );
      }
    }
    throw new RuntimeError(
      `Local function '${target.name}' not found in private file`
    );
  }

  throw new RuntimeError(`Unknown local function source`);
}

// ── Workspace function interpretation ────────────────────────────────────

export function interpretWorkspaceFunction(
  this: Interpreter,
  target: Extract<ResolvedTarget, { kind: "workspaceFunction" }>,
  args: unknown[],
  nargout: number
): unknown {
  const dotIdx = target.name.lastIndexOf(".");
  const primaryName = dotIdx >= 0 ? target.name.slice(dotIdx + 1) : target.name;

  const fn = this.findFunctionInWorkspaceFile(target.name, primaryName);
  if (!fn) {
    const entry = this.ctx.registry.filesByFuncName.get(target.name);
    if (entry) {
      const ast = this.ctx.getCachedAST(entry.fileName);
      return this.withFileContext(entry.fileName, undefined, undefined, () => {
        const savedEnv = this.env;
        this.env = new Environment();
        try {
          for (const stmt of ast.body) {
            if (stmt.type === "Function") continue;
            const signal = this.execStmt(stmt);
            if (signal instanceof ReturnSignal) break;
          }
          return this.ans;
        } finally {
          this.env = savedEnv;
        }
      });
    }
    throw new RuntimeError(`Workspace function '${target.name}' not found`);
  }

  return this.withFileContext(
    this.getWorkspaceFileName(target.name),
    undefined,
    undefined,
    () => this.callUserFunction(fn, args, nargout)
  );
}

// ── Class method interpretation ──────────────────────────────────────────

export function interpretClassMethod(
  this: Interpreter,
  target: Extract<ResolvedTarget, { kind: "classMethod" }>,
  args: unknown[],
  nargout: number
): unknown {
  const { className, methodName } = target;
  const definingClass = this.ctx.findDefiningClass(className, methodName);
  const classInfo = this.ctx.getClassInfo(definingClass);
  if (!classInfo) throw new RuntimeError(`Class '${definingClass}' not found`);

  if (methodName === classInfo.constructorName) {
    return this.interpretConstructor(classInfo, args, nargout);
  }

  const methodFn = this.findMethodInClass(classInfo, methodName);
  if (!methodFn) {
    const extFn = this.findExternalMethod(classInfo, methodName);
    if (extFn) {
      const actualArgs =
        target.stripInstance && args.length > 0 ? args.slice(1) : args;
      return this.withFileContext(
        classInfo.externalMethodFiles.get(methodName)?.fileName ??
          classInfo.fileName,
        definingClass,
        methodName,
        () => this.callUserFunction(extFn, actualArgs, nargout)
      );
    }
    throw new RuntimeError(
      `No method '${methodName}' for class '${className}'`
    );
  }

  const actualArgs =
    target.stripInstance && args.length > 0 ? args.slice(1) : args;

  return this.withFileContext(
    classInfo.fileName,
    definingClass,
    methodName,
    () => this.callUserFunction(methodFn, actualArgs, nargout)
  );
}

// ── Private function interpretation ──────────────────────────────────────

export function interpretPrivateFunction(
  this: Interpreter,
  target: Extract<ResolvedTarget, { kind: "privateFunction" }>,
  args: unknown[],
  nargout: number
): unknown {
  const entry = this.ctx.getPrivateFileEntry(target.callerFile, target.name);
  if (!entry)
    throw new RuntimeError(`Private function '${target.name}' not found`);

  const ast = this.ctx.getCachedAST(entry.fileName);
  for (const stmt of ast.body) {
    if (stmt.type === "Function" && stmt.name === target.name) {
      const fn = funcDefFromStmt(stmt);
      return this.withFileContext(entry.fileName, undefined, undefined, () =>
        this.callUserFunction(fn, args, nargout)
      );
    }
  }
  throw new RuntimeError(`Private function '${target.name}' not found`);
}

// ── Class instantiation ──────────────────────────────────────────────────

export function instantiateClass(
  this: Interpreter,
  className: string,
  args: unknown[],
  nargout: number
): unknown {
  const classInfo = this.ctx.getClassInfo(className);
  if (!classInfo) {
    return this.rt.callClassMethod(className, className, nargout, args);
  }

  const { propertyNames, propertyDefaults } =
    this.collectClassProperties(classInfo);

  const defaults = new Map<string, RuntimeValue>();
  for (const [propName, defaultExpr] of propertyDefaults) {
    try {
      defaults.set(propName, ensureRuntimeValue(this.evalExpr(defaultExpr)));
    } catch {
      // Default evaluation failed
    }
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
  const constructorName = classInfo.constructorName;
  if (!constructorName) return args[0];

  for (const member of classInfo.ast.members) {
    if (member.type !== "Methods") continue;
    for (const methodStmt of member.body) {
      if (
        methodStmt.type === "Function" &&
        methodStmt.name === constructorName
      ) {
        const outputName =
          methodStmt.outputs.length > 0 ? methodStmt.outputs[0] : "obj";
        const fn: FunctionDef = {
          ...funcDefFromStmt(methodStmt),
          params: [outputName, ...methodStmt.params],
        };
        // nargin in constructor counts user args, not obj
        const userArgCount = args.length - 1;
        return this.withFileContext(
          classInfo.fileName,
          classInfo.name,
          constructorName,
          () => this.callUserFunction(fn, args, nargout, userArgCount)
        );
      }
    }
  }

  const extFn = this.findExternalMethod(classInfo, constructorName);
  if (extFn) {
    const outputName = extFn.outputs.length > 0 ? extFn.outputs[0] : "obj";
    const fn: FunctionDef = {
      ...extFn,
      params: [outputName, ...extFn.params],
    };
    const userArgCount = args.length - 1;
    return this.withFileContext(
      classInfo.externalMethodFiles.get(constructorName)?.fileName ??
        classInfo.fileName,
      classInfo.name,
      constructorName,
      () => this.callUserFunction(fn, args, nargout, userArgCount)
    );
  }

  return args[0];
}

// ── Call user function (core execution) ──────────────────────────────────

export function callUserFunction(
  this: Interpreter,
  fn: FunctionDef,
  args: unknown[],
  nargout: number,
  narginOverride?: number
): unknown {
  // Try JIT compilation for eligible functions
  if (this.optimization >= 1 && narginOverride === undefined) {
    const jitResult = tryJitCall(this, fn, args, nargout);
    if (jitResult !== JIT_SKIP) return jitResult;
  }

  const fnEnv = new Environment();
  fnEnv.rt = this.rt;
  fnEnv.persistentFuncId = `${this.currentFile}:${fn.name}`;

  const processedArgs = this.processArgumentsBlocks(fn, args);

  const hasVarargin =
    fn.params.length > 0 && fn.params[fn.params.length - 1] === "varargin";
  const regularParams = hasVarargin ? fn.params.slice(0, -1) : fn.params;
  // Build set of output names for COW sharing (value class semantics)
  const outputSet = new Set(fn.outputs);
  for (let i = 0; i < regularParams.length; i++) {
    if (i < processedArgs.length) {
      let val = ensureRuntimeValue(processedArgs[i]);
      // When a parameter is also an output (e.g., function obj = method(obj)),
      // share it for value-class COW safety, matching codegen behavior.
      if (outputSet.has(regularParams[i])) {
        val = shareRuntimeValue(val);
      }
      fnEnv.set(regularParams[i], val);
    }
  }
  if (hasVarargin) {
    const extraArgs = processedArgs
      .slice(regularParams.length)
      .map(a => ensureRuntimeValue(a));
    fnEnv.set("varargin", RTV.cell(extraArgs, [1, extraArgs.length]));
  }
  fnEnv.set("$nargin", narginOverride ?? args.length);
  fnEnv.set("$nargout", nargout);

  // Pre-register nested function definitions (hoisted, like MATLAB)
  for (const stmt of fn.body) {
    if (stmt.type === "Function") {
      fnEnv.nestedFunctions.set(stmt.name, {
        fn: funcDefFromStmt(stmt),
        env: fnEnv,
      });
    }
  }

  const savedEnv = this.env;
  const savedCallerEnv = this.callerEnv;
  this.callerEnv = savedEnv;
  this.env = fnEnv;
  this.rt.pushCallFrame(fn.name);

  try {
    this.execStmts(fn.body);

    if (fnEnv.persistentFuncId) {
      for (const name of fnEnv.persistentNames) {
        const val = fnEnv.get(name);
        if (val !== undefined) {
          this.rt.setPersistent(fnEnv.persistentFuncId, name, val);
        }
      }
    }

    const hasVarargout =
      fn.outputs.length > 0 &&
      fn.outputs[fn.outputs.length - 1] === "varargout";
    const regularOutputs = hasVarargout ? fn.outputs.slice(0, -1) : fn.outputs;

    const outputs: RuntimeValue[] = [];
    for (let i = 0; i < Math.min(regularOutputs.length, nargout); i++) {
      const val = this.env.get(regularOutputs[i]);
      if (val === undefined && nargout >= i + 1) {
        throw new RuntimeError(
          `Output argument '${regularOutputs[i]}' (and maybe others) not assigned during call to '${fn.name}'`
        );
      }
      outputs.push(val ?? RTV.num(0));
    }

    if (hasVarargout && nargout > regularOutputs.length) {
      const varargoutVal = this.env.get("varargout");
      if (varargoutVal && isRuntimeCell(varargoutVal)) {
        const remaining = nargout - regularOutputs.length;
        for (
          let i = 0;
          i < Math.min(remaining, varargoutVal.data.length);
          i++
        ) {
          outputs.push(ensureRuntimeValue(varargoutVal.data[i]));
        }
      }
    }

    if (nargout <= 1) {
      return outputs[0];
    }
    return outputs;
  } catch (e) {
    this.rt.annotateError(e);
    throw e;
  } finally {
    this.rt.popCallFrame();
    this.env = savedEnv;
    this.callerEnv = savedCallerEnv;
  }
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

  const fnEnv = new Environment(parentEnv);
  fnEnv.isNested = true;
  fnEnv.rt = this.rt;
  fnEnv.persistentFuncId = `${this.currentFile}:${fn.name}`;

  const hasVarargin =
    fn.params.length > 0 && fn.params[fn.params.length - 1] === "varargin";
  const regularParams = hasVarargin ? fn.params.slice(0, -1) : fn.params;
  for (let i = 0; i < regularParams.length; i++) {
    if (i < args.length) {
      fnEnv.setLocal(regularParams[i], ensureRuntimeValue(args[i]));
    }
  }
  if (hasVarargin) {
    const extraArgs = args
      .slice(regularParams.length)
      .map(a => ensureRuntimeValue(a));
    fnEnv.setLocal("varargin", RTV.cell(extraArgs, [1, extraArgs.length]));
  }
  fnEnv.setLocal("$nargin", args.length);
  fnEnv.setLocal("$nargout", nargout);

  const savedEnv = this.env;
  this.env = fnEnv;

  try {
    this.execStmts(fn.body);

    if (fnEnv.persistentFuncId) {
      for (const name of fnEnv.persistentNames) {
        const val = fnEnv.get(name);
        if (val !== undefined) {
          this.rt.setPersistent(fnEnv.persistentFuncId, name, val);
        }
      }
    }

    const hasVarargout =
      fn.outputs.length > 0 &&
      fn.outputs[fn.outputs.length - 1] === "varargout";
    const regularOutputs = hasVarargout ? fn.outputs.slice(0, -1) : fn.outputs;

    const outputs: RuntimeValue[] = [];
    for (let i = 0; i < Math.min(regularOutputs.length, nargout); i++) {
      const val = this.env.get(regularOutputs[i]);
      if (val === undefined && nargout >= i + 1) {
        throw new RuntimeError(
          `Output argument '${regularOutputs[i]}' (and maybe others) not assigned during call to '${fn.name}'`
        );
      }
      outputs.push(val ?? RTV.num(0));
    }

    if (hasVarargout && nargout > regularOutputs.length) {
      const varargoutVal = this.env.get("varargout");
      if (varargoutVal && isRuntimeCell(varargoutVal)) {
        const remaining = nargout - regularOutputs.length;
        for (
          let i = 0;
          i < Math.min(remaining, varargoutVal.data.length);
          i++
        ) {
          outputs.push(ensureRuntimeValue(varargoutVal.data[i]));
        }
      }
    }

    if (nargout <= 1) {
      return outputs[0];
    }
    return outputs;
  } finally {
    this.env = savedEnv;
  }
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

// ── AST lookup helpers ───────────────────────────────────────────────────

export function getWorkspaceFileName(
  this: Interpreter,
  funcName: string
): string {
  const entry = this.ctx.registry.filesByFuncName.get(funcName);
  return entry?.fileName ?? funcName + ".m";
}

export function getClassFileName(this: Interpreter, className: string): string {
  const info = this.ctx.getClassInfo(className);
  return info?.fileName ?? className + ".m";
}

export function findFunctionInWorkspaceFile(
  this: Interpreter,
  wsName: string,
  funcName: string
): FunctionDef | null {
  const cacheKey = `ws:${wsName}:${funcName}`;
  const cached = this.functionDefCache.get(cacheKey);
  if (cached) return cached;

  const entry = this.ctx.registry.filesByFuncName.get(wsName);
  if (!entry) return null;

  const ast = this.ctx.getCachedAST(entry.fileName);
  for (const stmt of ast.body) {
    if (stmt.type === "Function" && stmt.name === funcName) {
      const fn = funcDefFromStmt(stmt);
      this.functionDefCache.set(cacheKey, fn);
      return fn;
    }
  }
  return null;
}

export function findFunctionInClassFile(
  this: Interpreter,
  className: string,
  funcName: string,
  methodScope?: string
): FunctionDef | null {
  const cacheKey = `cls:${className}:${funcName}:${methodScope ?? ""}`;
  const cached = this.functionDefCache.get(cacheKey);
  if (cached) return cached;

  const classInfo = this.ctx.getClassInfo(className);
  if (!classInfo) return null;

  // If methodScope is an external method, look ONLY in that file's subfunctions
  if (methodScope && classInfo.externalMethodFiles.has(methodScope)) {
    const mf = classInfo.externalMethodFiles.get(methodScope)!;
    const methodAst = this.ctx.getCachedAST(mf.fileName);
    for (const stmt of methodAst.body) {
      if (stmt.type === "Function" && stmt.name === funcName) {
        const fn = funcDefFromStmt(stmt);
        this.functionDefCache.set(cacheKey, fn);
        return fn;
      }
    }
    // Fall through to main classdef file if not found in external method file
  }

  // Look in main classdef file's subfunctions
  const ast = this.ctx.getCachedAST(classInfo.fileName);
  for (const stmt of ast.body) {
    if (stmt.type === "Function" && stmt.name === funcName) {
      const fn = funcDefFromStmt(stmt);
      this.functionDefCache.set(cacheKey, fn);
      return fn;
    }
  }

  // If no methodScope specified, also search all external method files
  if (!methodScope) {
    for (const [, mf] of classInfo.externalMethodFiles) {
      const methodAst = this.ctx.getCachedAST(mf.fileName);
      for (const stmt of methodAst.body) {
        if (stmt.type === "Function" && stmt.name === funcName) {
          const fn = funcDefFromStmt(stmt);
          this.functionDefCache.set(cacheKey, fn);
          return fn;
        }
      }
    }
  }

  return null;
}

export function findMethodInClass(
  this: Interpreter,
  classInfo: ClassInfo,
  methodName: string
): FunctionDef | null {
  const cacheKey = `method:${classInfo.name}:${methodName}`;
  const cached = this.functionDefCache.get(cacheKey);
  if (cached) return cached;

  for (const member of classInfo.ast.members) {
    if (member.type !== "Methods") continue;
    for (const methodStmt of member.body) {
      if (methodStmt.type === "Function" && methodStmt.name === methodName) {
        const fn = funcDefFromStmt(methodStmt);
        this.functionDefCache.set(cacheKey, fn);
        return fn;
      }
    }
  }

  if (classInfo.superClass) {
    const parentInfo = this.ctx.getClassInfo(classInfo.superClass);
    if (parentInfo) {
      return this.findMethodInClass(parentInfo, methodName);
    }
  }

  return null;
}

export function findExternalMethod(
  this: Interpreter,
  classInfo: ClassInfo,
  methodName: string
): FunctionDef | null {
  const mf = classInfo.externalMethodFiles.get(methodName);
  if (!mf) {
    if (classInfo.superClass) {
      const parentInfo = this.ctx.getClassInfo(classInfo.superClass);
      if (parentInfo) return this.findExternalMethod(parentInfo, methodName);
    }
    return null;
  }

  const ast = this.ctx.getCachedAST(mf.fileName);
  for (const stmt of ast.body) {
    if (stmt.type === "Function" && stmt.name === methodName) {
      return funcDefFromStmt(stmt);
    }
  }
  return null;
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

  let parentName = classInfo.superClass;
  while (parentName) {
    if (parentName === "handle") break;
    const parentInfo = this.ctx.getClassInfo(parentName);
    if (!parentInfo) break;
    for (const propName of parentInfo.propertyNames) {
      if (!propertyNames.includes(propName)) {
        propertyNames.push(propName);
        const defaultExpr = parentInfo.propertyDefaults.get(propName);
        if (defaultExpr && !propertyDefaults.has(propName)) {
          propertyDefaults.set(propName, defaultExpr);
        }
      }
    }
    parentName = parentInfo.superClass;
  }
  return { propertyNames, propertyDefaults };
}

export function isHandleClass(
  this: Interpreter,
  classInfo: ClassInfo
): boolean {
  let parentName = classInfo.superClass;
  while (parentName) {
    if (parentName === "handle") return true;
    const parentInfo = this.ctx.getClassInfo(parentName);
    if (!parentInfo) break;
    parentName = parentInfo.superClass;
  }
  return false;
}

// ── eval in local scope ──────────────────────────────────────────────────

export function evalInLocalScope(this: Interpreter, codeArg: unknown): unknown {
  const code = toString(ensureRuntimeValue(codeArg));
  const initialVars: Record<string, RuntimeValue> = {};
  for (const name of this.env.localNames()) {
    if (name.startsWith("$")) continue;
    const val = this.env.get(name);
    if (val !== undefined) initialVars[name] = val;
  }
  for (const name of this.env.globalNames) {
    const val = this.env.get(name);
    if (val !== undefined) initialVars[name] = val;
  }

  const cb = this.rt.evalLocalCallback;
  if (cb) {
    const result = cb(code, initialVars, (text: string) => {
      this.rt.output(text);
    });
    if (result.variableValues) {
      for (const [name, val] of Object.entries(result.variableValues)) {
        if (name.startsWith("$")) continue;
        this.env.set(name, val);
      }
    }
    return result.returnValue;
  }
  throw new RuntimeError("eval not available in this context");
}

// ── Arguments block processing ──────────────────────────────────────────

export function processArgumentsBlocks(
  this: Interpreter,
  fn: FunctionDef,
  args: unknown[]
): unknown[] {
  const argBlocks = fn.argumentsBlocks;
  if (!argBlocks || argBlocks.length === 0) return args;

  for (const block of argBlocks) {
    if (block.kind === "Output") continue;
    const entries = block.entries;
    if (!entries || entries.length === 0) continue;

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

    if (nvGroups.size > 0) {
      const processedArgs = [...args];
      const nvParamIndex = regularEntries.length;

      for (const [paramName, fields] of nvGroups) {
        const nvArgs = args.slice(nvParamIndex);
        const defaults: Record<string, unknown> = {};
        for (const { field, defaultExpr } of fields) {
          if (defaultExpr) {
            try {
              defaults[field] = this.evalExpr(defaultExpr);
            } catch {
              // skip
            }
          }
        }
        const struct = this.rt.buildNameValueStruct(nvArgs, defaults);
        const paramIdx = fn.params.indexOf(paramName);
        const targetIdx = paramIdx >= 0 ? paramIdx : nvParamIndex;
        processedArgs.length = Math.max(processedArgs.length, targetIdx + 1);
        processedArgs[targetIdx] = struct;
      }

      for (let i = 0; i < regularEntries.length; i++) {
        if (processedArgs[i] === undefined && regularEntries[i].defaultValue) {
          try {
            processedArgs[i] = this.evalExpr(regularEntries[i].defaultValue!);
          } catch {
            // skip
          }
        }
      }
      return processedArgs;
    }

    const processedArgs = [...args];
    for (let i = 0; i < entries.length; i++) {
      if (processedArgs[i] === undefined && entries[i].defaultValue) {
        try {
          processedArgs[i] = this.evalExpr(entries[i].defaultValue!);
        } catch {
          // skip
        }
      }
    }
    return processedArgs;
  }

  return args;
}
