/**
 * Function bodies: the `FunctionDef` a resolved target runs, and the file and
 * class context it runs in. Bodies are built once per function index and
 * reused, so every call of a function sees the same `FunctionDef` (the JIT's
 * per-function caches key on it).
 */

import type { AbstractSyntaxTree, Stmt } from "../parser/types.js";
import { funcDefFromStmt, type FunctionDef } from "../interpreter/types.js";
import type { ClassInfo } from "../workspace/classInfo.js";
import type { CodeIndex } from "../workspace/codeIndex.js";
import type { FunctionIndex } from "../workspace/functionIndex.js";
import type { ResolvedTarget } from "./resolveFunction.js";

type FunctionStmt = Stmt & { type: "Function" };

/** A function body and the context it runs in. */
export interface FunctionBody {
  fn: FunctionDef;
  /** The file the body runs in. */
  file: string;
  /** The class and method whose code the body is, for methods and the local
   *  functions of class files. */
  className?: string;
  methodName?: string;
}

const caches = new WeakMap<FunctionIndex, Map<string, FunctionBody | null>>();

function cached(
  code: CodeIndex,
  key: string,
  build: () => FunctionBody | null
): FunctionBody | null {
  const index = code.functionIndex;
  let cache = caches.get(index);
  if (cache === undefined) {
    cache = new Map();
    caches.set(index, cache);
  }
  let body = cache.get(key);
  if (body === undefined) {
    body = build();
    cache.set(key, body);
  }
  return body;
}

/** The body `target` runs, or null when it has none: builtins, `.numbl.js`
 *  functions, class constructors (see `constructorBodyFor`), the implicit
 *  `Class.empty`, scripts, and names with no definition. */
export function functionBodyFor(
  code: CodeIndex,
  target: ResolvedTarget
): FunctionBody | null {
  switch (target.kind) {
    case "localFunction": {
      const s = target.source;
      const where =
        s.from === "main"
          ? ""
          : s.from === "workspaceFile"
            ? s.wsName
            : s.from === "privateFile"
              ? s.callerFile
              : `${s.className}\u0000${s.methodScope ?? ""}`;
      return cached(
        code,
        `L\u0000${s.from}\u0000${where}\u0000${target.name}`,
        () => localFunctionBody(code, target)
      );
    }
    case "privateFunction": {
      const entry = code.getPrivateFileEntry(target.callerFile, target.name);
      if (!entry) return null;
      return cached(code, `P\u0000${entry.fileName}`, () =>
        bodyIn(code, entry.fileName, target.name)
      );
    }
    case "workspaceFunction":
      return cached(code, `W\u0000${target.name}`, () =>
        workspaceFunctionBody(code, target.name)
      );
    case "classMethod":
      return cached(
        code,
        `M\u0000${target.className}\u0000${target.methodName}`,
        () => classMethodBody(code, target.className, target.methodName)
      );
    case "builtin":
    case "builtinMethod":
    case "jsUserFunction":
    case "workspaceClassConstructor":
      return null;
  }
}

/** The constructor of a classdef class, taking the pre-built instance as its
 *  first parameter (named after the constructor's output). Null when the
 *  class has no explicit constructor. */
export function constructorBodyFor(
  code: CodeIndex,
  info: ClassInfo
): FunctionBody | null {
  return cached(code, `C\u0000${info.qualifiedName}`, () => {
    const ctorName = info.constructorName;
    if (!ctorName) return null;
    const withInstance = (
      fn: FunctionDef,
      outputs: string[],
      file: string
    ): FunctionBody => ({
      fn: { ...fn, params: [outputs[0] ?? "obj", ...fn.params] },
      file,
      className: info.qualifiedName,
      methodName: ctorName,
    });
    for (const member of info.ast?.members ?? []) {
      if (member.type !== "Methods") continue;
      for (const stmt of member.body) {
        if (stmt.type === "Function" && stmt.name === ctorName)
          return withInstance(
            funcDefFromStmt(stmt),
            stmt.outputs,
            info.fileName
          );
      }
    }
    const ext = externalMethodDef(code, info, ctorName);
    if (!ext) return null;
    return withInstance(
      ext,
      ext.outputs,
      info.externalMethodFiles.get(ctorName)?.fileName ?? info.fileName
    );
  });
}

/** The constructor function of an old-style (`@Name/Name.m`) class, which
 *  builds its own instance. */
export function oldStyleConstructorBodyFor(
  code: CodeIndex,
  info: ClassInfo
): FunctionBody | null {
  return cached(code, `O\u0000${info.qualifiedName}`, () => {
    const ctorName = info.constructorName ?? info.qualifiedName;
    const fn = externalMethodDef(code, info, ctorName);
    if (!fn) return null;
    return {
      fn,
      file: info.externalMethodFiles.get(ctorName)?.fileName ?? info.fileName,
      className: info.qualifiedName,
      methodName: ctorName,
    };
  });
}

// ── Lookup ───────────────────────────────────────────────────────────────

function functionNamed(
  ast: AbstractSyntaxTree,
  name: string
): FunctionStmt | null {
  for (const stmt of ast.body) {
    if (stmt.type === "Function" && stmt.name === name) return stmt;
  }
  return null;
}

function bodyIn(
  code: CodeIndex,
  fileName: string,
  name: string,
  className?: string,
  methodName?: string
): FunctionBody | null {
  const stmt = functionNamed(code.getCachedAST(fileName), name);
  if (!stmt) return null;
  return { fn: funcDefFromStmt(stmt), file: fileName, className, methodName };
}

function localFunctionBody(
  code: CodeIndex,
  target: Extract<ResolvedTarget, { kind: "localFunction" }>
): FunctionBody | null {
  const { name, source } = target;
  switch (source.from) {
    case "main": {
      const stmt = code.mainLocalFunctions.get(name);
      return stmt
        ? { fn: funcDefFromStmt(stmt), file: code.mainFileName }
        : null;
    }
    case "workspaceFile": {
      const entry = code.filesByFuncName.get(source.wsName);
      return entry ? bodyIn(code, entry.fileName, name) : null;
    }
    case "privateFile":
      return bodyIn(code, source.callerFile, name);
    case "classFile":
      return classFileLocalBody(
        code,
        source.className,
        name,
        source.methodScope
      );
  }
}

/** A local function of a class: from the external method file `methodScope`
 *  if it is one, else from the classdef file, else (with no method scope)
 *  from any external method file. It runs in the method scope's file. */
function classFileLocalBody(
  code: CodeIndex,
  className: string,
  name: string,
  methodScope: string | undefined
): FunctionBody | null {
  const info = code.classes.get(className);
  if (!info) return null;
  const scopeFile = info.externalMethodFiles.get(methodScope ?? "")?.fileName;
  const file = scopeFile ?? info.fileName;
  const found = (stmt: FunctionStmt | null): FunctionBody | null =>
    stmt
      ? {
          fn: funcDefFromStmt(stmt),
          file,
          className,
          methodName: methodScope,
        }
      : null;
  if (scopeFile) {
    const body = found(functionNamed(code.getCachedAST(scopeFile), name));
    if (body) return body;
  }
  const body = found(functionNamed(code.getCachedAST(info.fileName), name));
  if (body || methodScope) return body;
  for (const [, mf] of info.externalMethodFiles) {
    const extBody = found(functionNamed(code.getCachedAST(mf.fileName), name));
    if (extBody) return extBody;
  }
  return null;
}

/** The function a workspace function name calls: the file's function of that
 *  name, else, for a function file, its first function (MATLAB calls a
 *  function file's primary function whatever its declared name). A script
 *  (a file not starting with a function) has no body. */
function workspaceFunctionBody(
  code: CodeIndex,
  name: string
): FunctionBody | null {
  const entry = code.filesByFuncName.get(name);
  if (!entry) return null;
  const primaryName = name.slice(name.lastIndexOf(".") + 1);
  const named = bodyIn(code, entry.fileName, primaryName);
  if (named) return named;
  for (const stmt of code.getCachedAST(entry.fileName).body) {
    if (stmt.type === "Function")
      return { fn: funcDefFromStmt(stmt), file: entry.fileName };
    if (stmt.type !== "ClassDef" && stmt.type !== "Import") break;
  }
  return null;
}

/** A method of `className` or an ancestor, from a classdef body or an
 *  external method file. Runs in the defining class's context. */
function classMethodBody(
  code: CodeIndex,
  className: string,
  methodName: string
): FunctionBody | null {
  const definingClass = code.classes.findDefiningClass(className, methodName);
  const info = code.classes.get(definingClass);
  if (!info || methodName === info.constructorName) return null;
  const inBody = methodDefInClassdef(code, info, methodName);
  if (inBody)
    return {
      fn: inBody,
      file: info.fileName,
      className: definingClass,
      methodName,
    };
  const ext = externalMethodDef(code, info, methodName);
  if (ext)
    return {
      fn: ext,
      file: info.externalMethodFiles.get(methodName)?.fileName ?? info.fileName,
      className: definingClass,
      methodName,
    };
  return null;
}

/** A method declared in the classdef body of `info` or an ancestor. */
function methodDefInClassdef(
  code: CodeIndex,
  info: ClassInfo,
  methodName: string
): FunctionDef | null {
  // Old-style @folder classes have no classdef AST: all their methods are
  // external files.
  for (const member of info.ast?.members ?? []) {
    if (member.type !== "Methods") continue;
    for (const stmt of member.body) {
      if (stmt.type === "Function" && stmt.name === methodName)
        return funcDefFromStmt(stmt);
    }
  }
  for (const sc of info.superClasses) {
    const parent = code.classes.get(sc);
    const fn = parent && methodDefInClassdef(code, parent, methodName);
    if (fn) return fn;
  }
  return null;
}

/** A method defined by an `@Class/<methodName>.m` file of `info` or an
 *  ancestor. Dispatch is by file name: the method is the file's primary
 *  function, preferring one declared with the method's name. */
function externalMethodDef(
  code: CodeIndex,
  info: ClassInfo,
  methodName: string
): FunctionDef | null {
  const mf = info.externalMethodFiles.get(methodName);
  if (!mf) {
    for (const sc of info.superClasses) {
      const parent = code.classes.get(sc);
      const fn = parent && externalMethodDef(code, parent, methodName);
      if (fn) return fn;
    }
    return null;
  }
  let primary: FunctionDef | null = null;
  for (const stmt of code.getCachedAST(mf.fileName).body) {
    if (stmt.type !== "Function") continue;
    if (stmt.name === methodName) return funcDefFromStmt(stmt);
    primary ??= funcDefFromStmt(stmt);
  }
  return primary;
}
