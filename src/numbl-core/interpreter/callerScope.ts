/**
 * The calling scope that builtins declared with `scope: true` see (eval,
 * evalin, exist, nargin, inputname, ...), and running MATLAB text in a
 * frame.
 */

import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import type { RuntimeValue } from "../runtime/types.js";
import { RuntimeError } from "../runtime/error.js";
import { toString } from "../runtime/convert.js";
import {
  resolveName,
  definingFile,
  type NameResolution,
} from "../resolution/index.js";
import { parseMFile, type AbstractSyntaxTree } from "../parser/index.js";
import { SyntaxError } from "../parser/errors.js";

import { ReturnSignal, type Environment } from "./types.js";
import type { CallFrame } from "./callFrame.js";
import type { Interpreter } from "./interpreter.js";
import { evalExprNargout } from "./evalExpressions.js";
import { execStmt } from "./execStatements.js";
import { declaredNargin, declaredNargout } from "./calls.js";
import { classPublicMethods, classPublicProperties } from "./oop.js";

/** The value(s) of evaluated text: undefined when it ran as statements. */
type EvalResult = RuntimeValue | RuntimeValue[] | undefined;

/** A workspace file that provides a function or class name. */
export interface WorkspaceFileInfo {
  path: string;
  /** A .m function, a `.numbl.js` function (MEX-like), or a class. */
  kind: "function" | "jsfunction" | "class";
  /** The .m source for "function"/"class", or the help text registered by
   *  a "jsfunction" (empty if none). */
  source: string;
}

/** The scope a builtin was called from: the running frame and the code it
 *  can see. */
export interface CallerScope {
  /** The running frame's variables. */
  readonly env: Environment;
  /** Call-site names of the running function's arguments (`inputname`). */
  readonly inputNames: readonly string[] | undefined;
  /** The frame the running function was called from; undefined at the top
   *  level. */
  readonly callerFrame: CallFrame | undefined;
  /** The base workspace's frame. */
  readonly baseFrame: CallFrame;
  /** Run MATLAB text in the running frame. See `evalInLocalScope`. */
  evalText: (
    code: RuntimeValue | string,
    nargout?: number,
    fileName?: string
  ) => EvalResult;
  /** Run MATLAB text in another frame's workspace (`evalin`). */
  evalInFrame: (
    frame: CallFrame,
    code: RuntimeValue | string,
    nargout: number
  ) => EvalResult;
  /** What `name` means in this scope, and the file defining it. */
  lookupName: (name: string) => {
    resolution: NameResolution;
    file: string | undefined;
  };
  /** Declared input count of the user function `name` resolves to
   *  (negative with varargin), or undefined. */
  declaredNargin: (name: string) => number | undefined;
  /** Declared output count, likewise (negative with varargout). */
  declaredNargout: (name: string) => number | undefined;
  /** The workspace file providing function or class `name`, if any. */
  lookupWorkspaceFile: (name: string) => WorkspaceFileInfo | undefined;
  /** Public property names of a class, including inherited ones, in
   *  declaration order; undefined if `className` is not a known class. */
  classPublicProperties: (className: string) => string[] | undefined;
  /** Public method names of a class (including inherited, static and the
   *  constructor), or undefined if not a known class. */
  classPublicMethods: (className: string) => Set<string> | undefined;
}

const scopes = new WeakMap<Interpreter, CallerScope>();

/** The calling scope for builtins. One per interpreter; its frame-derived
 *  fields read the running frame. */
export function callerScope(interp: Interpreter): CallerScope {
  let scope = scopes.get(interp);
  if (scope) return scope;
  scope = {
    get env() {
      return interp.frame.env;
    },
    get inputNames() {
      return interp.frame.inputNames;
    },
    get callerFrame() {
      return interp.frame.caller;
    },
    baseFrame: interp.baseFrame,
    evalText: (code, nargout, fileName) =>
      evalInLocalScope(interp, code, fileName, nargout) as EvalResult,
    evalInFrame: (frame, code, nargout) =>
      interp.inFrame(
        interp.frame.inWorkspaceOf(frame),
        () => evalInLocalScope(interp, code, undefined, nargout) as EvalResult
      ),
    lookupName: n => {
      const frame = interp.frame;
      const r = resolveName(n, frame);
      return { resolution: r, file: definingFile(r, frame) };
    },
    declaredNargin: n => declaredNargin(interp, n),
    declaredNargout: n => declaredNargout(interp, n),
    lookupWorkspaceFile: n => {
      const code = interp.code;
      const entry = code.filesByFuncName.get(n);
      if (entry)
        return { path: entry.fileName, kind: "function", source: entry.source };
      const classInfo = code.classes.get(n);
      if (classInfo)
        return {
          path: classInfo.fileName,
          kind: "class",
          source: classInfo.source,
        };
      const jsEntry = code.jsUserFunctionsByName.get(n);
      if (jsEntry)
        return {
          path: jsEntry.fileName,
          kind: "jsfunction",
          source: jsEntry.help ?? "",
        };
      return undefined;
    },
    classPublicProperties: n => classPublicProperties(interp, n),
    classPublicMethods: n => classPublicMethods(interp, n),
  };
  scopes.set(interp, scope);
  return scope;
}

/** Run MATLAB text in the running frame (eval, evalin, run, input,
 *  str2num). With `nargout` > 0 the text must be one expression, whose
 *  value(s) are returned; otherwise its statements run (displaying
 *  unsuppressed results) and nothing is returned. `fileName` sets the file
 *  the code runs from, e.g. for `run(script)`. */
export function evalInLocalScope(
  interp: Interpreter,
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
    return evalExprNargout(interp, stmt.expr, nargout);
  }
  const exec = () => {
    for (const stmt of ast.body) {
      if (stmt.type === "Function") continue;
      if (execStmt(interp, stmt) instanceof ReturnSignal) break;
    }
  };
  if (fileName) interp.inFrame(interp.frame.at({ file: fileName }), exec);
  else exec();
  return undefined;
}
