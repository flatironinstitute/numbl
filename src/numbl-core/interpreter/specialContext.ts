/**
 * The interpreter side of the special builtins (eval, evalin, exist,
 * nargin, inputname, ...): the context they see, and running MATLAB text
 * in a frame.
 */

import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { toString } from "../runtime/convert.js";
import { resolveName, definingFile } from "../resolution/index.js";
import { parseMFile, type AbstractSyntaxTree } from "../parser/index.js";
import { SyntaxError } from "../parser/errors.js";
import {
  getInterpreterSpecialBuiltin,
  FALL_THROUGH,
  type InterpreterContext,
} from "./interpreterSpecialBuiltins.js";

import { ReturnSignal } from "./types.js";
import type { Interpreter } from "./interpreter.js";
import { evalExprNargout } from "./evalExpressions.js";
import { execStmt } from "./execStatements.js";
import { callByName, declaredNargin } from "./calls.js";
import { classPublicMethods, classPublicProperties } from "./oop.js";

/** Run the interpreter special builtin `name` (one that needs the caller's
 *  frame, such as eval or exist). Returns FALL_THROUGH if it declines, in
 *  which case the registered builtin handles the call. */
export function callInterpreterSpecial(
  interp: Interpreter,
  name: string,
  args: unknown[],
  nargout: number
): unknown {
  const handler = getInterpreterSpecialBuiltin(name);
  if (!handler) return FALL_THROUGH;
  return handler(specialContext(interp), args, nargout);
}

const contexts = new WeakMap<Interpreter, InterpreterContext>();

/** The context special builtins see. One per interpreter; its frame-derived
 *  fields read the running frame. */
function specialContext(interp: Interpreter): InterpreterContext {
  let ctx = contexts.get(interp);
  if (ctx) return ctx;
  ctx = {
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
    evalInLocalScope: (codeArg, fileName, nargout) =>
      evalInLocalScope(interp, codeArg, fileName, nargout),
    evalInFrame: (frame, codeArg, nargout) =>
      interp.inFrame(interp.frame.inWorkspaceOf(frame), () =>
        evalInLocalScope(interp, codeArg, undefined, nargout)
      ),
    callByName: (n, a, no) => callByName(interp, n, a, no),
    lookupName: n => {
      const scope = interp.frame;
      const r = resolveName(n, scope);
      return { resolution: r, file: definingFile(r, scope) };
    },
    declaredNargin: n => declaredNargin(interp, n),
    rt: interp.rt,
    get optimization() {
      return interp.optimization;
    },
    lookupWorkspaceFile: n => {
      const code = interp.code;
      const entry = code.filesByFuncName.get(n);
      if (entry)
        return {
          path: entry.fileName,
          kind: "function",
          source: entry.source,
        };
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
  contexts.set(interp, ctx);
  return ctx;
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
