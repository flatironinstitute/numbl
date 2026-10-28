/**
 * MException and its methods: MException, throw, rethrow, throwAsCaller,
 * getReport, addCause, addCorrection. See runtime/mexception.ts for the
 * value representation. The methods are builtins taking the exception
 * first, so `ME.getReport()` and `getReport(ME)` both reach them.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import {
  RuntimeClassInstance,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeStruct,
  isRuntimeStructArray,
} from "../../runtime/types.js";
import { RTV } from "../../runtime/constructors.js";
import type { RuntimeError } from "../../runtime/error.js";
import type { Runtime } from "../../runtime/runtime.js";
import {
  emptyStackField,
  exceptionText,
  isMException,
  makeMException,
  raiseException,
  stackFrames,
  withProperties,
} from "../../runtime/mexception.js";
import { sprintfFormat } from "../../helpers/string.js";
import { builtin } from "./builtin.js";
import { builtinError } from "./args.js";
import { undefinedForTypeError } from "./value-class.js";
import { MATLAB_ID_RE } from "./utility.js";

function textValue(v: RuntimeValue): string | null {
  if (isRuntimeChar(v)) return v.value;
  if (isRuntimeString(v)) return v;
  return null;
}

/** The MException a method was called on. */
function requireMException(
  rt: Runtime,
  v: RuntimeValue,
  fn: string
): RuntimeClassInstance {
  if (!isMException(v, c => rt.classAncestors(c))) {
    throw undefinedForTypeError(fn, v);
  }
  return v;
}

// ── MException ──────────────────────────────────────────────────────────

builtin({
  name: "MException",
  help: {
    signatures: [
      "ME = MException(errID, msg)",
      "ME = MException(errID, msg, A1, ..., An)",
    ],
    description:
      "Construct an exception object with an error identifier and message. The message is a format string (as for sprintf) filled in with any further arguments. ME has the properties identifier, message, cause, stack and Correction; its methods are throw, rethrow, throwAsCaller, getReport, addCause and addCorrection.",
  },
  nargin: [2, Infinity],
  apply: args => {
    const errID = textValue(args[0]);
    if (errID === null)
      throw builtinError(
        "MATLAB:dispatcher:noMatchingConstructor",
        "No constructor 'MException' with matching signature found."
      );
    if (errID !== "" && !MATLAB_ID_RE.test(errID))
      throw builtinError(
        "MATLAB:MException:badIdentifier",
        "Identifier must be a valid MATLAB message identifier ('component:mnemonic')."
      );
    const fmt = textValue(args[1]);
    if (fmt === null)
      throw builtinError(
        "MATLAB:MException:MessageMustBeChar",
        "MESSAGE input must be a character vector or string scalar."
      );
    // As in MATLAB, the message is always formatted, even without values.
    return makeMException(errID, sprintfFormat(fmt, args.slice(2)));
  },
});

// ── throw / throwAsCaller / rethrow ─────────────────────────────────────

builtin({
  name: "throw",
  help: {
    signatures: ["throw(ME)"],
    description:
      "Throw the exception ME. A catch receives ME (its class, causes and properties), with the stack of the throw site.",
  },
  nargin: 1,
  nargout: 0,
  apply: (args, _n, ctx) => {
    const me = requireMException(ctx.rt, args[0], "throw");
    throw raiseException(me, false);
  },
});

builtin({
  name: "throwAsCaller",
  help: {
    signatures: ["throwAsCaller(ME)"],
    description:
      "Throw the exception ME as if from the calling function: the stack omits the function that calls throwAsCaller.",
  },
  nargin: 1,
  nargout: 0,
  apply: (args, _n, ctx) => {
    const me = requireMException(ctx.rt, args[0], "throwAsCaller");
    throw raiseAsCaller(ctx.rt, me);
  },
});

/** Raise `me` with the stack of the running function's caller: the error
 *  is attributed to the site that called the running function. At the top
 *  level (no function frame) the stack is empty. */
function raiseAsCaller(rt: Runtime, me: RuntimeClassInstance): RuntimeError {
  const frames = rt.$callStack;
  if (frames.length === 0)
    return raiseException(
      withProperties(me, { stack: emptyStackField() }),
      true
    );
  const err = raiseException(me, false);
  const top = frames[frames.length - 1];
  err.callStack = frames.slice(0, -1);
  err.file = top.callerFile;
  err.line = top.callerLine > 0 ? top.callerLine : null;
  return err;
}

builtin({
  name: "rethrow",
  help: {
    signatures: ["rethrow(ME)", "rethrow(errStruct)"],
    description:
      "Rethrow a caught exception, keeping its original stack. With a struct, raise an error with its message and identifier fields, and its stack field (if present; otherwise the stack of the calling function's caller).",
  },
  nargin: 1,
  nargout: 0,
  apply: (args, _n, ctx) => {
    let v = args[0];
    if (isMException(v, c => ctx.rt.classAncestors(c))) {
      throw raiseException(v, true);
    }
    if (isRuntimeStructArray(v) && v.elements.length === 1) v = v.elements[0];
    if (isRuntimeStruct(v)) {
      const { identifier, message } = exceptionText(v);
      const stack = v.fields.get("stack");
      const me = makeMException(identifier, message, stack);
      // Without a stack field, MATLAB records the caller's stack.
      throw stack !== undefined
        ? raiseException(me, true)
        : raiseAsCaller(ctx.rt, me);
    }
    throw undefinedForTypeError("rethrow", v);
  },
});

// ── getReport ───────────────────────────────────────────────────────────

/** A stack frame's name as MATLAB's reports show it: a local function as
 *  `file>fn`. */
function frameLabel(frame: { file: string; name: string }): string {
  const base = frame.file.slice(frame.file.lastIndexOf("/") + 1);
  const stem = base.endsWith(".m") ? base.slice(0, -2) : "";
  if (frame.name === "") return stem;
  if (stem !== "" && stem !== frame.name && !frame.name.includes("."))
    return `${stem}>${frame.name}`;
  return frame.name;
}

function indent(s: string): string {
  return s
    .split("\n")
    .map(l => "    " + l)
    .join("\n");
}

function exceptionReport(me: RuntimeClassInstance, extended: boolean): string {
  const { message } = exceptionText(me);
  const frames = stackFrames(me.fields.get("stack"));
  const parts: string[] = [];
  let rest = frames;
  if (frames.length > 0 && frameLabel(frames[0]) !== "") {
    const where = frameLabel(frames[0]);
    const line = frames[0].line > 0 ? ` (line ${frames[0].line})` : "";
    parts.push(`Error using ${where}${line}\n${message}`);
    rest = frames.slice(1);
  } else {
    parts.push(message);
  }
  if (!extended) return parts[0];
  const errorIn = rest.map(f => {
    const where = frameLabel(f) || f.file;
    return `Error in ${where}` + (f.line > 0 ? ` (line ${f.line})` : "");
  });
  if (errorIn.length > 0) parts.push(errorIn.join("\n"));
  const cause = me.fields.get("cause");
  if (cause && isRuntimeCell(cause) && cause.data.length > 0) {
    const causes = cause.data.map(c =>
      c instanceof RuntimeClassInstance ? indent(exceptionText(c).message) : ""
    );
    parts.push("Caused by:\n" + causes.join("\n"));
  }
  return parts.join("\n\n");
}

builtin({
  name: "getReport",
  help: {
    signatures: [
      "S = getReport(ME)",
      "S = getReport(ME, TYPE)",
      "S = getReport(ME, TYPE, 'hyperlinks', VALUE)",
    ],
    description:
      "Return the error report of an exception as a char vector. TYPE 'basic' gives the message, headed by the function and line that raised it; 'extended' (the default) adds the rest of the stack and the causes. The 'hyperlinks' option is accepted and ignored (numbl's reports have no links).",
  },
  nargin: [1, 4],
  apply: (args, _n, ctx) => {
    const me = requireMException(ctx.rt, args[0], "getReport");
    let extended = true;
    if (args.length >= 2) {
      const t = textValue(args[1]);
      if (t === "basic") extended = false;
      else if (t !== "extended")
        throw builtinError(
          "MATLAB:MException:getReport:invalidReportOption",
          "getReport method can only accept 'basic' or 'extended' report types."
        );
    }
    return RTV.char(exceptionReport(me, extended));
  },
});

// ── addCause / addCorrection ────────────────────────────────────────────

builtin({
  name: "addCause",
  help: {
    signatures: ["ME = addCause(ME, causeException)"],
    description:
      "Return a copy of ME with causeException appended to its cause property (a column cell array).",
  },
  nargin: 2,
  apply: (args, _n, ctx) => {
    const rt = ctx.rt;
    const me = requireMException(rt, args[0], "addCause");
    const c = args[1];
    if (!isMException(c, k => rt.classAncestors(k)))
      throw builtinError(
        "MATLAB:class:RequireClass",
        "Invalid input for argument 2 (rhs2):\nValue must be 'MException scalar'."
      );
    const old = me.fields.get("cause");
    const causes = old && isRuntimeCell(old) ? [...old.data, c] : [c];
    return withProperties(me, {
      cause: RTV.cell(causes, [causes.length, 1]),
    });
  },
});

builtin({
  name: "addCorrection",
  help: {
    signatures: ["ME = addCorrection(ME, correction)"],
    description:
      "Return ME. numbl has no correction classes, so the correction is not recorded.",
  },
  nargin: 2,
  apply: (args, _n, ctx) => requireMException(ctx.rt, args[0], "addCorrection"),
});
