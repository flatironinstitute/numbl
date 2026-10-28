/**
 * MException values.
 *
 * An MException is a value-class `RuntimeClassInstance` with className
 * "MException" (or a user subclass, `classdef MyErr < MException`) and the
 * properties identifier, message, cause, stack and Correction, in MATLAB's
 * order. There is no classdef behind it: its methods (throw, rethrow,
 * getReport, addCause, ...) are builtins, which a method call on the
 * instance reaches through the usual class-method dispatch.
 *
 * A raised error travels as a `RuntimeError`. `catch ME` turns it back into
 * an MException with `errorToMException`: the exception a `throw` raised
 * (`RuntimeError.exception`, which keeps a subclass, its causes and extra
 * properties) or one built from the error's identifier and message, with the
 * stack of where it was raised.
 */

import { RTV } from "./constructors.js";
import { RuntimeError, type CallFrame } from "./error.js";
import {
  RuntimeClassInstance,
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeStruct,
  isRuntimeStructArray,
  type RuntimeStruct,
  type RuntimeValue,
} from "./types.js";

export const MEXCEPTION_CLASS = "MException";

/** MException's properties, in MATLAB's order. */
export const MEXCEPTION_PROPERTIES: readonly string[] = [
  "identifier",
  "message",
  "cause",
  "stack",
  "Correction",
];

const STACK_FIELDS = ["file", "name", "line"];

/** Empty MATLAB-style stack field: 0x1 struct with file/name/line fields. */
export function emptyStackField(): RuntimeValue {
  return RTV.structArray(STACK_FIELDS, []);
}

export function makeMException(
  identifier: string,
  message: string,
  stack?: RuntimeValue,
  cause?: RuntimeValue
): RuntimeClassInstance {
  return new RuntimeClassInstance(
    MEXCEPTION_CLASS,
    new Map<string, RuntimeValue>([
      ["identifier", RTV.char(identifier)],
      ["message", RTV.char(message)],
      ["cause", cause ?? RTV.cell([], [0, 0])],
      ["stack", stack ?? emptyStackField()],
      ["Correction", RTV.tensor(new Float64Array(0), [0, 0])],
    ]),
    false
  );
}

/** True if `v` is an MException or an instance of a subclass of it.
 *  `ancestors` gives a class's superclasses (`rt.classAncestors`). */
export function isMException(
  v: RuntimeValue,
  ancestors: (className: string) => string[]
): v is RuntimeClassInstance {
  if (!isRuntimeClassInstance(v)) return false;
  return (
    v.className === MEXCEPTION_CLASS ||
    ancestors(v.className).includes(MEXCEPTION_CLASS)
  );
}

function text(v: RuntimeValue | undefined): string {
  if (v === undefined) return "";
  if (isRuntimeChar(v)) return v.value;
  if (isRuntimeString(v)) return v;
  return "";
}

/** The identifier and message of an MException (or error struct). */
export function exceptionText(v: RuntimeClassInstance | RuntimeStruct): {
  identifier: string;
  message: string;
} {
  return {
    identifier: text(v.fields.get("identifier")),
    message: text(v.fields.get("message")),
  };
}

/** A copy of an MException with some properties replaced. */
export function withProperties(
  me: RuntimeClassInstance,
  changes: Record<string, RuntimeValue>
): RuntimeClassInstance {
  const fields = new Map(me.fields);
  for (const [k, v] of Object.entries(changes)) fields.set(k, v);
  return new RuntimeClassInstance(
    me.className,
    fields,
    me.isHandleClass,
    me._builtinData
  );
}

/** The frames of a stack struct array (innermost first). */
export function stackFrames(
  stack: RuntimeValue | undefined
): { file: string; name: string; line: number }[] {
  if (stack === undefined) return [];
  const els: RuntimeStruct[] = isRuntimeStructArray(stack)
    ? stack.elements
    : isRuntimeStruct(stack)
      ? [stack]
      : [];
  return els.map(el => {
    const line = el.fields.get("line");
    return {
      file: text(el.fields.get("file")),
      name: text(el.fields.get("name")),
      line: line !== undefined && isRuntimeNumber(line) ? line : 0,
    };
  });
}

/** The stack frame name of code at the top level of a file: the script's
 *  name, as MATLAB reports it. */
function fileStem(file: string | null): string {
  if (!file) return "";
  const base = file.slice(file.lastIndexOf("/") + 1);
  return base.endsWith(".m") ? base.slice(0, -2) : "";
}

function stackFrame(file: string, name: string, line: number): RuntimeStruct {
  return RTV.struct(
    new Map<string, RuntimeValue>([
      ["file", RTV.char(file)],
      ["name", RTV.char(name)],
      ["line", line],
    ])
  );
}

/**
 * Build a MATLAB-style stack struct array from a RuntimeError's callStack.
 * Index 1 is the innermost frame (where the error was thrown).
 */
export function buildStackField(e: RuntimeError): RuntimeValue {
  const frames = e.callStack;
  if (!frames || frames.length === 0) {
    // No function frames: a single frame at the error's own location.
    if (e.file || (e.line !== null && e.line > 0)) {
      return RTV.structArray(STACK_FIELDS, [
        stackFrame(e.file ?? "", fileStem(e.file), e.line ?? 0),
      ]);
    }
    return emptyStackField();
  }

  // Build frames innermost-first to match MATLAB convention.
  const elements: RuntimeStruct[] = [];
  const N = frames.length;
  for (let i = N - 1; i >= 0; i--) {
    let frameFile: string | null;
    let frameLine: number;
    if (i === N - 1) {
      frameFile = e.file;
      frameLine = e.line ?? 0;
    } else {
      const callerFrame = frames[i + 1];
      frameFile = callerFrame.callerFile;
      frameLine = callerFrame.callerLine;
    }
    elements.push(stackFrame(frameFile ?? "", frames[i].name, frameLine));
  }

  // Append the outermost caller (e.g. top-level script) if present.
  const outermost = frames[0];
  if (outermost.callerFile && outermost.callerLine > 0) {
    elements.push(
      stackFrame(
        outermost.callerFile,
        fileStem(outermost.callerFile),
        outermost.callerLine
      )
    );
  }

  return RTV.structArray(STACK_FIELDS, elements);
}

/** Point a RuntimeError's location and call stack at the frames of a stack
 *  struct array (as `rethrow` does), so an uncaught rethrown error reports
 *  where it was first raised. An empty stack leaves the error unannotated. */
export function setErrorStack(e: RuntimeError, stack: RuntimeValue): void {
  const frames = stackFrames(stack);
  if (frames.length === 0) return;
  e.file = frames[0].file || null;
  e.line = frames[0].line > 0 ? frames[0].line : null;
  const callStack: CallFrame[] = [];
  for (let i = frames.length - 1; i >= 0; i--) {
    const caller = frames[i + 1];
    callStack.push({
      name: frames[i].name,
      callerFile: caller ? caller.file || null : null,
      callerLine: caller ? caller.line : 0,
    });
  }
  e.callStack = callStack;
}

/** The MException `catch` binds for an error. */
export function errorToMException(e: unknown): RuntimeClassInstance {
  if (e instanceof RuntimeError) {
    const thrown = e.exception;
    if (thrown) {
      if (e.keepStack) return withProperties(thrown, {});
      return withProperties(thrown, { stack: buildStackField(e) });
    }
    return makeMException(e.identifier, e.message, buildStackField(e));
  }
  return makeMException("", e instanceof Error ? e.message : String(e));
}

/** A RuntimeError that raises `me` (throw / rethrow / error(struct)). With
 *  `keepStack`, a catch sees `me`'s own stack rather than the raise site's. */
export function raiseException(
  me: RuntimeClassInstance,
  keepStack: boolean
): RuntimeError {
  const { identifier, message } = exceptionText(me);
  const err = new RuntimeError(message, null, { identifier });
  err.exception = me;
  err.keepStack = keepStack;
  if (keepStack)
    setErrorStack(err, me.fields.get("stack") ?? emptyStackField());
  return err;
}
