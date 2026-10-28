// JS sibling of `raise.h`: the raise channel. Runtime errors and bails in
// emitted JS throw a `JitRaise` (recognized by its `jitRaise` tag, since
// each emitted module defines its own copy of the class). The host
// (executors/jit/raise.ts) maps it: a bail re-runs the unit on the
// interpreter; an error becomes numbl's RuntimeError, or, when `message`
// is null, is left to the interpreter's re-run to word. Anything else
// thrown by emitted code is a JIT bug.

export class JitRaise extends Error {
  constructor(kind, identifier, message) {
    super(message === null ? `JIT ${kind}` : message);
    this.jitRaise = true;
    this.kind = kind;
    this.identifier = identifier;
    this.raiseMessage = message;
  }
}

/** A JIT limitation: the interpreter re-runs the unit. */
export function jit_raise_bail(why) {
  throw new JitRaise("bail", "", why);
}

/** A MATLAB error with the interpreter's exact message and identifier. */
export function jit_raise_error(identifier, message) {
  throw new JitRaise("error", identifier, message);
}

/** A MATLAB error whose wording the JIT does not reproduce. */
export function jit_raise_error_deferred() {
  throw new JitRaise("error", "", null);
}

/** A read of a local that may be unassigned: the interpreter raises its
 *  undefined-variable error. */
export function jit_bound(v) {
  if (v === undefined) jit_raise_error_deferred();
  return v;
}

/** numbl's shape spelling in error messages: "[2,3]". */
export function jit_fmt_shape(shape) {
  return `[${shape.join(",")}]`;
}
