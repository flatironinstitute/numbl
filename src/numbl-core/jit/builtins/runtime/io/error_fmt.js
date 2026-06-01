// JS sibling of `error_fmt.h`. Format the message + arguments and
// throw a runtime error. Mirrors numbl's `error(...)` behavior
// (stderr + halt).
//
// The thrown error carries an `identifier` string property so a
// `try / catch ME` arm in the interpreter can populate the `ME`
// struct's `identifier` field. When the caller passes no identifier
// (the common `error('msg')` form), it's the empty string.

import { mtoc2_sprintf_format } from "./format_engine.js";

export function mtoc2_error_fmt(fmt, ...args) {
  const msg = mtoc2_sprintf_format(fmt, args);
  // Numbl writes to stderr; in the JS host we route through the same
  // `$write` channel by emitting a "stderr" sentinel prefix the host
  // can split on. For now, throw — the CLI's runner formats the
  // exception to stderr, matching the user-visible result.
  const err = new Error(msg);
  err.identifier = "";
  throw err;
}

export function mtoc2_error_fmt_id(identifier, fmt, ...args) {
  const msg = mtoc2_sprintf_format(fmt, args);
  const err = new Error(msg);
  err.identifier = identifier;
  throw err;
}
