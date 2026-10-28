// JS sibling of `error_fmt.h`: format the message and throw. Any throw
// from emitted code makes the executor bail, and the interpreter re-runs
// the unit to raise numbl's canonical error (with its identifier), so
// the error built here is never what the user sees.

import { mtoc2_sprintf_format } from "./format_engine.js";

export function mtoc2_error_fmt(fmt, ...args) {
  const msg = mtoc2_sprintf_format(fmt, args);
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
