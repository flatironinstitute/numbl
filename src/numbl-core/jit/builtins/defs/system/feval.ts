/**
 * feval(handle_or_name, args...) — invoke a function handle (or a
 * function named by a char) with the given args.
 *
 * The lowerer intercepts `feval(...)` in `lowerFuncCall` and
 * `lowerMultiAssign` and rewrites it to a direct call on the
 * underlying handle / function name (see `tryLowerFevalCall`), since
 * invoking a user function needs specialization that a builtin emit
 * hook can't do. This def exists to keep `feval` in
 * `allBuiltinNames()` and as a safety net: every hook raises
 * `UnsupportedConstruct`.
 */

import { UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Builtin } from "../../registry.js";

const NOT_INTERCEPTED =
  "'feval' must be intercepted by the lowerer " +
  "(see jit/lowering/lowerFuncCall.ts); reaching the builtin " +
  "registry hook indicates a routing bug or an unsupported first " +
  "argument shape (only @name handles, in-scope handle vars, and " +
  "char-name strings are supported — runtime-computed handle " +
  "expressions are deferred)";

export const feval: Builtin = {
  name: "feval",
  transfer(argTypes) {
    if (argTypes.length < 1) {
      throw new UnsupportedConstruct(
        `'feval' expects at least 1 argument (the function handle or ` +
          `name), got 0`
      );
    }
    throw new UnsupportedConstruct(NOT_INTERCEPTED);
  },
  emitC() {
    throw new UnsupportedConstruct(NOT_INTERCEPTED);
  },
  emitJs() {
    throw new UnsupportedConstruct(NOT_INTERCEPTED);
  },
};
