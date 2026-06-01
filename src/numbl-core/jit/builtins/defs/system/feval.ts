/**
 * feval(handle_or_name, args...) — invoke a function handle (or a
 * function named by a char) with the given args.
 *
 * Implementation strategy:
 *
 *   The c-aot / js-aot backends cannot route through a builtin
 *   emitC / emitJs hook because invoking a user function requires
 *   specialization (the target's mangled C name is created at
 *   lowering time, with the actual arg types) — information a
 *   builtin emit hook doesn't have access to. So the lowerer
 *   intercepts `feval(...)` in `lowerFuncCall` and
 *   `lowerMultiAssign` and rewrites it to a direct call on the
 *   underlying handle / function name (see `tryLowerFevalCall` in
 *   `src/lowering/lowerFuncCall.ts`). Once the rewrite fires, the
 *   builtin's emit hooks are never reached.
 *
 *   The interpreter parallels this with a special-case in
 *   `callByName` (next to the existing `struct` / `cell`
 *   special-cases) — it consumes args[0] as a handle or char,
 *   dispatches via `callHandle` or recursively via `callByName`,
 *   and threads `nargout` through. Once that fires, this builtin's
 *   `call` hook is unreachable.
 *
 *   This file therefore exists primarily to:
 *     - keep `feval` in `allBuiltinNames()` for syntax highlighting
 *       and the `@feval` rejection path,
 *     - validate arity / argument shape via `transfer` as a safety
 *       net if a future codepath bypasses the lowerer / interpreter
 *       interception.
 *
 *   All four hooks raise `UnsupportedConstruct` with a message
 *   pointing the reader at where the real implementation lives.
 */

import { UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Builtin } from "../../registry.js";

const NOT_INTERCEPTED =
  "'feval' must be intercepted by the lowerer or interpreter " +
  "(see src/lowering/lowerFuncCall.ts and " +
  "src/interpreter/interpreterFunctions.ts); reaching the builtin " +
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
  call() {
    throw new UnsupportedConstruct(NOT_INTERCEPTED);
  },
};
