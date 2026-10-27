/**
 * Registration helpers for builtins that need the active Runtime. They are
 * ordinary IBuiltins; each call gets the runtime from the current-runtime
 * stack.
 */

import type { Runtime } from "./runtime.js";
import type { RuntimeValue } from "./types.js";
import { RuntimeError } from "./error.js";
import { getCurrentRuntime } from "./refcount.js";
import { registerIBuiltin } from "../interpreter/builtins/types.js";

/** The runtime executing the current call. */
export function currentRuntime(): Runtime {
  const rt = getCurrentRuntime();
  if (!rt) throw new RuntimeError("no active runtime");
  return rt as Runtime;
}

/** Register a builtin that accepts any args and gets the active runtime. */
export function registerSpecial(
  name: string,
  fn: (nargout: number, args: RuntimeValue[], rt: Runtime) => unknown
): void {
  registerIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [{ kind: "unknown" as const }],
      apply: (args: RuntimeValue[], nargout: number) =>
        fn(nargout, args, currentRuntime()) as RuntimeValue | RuntimeValue[],
    }),
  });
}

/** Register a builtin with no outputs. The call site raises "Too many output
 *  arguments." when nargout > 0, and `ans` is not set when called as a
 *  statement. */
export function registerSpecialVoid(
  name: string,
  fn: (args: RuntimeValue[], rt: Runtime) => void
): void {
  registerIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [],
      apply: (args: RuntimeValue[]) => {
        fn(args, currentRuntime());
        return undefined as unknown as RuntimeValue;
      },
    }),
  });
}
