/**
 * Mode-driven executor registration.
 *
 * `registerExecutorsForOpt(registry, opt)` is the single switch from
 * an `--opt` value to a set of registered executors. Adding a new
 * mode means extending the switch here — no other call-site changes.
 *
 * The AST interpreter is the dispatcher's hardcoded last-resort
 * fallback (see `Registry.dispatch`); it doesn't need to be a
 * registered executor.
 */

import type { Registry } from "./registry.js";
import { jsJitTopLevelExecutor } from "./jsJit/topLevelExecutor.js";
import { jsJitLoopExecutor } from "./jsJit/loopExecutor.js";
import { jsJitCallExecutor } from "./jsJit/callExecutor.js";
import { cJitLoopExecutor } from "./cJit/loopExecutor.js";
import { cJitFuseExecutor } from "./cJit/fuseExecutor.js";

/** Optimization mode label.
 *
 *   - `"0"`  — pure AST interpreter, no executors registered.
 *   - `"1"`  — JS-JIT suite (top-level / loop / call).
 *   - `"e3"` — C-JIT scalar-loop only. Targets compute-bound scalar
 *     loops by compiling the loop body to C and loading via koffi.
 *     Does NOT register the JS-JIT suite — the C-JIT loop executor
 *     either matches (and runs in C) or falls back to the AST
 *     interpreter. */
export type OptLevel = "0" | "1" | "e3";

export const OPT_LEVELS: readonly OptLevel[] = ["0", "1", "e3"];

export function isOptLevel(s: string): s is OptLevel {
  return (OPT_LEVELS as readonly string[]).includes(s);
}

/** Register the executors for a given optimization mode. */
export function registerExecutorsForOpt(
  registry: Registry,
  opt: OptLevel
): void {
  switch (opt) {
    case "0":
      return;
    case "1":
      registry.registerWholeScope(jsJitTopLevelExecutor);
      registry.register(jsJitLoopExecutor);
      registry.register(jsJitCallExecutor);
      return;
    case "e3":
      registry.register(cJitLoopExecutor);
      registry.register(cJitFuseExecutor);
      return;
  }
}
