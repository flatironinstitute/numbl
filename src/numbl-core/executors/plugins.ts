/**
 * Mode-driven executor registration.
 *
 * `registerExecutorsForOpt(registry, opt)` is the single switch from
 * an `--opt` level to a set of registered executors. Adding a new
 * mode (e.g. `--opt 2` for the C-JIT plugin) means extending the
 * switch here — no other call-site changes.
 *
 * The AST interpreter is the dispatcher's hardcoded last-resort
 * fallback (see `Registry.dispatch`); it doesn't need to be a
 * registered executor.
 */

import type { Registry } from "./registry.js";
import { jsJitTopLevelExecutor } from "./jsJit/topLevelExecutor.js";
import { jsJitLoopExecutor } from "./jsJit/loopExecutor.js";
import { jsJitCallExecutor } from "./jsJit/callExecutor.js";

/** Register the executors for a given `--opt` level.
 *
 *   - 0 — no executors; the AST interpreter handles everything.
 *   - 1 — JS-JIT (top-level / loop / call).
 *   - 2 — JS-JIT plus the C-JIT optimizer plugin (added by e3). */
export function registerExecutorsForOpt(registry: Registry, opt: number): void {
  if (opt >= 1) {
    registry.register(jsJitTopLevelExecutor);
    registry.register(jsJitLoopExecutor);
    registry.register(jsJitCallExecutor);
  }
}
