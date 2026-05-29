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
import { jitCallExecutor } from "./jit/callExecutor.js";
import { jitLoopExecutor } from "./jit/loopExecutor.js";
import { jitTopLevelExecutor } from "./jit/topLevelExecutor.js";
import { cJitCallExecutor } from "./jit/cJitCallExecutor.js";
import { cJitLoopExecutor } from "./jit/cJitLoopExecutor.js";
import { cJitTopLevelExecutor } from "./jit/cJitTopLevelExecutor.js";

/** Optimization mode label.
 *
 *   - `"0"` — pure AST interpreter, no executors registered.
 *   - `"1"` — mtoc2 JS-JIT: all three shapes (top-level, loop,
 *     call) emit JS via `compileSpec`. mtoc2 declines
 *     (`UnsupportedConstruct` / `JitTypeError`) fall back to the
 *     interpreter cleanly.
 *   - `"2"` — mtoc2 C-JIT-first with JS-JIT fallback. Both backends
 *     register their executors; the dispatcher picks based on cost,
 *     so the C path wins where it applies (scalar / tensor numeric
 *     types with a wired `nativeBridge`) and the JS path picks up
 *     the slack (struct / class / handle / no-cc-available).
 *     Requires `cc` on the PATH and `koffi` installed; without
 *     either, the C executors decline and `"2"` collapses to the
 *     same behaviour as `"1"`. */
export type OptLevel = "0" | "1" | "2";

export const OPT_LEVELS: readonly OptLevel[] = ["0", "1", "2"];

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
      registry.registerWholeScope(jitTopLevelExecutor);
      registry.register(jitLoopExecutor);
      registry.register(jitCallExecutor);
      return;
    case "2":
      // C-JIT first, JS-JIT as fallback. Both compete via the
      // dispatcher's cost model — C-JIT proposes only where it can
      // marshal the types, and its lower per-call/run cost wins
      // when both match. Outside its acceptance set, JS-JIT picks
      // up unchanged.
      registry.registerWholeScope(cJitTopLevelExecutor);
      registry.registerWholeScope(jitTopLevelExecutor);
      registry.register(cJitCallExecutor);
      registry.register(cJitLoopExecutor);
      registry.register(jitLoopExecutor);
      registry.register(jitCallExecutor);
      return;
  }
}
