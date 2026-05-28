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
import { mtoc2CallExecutor } from "./mtoc2/callExecutor.js";
import { mtoc2LoopExecutor } from "./mtoc2/loopExecutor.js";
import { mtoc2TopLevelExecutor } from "./mtoc2/topLevelExecutor.js";
import { mtoc2CJitCallExecutor } from "./mtoc2/cJitCallExecutor.js";
import { mtoc2CJitLoopExecutor } from "./mtoc2/cJitLoopExecutor.js";
import { mtoc2CJitTopLevelExecutor } from "./mtoc2/cJitTopLevelExecutor.js";

/** Optimization mode label.
 *
 *   - `"0"` — pure AST interpreter, no executors registered.
 *   - `"1"` — mtoc2 JS-JIT: all three shapes (top-level, loop,
 *     call) emit JS via `compileSpec`. mtoc2 declines
 *     (`UnsupportedConstruct` / `Mtoc2TypeError`) fall back to the
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
      registry.registerWholeScope(mtoc2TopLevelExecutor);
      registry.register(mtoc2LoopExecutor);
      registry.register(mtoc2CallExecutor);
      return;
    case "2":
      // C-JIT first, JS-JIT as fallback. Both compete via the
      // dispatcher's cost model — C-JIT proposes only where it can
      // marshal the types, and its lower per-call/run cost wins
      // when both match. Outside its acceptance set, JS-JIT picks
      // up unchanged.
      registry.registerWholeScope(mtoc2CJitTopLevelExecutor);
      registry.registerWholeScope(mtoc2TopLevelExecutor);
      registry.register(mtoc2CJitCallExecutor);
      registry.register(mtoc2CJitLoopExecutor);
      registry.register(mtoc2LoopExecutor);
      registry.register(mtoc2CallExecutor);
      return;
  }
}
