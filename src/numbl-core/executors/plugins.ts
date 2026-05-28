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
import { mtoc2TopLevelExecutor } from "./mtoc2/topLevelExecutor.js";

/** Optimization mode label.
 *
 *   - `"0"` — pure AST interpreter, no executors registered.
 *   - `"1"` — mtoc2 JIT: top-level + call shapes. Whole-script
 *     bodies and user-function calls get type-specialized JS via
 *     `compileSpec`; mtoc2 declines (`UnsupportedConstruct` /
 *     `Mtoc2TypeError`) fall back to the interpreter cleanly. */
export type OptLevel = "0" | "1";

export const OPT_LEVELS: readonly OptLevel[] = ["0", "1"];

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
      registry.register(mtoc2CallExecutor);
      return;
  }
}
