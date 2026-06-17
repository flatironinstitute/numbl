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
import type { Executor } from "./types.js";
import { jitCallExecutor } from "./jit/callExecutor.js";
import { jitLoopExecutor } from "./jit/loopExecutor.js";
import { jitTopLevelExecutor } from "./jit/topLevelExecutor.js";

/** The three C-JIT executors, registered at Node bootstrap. */
export interface CJitExecutors {
  topLevel: Executor;
  loop: Executor;
  call: Executor;
}

/**
 * C-JIT executors, injected at Node bootstrap rather than imported
 * statically. They shell out to `cc` + koffi and pull in the C
 * compile/codegen graph (`compileSpecC`, the C type/value adapters, the
 * marshaling helpers) — all Node-only. Keeping them out of this module's
 * static imports means the browser worker bundle never pulls that graph in;
 * `registerNodeCompileC` (`executors/jit/compileC.node.ts`) registers them.
 * When unregistered (browser), `--opt 2` collapses to `--opt 1`.
 */
let cJitExecutors: CJitExecutors | null = null;

/** Register the C-JIT executors. Called once at Node bootstrap. */
export function registerCJitExecutors(execs: CJitExecutors): void {
  cJitExecutors = execs;
}

/** Optimization mode label.
 *
 *   - `"0"` — pure AST interpreter, no executors registered.
 *   - `"1"` — JS-JIT: all three shapes (top-level, loop,
 *     call) emit JS via `compileSpec`. JIT declines
 *     (`UnsupportedConstruct` / `JitTypeError`) fall back to the
 *     interpreter cleanly.
 *   - `"2"` — C-JIT-first with JS-JIT fallback. Both backends
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
      // up unchanged. When the C executors aren't registered (browser),
      // only the JS-JIT executors register, so `--opt 2` == `--opt 1`.
      if (cJitExecutors) registry.registerWholeScope(cJitExecutors.topLevel);
      registry.registerWholeScope(jitTopLevelExecutor);
      if (cJitExecutors) {
        registry.register(cJitExecutors.call);
        registry.register(cJitExecutors.loop);
      }
      registry.register(jitLoopExecutor);
      registry.register(jitCallExecutor);
      return;
  }
}
