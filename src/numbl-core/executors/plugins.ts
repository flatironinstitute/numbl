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
 *
 * The C-JIT (e3) executors are registered via an injected callback
 * (`setCJitRegistrar`). A Node-only entry point (`cli.ts`) imports
 * `executors/cJit/register.ts`, which calls `setCJitRegistrar` at
 * load time. The browser worker never imports that module, so the
 * cJit subtree (which pulls in `node:fs`/`node:os`/`node:child_process`
 * via `compile.ts`) stays out of the web bundle.
 */

import type { Registry } from "./registry.js";
import { mtoc2CallExecutor } from "./mtoc2/callExecutor.js";
import { mtoc2TopLevelExecutor } from "./mtoc2/topLevelExecutor.js";

/** Optimization mode label.
 *
 *   - `"0"`  — pure AST interpreter, no executors registered.
 *   - `"1"`  — mtoc2 JIT (call shape only). Hot-loop and top-level
 *     triggers are intentionally dropped vs the legacy JS-JIT: mtoc2
 *     has no natural loop-body entry point, and top-level scripts run
 *     well enough on the interpreter. Function calls get type-
 *     specialized JS via `compileSpec`.
 *   - `"e3"` — C-JIT scalar-loop only. Targets compute-bound scalar
 *     loops by compiling the loop body to C and loading via koffi.
 *     Does NOT register the mtoc2 suite — the C-JIT loop executor
 *     either matches (and runs in C) or falls back to the AST
 *     interpreter. */
export type OptLevel = "0" | "1" | "e3";

export const OPT_LEVELS: readonly OptLevel[] = ["0", "1", "e3"];

export function isOptLevel(s: string): s is OptLevel {
  return (OPT_LEVELS as readonly string[]).includes(s);
}

type CJitRegistrar = (registry: Registry) => void;
let cJitRegistrar: CJitRegistrar | null = null;

/** Wire up the C-JIT (e3) executors. Called from a Node-only entry
 *  point at startup so the browser bundle never reaches the cJit
 *  module graph. */
export function setCJitRegistrar(fn: CJitRegistrar): void {
  cJitRegistrar = fn;
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
    case "e3":
      if (!cJitRegistrar) {
        throw new Error(
          "--opt e3 (C-JIT) is only available in the Node CLI; " +
            "cJit registrar not set"
        );
      }
      cJitRegistrar(registry);
      return;
  }
}
