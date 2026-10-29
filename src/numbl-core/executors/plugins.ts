/**
 * Mode-driven executor registration.
 *
 * `registerExecutorsForOpt(registry, opt, host)` is the single switch from an
 * `--opt` value to a set of registered executors. Each JIT backend
 * contributes one executor per shape (top-level, loop, call); the
 * registry tries them in registration order. The AST interpreter is the
 * registry's last-resort fallback and isn't registered.
 */

import type { Registry } from "./registry.js";
import type { JitBackend } from "./jit/backend.js";
import { jsBackend } from "./jit/jsBackend.js";
import { makeTopLevelExecutor } from "./jit/topLevelExecutor.js";
import { makeLoopExecutor } from "./jit/loopExecutor.js";
import { makeCallExecutor } from "./jit/callExecutor.js";
import type { NumblHost } from "../host.js";

/** Optimization mode label.
 *
 *   - `"0"` — pure AST interpreter, no executors registered.
 *   - `"1"` — JS-JIT for all three shapes.
 *   - `"2"` — C-JIT first, JS-JIT as fallback, per shape. C-JIT takes
 *     what it can marshal (scalar / tensor numeric types with a wired
 *     `nativeBridge`); JS-JIT picks up the rest (struct / class / handle).
 *     Without `cc` + `koffi` the C executors decline and `"2"` behaves
 *     like `"1"`. */
export type OptLevel = "0" | "1" | "2";

export const OPT_LEVELS: readonly OptLevel[] = ["0", "1", "2"];

export function isOptLevel(s: string): s is OptLevel {
  return (OPT_LEVELS as readonly string[]).includes(s);
}

function registerBackend(registry: Registry, backend: JitBackend): void {
  registry.register(makeTopLevelExecutor(backend));
  registry.register(makeLoopExecutor(backend));
  registry.register(makeCallExecutor(backend));
}

/** Register the executors for a given optimization mode. The C backend
 *  comes from the run's host (`host.cjit`, Node only); without it `"2"`
 *  behaves like `"1"`. */
export function registerExecutorsForOpt(
  registry: Registry,
  opt: OptLevel,
  host: NumblHost
): void {
  if (opt === "0") return;
  if (opt === "2" && host.cjit)
    registerBackend(registry, host.cjit as JitBackend);
  registerBackend(registry, jsBackend);
}
