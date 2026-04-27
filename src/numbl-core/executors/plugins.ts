/**
 * Plugin registration entry points.
 *
 * A "plugin" here is just a function that registers one or more
 * executors with the registry. Different `--opt` modes call different
 * subsets of these. The browser bundle simply omits modules whose
 * dependencies it can't satisfy (e.g., the C-kernel plugin is
 * Node-only).
 *
 * For the initial skeleton, only the interpreter plugin exists. As
 * executors are ported, each gets its own plugin module that lands
 * here.
 */

import type { Registry } from "./registry.js";
import { interpreterExecutor } from "./interpreter/interpreterExecutor.js";
import { chainCKernelExecutor } from "./e2/chainCKernelExecutor.js";
import { loopCKernelExecutor } from "./e2/loopCKernelExecutor.js";
import { jsJitLoopExecutor } from "./jsJit/loopExecutor.js";
import { jsJitTopLevelExecutor } from "./jsJit/topLevelExecutor.js";

/** Register the AST interpreter as a regular executor.
 *
 *  Normally not needed: the dispatcher already calls
 *  `interp.execStmt` as a hardcoded last-resort fallback, which is
 *  faster than going through the executor protocol on the hot path.
 *  This entry point exists for tests that want the interpreter
 *  visible as a registered candidate, or for diagnostics. */
export function registerInterpreterPlugin(registry: Registry): void {
  registry.register(interpreterExecutor);
}

/** `--opt 1` / `--opt e1` JS-JIT plugins. Registers the loop executor
 *  (For + While) and the whole-script top-level executor. The
 *  user-function call path (`tryJitCall`) is still inline; it lives
 *  at expression-evaluation time rather than stmt-dispatch time and
 *  doesn't fit the registry interface as currently shaped. */
export function registerJsJitPlugin(registry: Registry): void {
  registry.register(jsJitLoopExecutor);
  registry.register(jsJitTopLevelExecutor);
}

/** `--opt e2` plugins. Registers the per-assign / chain C-kernel
 *  executor and the whole-loop C-kernel executor. Both are currently
 *  shims around the legacy `tryE2Assign` / `tryE2Loop`. The
 *  scalar-function kernel is still inline; it'll be ported in a
 *  subsequent commit. */
export function registerE2Plugin(registry: Registry): void {
  registry.register(chainCKernelExecutor);
  registry.register(loopCKernelExecutor);
}
