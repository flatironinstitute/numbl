/**
 * Plugin registration entry points.
 *
 * A "plugin" here is just a function that registers one or more
 * executors with the registry. Different `--opt` modes call different
 * subsets of these.
 */

import type { Registry } from "./registry.js";
import { interpreterExecutor } from "./interpreter/interpreterExecutor.js";
import { jsJitTopLevelExecutor } from "./jsJit/topLevelExecutor.js";
import { jsJitLoopExecutor } from "./jsJit/loopExecutor.js";
import { jsJitCallExecutor } from "./jsJit/callExecutor.js";

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

/** `--opt 1` JS-JIT plugin. Registers:
 *
 *   - js-jit-top-level — JS codegen for the top-level shape.
 *   - js-jit-loop      — JS codegen for the loop shape.
 *   - js-jit-call      — JS codegen for the call shape. */
export function registerJsJitPlugin(registry: Registry): void {
  registry.register(jsJitTopLevelExecutor);
  registry.register(jsJitLoopExecutor);
  registry.register(jsJitCallExecutor);
}
