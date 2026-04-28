/**
 * Plugin registration entry points.
 *
 * A "plugin" here is just a function that registers one or more
 * executors with the registry. Different `--opt` modes call different
 * subsets of these.
 *
 * The AST interpreter is the dispatcher's hardcoded last-resort
 * fallback (see `Registry.dispatch`); it doesn't need to be a
 * registered executor.
 */

import type { Registry } from "./registry.js";
import { jsJitTopLevelExecutor } from "./jsJit/topLevelExecutor.js";
import { jsJitLoopExecutor } from "./jsJit/loopExecutor.js";
import { jsJitCallExecutor } from "./jsJit/callExecutor.js";

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
