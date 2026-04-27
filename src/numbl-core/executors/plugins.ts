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
import { scalarFnCKernelExecutor } from "./e2/scalarFnCKernelExecutor.js";
import { jsJitLoopExecutor } from "./jsJit/loopExecutor.js";
import { jsJitTopLevelExecutor } from "./jsJit/topLevelExecutor.js";
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

/** `--opt 1` / `--opt e1` JS-JIT plugins. Registers:
 *
 *   - js-jit-loop  — wraps tryJitFor / tryJitWhile (stmt-level).
 *   - js-jit-top-level — wraps tryJitTopLevel (stmt-level).
 *   - js-jit-call  — wraps tryJitCall (function-call dispatch). */
export function registerJsJitPlugin(registry: Registry): void {
  registry.register(jsJitLoopExecutor);
  registry.register(jsJitTopLevelExecutor);
  registry.registerCall(jsJitCallExecutor);
}

/** `--opt e2` plugins. Registers:
 *
 *   - chain-c-kernel — wraps tryE2Assign (stmt-level).
 *   - loop-c-kernel  — wraps tryE2Loop (stmt-level).
 *   - scalar-fn-c-kernel — wraps tryE2ScalarFn (function-call). */
export function registerE2Plugin(registry: Registry): void {
  registry.register(chainCKernelExecutor);
  registry.register(loopCKernelExecutor);
  registry.registerCall(scalarFnCKernelExecutor);
}
