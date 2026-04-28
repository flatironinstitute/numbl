/**
 * Executor registry — public surface.
 *
 * See docs/developer_reference/executors.md for the design.
 */

export type {
  Executor,
  CostEstimate,
  BailReason,
  RunResult,
  Proposal,
  CallExecutor,
  CallProposal,
  CallRunResult,
} from "./types.js";
export { Registry, makeRootContext } from "./registry.js";
export type { DispatchResult, CallDispatchResult } from "./registry.js";
export { DispatchContext } from "./context.js";
export type { DispatchScope } from "./context.js";
export { ExecutorCache } from "./cache.js";
export type { TypeInfo, SignCategory } from "./typeInfo.js";
export {
  typeInfoKey,
  unifyTypeInfo,
  inferTypeInfo,
  isScalarType,
  isTensorType,
  isComplexType,
  isKnownInteger,
} from "./typeInfo.js";
export { interpreterExecutor } from "./interpreter/interpreterExecutor.js";
export { chainCKernelExecutor } from "./e2/chainCKernelExecutor.js";
export { loopCKernelExecutor } from "./e2/loopCKernelExecutor.js";
export { scalarFnCKernelExecutor } from "./e2/scalarFnCKernelExecutor.js";
export { jsJitLoopExecutor } from "./jsJit/loopExecutor.js";
export { jsJitTopLevelExecutor } from "./jsJit/topLevelExecutor.js";
export { jsJitCallExecutor } from "./jsJit/callExecutor.js";
export {
  registerInterpreterPlugin,
  registerJsJitPlugin,
  registerE2Plugin,
} from "./plugins.js";
