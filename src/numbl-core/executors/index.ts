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
} from "./types.js";
export { Registry, makeRootContext } from "./registry.js";
export type { DispatchResult, CallDispatchResult } from "./registry.js";
export { DispatchContext } from "./context.js";
export type { DispatchScope } from "./context.js";
export { ExecutorCache } from "./cache.js";
export type { JitType, SignCategory } from "../jitTypes.js";
export {
  jitTypeKey,
  unifyJitTypes,
  isScalarType,
  isTensorType,
  isComplexType,
  isKnownInteger,
} from "../jitTypes.js";
export { inferJitType } from "../interpreter/builtins/types.js";
export { jsJitTopLevelExecutor } from "./jsJit/topLevelExecutor.js";
export { jsJitLoopExecutor } from "./jsJit/loopExecutor.js";
export { jsJitCallExecutor } from "./jsJit/callExecutor.js";
export { registerExecutorsForOpt } from "./plugins.js";
