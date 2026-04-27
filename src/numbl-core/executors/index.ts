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
  MatchResult,
} from "./types.js";
export { Registry, makeRootContext } from "./registry.js";
export type { DispatchResult } from "./registry.js";
export { DispatchContext } from "./context.js";
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
export { interpreterExecutor } from "./interpreterExecutor.js";
export { chainCKernelExecutor } from "./chainCKernelExecutor.js";
export { registerInterpreterPlugin, registerE2Plugin } from "./plugins.js";
