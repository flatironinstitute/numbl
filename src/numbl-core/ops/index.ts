/**
 * Tensor-ops layer: pointer-based, op-code dispatched, native + TS parity.
 *
 * Every op has identical signatures across native (C) and TS implementations.
 * Caller owns input AND output memory; ops never allocate output.
 */

export { tensorOps } from "./dispatch.js";
export { OpRealBin, OpComplexBin, OpUnary } from "./opCodes.js";
