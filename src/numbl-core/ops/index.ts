/**
 * Tensor-ops layer. `tensorOps` holds the op-code dispatched kernels with
 * native and TS implementations (caller-allocated output); `elementwise.ts`
 * is the element-wise kernel layer built on them (broadcasting, real and
 * complex rules, fresh results).
 */

export { tensorOps } from "./dispatch.js";
export { OpBin, OpUnary, OpCmp, OpReduce, OpBessel } from "./opCodes.js";
export {
  elementwiseBinary,
  elementwiseCompare,
  elementwiseUnary,
  mapBinary,
  binaryScalar,
  compareScalar,
  realElem,
  type Operand,
  type DenseOperand,
  type ElemResult,
} from "./elementwise.js";
export {
  broadcastShape,
  tryBroadcastShape,
  broadcastIterate,
  forEachBroadcastRow,
  sizeMismatchError,
} from "./broadcast.js";
