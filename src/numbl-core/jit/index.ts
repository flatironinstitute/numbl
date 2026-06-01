/**
 * Public surface of numbl's vendored JIT compiler. This is the single
 * module the executor registry (`src/numbl-core/executors/jit/*`) imports
 * from — it replaces the former cross-repo `src/numbl-core/mtoc2/index.ts`
 * bridge. Everything the JIT exposes to the rest of numbl appears here
 * exactly once.
 */

export {
  compileSpec,
  type CompileSpecArgs,
  type CompileSpecResult,
} from "./compileSpec.js";

export {
  compileSpecC,
  type CompileSpecCArgs,
  type CompileSpecCResult,
  type SpecCSignature,
  type SpecCParam,
  type SpecCOutput,
} from "./compileSpecC.js";

export { Workspace } from "./workspace/workspace.js";
export type { WorkspaceFile } from "./workspace/workspace.js";

export { Lowerer } from "./lowering/lower.js";

export {
  UnsupportedConstruct,
  TypeError as JitTypeError,
} from "./lowering/errors.js";

export {
  scalarDouble,
  scalarComplex,
  scalarLogical,
  tensorDouble,
  tensorComplex,
  tensorDoubleFromDims,
  tensorComplexFromDims,
  withoutExact,
  isMultiElement,
  UNKNOWN,
  VOID,
  DIM_ONE,
} from "./lowering/types.js";

export type { Type, DimInfo, NumericType } from "./lowering/types.js";
