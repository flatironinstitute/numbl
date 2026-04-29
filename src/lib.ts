/**
 * numbl library entry point.
 *
 * Usage:
 *   import { executeCode } from "numbl";
 *   const result = executeCode('disp("hello")');
 *   console.log(result.output); // ["hello"]
 */

// Side-effect: registers the C-JIT (e3) executors so Node consumers
// can pass `optimization: "e3"` to executeCode. The browser worker
// uses a different entry that does not import this module.
import "./numbl-core/executors/cJit/register.js";

export { executeCode } from "./numbl-core/executeCode.js";
export type {
  ExecOptions,
  ExecResult,
  ProfileData,
  BuiltinProfileEntry,
  BuiltinProfileBreakdown,
} from "./numbl-core/executeCode.js";

export type { FileIOAdapter } from "./numbl-core/fileIOAdapter.js";
export type { SystemAdapter } from "./numbl-core/systemAdapter.js";
export type {
  WorkspaceFile,
  NativeBridge,
} from "./numbl-core/workspace/index.js";
export type { PlotInstruction } from "./graphics/types.js";

export type {
  RuntimeValue,
  RuntimeTensor,
  RuntimeString,
  RuntimeLogical,
  RuntimeCell,
  RuntimeStruct,
  RuntimeFunction,
  RuntimeDictionary,
} from "./numbl-core/runtime/index.js";

export { RuntimeError } from "./numbl-core/runtime/index.js";
export { RTV } from "./numbl-core/runtime/constructors.js";
export { displayValue } from "./numbl-core/runtime/display.js";
export { toNumber, toBool, toString } from "./numbl-core/runtime/convert.js";
