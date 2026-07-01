/**
 * numbl library entry point.
 *
 * Usage:
 *   import { executeCode } from "numbl";
 *   const result = executeCode('disp("hello")');
 *   console.log(result.output); // ["hello"]
 */

export { executeCode } from "./numbl-core/executeCode.js";

// Delaunay triangulation (delaunay/delaunayn) is backed by the qhull WASM
// module, which loads asynchronously. Call loadDelaunayBackend() once before
// running code that uses those builtins; otherwise they throw.
// setDelaunayBackend lets you supply a custom backend (e.g. a browser loader).
export { loadQhullNodeBackend as loadDelaunayBackend } from "./numbl-core/native/qhull-node.js";
export { setDelaunayBackend } from "./numbl-core/native/geometry-bridge.js";

export type {
  ExecOptions,
  ExecResult,
  ProfileData,
  BuiltinProfileEntry,
  BuiltinProfileBreakdown,
} from "./numbl-core/executeCode.js";

export type { FileIOAdapter } from "./numbl-core/fileIOAdapter.js";
export type { SystemAdapter } from "./numbl-core/systemAdapter.js";

// Browser embedding: the in-memory filesystem and its adapters, as used by
// numbl's own web worker. Exporting them lets a third-party page run scripts
// in its own worker (executeCode + a VFS of project files) without vendoring
// numbl source. See ExecOptions.onHtmlSourceEvent and ExecResult.uihtmlSession
// for the two halves of the uihtml host bridge.
export { VirtualFileSystem } from "./vfs/VirtualFileSystem.js";
export { BrowserFileIOAdapter } from "./vfs/BrowserFileIOAdapter.js";
export { BrowserSystemAdapter } from "./vfs/BrowserSystemAdapter.js";
export type { UihtmlSession } from "./numbl-core/executeCode.js";
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
