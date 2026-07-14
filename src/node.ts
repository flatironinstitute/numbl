/**
 * numbl Node entry point ("numbl/node").
 *
 * Node.js implementations of the platform adapters consumed by executeCode
 * (ExecOptions.fileIO / ExecOptions.system), plus the directory scanner used
 * to build the workspace file set from search-path directories. These are
 * the same implementations the numbl CLI uses; they are exported here so
 * Node hosts embedding numbl do not have to vendor them.
 *
 * Usage:
 *   import { executeCode } from "numbl";
 *   import {
 *     NodeFileIOAdapter,
 *     NodeSystemAdapter,
 *     scanMFiles,
 *   } from "numbl/node";
 *
 *   const searchPaths = ["/path/to/matlab/code"];
 *   const workspaceFiles = searchPaths.flatMap(p => scanMFiles(p));
 *   executeCode(
 *     'disp("hello")',
 *     { fileIO: new NodeFileIOAdapter(), system: new NodeSystemAdapter() },
 *     workspaceFiles,
 *     "eval.m",
 *     searchPaths
 *   );
 */

export { NodeFileIOAdapter } from "./cli-fileio.js";
export { NodeSystemAdapter } from "./cli-system.js";
export { scanMFiles } from "./cli-scan.js";
