/**
 * numbl Node entry point ("numbl/node").
 *
 * Node.js implementations of the platform adapters consumed by executeCode
 * (ExecOptions.fileIO / ExecOptions.system), the directory scanner used
 * to build the workspace file set from search-path directories, and
 * initNodePlatform, which loads the native LAPACK addon and sets up the
 * C-JIT for optimization "2". These are the same implementations the numbl
 * CLI uses; they are exported here so Node hosts embedding numbl do not
 * have to vendor them.
 *
 * Usage:
 *   import { executeCode } from "numbl";
 *   import {
 *     initNodePlatform,
 *     NodeFileIOAdapter,
 *     NodeSystemAdapter,
 *     scanMFiles,
 *   } from "numbl/node";
 *
 *   const { nativeBridge } = await initNodePlatform();
 *   const searchPaths = ["/path/to/matlab/code"];
 *   const workspaceFiles = searchPaths.flatMap(p => scanMFiles(p));
 *   executeCode(
 *     'disp("hello")',
 *     {
 *       fileIO: new NodeFileIOAdapter(),
 *       system: new NodeSystemAdapter(),
 *       optimization: "2",
 *     },
 *     workspaceFiles,
 *     "eval.m",
 *     searchPaths,
 *     nativeBridge
 *   );
 */

export { NodeFileIOAdapter } from "./cli-fileio.js";
export { NodeSystemAdapter } from "./cli-system.js";
export { scanMFiles } from "./cli-scan.js";
export { initNodePlatform } from "./node-platform.js";
export type { NodePlatform, NodePlatformOptions } from "./node-platform.js";
