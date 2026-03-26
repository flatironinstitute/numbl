/**
 * Entry point for code execution.
 *
 * Delegates to the interpreter (AST-walk + JIT).
 */

import { type ExecOptions, type ExecResult } from "./executor/types.js";
import type {
  WorkspaceFile,
  NativeBridge,
} from "../numbl-core/workspace/index.js";
import { interpretCode } from "./interpretCode.js";

export function executeCode(
  source: string,
  options: ExecOptions = {},
  workspaceFiles?: WorkspaceFile[],
  mainFileName: string = "script.m",
  searchPaths?: string[],
  nativeBridge?: NativeBridge
): ExecResult {
  return interpretCode(
    source,
    options,
    workspaceFiles,
    mainFileName,
    searchPaths,
    nativeBridge
  );
}
