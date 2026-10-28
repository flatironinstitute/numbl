/**
 * REPL-style session over `executeCode`. Each call builds a fresh runtime, so
 * the state a session carries from one call to the next is threaded through
 * here: workspace variables, plot hold state, figure handles, search paths and
 * workspace files (as changed by addpath/rmpath), the implicit cwd, and the
 * uihtml session left by the latest run. Used by the CLI REPL and the
 * browser workers, and exported from "numbl" for other hosts.
 */

import {
  executeCode,
  type ExecOptions,
  type ExecResult,
  type FigureHandles,
  type UihtmlSession,
} from "./executeCode.js";
import type { RuntimeValue } from "./runtime/index.js";
import type { NativeBridge, WorkspaceFile } from "./workspace/types.js";

export interface ExecSessionInit {
  workspaceFiles?: WorkspaceFile[];
  searchPaths?: string[];
  /** From initNodePlatform in Node; enables the C-JIT and native libraries. */
  nativeBridge?: NativeBridge;
}

export class ExecSession {
  variableValues: Record<string, RuntimeValue> = {};
  holdState = false;
  figureHandles: FigureHandles = { current: 0, max: 0 };
  workspaceFiles: WorkspaceFile[];
  searchPaths: string[] | undefined;
  implicitCwdPath: string | null | undefined;
  /** Live uihtml callbacks from the latest run, if any (see UihtmlSession). */
  uihtmlSession: UihtmlSession | null = null;
  /** Appended to the search paths of every `execute` unless already present
   *  (e.g. the mip core directory). */
  extraSearchPaths: string[] = [];
  nativeBridge: NativeBridge | undefined;

  constructor(init: ExecSessionInit = {}) {
    this.workspaceFiles = init.workspaceFiles ?? [];
    this.searchPaths = init.searchPaths;
    this.nativeBridge = init.nativeBridge;
  }

  /** Run `code` against the carried state and adopt the result's state. A
   *  new run supersedes the previous uihtml session; on error the rest of
   *  the state is left unchanged. */
  execute(
    code: string,
    options: ExecOptions = {},
    mainFileName = "repl"
  ): ExecResult {
    this.disposeUihtmlSession();
    const searchPaths = [...(this.searchPaths ?? [])];
    for (const p of this.extraSearchPaths) {
      if (!searchPaths.includes(p)) searchPaths.push(p);
    }
    const result = executeCode(
      code,
      {
        initialVariableValues: this.variableValues,
        initialHoldState: this.holdState,
        initialFigureHandles: this.figureHandles,
        implicitCwdPath: this.implicitCwdPath,
        ...options,
      },
      this.workspaceFiles,
      mainFileName,
      searchPaths,
      this.nativeBridge
    );
    this.adopt(result);
    return result;
  }

  /** Carry a successful run's state forward. */
  adopt(result: ExecResult): void {
    this.variableValues = result.variableValues;
    this.holdState = result.holdState;
    this.figureHandles = result.figureHandles;
    if (result.searchPaths) {
      this.searchPaths = result.searchPaths;
      this.workspaceFiles = result.workspaceFiles ?? [];
    }
    if (result.implicitCwdPath !== undefined) {
      this.implicitCwdPath = result.implicitCwdPath;
    }
    this.uihtmlSession = result.uihtmlSession ?? null;
  }

  /** Forget variables, hold state and figures (paths and cwd are kept). */
  clearWorkspace(): void {
    this.disposeUihtmlSession();
    this.variableValues = {};
    this.holdState = false;
    this.figureHandles = { current: 0, max: 0 };
  }

  disposeUihtmlSession(): void {
    this.uihtmlSession?.dispose();
    this.uihtmlSession = null;
  }
}
