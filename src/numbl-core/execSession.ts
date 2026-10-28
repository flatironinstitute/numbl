/**
 * REPL-style session over `executeCode`. Each call builds a fresh runtime;
 * what MATLAB keeps between commands lives in the session's `SessionState`
 * (the base workspace, globals, persistents, warning and PRNG state, tic,
 * appdata, ...), which every run is given. The session also threads plot
 * hold state, figure handles, search paths and workspace files (as changed
 * by addpath/rmpath), the implicit cwd, and the uihtml session left by the
 * latest run. Used by the CLI REPL and the browser workers, and exported
 * from "numbl" for other hosts.
 */

import {
  executeCode,
  type ExecOptions,
  type ExecResult,
  type FigureHandles,
  type UihtmlSession,
} from "./executeCode.js";
import type { RuntimeValue } from "./runtime/index.js";
import { SessionState } from "./sessionState.js";
import type { NativeBridge, WorkspaceFile } from "./workspace/types.js";

export interface ExecSessionInit {
  workspaceFiles?: WorkspaceFile[];
  searchPaths?: string[];
  /** From initNodePlatform in Node; enables the C-JIT and native libraries. */
  nativeBridge?: NativeBridge;
}

export class ExecSession {
  /** State every run of the session shares (see SessionState). */
  state = new SessionState();
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

  /** The base workspace's variables (including the globals it declares). */
  get variableValues(): Record<string, RuntimeValue> {
    return this.state.baseEnv.toRecord();
  }

  /** Run `code` against the session and adopt the result's state. A new
   *  run supersedes the previous uihtml session. On error, changes the run
   *  made to variables, globals and the like stay, as at the MATLAB prompt;
   *  the rest (hold state, figures, paths) is left unchanged. */
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
        session: this.state,
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

  /** Carry a successful run's state forward (its variables are already in
   *  `state`). */
  adopt(result: ExecResult): void {
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

  /** Start over: forget variables, globals, persistents and the rest of
   *  the session state, hold state and figures (paths and cwd are kept). */
  clearWorkspace(): void {
    this.disposeUihtmlSession();
    this.state = new SessionState();
    this.holdState = false;
    this.figureHandles = { current: 0, max: 0 };
  }

  disposeUihtmlSession(): void {
    this.uihtmlSession?.dispose();
    this.uihtmlSession = null;
  }
}
