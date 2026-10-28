/**
 * Where the IDE runs a script: in the browser worker
 * (`WorkerExecutionBackend`) or on a numbl execution server
 * (`RemoteExecutionBackend`). Both report a run through the same events.
 */

import type { OptLevel } from "../numbl-core/executors/plugins.js";
import type { PlotInstruction } from "../numbl-core/plotting/types.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";
import type { WorkerVfsFile, WorkspaceRep } from "../numblWorkerProtocol.js";
import type { StreamDiagnostic } from "../streamProtocol.js";
import type { VfsChanges } from "../vfs/VirtualFileSystem.js";

/** Structured error of a failed run; format it with `formatDiagnostic`. */
export type ExecutionDiagnostic = StreamDiagnostic;

export interface ExecutionRequest {
  /** Source of the script to run. */
  code: string;
  /** Project-relative path of the script. */
  mainFileName: string;
  /** Project and system files, as built by `buildWorkerFiles`. */
  vfsFiles: WorkerVfsFile[];
  /** Text .m files other than the script, as built by `buildWorkerFiles`. */
  workspaceFiles: WorkspaceFile[];
  optimization: OptLevel;
  /** Share variables and files with the REPL (backends with `repl`). */
  persistent: boolean;
}

export interface ExecutionResult {
  status: "success" | "error" | "cancelled";
  generatedJS?: string;
  /** Parsed sources for the AST view (worker only). */
  workspaceRep?: WorkspaceRep | null;
  dispatchUnknownCounts?: Record<string, number>;
}

export interface ExecutionEvents {
  onOutput(text: string): void;
  onPlotInstructions(instructions: PlotInstruction[]): void;
  /** The program called input(); answer with `sendInput`. Without this
   *  handler the worker backend asks with window.prompt. */
  onInputRequest?(prompt: string): void;
  /** One call per diagnostic of a failed run, before `onDone`. */
  onError(diagnostic: ExecutionDiagnostic): void;
  /** Files the run created, modified or deleted (worker only). */
  onVfsChanges?(changes: VfsChanges): void;
  /** The run ended. Called exactly once per run, last. */
  onDone(result: ExecutionResult): void;
}

export interface ExecutionCapabilities {
  /** input() is answered interactively. */
  input: boolean;
  /** uihtml figures are interactive. */
  uihtml: boolean;
  /** A REPL and persistent workspace share state with runs. */
  repl: boolean;
}

export interface ExecutionBackend {
  readonly kind: "worker" | "remote";
  readonly capabilities: ExecutionCapabilities;
  /** Start a run; one at a time. */
  run(request: ExecutionRequest, events: ExecutionEvents): void;
  /**
   * Stop the current run. "stopped": the run is over and its `onDone`
   * (status "cancelled") has been called. "stopping": cancellation was
   * requested and `onDone` follows when the code reaches a check.
   */
  stop(): "stopped" | "stopping";
  /** Answer the pending input() request. */
  sendInput(text: string): void;
}
