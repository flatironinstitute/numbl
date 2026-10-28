/**
 * Messages between the web app (IDE, embeds, figure page, gallery,
 * benchmarks) and its worker, `numbl-worker.ts`. The embedding API's worker
 * has its own protocol in `browser/protocol.ts`.
 */

import type { PlotInstruction } from "./numbl-core/plotting/types.js";
import type { UihtmlPageEvent } from "./graphics/uihtmlRelay.js";
import type { OptLevel } from "./numbl-core/executors/plugins.js";
import type { CallFrame } from "./numbl-core/runtime/error.js";
import type { ChannelMessage } from "./numbl-core/runtime/blockingChannel.js";
import type { WorkspaceFile } from "./numbl-core/workspace/index.js";
import type { VfsChanges } from "./vfs/VirtualFileSystem.js";

/** A file written into the worker's virtual file system. */
export interface WorkerVfsFile {
  path: string;
  content: Uint8Array;
}

export interface RunOptions {
  displayResults?: boolean;
  maxIterations?: number;
  optimization?: OptLevel;
}

export interface RunMessage {
  type: "run";
  code: string;
  /** Setup code run first in the same VFS; its output is shown only if it
   *  fails (see `preamble_done` / `preamble_error`). */
  preamble?: string;
  options?: RunOptions;
  workspaceFiles: WorkspaceFile[];
  /** Project-relative path of the script; defaults to "script.m". */
  mainFileName?: string;
  searchPaths?: string[];
  vfsFiles?: WorkerVfsFile[];
  /** Share variables, hold state and VFS with the REPL (`execute`). */
  persistent?: boolean;
  /** Cancel buffer (`createCancelBuffer`): Int32[0] != 0 cancels the run
   *  cooperatively; input() blocks on the channel bound to it. */
  cancelSAB?: SharedArrayBuffer;
}

export type ToNumblWorker =
  | RunMessage
  | { type: "execute"; code: string; cancelSAB?: SharedArrayBuffer }
  | { type: "set_optimization"; optimization: OptLevel }
  /** null/"" uninstalls the WASM LAPACK accelerator. */
  | { type: "set_wasm_bridge"; url: string | null }
  | {
      type: "update_workspace";
      workspaceFiles: WorkspaceFile[];
      vfsFiles?: WorkerVfsFile[];
      searchPaths?: string[];
    }
  /** Whether the host answers input() requests (channel "input"). */
  | { type: "set_input"; enabled: boolean }
  | { type: "clear" }
  | ({ type: "html_event" } & UihtmlPageEvent);

/** Structured error, formatted on the main thread with `formatDiagnostic`. */
export interface WorkerDiagnostic {
  message: string;
  errorType: "syntax" | "runtime" | "unknown";
  /** MATLAB error identifier, if any. */
  identifier?: string;
  file: string | null;
  line: number | null;
  snippet: string | null;
  callStack: CallFrame[] | null;
}

export interface FileRep {
  name: string;
  ast: unknown;
  irProgram: unknown;
}

/** Parsed sources of a run, shown in the IDE's AST view. */
export interface WorkspaceRep {
  mainFile: FileRep;
  workspaceFiles: FileRep[];
  fileSources: Map<string, string>;
}

export interface DoneMessage {
  type: "done";
  generatedJS: string | undefined;
  outputCount: number;
  /** null when the run was cancelled. */
  workspaceRep: WorkspaceRep | null;
  plotInstructions: PlotInstruction[];
  dispatchUnknownCounts?: Record<string, number>;
  vfsChanges?: VfsChanges;
}

/** One per diagnostic, so a failed run can post several; `last` marks the
 *  final one, which alone carries the run's VFS changes. */
export interface ErrorMessage extends WorkerDiagnostic {
  type: "error";
  last: boolean;
  generatedJS: string | undefined;
  workspaceRep: WorkspaceRep;
  vfsChanges?: VfsChanges;
}

export interface PreambleErrorMessage extends WorkerDiagnostic {
  type: "preamble_error";
  /** The preamble's captured console output. */
  text: string;
}

/** Completion of an `execute` (REPL) command. */
export type ResultMessage =
  | {
      type: "result";
      success: true;
      output: string;
      plotInstructions: PlotInstruction[];
      vfsChanges: VfsChanges;
    }
  | { type: "result"; success: false; error: string; vfsChanges: VfsChanges };

export type FromNumblWorker =
  | { type: "output"; text: string }
  // Blocking-channel traffic: input() requests (see inputChannel.ts).
  | ChannelMessage
  | { type: "drawnow"; plotInstructions: PlotInstruction[] }
  // sendEventToHTMLSource: relay to the uihtml iframe with this compId.
  | {
      type: "html_source_event";
      compId: string;
      name: string;
      dataJson: string;
    }
  | { type: "preamble_done" }
  | PreambleErrorMessage
  | DoneMessage
  | ErrorMessage
  | ResultMessage
  | { type: "cleared" };
