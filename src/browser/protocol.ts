/** Messages between the NumblSession host and its worker. */

import type { PlotInstruction } from "../graphics/types.js";

export interface BootFile {
  path: string;
  content: string | Uint8Array;
}

export interface BootMessage {
  type: "boot";
  files: BootFile[];
  /** Script to run at boot. When omitted, the session boots idle and code is
   *  run incrementally via `execute`. */
  mainFile?: string;
  mip: boolean;
  persistSystem: boolean;
  systemInactivityMs: number;
  optimization: "0" | "1";
  maxIterations: number;
  displayResults: boolean;
  /**
   * Shared cancellation flag (Int32[0] != 0 ⇒ cancel the running code). The
   * worker keeps this reference and passes it to every executeCode call, so
   * the host can interrupt a runaway run cooperatively. Only present when the
   * page is cross-origin isolated (SharedArrayBuffer available).
   */
  cancelSAB?: SharedArrayBuffer;
  /**
   * Shared channel for synchronous `input()` (stdin). The worker blocks on it
   * (Atomics.wait) after asking the host for a line via a `request-input`
   * message; the host writes the reply back through it (see syncInputChannel).
   * Only present when the page is cross-origin isolated.
   */
  inputSAB?: SharedArrayBuffer;
}

export type ToWorker =
  | BootMessage
  | { type: "writeFile"; path: string; content: string | Uint8Array }
  | { type: "readFile"; id: number; path: string }
  | { type: "execute"; id: number; code: string }
  | {
      type: "dispatch";
      id: number;
      compId: string;
      name: string;
      data: unknown;
    };

export interface UihtmlComponent {
  compId: string;
  /** The component's HTML markup (render with buildUihtmlSrcDoc). */
  html: string;
  /** The component's Data, JSON-encoded. */
  dataJson: string;
}

export interface ExecuteResult {
  ok: boolean;
  /** Concatenated console output (also streamed live via onOutput). */
  output: string;
  /** Every plot instruction the execution produced, uihtml included. */
  plotInstructions: PlotInstruction[];
  /** Formatted error message when `ok` is false. */
  error?: string;
  /**
   * True when the run ended because it was interrupted via
   * `NumblSession.interrupt()` (rather than finishing or erroring). The
   * workspace is left at its pre-run state, so variables survive the abort.
   */
  aborted?: boolean;
}

export type FromWorker =
  | { type: "progress"; message: string }
  | { type: "output"; text: string }
  | { type: "uihtml"; compId: string; html: string; dataJson: string }
  | {
      type: "ready";
      hasUihtmlSession: boolean;
      components: UihtmlComponent[];
    }
  | { type: "bootError"; message: string }
  // Emitted by workerOnInput when running code calls `input()`: the worker has
  // already blocked on inputSAB and is waiting for the host to write the reply.
  | { type: "request-input"; prompt: string }
  | { type: "htmlSourceEvent"; compId: string; name: string; dataJson: string }
  | { type: "executeResult"; id: number; result: ExecuteResult }
  | { type: "dispatchResult"; id: number; ok: boolean; message?: string }
  | {
      type: "readFileResult";
      id: number;
      ok: boolean;
      content?: Uint8Array;
      message?: string;
    };
