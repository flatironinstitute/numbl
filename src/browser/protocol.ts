/** Messages between the NumblSession host and its worker. */

import type { PlotInstruction } from "../numbl-core/plotting/types.js";
import type { ChannelMessage } from "../numbl-core/runtime/blockingChannel.js";

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
   * Cancel buffer (`createCancelBuffer`; Int32[0] != 0 cancels the running
   * code). The worker passes it to every executeCode call, so the host can
   * interrupt a run cooperatively, and `input()` blocks on the channel bound
   * to it (see inputChannel.ts). Only present when the page is cross-origin
   * isolated (SharedArrayBuffer available).
   */
  cancelSAB?: SharedArrayBuffer;
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
   * workspace survives the abort, with what the run did before it stopped.
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
  // Blocking-channel traffic: input() requests (see inputChannel.ts).
  | ChannelMessage
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
