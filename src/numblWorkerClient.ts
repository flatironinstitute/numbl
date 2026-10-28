/**
 * Main-thread side of numbl-worker.ts: owns the Worker and the handling every
 * host shares (input() prompts over a SharedArrayBuffer, cooperative
 * cancellation, the uihtml iframe relay, plot instructions, VFS changes).
 * Hosts supply callbacks for the rest; React code uses `useNumblWorker`.
 */

import type { PlotInstruction } from "./numbl-core/plotting/types.js";
import {
  listenForUihtmlEvents,
  postToUihtmlFrames,
  type UihtmlPageEvent,
} from "./graphics/uihtmlRelay.js";
import type { OptLevel } from "./numbl-core/executors/plugins.js";
import type { WorkspaceFile } from "./numbl-core/workspace/index.js";
import type {
  DoneMessage,
  ErrorMessage,
  FromNumblWorker,
  PreambleErrorMessage,
  ResultMessage,
  RunMessage,
  ToNumblWorker,
  WorkerVfsFile,
} from "./numblWorkerProtocol.js";
import { createInputSAB, mainThreadRespond } from "./syncInputChannel.js";
import type { VfsChanges } from "./vfs/VirtualFileSystem.js";

export interface NumblWorkerHandlers {
  onOutput?(text: string): void;
  /** The running code called input(). When set, the host must answer with
   *  `respondToInput`; otherwise the client asks with window.prompt. */
  onInputRequest?(prompt: string): void;
  /** Instructions from drawnow and those returned with done/result. */
  onPlotInstructions?(instructions: PlotInstruction[]): void;
  onPreambleDone?(): void;
  onPreambleError?(msg: PreambleErrorMessage): void;
  /** A run finished (or was cancelled cooperatively). */
  onDone?(msg: DoneMessage): void;
  /** A run failed; called once per diagnostic. */
  onError?(msg: ErrorMessage): void;
  /** A REPL command finished. */
  onResult?(msg: ResultMessage): void;
  /** Non-empty file changes made by a run or REPL command. */
  onVfsChanges?(changes: VfsChanges): void;
  onWorkerError?(ev: ErrorEvent): void;
}

export interface NumblWorkerOptions {
  /** Answer input() with window.prompt (default true). */
  interactiveInput?: boolean;
  /** Relay events between uihtml iframes and the worker (default true). */
  uihtml?: boolean;
  /** WASM LAPACK accelerator URL sent to each new worker. */
  wasmBridgeUrl?: () => string | null;
}

export class NumblWorkerClient {
  private worker: Worker | null = null;
  private unlistenUihtml: (() => void) | null = null;
  private readonly inputSAB: SharedArrayBuffer | null;
  private readonly cancelSAB: SharedArrayBuffer | null;
  private optimization: OptLevel | undefined;

  constructor(
    private handlers: NumblWorkerHandlers = {},
    private readonly options: NumblWorkerOptions = {}
  ) {
    this.inputSAB =
      options.interactiveInput === false ? null : createInputSAB();
    this.cancelSAB =
      typeof SharedArrayBuffer !== "undefined"
        ? new SharedArrayBuffer(4)
        : null;
  }

  /** Replace the callbacks; later messages go to the new ones. */
  setHandlers(handlers: NumblWorkerHandlers): void {
    this.handlers = handlers;
  }

  /** Create the worker (no-op if running). */
  start(): void {
    if (this.worker) return;
    const worker = new Worker(new URL("./numbl-worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker = worker;
    worker.onmessage = (e: MessageEvent<FromNumblWorker>) =>
      this.handleMessage(e.data);
    worker.onerror = ev => this.handlers.onWorkerError?.(ev);

    if (this.options.wasmBridgeUrl) {
      this.post({
        type: "set_wasm_bridge",
        url: this.options.wasmBridgeUrl(),
      });
    }
    if (this.inputSAB) {
      this.post({ type: "set_input_sab", inputSAB: this.inputSAB });
    }
    if (this.optimization) {
      this.post({ type: "set_optimization", optimization: this.optimization });
    }
    if (this.options.uihtml !== false) {
      this.unlistenUihtml = listenForUihtmlEvents(ev =>
        this.sendUihtmlEvent(ev)
      );
    }
  }

  terminate(): void {
    this.unlistenUihtml?.();
    this.unlistenUihtml = null;
    this.worker?.terminate();
    this.worker = null;
  }

  /** Kill a running script by replacing the worker; worker state is lost. */
  restart(): void {
    this.terminate();
    this.start();
  }

  post(msg: ToNumblWorker): void {
    this.worker?.postMessage(msg);
  }

  run(msg: Omit<RunMessage, "type" | "cancelSAB">): void {
    this.resetCancel();
    this.post({
      type: "run",
      ...msg,
      cancelSAB: this.cancelSAB ?? undefined,
    });
  }

  execute(code: string): void {
    this.resetCancel();
    this.post({
      type: "execute",
      code,
      cancelSAB: this.cancelSAB ?? undefined,
    });
  }

  /** Ask the running code to stop at its next check; the worker keeps its
   *  state and reports the run as done. */
  cancel(): void {
    if (this.cancelSAB) Atomics.store(new Int32Array(this.cancelSAB), 0, 1);
  }

  updateWorkspace(
    workspaceFiles: WorkspaceFile[],
    vfsFiles?: WorkerVfsFile[]
  ): void {
    this.post({ type: "update_workspace", workspaceFiles, vfsFiles });
  }

  /** Default level for REPL commands and runs without one; kept across
   *  restarts. */
  setOptimization(optimization: OptLevel): void {
    this.optimization = optimization;
    this.post({ type: "set_optimization", optimization });
  }

  setWasmBridge(url: string | null): void {
    this.post({ type: "set_wasm_bridge", url });
  }

  /** Clear the persistent (REPL) workspace. */
  clear(): void {
    this.post({ type: "clear" });
  }

  /** Answer the pending input() request. */
  respondToInput(text: string): void {
    if (this.inputSAB) mainThreadRespond(this.inputSAB, text);
  }

  sendUihtmlEvent(ev: UihtmlPageEvent): void {
    this.post({ type: "html_event", ...ev });
  }

  private resetCancel(): void {
    if (this.cancelSAB) Atomics.store(new Int32Array(this.cancelSAB), 0, 0);
  }

  private handleMessage(msg: FromNumblWorker): void {
    const h = this.handlers;
    switch (msg.type) {
      case "request-input":
        if (h.onInputRequest) h.onInputRequest(msg.prompt ?? "");
        else this.respondToInput(prompt(msg.prompt ?? "") ?? "");
        return;
      case "output":
        h.onOutput?.(msg.text);
        return;
      case "drawnow":
        this.plot(h, msg.plotInstructions);
        return;
      case "html_source_event":
        postToUihtmlFrames(msg.compId, msg.name, msg.dataJson);
        return;
      case "preamble_done":
        h.onPreambleDone?.();
        return;
      case "preamble_error":
        h.onPreambleError?.(msg);
        return;
      // File changes are reported before the completion callback.
      case "done":
        this.plot(h, msg.plotInstructions);
        this.vfsChanges(h, msg.vfsChanges);
        h.onDone?.(msg);
        return;
      case "error":
        this.vfsChanges(h, msg.vfsChanges);
        h.onError?.(msg);
        return;
      case "result":
        if (msg.success) this.plot(h, msg.plotInstructions);
        this.vfsChanges(h, msg.vfsChanges);
        h.onResult?.(msg);
        return;
      case "cleared":
        return;
    }
  }

  private plot(h: NumblWorkerHandlers, instructions?: PlotInstruction[]) {
    if (instructions?.length) h.onPlotInstructions?.(instructions);
  }

  private vfsChanges(h: NumblWorkerHandlers, changes?: VfsChanges) {
    if (!changes) return;
    const { created, modified, deleted } = changes;
    if (created.length + modified.length + deleted.length === 0) return;
    h.onVfsChanges?.(changes);
  }
}
