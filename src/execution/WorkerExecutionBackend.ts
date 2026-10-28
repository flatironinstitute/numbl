/**
 * Runs scripts in the browser worker (numbl-worker.ts) through a
 * NumblWorkerClient. Besides `ExecutionBackend`, it offers the REPL, which
 * shares the worker's state with persistent runs.
 */

import type { PlotInstruction } from "../numbl-core/plotting/types.js";
import type {
  NumblWorkerClient,
  NumblWorkerHandlers,
} from "../numblWorkerClient.js";
import type { VfsChanges } from "../vfs/VirtualFileSystem.js";
import type {
  ExecutionBackend,
  ExecutionEvents,
  ExecutionRequest,
  ExecutionResult,
} from "./ExecutionBackend.js";

export interface ReplEvents {
  onOutput(text: string): void;
  onPlotInstructions(instructions: PlotInstruction[]): void;
  onInputRequest?(prompt: string): void;
  onVfsChanges?(changes: VfsChanges): void;
  /** The command finished; called exactly once, last. */
  onResult(result: { success: true } | { success: false; error: string }): void;
}

type Active =
  | { kind: "run"; events: ExecutionEvents; persistent: boolean }
  | { kind: "repl"; events: ReplEvents };

export class WorkerExecutionBackend implements ExecutionBackend {
  readonly kind = "worker";
  readonly capabilities = { input: true, uihtml: true, repl: true };
  private active: Active | null = null;

  /** Takes over the client's handlers. */
  constructor(readonly client: NumblWorkerClient) {
    client.setHandlers(this.handlers());
  }

  run(request: ExecutionRequest, events: ExecutionEvents): void {
    this.active = { kind: "run", events, persistent: request.persistent };
    // The REPL sees the same files as the run.
    this.client.updateWorkspace(request.workspaceFiles, request.vfsFiles);
    this.client.run({
      code: request.code,
      workspaceFiles: request.workspaceFiles,
      mainFileName: request.mainFileName,
      options: {
        displayResults: true,
        maxIterations: 10000000,
        optimization: request.optimization,
      },
      vfsFiles: request.vfsFiles,
      persistent: request.persistent,
    });
  }

  /** A persistent run is cancelled cooperatively so the worker keeps its
   *  state; anything else is stopped by replacing the worker, which loses
   *  its state (including the workspace files). */
  stop(): "stopped" | "stopping" {
    const active = this.active;
    if (active?.kind === "run" && active.persistent) {
      this.client.cancel();
      return "stopping";
    }
    this.active = null;
    this.client.restart();
    if (active?.kind === "run") active.events.onDone({ status: "cancelled" });
    else active?.events.onResult({ success: false, error: "Stopped" });
    return "stopped";
  }

  sendInput(text: string): void {
    this.client.respondToInput(text);
  }

  /** Run a REPL command in the persistent workspace. */
  execute(command: string, events: ReplEvents): void {
    this.active = { kind: "repl", events };
    this.client.execute(command);
  }

  /** Clear the persistent (REPL) workspace. */
  clear(): void {
    this.client.clear();
  }

  private finishRun(result: ExecutionResult): void {
    const active = this.active;
    if (active?.kind !== "run") return;
    this.active = null;
    active.events.onDone(result);
  }

  private handlers(): NumblWorkerHandlers {
    const runEvents = () =>
      this.active?.kind === "run" ? this.active.events : undefined;
    return {
      onOutput: text => this.active?.events.onOutput(text),
      onPlotInstructions: instructions =>
        this.active?.events.onPlotInstructions(instructions),
      onInputRequest: prompt => {
        const ask = this.active?.events.onInputRequest;
        if (ask) ask(prompt);
        else this.sendInput(window.prompt(prompt) ?? "");
      },
      onVfsChanges: changes => this.active?.events.onVfsChanges?.(changes),
      onDone: msg =>
        this.finishRun({
          // The worker reports a cancelled run as done without an AST.
          status: msg.workspaceRep === null ? "cancelled" : "success",
          generatedJS: msg.generatedJS,
          workspaceRep: msg.workspaceRep,
          dispatchUnknownCounts: msg.dispatchUnknownCounts,
        }),
      onError: msg => {
        runEvents()?.onError({
          message: msg.message,
          errorType: msg.errorType,
          identifier: msg.identifier,
          file: msg.file,
          line: msg.line,
          snippet: msg.snippet,
          callStack: msg.callStack,
        });
        if (msg.last) {
          this.finishRun({
            status: "error",
            generatedJS: msg.generatedJS,
            workspaceRep: msg.workspaceRep,
          });
        }
      },
      onResult: msg => {
        const active = this.active;
        if (active?.kind !== "repl") return;
        this.active = null;
        active.events.onResult(
          msg.success ? { success: true } : { success: false, error: msg.error }
        );
      },
      onWorkerError: ev => {
        const active = this.active;
        this.active = null;
        const error = `Worker error: ${ev.message}`;
        if (active?.kind === "run") {
          active.events.onError({
            message: error,
            errorType: "unknown",
            file: null,
            line: null,
            snippet: null,
          });
          active.events.onDone({ status: "error" });
        } else if (active?.kind === "repl") {
          active.events.onResult({ success: false, error });
        } else {
          console.error(error);
        }
      },
    };
  }
}
