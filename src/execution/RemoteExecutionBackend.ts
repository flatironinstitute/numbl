/**
 * Runs scripts on a numbl execution server (`numbl serve`), which runs
 * `numbl run --stream` on a copy of the project and relays its stream
 * messages (streamProtocol.ts) as server-sent events.
 *
 * Gaps relative to the worker: input() cannot be answered (the server's CLI
 * reads EOF, so it returns empty), uihtml figures are not interactive, and
 * every run starts from a fresh process (no REPL or persistent workspace).
 */

import { decodeStreamMessage } from "../streamProtocol.js";
import type {
  ExecutionBackend,
  ExecutionDiagnostic,
  ExecutionEvents,
  ExecutionRequest,
  ExecutionResult,
} from "./ExecutionBackend.js";

export interface RemoteServerConfig {
  serviceUrl: string;
  passkey: string;
}

/** Body of POST /execute. */
interface RemoteExecutionRequest {
  files: { name: string; content: string }[];
  mainScript: string;
  optimization?: string;
}

const failure = (message: string): ExecutionDiagnostic => ({
  message,
  errorType: "unknown",
  file: null,
  line: null,
  snippet: null,
});

/** Text project files to upload: no system files (the server has its own
 *  numbl) and no binary files. The script is sent with `request.code`. */
function projectFiles(request: ExecutionRequest) {
  const decoder = new TextDecoder("utf-8");
  const files = request.vfsFiles
    .filter(
      f =>
        !f.path.startsWith("/") &&
        f.path !== "system" &&
        !f.path.startsWith("system/") &&
        f.path !== request.mainFileName &&
        !f.content.includes(0)
    )
    .map(f => ({ name: f.path, content: decoder.decode(f.content) }));
  files.push({ name: request.mainFileName, content: request.code });
  return files;
}

export class RemoteExecutionBackend implements ExecutionBackend {
  readonly kind = "remote";
  readonly capabilities = { input: false, uihtml: false, repl: false };
  private current: { abort: AbortController; events: ExecutionEvents } | null =
    null;

  constructor(
    private config: RemoteServerConfig,
    private readonly fetchFn: typeof fetch = (...args) => fetch(...args)
  ) {}

  /** Server for later runs; a run in progress keeps its server. */
  configure(config: RemoteServerConfig): void {
    this.config = config;
  }

  run(request: ExecutionRequest, events: ExecutionEvents): void {
    const current = { abort: new AbortController(), events };
    this.current = current;
    // Drop anything that arrives after stop().
    const live = () => this.current === current;
    const guarded: ExecutionEvents = {
      onOutput: t => {
        if (live()) events.onOutput(t);
      },
      onPlotInstructions: i => {
        if (live()) events.onPlotInstructions(i);
      },
      onError: d => {
        if (live()) events.onError(d);
      },
      onDone: () => {},
    };
    void this.stream(request, guarded, current.abort.signal).then(result => {
      if (!live()) return;
      this.current = null;
      events.onDone(result);
    });
  }

  stop(): "stopped" {
    const current = this.current;
    this.current = null;
    if (current) {
      current.abort.abort();
      current.events.onDone({ status: "cancelled" });
    }
    return "stopped";
  }

  /** input() is not supported remotely; see the module comment. */
  sendInput(): void {}

  private async stream(
    request: ExecutionRequest,
    events: ExecutionEvents,
    signal: AbortSignal
  ): Promise<ExecutionResult> {
    const { serviceUrl, passkey } = this.config;
    const body: RemoteExecutionRequest = {
      files: projectFiles(request),
      mainScript: request.mainFileName,
      optimization: request.optimization,
    };
    const fail = (message: string): ExecutionResult => {
      events.onError(failure(message));
      return { status: "error" };
    };

    let response: Response;
    try {
      response = await this.fetchFn(`${serviceUrl}/execute`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(passkey ? { Authorization: `Bearer ${passkey}` } : {}),
        },
        body: JSON.stringify(body),
        signal,
      });
    } catch (error) {
      if (signal.aborted) return { status: "cancelled" };
      return fail(
        `Connection failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      return fail(data.error || `Server error: ${response.status}`);
    }
    const reader = response.body?.getReader();
    if (!reader) return fail("No response body");

    let failed = false;
    let finished = false;
    let generatedJS: string | undefined;
    const handle = (payload: string) => {
      const msg = decodeStreamMessage(payload);
      if (!msg) return;
      switch (msg.type) {
        case "output":
          events.onOutput(msg.text);
          return;
        case "drawnow":
          events.onPlotInstructions(msg.plotInstructions);
          return;
        case "input_request":
          events.onOutput(
            `${msg.prompt}\n[input() is not available with remote execution; it returned an empty value]\n`
          );
          return;
        case "error": {
          failed = true;
          events.onError({
            message: msg.message,
            errorType: msg.errorType,
            identifier: msg.identifier,
            file: msg.file,
            line: msg.line,
            snippet: msg.snippet,
            callStack: msg.callStack,
          });
          return;
        }
        case "done":
          finished = true;
          generatedJS = msg.generatedJS;
          if (!msg.success) failed = true;
          return;
        case "close":
          finished = true;
          if (!msg.success) failed = true;
          return;
      }
    };

    // Server-sent events: "data: <json>\n\n".
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";
        for (const chunk of chunks) {
          const line = chunk.trim();
          if (line.startsWith("data: ")) handle(line.slice(6));
        }
      }
    } catch (error) {
      if (signal.aborted) return { status: "cancelled" };
      return fail(
        `Connection lost: ${error instanceof Error ? error.message : String(error)}`
      );
    }
    if (!finished) {
      return fail("The execution server closed the connection early");
    }
    return { status: failed ? "error" : "success", generatedJS };
  }
}
