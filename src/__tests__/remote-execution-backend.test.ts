import { describe, it, expect } from "vitest";
import type {
  ExecutionDiagnostic,
  ExecutionEvents,
  ExecutionRequest,
  ExecutionResult,
} from "../execution/ExecutionBackend.js";
import { RemoteExecutionBackend } from "../execution/RemoteExecutionBackend.js";
import { encodeStreamMessage, type StreamMessage } from "../streamProtocol.js";

const enc = new TextEncoder();

/** A fetch that answers POST /execute with these server-sent events, split
 *  into small chunks to exercise buffering. */
function sseFetch(messages: StreamMessage[], captured: { body?: unknown }) {
  return (async (_url: RequestInfo | URL, init?: RequestInit) => {
    captured.body = JSON.parse(String(init?.body));
    const text = messages
      .map(m => `data: ${encodeStreamMessage(m)}\n\n`)
      .join("");
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (let i = 0; i < text.length; i += 7) {
          controller.enqueue(enc.encode(text.slice(i, i + 7)));
        }
        controller.close();
      },
    });
    return new Response(stream, { status: 200 });
  }) as typeof fetch;
}

function collect() {
  const log = {
    output: "",
    errors: [] as ExecutionDiagnostic[],
    plots: 0,
    done: [] as ExecutionResult[],
  };
  let resolveDone: () => void;
  const finished = new Promise<void>(r => (resolveDone = r));
  const events: ExecutionEvents = {
    onOutput: t => (log.output += t),
    onPlotInstructions: i => (log.plots += i.length),
    onError: d => log.errors.push(d),
    onDone: r => {
      log.done.push(r);
      resolveDone();
    },
  };
  return { log, events, finished };
}

const request: ExecutionRequest = {
  code: "disp(1)",
  mainFileName: "main.m",
  vfsFiles: [
    { path: "main.m", content: enc.encode("old") },
    { path: "lib/f.m", content: enc.encode("function f\nend") },
    { path: "data.bin", content: new Uint8Array([1, 0, 2]) },
    { path: "/system/x.m", content: enc.encode("x") },
  ],
  workspaceFiles: [],
  optimization: "1",
  persistent: false,
};

const config = { serviceUrl: "http://server", passkey: "k" };

describe("RemoteExecutionBackend", () => {
  it("relays output and plots and reports success", async () => {
    const captured: { body?: unknown } = {};
    const backend = new RemoteExecutionBackend(
      config,
      sseFetch(
        [
          { type: "output", text: "1\n" },
          { type: "drawnow", plotInstructions: [{ type: "clf" }] },
          { type: "done", success: true, generatedJS: "// js" },
          { type: "close", success: true },
        ],
        captured
      )
    );
    const { log, events, finished } = collect();
    backend.run(request, events);
    await finished;
    expect(log.output).toBe("1\n");
    expect(log.plots).toBe(1);
    expect(log.errors).toEqual([]);
    expect(log.done).toEqual([{ status: "success", generatedJS: "// js" }]);
    // Text project files only, with the script's current source.
    expect(captured.body).toEqual({
      files: [
        { name: "lib/f.m", content: "function f\nend" },
        { name: "main.m", content: "disp(1)" },
      ],
      mainScript: "main.m",
      optimization: "1",
    });
  });

  it("reports errors as diagnostics and the run as failed", async () => {
    const backend = new RemoteExecutionBackend(
      config,
      sseFetch(
        [
          {
            type: "error",
            message: "boom",
            errorType: "runtime",
            identifier: "my:id",
            file: "main.m",
            line: 1,
            snippet: null,
          },
          { type: "done", success: false },
          { type: "close", success: false },
        ],
        {}
      )
    );
    const { log, events, finished } = collect();
    backend.run(request, events);
    await finished;
    expect(log.errors).toEqual([
      {
        message: "boom",
        errorType: "runtime",
        identifier: "my:id",
        file: "main.m",
        line: 1,
        snippet: null,
        callStack: undefined,
      },
    ]);
    expect(log.done.map(d => d.status)).toEqual(["error"]);
  });

  it("treats an error from an older CLI (exit 0, no done) as a failure", async () => {
    const backend = new RemoteExecutionBackend(
      config,
      sseFetch(
        [
          {
            type: "error",
            message: "boom",
            errorType: "runtime",
            file: null,
            line: null,
            snippet: null,
          },
          { type: "close", success: true },
        ],
        {}
      )
    );
    const { log, events, finished } = collect();
    backend.run(request, events);
    await finished;
    expect(log.done.map(d => d.status)).toEqual(["error"]);
  });

  it("reports a connection failure", async () => {
    const backend = new RemoteExecutionBackend(config, (async () => {
      throw new Error("refused");
    }) as typeof fetch);
    const { log, events, finished } = collect();
    backend.run(request, events);
    await finished;
    expect(log.errors.map(e => e.message)).toEqual([
      "Connection failed: refused",
    ]);
    expect(log.done.map(d => d.status)).toEqual(["error"]);
  });

  it("stop ends the run at once and drops later events", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>(r => (release = r));
    const backend = new RemoteExecutionBackend(config, (async () => {
      await gate;
      return new Response(
        `data: ${encodeStreamMessage({ type: "output", text: "late" })}\n\n`
      );
    }) as typeof fetch);
    const { log, events } = collect();
    backend.run(request, events);
    expect(backend.stop()).toBe("stopped");
    expect(log.done).toEqual([{ status: "cancelled" }]);
    release();
    await new Promise(r => setTimeout(r, 10));
    expect(log.output).toBe("");
    expect(log.done).toHaveLength(1);
  });
});
