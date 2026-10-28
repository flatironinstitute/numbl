import { describe, it, expect } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  STREAM_PROTOCOL_VERSION,
  decodeStreamMessage,
  encodeStreamMessage,
  type StreamMessage,
} from "../streamProtocol.js";

describe("StreamMessage encode/decode", () => {
  const messages: StreamMessage[] = [
    { type: "output", text: "x = 3\n" },
    {
      type: "drawnow",
      plotInstructions: [{ type: "set_hold", value: true }],
    },
    { type: "input_request", prompt: "name? " },
    {
      type: "error",
      message: "boom",
      errorType: "runtime",
      identifier: "my:id",
      file: "f.m",
      line: 2,
      snippet: ">   2 | error('my:id', 'boom')",
      callStack: [{ name: "f", callerFile: "main.m", callerLine: 1 }],
    },
    {
      type: "error",
      message: "t",
      errorType: "runtime",
      file: null,
      line: null,
      snippet: null,
      timedOut: true,
    },
    { type: "done", success: true, generatedJS: "// js" },
    { type: "done", success: false },
    { type: "close", success: false },
  ];

  it("round-trips every message type with a version field", () => {
    for (const msg of messages) {
      const line = encodeStreamMessage(msg);
      expect(line).not.toContain("\n");
      expect(JSON.parse(line).v).toBe(STREAM_PROTOCOL_VERSION);
      expect(decodeStreamMessage(line)).toEqual(msg);
    }
  });

  it("accepts the unversioned format", () => {
    expect(decodeStreamMessage('{"type":"output","text":"hi"}')).toEqual({
      type: "output",
      text: "hi",
    });
    // Before v1, `done` was sent only after a successful run.
    expect(decodeStreamMessage('{"type":"done"}')).toEqual({
      type: "done",
      success: true,
    });
    expect(
      decodeStreamMessage(
        '{"type":"error","message":"m","errorType":"syntax","file":"a.m","line":1,"snippet":null}'
      )
    ).toEqual({
      type: "error",
      message: "m",
      errorType: "syntax",
      file: "a.m",
      line: 1,
      snippet: null,
    });
  });

  it("rejects malformed, unknown and newer messages", () => {
    expect(decodeStreamMessage("not json")).toBeNull();
    expect(decodeStreamMessage("42")).toBeNull();
    expect(decodeStreamMessage('{"type":"output"}')).toBeNull();
    expect(decodeStreamMessage('{"type":"error"}')).toBeNull();
    expect(decodeStreamMessage('{"type":"close"}')).toBeNull();
    expect(decodeStreamMessage('{"type":"mystery"}')).toBeNull();
    expect(
      decodeStreamMessage('{"v":2,"type":"output","text":"hi"}')
    ).toBeNull();
  });
});

describe("numbl --stream", () => {
  const root = resolve(import.meta.dirname, "../..");

  function runStream(source: string) {
    const dir = mkdtempSync(join(tmpdir(), "numbl-stream-"));
    try {
      writeFileSync(join(dir, "main.m"), source);
      writeFileSync(
        join(dir, "fail.m"),
        "function fail()\n  error('my:id', 'boom %d', 7);\nend\n"
      );
      const proc = spawnSync(
        process.execPath,
        [
          "--import",
          "tsx",
          join(root, "src/cli.ts"),
          "run",
          join(dir, "main.m"),
          "--stream",
          "--no-native",
        ],
        { cwd: root, encoding: "utf8", input: "Ada\n" }
      );
      const messages = proc.stdout
        .split("\n")
        .filter(line => line.trim())
        .map(line => {
          const msg = decodeStreamMessage(line);
          expect(msg, `not a stream message: ${line}`).not.toBeNull();
          return msg!;
        });
      return { status: proc.status, messages };
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }

  it("streams output, input requests and plots, then done, exit 0", () => {
    const { status, messages } = runStream(
      "disp(42)\nname = input('Name? ', 's');\nfprintf('hi %s\\n', name);\nplot(1:3)\n"
    );
    expect(status).toBe(0);
    expect(messages.map(m => m.type)).toEqual([
      "output",
      "input_request",
      "output",
      "drawnow",
      "done",
    ]);
    expect(messages[0]).toEqual({ type: "output", text: "42\n" });
    expect(messages[1]).toEqual({ type: "input_request", prompt: "Name? " });
    expect(messages[2]).toEqual({ type: "output", text: "hi Ada\n" });
    expect(messages[4]).toMatchObject({ type: "done", success: true });
  }, 60000);

  it("reports a runtime error with identifier and stack, then fails", () => {
    const { status, messages } = runStream("disp(1)\nfail()\n");
    expect(status).toBe(1);
    expect(messages.map(m => m.type)).toEqual(["output", "error", "done"]);
    const err = messages[1];
    if (err.type !== "error") throw new Error("expected error");
    expect(err.message).toBe("boom 7");
    expect(err.errorType).toBe("runtime");
    expect(err.identifier).toBe("my:id");
    expect(err.file).toMatch(/fail\.m$/);
    expect(err.line).toBe(2);
    expect(err.callStack?.map(f => f.name)).toEqual(["fail"]);
    expect(messages[2]).toEqual({ type: "done", success: false });
  }, 60000);

  it("reports a syntax error and fails", () => {
    const { status, messages } = runStream("x = (1 + \n");
    expect(status).toBe(1);
    expect(messages.map(m => m.type)).toEqual(["error", "done"]);
    expect(messages[0]).toMatchObject({ type: "error", errorType: "syntax" });
  }, 60000);
});
