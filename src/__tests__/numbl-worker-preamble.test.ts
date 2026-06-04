/**
 * The browser worker's "run" handler supports an optional `preamble`: setup
 * code (e.g. `mip load --install ...`) that runs before the visible script on
 * every run, sharing the same VFS. Its console output is captured (not
 * forwarded) and surfaced only if the preamble itself errors. On success a
 * `preamble_done` message is posted and the preamble's resulting variables /
 * search paths / VFS are available to the main run.
 */

import { beforeAll, describe, expect, it } from "vitest";

type WorkerMessage = Record<string, unknown>;

let onmessage: (e: { data: WorkerMessage }) => void;
let posted: WorkerMessage[] = [];

const enc = new TextEncoder();
const file = (path: string, source: string) => ({
  path,
  content: enc.encode(source),
});

/** Drive one "run" message and return the concatenated stdout + posted msgs. */
function run(message: WorkerMessage): {
  output: string;
  msgs: WorkerMessage[];
} {
  posted = [];
  onmessage({ data: { type: "run", ...message } });
  const output = posted
    .filter(m => m.type === "output")
    .map(m => m.text as string)
    .join("");
  return { output, msgs: posted };
}

beforeAll(async () => {
  (globalThis as unknown as { self: unknown }).self = {
    postMessage: (m: WorkerMessage) => posted.push(m),
    onmessage: null,
  };
  await import("../numbl-worker.js");
  onmessage = (
    globalThis as unknown as { self: { onmessage: typeof onmessage } }
  ).self.onmessage;
});

describe("worker run: preamble", () => {
  it("runs the preamble first, hides its output, and threads variables into the main run", () => {
    const main = "disp(x)\ndisp('MAIN_RAN')";
    const { output, msgs } = run({
      code: main,
      preamble: "disp('PREAMBLE_OUTPUT')\nx = 42;",
      mainFileName: "script.m",
      options: { displayResults: true, optimization: "0" },
      workspaceFiles: [{ name: "script.m", source: main }],
      vfsFiles: [file("script.m", main)],
      persistent: false,
    });

    expect(msgs.some(m => m.type === "error")).toBe(false);
    // Preamble ran OK and signalled the transition to the main phase.
    expect(msgs.some(m => m.type === "preamble_done")).toBe(true);
    expect(msgs.some(m => m.type === "done")).toBe(true);
    // The preamble's console output is hidden on success.
    expect(output).not.toContain("PREAMBLE_OUTPUT");
    // The preamble's variables are available to the main run.
    expect(output).toContain("42");
    expect(output).toContain("MAIN_RAN");
  });

  it("shares the VFS so files written by the preamble are visible to the main run", () => {
    const main = [
      "fid = fopen('made.txt', 'r');",
      "disp(fgetl(fid))",
      "fclose(fid);",
      "disp('READ_DONE')",
    ].join("\n");
    const { output, msgs } = run({
      code: main,
      preamble: [
        "fid = fopen('made.txt', 'w');",
        "fprintf(fid, 'from-preamble');",
        "fclose(fid);",
      ].join("\n"),
      mainFileName: "script.m",
      options: { displayResults: true, optimization: "0" },
      workspaceFiles: [{ name: "script.m", source: main }],
      vfsFiles: [file("script.m", main)],
      persistent: false,
    });

    expect(msgs.some(m => m.type === "error")).toBe(false);
    expect(output).toContain("from-preamble");
    expect(output).toContain("READ_DONE");
  });

  it("on a preamble error, reports preamble_error with its output and does not run the main script", () => {
    const main = "disp('MAIN_RAN')";
    const { output, msgs } = run({
      code: main,
      preamble: "disp('PRE_RAN')\nerror('boom')",
      mainFileName: "script.m",
      options: { displayResults: true, optimization: "0" },
      workspaceFiles: [{ name: "script.m", source: main }],
      vfsFiles: [file("script.m", main)],
      persistent: false,
    });

    const preErr = msgs.find(m => m.type === "preamble_error");
    expect(preErr).toBeDefined();
    // The captured preamble output is attached for display.
    expect(preErr?.text as string).toContain("PRE_RAN");
    expect(preErr?.message as string).toContain("boom");
    // The main script never ran, and no normal completion was reported.
    expect(output).not.toContain("MAIN_RAN");
    expect(msgs.some(m => m.type === "done")).toBe(false);
    expect(msgs.some(m => m.type === "preamble_done")).toBe(false);
  });

  it("behaves exactly as before when no preamble is given", () => {
    const main = "disp('NO_PREAMBLE')";
    const { output, msgs } = run({
      code: main,
      mainFileName: "script.m",
      options: { displayResults: true, optimization: "0" },
      workspaceFiles: [{ name: "script.m", source: main }],
      vfsFiles: [file("script.m", main)],
      persistent: false,
    });

    expect(msgs.some(m => m.type === "preamble_done")).toBe(false);
    expect(msgs.some(m => m.type === "preamble_error")).toBe(false);
    expect(msgs.some(m => m.type === "done")).toBe(true);
    expect(output).toContain("NO_PREAMBLE");
  });
});
