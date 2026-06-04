/**
 * The browser worker's "run" handler should treat the directory of the script
 * being run as the current working directory — mirroring the CLI `run` command
 * (which chdir's into dirname(filepath)). This makes sibling functions and
 * relative file I/O resolve against the script's folder when a driver script
 * lives in a subdirectory.
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
  // The worker registers `self.onmessage` at import time and posts results
  // via `self.postMessage`, so a mock `self` must exist before importing.
  (globalThis as unknown as { self: unknown }).self = {
    postMessage: (m: WorkerMessage) => posted.push(m),
    onmessage: null,
  };
  await import("../numbl-worker.js");
  onmessage = (
    globalThis as unknown as { self: { onmessage: typeof onmessage } }
  ).self.onmessage;
});

describe("worker run: cwd is the script's directory", () => {
  it("resolves a sibling function and relative file for a subdirectory script", () => {
    const driver = [
      "disp(pwd)",
      "y = helper(21);",
      "disp(y)",
      "fid = fopen('data.txt', 'r');",
      "line = fgetl(fid);",
      "fclose(fid);",
      "disp(line)",
      "disp('DONE')",
    ].join("\n");
    const helper = "function y = helper(n)\ny = n * 2;\nend\n";

    const { output, msgs } = run({
      code: driver,
      mainFileName: "folder1/driver.m",
      options: { displayResults: true, optimization: "0" },
      workspaceFiles: [
        { name: "folder1/driver.m", source: driver },
        { name: "folder1/helper.m", source: helper },
      ],
      vfsFiles: [
        file("folder1/driver.m", driver),
        file("folder1/helper.m", helper),
        file("folder1/data.txt", "hello-sibling\n"),
      ],
      persistent: false,
    });

    expect(msgs.some(m => m.type === "error")).toBe(false);
    // cwd reported by pwd is the script's directory, not the project root.
    expect(output).toContain("/project/folder1");
    expect(output).not.toContain("/project/folder1/folder1");
    // Sibling function resolved (cwd became a search path).
    expect(output).toContain("42");
    // Relative file I/O resolved against the script's directory.
    expect(output).toContain("hello-sibling");
    expect(output).toContain("DONE");
  });

  it("mfilename('fullpath') resolves to the script's absolute VFS path", () => {
    // Regression: a bare mainFileName left fileparts(mfilename('fullpath'))
    // empty, so e.g. fullfile(here,'app','dist','index.html') resolved to
    // /app/... instead of /project/app/...
    const driver = [
      "fp = mfilename('fullpath');",
      "disp(fp)",
      "disp(fileparts(fp))",
      "disp('MFDONE')",
    ].join("\n");
    const { output, msgs } = run({
      code: driver,
      mainFileName: "refine_demo.m",
      options: { displayResults: true, optimization: "0" },
      workspaceFiles: [{ name: "refine_demo.m", source: driver }],
      vfsFiles: [file("refine_demo.m", driver)],
      persistent: false,
    });
    expect(msgs.some(m => m.type === "error")).toBe(false);
    expect(output).toContain("/project/refine_demo");
    expect(output).toContain("/project\n");
    expect(output).toContain("MFDONE");
  });

  it("leaves the cwd at the project root for a root-level script", () => {
    const driver = "disp(pwd)\ndisp('ROOTDONE')";
    const { output, msgs } = run({
      code: driver,
      mainFileName: "main.m",
      options: { displayResults: true, optimization: "0" },
      workspaceFiles: [{ name: "main.m", source: driver }],
      vfsFiles: [file("main.m", driver)],
      persistent: false,
    });

    expect(msgs.some(m => m.type === "error")).toBe(false);
    expect(output).toContain("/project\n");
    expect(output).toContain("ROOTDONE");
  });
});
