import { describe, it, expect } from "vitest";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { ExecSession } from "../numbl-core/execSession.js";
import { executeCode } from "../numbl-core/executeCode.js";
import type { ExecOptions } from "../numbl-core/executeCode.js";
import type { RuntimeValue } from "../numbl-core/runtime/index.js";

const opts: ExecOptions = { implicitCwdPath: null };

function run(s: ExecSession, code: string, o: ExecOptions = {}) {
  return s.execute(code, { ...opts, ...o });
}

/** The value of `name` after running `code` in the session. */
function value(s: ExecSession, code: string, name: string): RuntimeValue {
  return run(s, code).variableValues[name];
}

const data = (v: RuntimeValue) =>
  Array.from((v as { data: Float64Array }).data);
const rc = (v: RuntimeValue) => (v as { _rc: number })._rc;

describe("ExecSession", () => {
  it("carries variables, hold state and figures across calls", () => {
    const s = new ExecSession();
    s.execute("x = 3; figure; hold on;");
    const r = s.execute("y = x + 1; h = figure;");
    expect(r.variableValues.y).toBe(4);
    expect(r.variableValues.h).toBe(2);
    expect(s.holdState).toBe(true);
    expect(s.figureHandles).toEqual({ current: 2, max: 2 });
  });

  it("keeps what a failing call did before the error, as MATLAB does", () => {
    const s = new ExecSession();
    run(s, "x = 1;");
    expect(() => run(s, "x = 2; error('boom');")).toThrow();
    expect(value(s, "y = x;", "y")).toBe(2);
  });

  it("clearWorkspace forgets variables and globals", () => {
    const s = new ExecSession();
    run(s, "x = 1; global G; G = 5;");
    s.clearWorkspace();
    expect(s.variableValues).toEqual({});
    expect(() => run(s, "y = x;")).toThrow();
    expect(value(s, "global G; e = isempty(G);", "e")).toBe(true);
  });

  it("keeps globals across calls until clear global / clear all", () => {
    const s = new ExecSession();
    run(s, "global G; G = 5;");
    expect(data(value(s, "global G; n = size(G);", "n"))).toEqual([1, 1]);
    run(s, "clear");
    expect(value(s, "e = exist('G', 'var');", "e")).toBe(0);
    expect(value(s, "global G; v = G;", "v")).toBe(5);
    run(s, "clear global");
    expect(value(s, "global G; e = isempty(G);", "e")).toBe(true);
    run(s, "global G; G = 6;");
    run(s, "clear all");
    expect(value(s, "global G; e = isempty(G);", "e")).toBe(true);
  });

  it("keeps persistents across calls until the file changes or is cleared", () => {
    const counter = (step: number) => ({
      name: "counter.m",
      source: `function n = counter()\npersistent c\nif isempty(c), c = 0; end\nc = c + ${step};\nn = c;\nend\n`,
    });
    const s = new ExecSession({ workspaceFiles: [counter(1)] });
    run(s, "counter(); counter();");
    expect(value(s, "n = counter();", "n")).toBe(3);
    run(s, "clear");
    expect(value(s, "n = counter();", "n")).toBe(4);
    run(s, "clear counter");
    expect(value(s, "n = counter();", "n")).toBe(1);
    run(s, "clear functions");
    expect(value(s, "n = counter();", "n")).toBe(1);
    // Editing the file clears its persistents, as MATLAB does.
    s.workspaceFiles = [counter(10)];
    expect(value(s, "n = counter();", "n")).toBe(10);
    expect(value(s, "n = counter();", "n")).toBe(20);
  });

  it("keeps warning state, PRNG stream, tic, appdata and spparms", () => {
    const s = new ExecSession();
    for (const optimization of ["0", "1"] as const) {
      run(s, "warning('off', 'my:id');", { optimization });
      const out = run(s, "warning('my:id', 'hidden'); warning('shown');", {
        optimization,
      }).output.join("");
      expect(out).toBe("Warning: shown\n");
    }

    run(s, "rng(3); a = rand;");
    const b = value(s, "b = rand;", "b");
    const ref = executeCode("rng(3); a = rand; b = rand;", opts);
    expect(b).toBe(ref.variableValues.b);
    expect(s.variableValues.a).toBe(ref.variableValues.a);

    run(s, "tic;");
    const t = value(s, "t = toc;", "t") as number;
    expect(t).toBeGreaterThanOrEqual(0);
    expect(t).toBeLessThan(60);

    run(s, "setappdata(0, 'k', [1 2]);");
    expect(data(value(s, "v = getappdata(0, 'k');", "v"))).toEqual([1, 2]);

    run(s, "spparms('spumoni', 2);");
    expect(value(s, "v = spparms('spumoni');", "v")).toBe(2);
  });

  it("keeps event listeners across calls", () => {
    const src = "classdef Src < handle\n  events\n    Ev\n  end\nend\n";
    const s = new ExecSession({
      workspaceFiles: [{ name: "Src.m", source: src }],
    });
    run(s, "h = Src; l = addlistener(h, 'Ev', @(~, ~) disp('fired'));");
    expect(run(s, "notify(h, 'Ev')").output.join("")).toBe("fired\n");
  });

  it("gives one-shot executeCode runs fresh state", () => {
    executeCode("global G; G = 5; rng(3); setappdata(0, 'k', 1);", opts);
    const r = executeCode(
      "global G; e = isempty(G); a = isappdata(0, 'k');",
      opts
    );
    expect(r.variableValues.e).toBe(true);
    expect(r.variableValues.a).toBe(false);
  });

  it("keeps refcounts stable across calls", () => {
    for (const optimization of ["0", "1"] as const) {
      const s = new ExecSession();
      run(s, "x = [1 2 3]; c = {x, 'a'}; st.f = x;", { optimization });
      const x = s.variableValues.x;
      const c = s.variableValues.c;
      const st = s.variableValues.st;
      // x is held by the variable, the cell and the struct field.
      expect([rc(x), rc(c), rc(st)]).toEqual([3, 1, 1]);
      for (let i = 0; i < 5; i++) run(s, `y = ${i};`, { optimization });
      expect([rc(x), rc(c), rc(st)]).toEqual([3, 1, 1]);
      expect(s.variableValues.x).toBe(x);
      run(s, "clear c st", { optimization });
      expect(rc(x)).toBe(1);
    }
  });

  it("isolates two sessions from each other", () => {
    const a = new ExecSession();
    const b = new ExecSession();
    run(a, "global G; G = 1; rng(3); warning('off', 'my:id');");
    run(b, "global G; G = 2; rng(5); setappdata(0, 'k', 1);");
    run(b, "tic; spparms('spumoni', 2); x = rand;");

    // a's stream continues from rng(3) whatever b did.
    const ref = executeCode("rng(3); r = rand;", opts).variableValues.r;
    expect(value(a, "r = rand;", "r")).toBe(ref);
    expect(value(a, "global G; v = G;", "v")).toBe(1);
    expect(value(b, "global G; v = G;", "v")).toBe(2);
    expect(value(a, "k = isappdata(0, 'k');", "k")).toBe(false);
    expect(value(a, "v = spparms('spumoni');", "v")).toBe(0);
    const warn = "warning('my:id', 'w');";
    expect(run(a, warn).output.join("")).toBe("");
    expect(run(b, warn).output.join("")).toBe("Warning: w\n");
  });
});

describe("CLI REPL", () => {
  it("keeps globals, and variables set before an error, across commands", () => {
    const root = resolve(import.meta.dirname, "../..");
    const input = [
      "global G; G = 5;",
      "global G; disp(size(G))",
      "x = 2; error('boom')",
      "disp(x)",
    ].join("\n");
    const proc = spawnSync(
      process.execPath,
      ["--import", "tsx", resolve(root, "src/cli.ts"), "--no-native"],
      { cwd: root, encoding: "utf8", input: input + "\n" }
    );
    // Output lines follow the prompts (">> ") of the commands before them.
    expect(proc.stdout).toMatch(/>> {4}1 {3}1\n/);
    expect(proc.stdout).toMatch(/>> 2\n/);
    expect(proc.stderr).toContain("boom");
  }, 60000);
});
