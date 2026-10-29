// Library-path use of the Node platform: only the public entry points
// ("numbl" = src/lib.ts, "numbl/node" = src/node.ts) are imported.
import { describe, it, expect } from "vitest";
import { execSync } from "node:child_process";
import { createHost, defaultHost, executeCode, ExecSession } from "../lib.js";
import { initNodePlatform } from "../node.js";

const platform = await initNodePlatform({ native: false });

function hasCCompiler(): boolean {
  try {
    execSync(`${process.env.NUMBL_CC || "cc"} --version`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}
const canCJit = platform.nativeBridge !== undefined && hasCCompiler();

const LOOP = "s = 0;\nfor i = 1:100\n  s = s + i * i;\nend";

describe("initNodePlatform", () => {
  it("does not add the mip core package unless asked", () => {
    expect(platform.searchPaths).toEqual([]);
    expect(platform.nativeAddonLoaded).toBe(false);
  });

  it.skipIf(!canCJit)('runs code through the C-JIT at optimization "2"', () => {
    const langs: string[] = [];
    const result = executeCode(
      LOOP,
      {
        optimization: "2",
        onJitCompile: (_desc, _code, lang) => langs.push(lang),
      },
      [],
      "script.m",
      [],
      platform.nativeBridge
    );
    expect(result.variableValues.s).toBe(338350);
    expect(langs).toContain("c");
  });

  it.skipIf(!canCJit)("fills the default host, so runs need no bridge", () => {
    expect(platform.host).toBe(defaultHost());
    const langs: string[] = [];
    executeCode(LOOP, {
      optimization: "2",
      onJitCompile: (_desc, _code, lang) => langs.push(lang),
    });
    expect(langs).toContain("c");
  });

  it("fills a separate host without touching the default", async () => {
    const host = createHost();
    const p = await initNodePlatform({ host, native: false, cjit: false });
    expect(p.host).toBe(host);
    expect(host.cjit).toBeUndefined();
    expect(host.geometry.delaunay).toBeDefined();
    expect(defaultHost().cjit).toBeDefined();
    const langs: string[] = [];
    const r = executeCode(LOOP, {
      host,
      optimization: "2",
      onJitCompile: (_desc, _code, lang) => langs.push(lang),
    });
    expect(r.variableValues.s).toBe(338350);
    expect(langs).not.toContain("c");
  });
});

describe("ExecSession (library entry point)", () => {
  it("keeps variables across executions", () => {
    const session = new ExecSession({ nativeBridge: platform.nativeBridge });
    session.execute("x = 3;", { optimization: "2" });
    const r = session.execute("y = x + 1; disp(y)", { optimization: "2" });
    expect(r.variableValues.y).toBe(4);
    expect(r.output.join("")).toBe("4\n");
  });
});
