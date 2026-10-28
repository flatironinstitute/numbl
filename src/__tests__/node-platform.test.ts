// Library-path use of the Node platform: only the public entry points
// ("numbl" = src/lib.ts, "numbl/node" = src/node.ts) are imported.
import { describe, it, expect } from "vitest";
import { execSync } from "node:child_process";
import { executeCode, ExecSession } from "../lib.js";
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
