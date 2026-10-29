/// <reference types="node" />

import { describe, it, expect } from "vitest";
import { createRequire } from "node:module";
import { executeCode } from "../numbl-core/executeCode.js";
import { registerNodeCompileC } from "../numbl-core/executors/jit/cjit.node.js";
import type { OptLevel } from "../numbl-core/executors/plugins.js";
import type { NativeBridge } from "../numbl-core/workspace/types.js";

// A JIT'd unit that bails at runtime (indexed-store growth at k = 4) is
// re-run from the start by the interpreter, so any side effect it produced
// before the bail must be rolled back.

registerNodeCompileC();
let nativeBridge: NativeBridge | undefined;
try {
  const koffi = createRequire(import.meta.url)("koffi");
  nativeBridge = { load: (path: string) => koffi.load(path), koffi };
} catch {
  // koffi unavailable: --opt 2 behaves like --opt 1.
}

function run(code: string, optimization: OptLevel) {
  const bails: string[] = [];
  const r = executeCode(
    code,
    { optimization, onJitBail: m => bails.push(m) },
    [],
    "script",
    undefined,
    nativeBridge
  );
  return { ...r, bails };
}

const OPTS: OptLevel[] = ["0", "1", "2"];

describe("JIT bail side effects", () => {
  it("plots from a unit that bails are not duplicated", () => {
    const code = `
v = zeros(1,3);
for k = 1:5
  figure;
  plot(k, k);
  v(k) = k;
end
`;
    const ref = run(code, "0");
    expect(ref.plotInstructions.length).toBeGreaterThan(0);
    for (const opt of OPTS.slice(1)) {
      const r = run(code, opt);
      expect(r.bails.length).toBeGreaterThan(0);
      expect(r.plotInstructions).toEqual(ref.plotInstructions);
      expect(r.figureHandles).toEqual(ref.figureHandles);
    }
  });

  it("rand draws from a unit that bails are rewound", () => {
    const code = `
rng(7);
v = zeros(1,3);
s = 0;
for k = 1:5
  s = s + rand();
  v(k) = k;
end
t = rand();
`;
    const ref = run(code, "0");
    for (const opt of OPTS.slice(1)) {
      const r = run(code, opt);
      expect(r.bails.length).toBeGreaterThan(0);
      expect(r.variableValues.s).toBe(ref.variableValues.s);
      expect(r.variableValues.t).toBe(ref.variableValues.t);
    }
  });
});
