/// <reference types="node" />

import { describe, it, expect } from "vitest";
import { createRequire } from "node:module";
import { Worker } from "node:worker_threads";
import { executeCode } from "../numbl-core/executeCode.js";
import { CancellationError } from "../numbl-core/runtime/index.js";
import { Runtime } from "../numbl-core/runtime/runtime.js";
import { registerNodeCompileC } from "../numbl-core/executors/jit/cjit.node.js";
import type { OptLevel } from "../numbl-core/executors/plugins.js";
import type { NativeBridge } from "../numbl-core/workspace/types.js";

// Stop (the cancel flag) must interrupt JIT'd loops, and output from a JIT'd
// loop must arrive while it runs. A run is synchronous, so the flag is set
// from another thread, as the IDE's main thread does for its worker.

registerNodeCompileC();
let nativeBridge: NativeBridge | undefined;
try {
  const koffi = createRequire(import.meta.url)("koffi");
  nativeBridge = { load: (path: string) => koffi.load(path), koffi };
} catch {
  // koffi unavailable: the --opt 2 cases run the JS backend.
}

/** Set word 0 of `sab` to 1 after `ms`, from another thread. */
function cancelAfter(sab: SharedArrayBuffer, ms: number): Worker {
  return new Worker(
    `const { workerData } = require("node:worker_threads");
     setTimeout(() => Atomics.store(new Int32Array(workerData), 0, 1), ${ms});`,
    { eval: true, workerData: sab }
  );
}

interface CancelRun {
  error: unknown;
  /** From the cancel signal (or the end of compiling, if later) to the stop. */
  ms: number;
  output: string;
  compiled: string[];
  bails: string[];
}

const CANCEL_AFTER_MS = 300;

function runCancelled(code: string, opt: OptLevel): CancelRun {
  const sab = new SharedArrayBuffer(4);
  const worker = cancelAfter(sab, CANCEL_AFTER_MS);
  const out: string[] = [];
  const compiled: string[] = [];
  const bails: string[] = [];
  const t0 = performance.now();
  // Compiling (cc at --opt 2, with a cold cache in CI) can outlast the
  // cancel delay, so time the stop from whichever comes later.
  let tCompiled = t0;
  let error: unknown = null;
  try {
    executeCode(
      code,
      {
        optimization: opt,
        cancelSAB: sab,
        onOutput: t => out.push(t),
        onJitCompile: (d, _c, lang) => {
          compiled.push(`${lang}:${d}`);
          tCompiled = performance.now();
        },
        onJitBail: m => bails.push(m),
      },
      [],
      "script",
      undefined,
      nativeBridge
    );
  } catch (e) {
    error = e;
  }
  const ms = performance.now() - Math.max(t0 + CANCEL_AFTER_MS, tCompiled);
  void worker.terminate();
  return { error, ms, output: out.join(""), compiled, bails };
}

const OPTS: OptLevel[] = ["1", "2"];
const lang = (opt: OptLevel) => (opt === "2" && nativeBridge ? "c" : "js");

// Each loop runs for minutes uncancelled (and still terminates if a poll is
// missing, so a regression fails rather than hangs).
const LOOPS: Record<string, string> = {
  "scalar for": `
s = 0;
for k = 1:1e11
  s = s + k;
end
disp(s);
`,
  "for with output": `
s = 0;
disp('start');
for k = 1:1e11
  s = s + sqrt(k);
  if mod(k, 1e9) == 0
    fprintf('%g\\n', s);
  end
end
`,
  while: `
s = 0;
k = 0;
while k < 1e11
  k = k + 1;
  s = s + k * 0.5;
end
disp(s);
`,
  recursion: `
disp(fib(60));
function r = fib(n)
  if n < 2
    r = n;
  else
    r = fib(n - 1) + fib(n - 2);
  end
end
`,
};

// Generous timeouts: these run long JIT'd loops on a possibly busy machine.
describe("JIT cancellation", { timeout: 60_000 }, () => {
  for (const opt of OPTS) {
    for (const [name, code] of Object.entries(LOOPS)) {
      it(`Stop interrupts a JIT'd ${name} at --opt ${opt}`, () => {
        const r = runCancelled(code, opt);
        expect(r.error).toBeInstanceOf(CancellationError);
        expect(r.ms).toBeLessThan(5000);
        // The unit ran compiled (and did not bail to the interpreter).
        expect(r.compiled.some(d => d.startsWith(`${lang(opt)}:`))).toBe(true);
        expect(r.bails).toEqual([]);
        if (name === "for with output") expect(r.output).toBe("start\n");
      });
    }
  }
});

describe("JIT output streaming", { timeout: 60_000 }, () => {
  for (const opt of OPTS) {
    it(`output from a JIT'd loop arrives while it runs at --opt ${opt}`, () => {
      // Four lines, tens of milliseconds apart.
      const code = `
for k = 1:4
  s = 0;
  for j = 1:5e7
    s = s + j;
  end
  fprintf('%d %g\\n', k, s);
end
`;
      const stamps: number[] = [];
      const compiled: string[] = [];
      const r = executeCode(
        code,
        {
          optimization: opt,
          // Compiled C may write a line in several pieces.
          onOutput: t => {
            if (t.endsWith("\n")) stamps.push(performance.now());
          },
          onJitCompile: (d, _c, l) => compiled.push(`${l}:${d}`),
        },
        [],
        "script",
        undefined,
        nativeBridge
      );
      const end = performance.now();
      expect(compiled.some(d => d.startsWith(`${lang(opt)}:`))).toBe(true);
      expect(r.output.join("")).toBe(
        [1, 2, 3, 4].map(k => `${k} 1.25e+15\n`).join("")
      );
      expect(stamps.length).toBe(4);
      // Buffered output would arrive at once, at the end of the unit, so
      // the lines would be well under a millisecond apart.
      const spread = stamps[3] - stamps[0];
      expect(spread).toBeGreaterThan(20);
      expect(spread).toBeGreaterThan((end - stamps[0]) / 2);
    });

    it(`a streaming unit that bails does not duplicate output at --opt ${opt}`, () => {
      // The store at k = 4 grows `v`: the JIT bails after printing three
      // lines and the interpreter re-runs the loop from the start.
      const code = `
v = zeros(1, 3);
for k = 1:5
  fprintf('k=%d\\n', k);
  v(k) = k;
end
disp(v);
`;
      const out: string[] = [];
      const bails: string[] = [];
      const r = executeCode(
        code,
        {
          optimization: opt,
          onOutput: t => out.push(t),
          onJitBail: m => bails.push(m),
        },
        [],
        "script",
        undefined,
        nativeBridge
      );
      const ref = executeCode(code, { optimization: "0" }).output.join("");
      expect(bails.length).toBeGreaterThan(0);
      expect(out.join("")).toBe(ref);
      expect(r.output.join("")).toBe(ref);
    });

    it(`an error re-run inside try does not duplicate output at --opt ${opt}`, () => {
      // The out-of-range read raises inside \`try\`, where the catch block
      // could see the loop's partial state: the unit re-runs on the
      // interpreter, which reprints the lines the JIT streamed.
      const code = `
v = [1 2 3];
s = 0;
try
  for k = 1:5
    fprintf('k=%d\\n', k);
    s = s + v(k);
  end
catch e
  disp(e.message);
end
disp(s);
`;
      const out: string[] = [];
      executeCode(
        code,
        { optimization: opt, onOutput: t => out.push(t) },
        [],
        "script",
        undefined,
        nativeBridge
      );
      const ref = executeCode(code, { optimization: "0" }).output.join("");
      expect(out.join("")).toBe(ref);
    });
  }
});

describe("Runtime output replay", () => {
  function runtime() {
    const out: string[] = [];
    const rt = new Runtime({ onOutput: t => out.push(t) });
    return { rt, text: () => out.join("") };
  }

  it("skips a re-run's output up to what a discarded attempt emitted", () => {
    const { rt, text } = runtime();
    rt.output("before\n");
    rt.beginOutputCapture(true);
    rt.output("a\n");
    rt.output("b\n");
    expect(text()).toBe("before\na\nb\n");
    rt.discardOutputCapture();
    // The re-run, in different chunks, then past the attempt.
    rt.output("a");
    rt.output("\nb\nc");
    rt.output("\n");
    expect(text()).toBe("before\na\nb\nc\n");
  });

  it("rewinds nested attempts and buffers inside a buffering frame", () => {
    const { rt, text } = runtime();
    rt.beginOutputCapture(true);
    rt.output("1");
    rt.beginOutputCapture(true);
    rt.output("2");
    rt.discardOutputCapture();
    rt.output("2");
    rt.output("3");
    rt.commitOutputCapture();
    expect(text()).toBe("123");
    rt.beginOutputCapture(false);
    rt.beginOutputCapture(true);
    rt.output("x");
    rt.discardOutputCapture();
    rt.output("y");
    expect(text()).toBe("123");
    rt.commitOutputCapture();
    expect(text()).toBe("123y");
  });

  it("emits the rest when the re-run diverges", () => {
    const { rt, text } = runtime();
    rt.beginOutputCapture(true);
    rt.output("t=0.1\n");
    rt.discardOutputCapture();
    rt.output("t=0.2\n");
    expect(text()).toBe("t=0.1\n2\n");
    rt.output("next\n");
    expect(text()).toBe("t=0.1\n2\nnext\n");
  });
});
