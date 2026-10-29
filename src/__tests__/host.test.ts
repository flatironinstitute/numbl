// Two sessions in one process with different hosts: each run reads its
// capabilities from its own host, not from process-wide state.
import { describe, it, expect } from "vitest";
import {
  createHost,
  defaultHost,
  executeCode,
  ExecSession,
  type LapackProvider,
} from "../lib.js";
import { currentHost } from "../numbl-core/host.js";

/** A matmul provider that records its calls and answers with all 7s. */
function recordingProvider(calls: string[]): LapackProvider {
  return {
    name: "stub",
    priority: 1000, // ahead of the inline provider
    entries: {
      matmul: (_a, m, k, _b, n) => {
        calls.push(`${m}x${k}*${k}x${n}`);
        return new Float64Array(m * n).fill(7);
      },
    },
  };
}

// Above the inline size limit, so every --opt level reaches the stack.
const PRODUCT = "C = ones(20, 30) * ones(30, 10); c = C(1, 1);";

describe("NumblHost", () => {
  for (const optimization of ["0", "1"] as const) {
    it(`keeps numeric providers per session (opt ${optimization})`, () => {
      const calls: string[] = [];
      const stubbed = createHost();
      stubbed.numeric.register(recordingProvider(calls));
      const a = new ExecSession({ host: stubbed });
      const b = new ExecSession({ host: createHost() });

      expect(a.execute(PRODUCT, { optimization }).variableValues.c).toBe(7);
      expect(b.execute(PRODUCT, { optimization }).variableValues.c).toBe(30);
      expect(a.execute(PRODUCT, { optimization }).variableValues.c).toBe(7);
      expect(calls).toEqual(["20x30*30x10", "20x30*30x10"]);

      // Runs without a host use the process default, which has no stub.
      expect(executeCode(PRODUCT, { optimization }).variableValues.c).toBe(30);
      expect(calls).toHaveLength(2);
    });
  }

  it("keeps geometry backends per host", () => {
    const withHull = createHost({
      geometry: { delaunay: () => [[0, 1, 2]] },
    });
    const without = createHost();
    const code = "t = delaunay([0 1 0 5], [0 0 1 5]);";
    const r = executeCode(code, { host: withHull });
    expect(r.variableValues.t).toMatchObject({ shape: [1, 3] });
    expect(() => executeCode(code, { host: without })).toThrow(
      /no triangulation backend/
    );
  });

  it("gives .numbl.js files the host's gpu and makes it current", () => {
    const seen: unknown[] = [];
    const probe = {
      name: "probe.numbl.js",
      source: `register({
        resolve: () => ({
          outputTypes: [{ kind: 'number' }],
          apply: () => { globalThis.__hostProbe(gpu); return gpu ? gpu.tag : -1; }
        })
      });`,
    };
    const g = globalThis as { __hostProbe?: (gpu: unknown) => void };
    g.__hostProbe = () => seen.push(currentHost());
    try {
      const host = createHost({ gpu: { tag: 42 } });
      const r = executeCode("v = probe();", { host }, [probe]);
      expect(r.variableValues.v).toBe(42);
      expect(seen).toEqual([host]);
      const d = executeCode("v = probe();", {}, [probe]);
      expect(d.variableValues.v).toBe(-1);
      expect(seen[1]).toBe(defaultHost());
      expect(currentHost()).toBe(defaultHost());
    } finally {
      delete g.__hostProbe;
    }
  });

  it("supplies fileIO and system defaults from the host", () => {
    let asked = false;
    const host = createHost({
      system: {
        getEnv: (name: string) => {
          asked = true;
          return name === "NUMBL_HOST_TEST" ? "yes" : undefined;
        },
        getAllEnv: () => ({}),
        setEnv: () => {},
        cwd: () => "/",
        chdir: () => {},
        platform: () => "linux",
        arch: () => "x64",
      },
    });
    const r = executeCode(
      "v = double(strcmp(getenv('NUMBL_HOST_TEST'), 'yes'));",
      {
        host,
        implicitCwdPath: null,
      }
    );
    expect(asked).toBe(true);
    expect(r.variableValues.v).toBe(1);
  });
});
