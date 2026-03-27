import { Bench } from "tinybench";
import type { LapackBridge } from "../numbl-core/native/lapack-bridge.js";

function splitmix32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x9e3779b9) | 0;
    let t = seed ^ (seed >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    t = t ^ (t >>> 15);
    return (t >>> 0) / 4294967296;
  };
}

function randomMatrix(
  rows: number,
  cols: number,
  rng: () => number
): Float64Array {
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < data.length; i++) data[i] = rng() * 2 - 1;
  return data;
}

/** Generate a symmetric positive definite matrix: A = M'M + n*I */
function randomSPD(n: number, rng: () => number): Float64Array {
  const m = randomMatrix(n, n, rng);
  const out = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += m[k + i * n] * m[k + j * n];
      }
      out[i + j * n] = s;
    }
  }
  for (let i = 0; i < n; i++) out[i + i * n] += n;
  return out;
}

export interface BenchScenario {
  id: string;
  label: string;
  execute(bridge: LapackBridge): unknown;
}

function createMatmulScenario(m: number, k: number, n: number): BenchScenario {
  const rng = splitmix32(m * 1000 + k * 100 + n);
  const A = randomMatrix(m, k, rng);
  const B = randomMatrix(k, n, rng);
  return {
    id: `matmul-${m}x${k}x${n}`,
    label: `matmul ${m}×${k} * ${k}×${n}`,
    execute: bridge => bridge.matmul!(A, m, k, B, n),
  };
}

function createInvScenario(n: number): BenchScenario {
  const rng = splitmix32(n * 7 + 1);
  const data = randomMatrix(n, n, rng);
  for (let i = 0; i < n; i++) data[i + i * n] += n;
  return {
    id: `inv-${n}`,
    label: `inv ${n}×${n}`,
    execute: bridge => bridge.inv(data, n),
  };
}

function createQrScenario(m: number, n: number): BenchScenario {
  const rng = splitmix32(m * 13 + n);
  const data = randomMatrix(m, n, rng);
  return {
    id: `qr-${m}x${n}`,
    label: `qr ${m}×${n} econ`,
    execute: bridge => bridge.qr!(data, m, n, true, true),
  };
}

function createLinsolveScenario(n: number, nrhs: number): BenchScenario {
  const rng = splitmix32(n * 17 + nrhs);
  const A = randomMatrix(n, n, rng);
  for (let i = 0; i < n; i++) A[i + i * n] += n;
  const B = randomMatrix(n, nrhs, rng);
  return {
    id: `linsolve-${n}-nrhs${nrhs}`,
    label: `linsolve ${n}×${n}, nrhs=${nrhs}`,
    execute: bridge => bridge.linsolve!(A, n, n, B, nrhs),
  };
}

function createEigScenario(n: number): BenchScenario {
  const rng = splitmix32(n * 23);
  const data = randomMatrix(n, n, rng);
  return {
    id: `eig-${n}`,
    label: `eig ${n}×${n}`,
    execute: bridge => bridge.eig!(data, n, false, true, true),
  };
}

function createLuScenario(n: number): BenchScenario {
  const rng = splitmix32(n * 29);
  const data = randomMatrix(n, n, rng);
  return {
    id: `lu-${n}`,
    label: `lu ${n}×${n}`,
    execute: bridge => bridge.lu!(data, n, n),
  };
}

function createSvdScenario(m: number, n: number): BenchScenario {
  const rng = splitmix32(m * 31 + n);
  const data = randomMatrix(m, n, rng);
  return {
    id: `svd-${m}x${n}`,
    label: `svd ${m}×${n} econ`,
    execute: bridge => bridge.svd!(data, m, n, true, true),
  };
}

function createCholScenario(n: number): BenchScenario {
  const rng = splitmix32(n * 37);
  const data = randomSPD(n, rng);
  return {
    id: `chol-${n}`,
    label: `chol ${n}×${n}`,
    execute: bridge => bridge.chol!(data, n, true),
  };
}

export function buildScenarios(): BenchScenario[] {
  return [
    createMatmulScenario(128, 128, 128),
    createMatmulScenario(256, 256, 256),
    createMatmulScenario(512, 512, 512),
    createMatmulScenario(1000, 1000, 1000),

    createInvScenario(64),
    createInvScenario(128),
    createInvScenario(256),

    createQrScenario(128, 128),
    createQrScenario(256, 256),
    createQrScenario(512, 256),

    createLinsolveScenario(128, 4),
    createLinsolveScenario(256, 1),

    createEigScenario(64),
    createEigScenario(128),
    createEigScenario(256),

    createLuScenario(128),
    createLuScenario(256),

    createSvdScenario(64, 64),
    createSvdScenario(128, 128),
    createSvdScenario(256, 128),

    createCholScenario(64),
    createCholScenario(128),
    createCholScenario(256),
  ];
}

export interface BenchTimingResult {
  scenarioId: string;
  scenarioLabel: string;
  backendId: string;
  meanMs: number;
  medianMs: number;
  p75Ms: number;
  p99Ms: number;
  stdDevMs: number;
  samples: number;
  opsPerSec: number;
  error?: string;
}

export interface BenchBackend {
  id: string;
  label: string;
  bridge: LapackBridge;
}

export interface RunOptions {
  warmup?: number;
  iterations?: number;
  time?: number;
  onProgress?: (scenarioId: string, backendId: string) => void;
}

export async function runAllScenarios(
  scenarios: BenchScenario[],
  backends: BenchBackend[],
  opts: RunOptions = {}
): Promise<BenchTimingResult[]> {
  const results: BenchTimingResult[] = [];

  for (const scenario of scenarios) {
    const bench = new Bench({
      warmupIterations: opts.warmup ?? 3,
      iterations: opts.iterations ?? 10,
      time: opts.time ?? 0,
    });

    for (const backend of backends) {
      bench.add(`${scenario.id}|${backend.id}`, () => {
        scenario.execute(backend.bridge);
      });
    }

    opts.onProgress?.(scenario.id, "all");
    await bench.run();

    for (const task of bench.tasks) {
      const [scenarioId, backendId] = task.name.split("|");
      const r = task.result!;
      const base = { scenarioId, scenarioLabel: scenario.label, backendId };

      if ("error" in r) {
        const err = r.error;
        results.push({
          ...base,
          meanMs: 0,
          medianMs: 0,
          p75Ms: 0,
          p99Ms: 0,
          stdDevMs: 0,
          samples: 0,
          opsPerSec: 0,
          error: err instanceof Error ? err.message : String(err),
        });
      } else if ("latency" in r) {
        const { mean, p50, p75, p99, sd, samplesCount } = r.latency;
        results.push({
          ...base,
          meanMs: mean,
          medianMs: p50,
          p75Ms: p75,
          p99Ms: p99,
          stdDevMs: sd,
          samples: samplesCount,
          opsPerSec: r.throughput.p50,
        });
      }
    }
  }

  return results;
}

function fmtMs(v: number): string {
  if (v === 0) return "-";
  if (v >= 1000) return (v / 1000).toFixed(2) + "s";
  if (v >= 100) return v.toFixed(1);
  if (v >= 1) return v.toFixed(3);
  return (v * 1000).toFixed(1) + "µs";
}

export function formatResultsTable(
  results: BenchTimingResult[],
  backends: BenchBackend[]
): string {
  const legend = backends.map(b => `  ${b.id}: ${b.label}`).join("\n");

  const byScenario = new Map<string, Map<string, BenchTimingResult>>();
  for (const r of results) {
    if (!byScenario.has(r.scenarioId)) byScenario.set(r.scenarioId, new Map());
    byScenario.get(r.scenarioId)!.set(r.backendId, r);
  }

  const backendIds = backends.map(b => b.id);
  const tsId = backendIds.includes("ts-lapack") ? "ts-lapack" : undefined;
  const otherIds = backendIds.filter(id => id !== "ts-lapack");

  const header = [
    "Scenario",
    ...backendIds.map(id => `${id} (median)`),
    ...otherIds.map(id => `${id} vs ts`),
  ];
  const rows: string[][] = [];

  for (const [, backendMap] of byScenario) {
    const first = backendMap.values().next().value!;
    const row = [first.scenarioLabel];

    for (const bid of backendIds) {
      const r = backendMap.get(bid);
      if (!r) row.push("-");
      else if (r.error) row.push("ERR");
      else row.push(fmtMs(r.medianMs));
    }

    const tsResult = tsId ? backendMap.get(tsId) : undefined;
    for (const bid of otherIds) {
      const r = backendMap.get(bid);
      if (
        !r ||
        !tsResult ||
        r.error ||
        tsResult.error ||
        tsResult.medianMs === 0 ||
        r.medianMs === 0
      ) {
        row.push("-");
      } else {
        const ratio = tsResult.medianMs / r.medianMs;
        row.push(`${ratio.toFixed(1)}x`);
      }
    }

    rows.push(row);
  }

  const widths = header.map((h, i) =>
    Math.max(h.length, ...rows.map(r => r[i].length))
  );
  const pad = (s: string, w: number) => s.padEnd(w);
  const sep = widths.map(w => "-".repeat(w)).join(" | ");

  const table = [
    header.map((h, i) => pad(h, widths[i])).join(" | "),
    sep,
    ...rows.map(r => r.map((c, i) => pad(c, widths[i])).join(" | ")),
  ].join("\n");

  return `Backends:\n${legend}\n\n${table}`;
}
