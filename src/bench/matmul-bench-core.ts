import { Bench } from "tinybench";

export interface MatmulFn {
  (
    A: Float64Array,
    m: number,
    k: number,
    B: Float64Array,
    n: number
  ): Float64Array;
}

export interface MatmulBackend {
  id: string;
  label: string;
  matmul: MatmulFn;
}

export interface MatmulScenario {
  id: string;
  label: string;
  m: number;
  k: number;
  n: number;
  A: Float64Array;
  B: Float64Array;
}

export interface MatmulResult {
  scenarioId: string;
  backendId: string;
  medianMs: number;
  p75Ms: number;
  p99Ms: number;
  samples: number;
  opsPerSec: number;
  error?: string;
  maxDiff?: number;
}

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

function randomMatrix(rows: number, cols: number, seed: number): Float64Array {
  const rng = splitmix32(seed);
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < data.length; i++) data[i] = rng() * 2 - 1;
  return data;
}

export function buildScenarios(): MatmulScenario[] {
  const sizes = [
    [128, 128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [1000, 1000, 1000],
  ];
  return sizes.map(([m, k, n]) => ({
    id: `matmul-${m}x${k}x${n}`,
    label: `${m}×${k} * ${k}×${n}`,
    m,
    k,
    n,
    A: randomMatrix(m, k, m * 1000 + k * 100 + n),
    B: randomMatrix(k, n, m * 1000 + k * 100 + n + 1),
  }));
}

function maxAbsDiff(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

export function validateBackend(
  backend: MatmulBackend,
  ref: MatmulBackend,
  scenarios: MatmulScenario[],
  tol = 1e-10
): { ok: boolean; errors: string[] } {
  const errors: string[] = [];
  for (const s of scenarios) {
    const refResult = ref.matmul(s.A, s.m, s.k, s.B, s.n);
    const testResult = backend.matmul(s.A, s.m, s.k, s.B, s.n);
    const diff = maxAbsDiff(refResult, testResult);
    if (diff > tol) errors.push(`${s.id}: maxdiff=${diff.toExponential(2)}`);
  }
  return { ok: errors.length === 0, errors };
}

export async function runBenchmarks(
  scenarios: MatmulScenario[],
  backends: MatmulBackend[],
  opts: {
    warmup?: number;
    iterations?: number;
    onProgress?: (msg: string) => void;
  } = {}
): Promise<MatmulResult[]> {
  const results: MatmulResult[] = [];
  const ref = backends[0]; // ts-lapack is always first

  for (const scenario of scenarios) {
    const bench = new Bench({
      warmupIterations: opts.warmup ?? 3,
      iterations: opts.iterations ?? 10,
      time: 0,
    });

    for (const backend of backends) {
      bench.add(`${scenario.id}|${backend.id}`, () => {
        backend.matmul(
          scenario.A,
          scenario.m,
          scenario.k,
          scenario.B,
          scenario.n
        );
      });
    }

    opts.onProgress?.(`${scenario.id}...`);
    await bench.run();

    // Compute correctness for each backend
    const refResult = ref.matmul(
      scenario.A,
      scenario.m,
      scenario.k,
      scenario.B,
      scenario.n
    );

    for (const task of bench.tasks) {
      const [, backendId] = task.name.split("|");
      const r = task.result!;
      const backend = backends.find(b => b.id === backendId)!;

      if ("error" in r) {
        results.push({
          scenarioId: scenario.id,
          backendId,
          medianMs: 0,
          p75Ms: 0,
          p99Ms: 0,
          samples: 0,
          opsPerSec: 0,
          error: r.error instanceof Error ? r.error.message : String(r.error),
        });
      } else if ("latency" in r) {
        const testResult = backend.matmul(
          scenario.A,
          scenario.m,
          scenario.k,
          scenario.B,
          scenario.n
        );
        const diff = maxAbsDiff(refResult, testResult);
        results.push({
          scenarioId: scenario.id,
          backendId,
          medianMs: r.latency.p50,
          p75Ms: r.latency.p75,
          p99Ms: r.latency.p99,
          samples: r.latency.samplesCount,
          opsPerSec: r.throughput.p50,
          maxDiff: diff,
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

export function formatResults(
  results: MatmulResult[],
  scenarios: MatmulScenario[],
  backends: MatmulBackend[]
): string {
  const legend = backends.map(b => `  ${b.id}: ${b.label}`).join("\n");

  const byScenario = new Map<string, Map<string, MatmulResult>>();
  for (const r of results) {
    if (!byScenario.has(r.scenarioId)) byScenario.set(r.scenarioId, new Map());
    byScenario.get(r.scenarioId)!.set(r.backendId, r);
  }

  const bids = backends.map(b => b.id);
  const tsId = "ts-lapack";
  const others = bids.filter(id => id !== tsId);

  const header = [
    "Size",
    ...bids.map(id => id),
    ...others.map(id => `${id} vs ts`),
  ];
  const rows: string[][] = [];

  for (const s of scenarios) {
    const bmap = byScenario.get(s.id);
    if (!bmap) continue;
    const row = [s.label];

    for (const bid of bids) {
      const r = bmap.get(bid);
      if (!r) row.push("-");
      else if (r.error) row.push("ERR");
      else row.push(fmtMs(r.medianMs));
    }

    const tsR = bmap.get(tsId);
    for (const bid of others) {
      const r = bmap.get(bid);
      if (
        !r ||
        !tsR ||
        r.error ||
        tsR.error ||
        r.medianMs === 0 ||
        tsR.medianMs === 0
      )
        row.push("-");
      else row.push(`${(tsR.medianMs / r.medianMs).toFixed(1)}x`);
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
