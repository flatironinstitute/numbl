#!/usr/bin/env tsx

import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Bench } from "tinybench";
import {
  buildAllocBenchScenarios,
  createJsAllocBackend,
  QUICK_ALLOC_BENCH_SCENARIO_IDS,
  type AllocBenchBackend,
  type AllocBenchScenario,
} from "../src/bench/alloc-bench-core.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = resolve(__dirname, "..");

const DEFAULT_BENCH_TIME_MS = 750;
const DEFAULT_WARMUP_TIME_MS = 200;
const QUICK_BENCH_TIME_MS = 250;
const QUICK_WARMUP_TIME_MS = 100;

interface RunResult {
  scenarioId: string;
  scenarioLabel: string;
  backendId: string;
  backendLabel: string;
  medianMs?: number;
  meanMs?: number;
  minMs?: number;
  maxMs?: number;
  p95Ms?: number;
  opsPerSec?: number;
  samples?: number;
}

function parseArgs(): {
  list: boolean;
  quick: boolean;
  scenarioFilters: string[];
  outputPath?: string;
  markdownPath?: string;
} {
  const args = process.argv.slice(2);
  const result = {
    list: false,
    quick: false,
    scenarioFilters: [] as string[],
    outputPath: undefined as string | undefined,
    markdownPath: undefined as string | undefined,
  };
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--list") result.list = true;
    else if (arg === "--quick") result.quick = true;
    else if (arg === "--scenario" && i + 1 < args.length)
      result.scenarioFilters = args[++i].split(",");
    else if (arg === "--output" && i + 1 < args.length)
      result.outputPath = args[++i];
    else if (arg === "--markdown" && i + 1 < args.length)
      result.markdownPath = args[++i];
    else if (arg === "--help") {
      console.log(`Usage: tsx bench/run-alloc-bench.ts [options]

Options:
  --list             List available scenarios
  --quick            Reduced scenario set + shorter runs
  --scenario <csv>   Filter scenarios by id
  --output <path>    Write JSON results
  --markdown <path>  Write markdown report
`);
      process.exit(0);
    }
  }
  return result;
}

async function probeWasmBackend(): Promise<AllocBenchBackend | null> {
  const wasmPath = join(
    repoRoot,
    "public",
    "wasm-kernels",
    "numbl_allocator.wasm"
  );
  if (!existsSync(wasmPath)) return null;

  try {
    const wasmBytes = readFileSync(wasmPath);
    const importObject = {
      wasi_snapshot_preview1: {
        fd_write: () => 0,
        fd_read: () => 0,
        fd_close: () => 0,
        fd_seek: () => 0,
        fd_fdstat_get: () => 0,
        proc_exit: () => {},
        environ_sizes_get: () => 0,
        environ_get: () => 0,
        clock_time_get: () => 0,
        args_sizes_get: () => 0,
        args_get: () => 0,
      },
      env: { emscripten_notify_memory_growth: () => {} },
    };
    const { instance } = await WebAssembly.instantiate(wasmBytes, importObject);
    const exports = instance.exports as Record<string, unknown>;
    const memory = exports.memory as WebAssembly.Memory;
    const init = exports.numbl_alloc_init as (n: number) => number;
    const alloc = exports.numbl_alloc as (n: number) => number;
    const free = exports.numbl_free as (offset: number) => void;
    const initialize = exports._initialize as (() => void) | undefined;

    if (initialize) initialize();
    if (init(0) !== 0) throw new Error("numbl_alloc_init failed");

    const offsets = new Map<Float64Array, number>();

    return {
      id: "wasm-tlsf",
      label: "WASM TLSF allocator",
      alloc: (n: number) => {
        const bytes = n * 8;
        const offset = alloc(bytes);
        if (offset === 0) throw new Error("WASM alloc failed");
        const arr = new Float64Array(memory.buffer, offset, n);
        offsets.set(arr, offset);
        return arr;
      },
      free: (arr: Float64Array) => {
        const offset = offsets.get(arr);
        if (offset !== undefined) {
          free(offset);
          offsets.delete(arr);
        }
      },
    };
  } catch (e) {
    console.error(`  Warning: WASM allocator probe failed: ${e}`);
    return null;
  }
}

function percentile(sortedValues: number[], fraction: number): number {
  if (sortedValues.length === 0) return 0;
  const index = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.ceil(sortedValues.length * fraction) - 1)
  );
  return sortedValues[index];
}

async function main() {
  const opts = parseArgs();
  const allScenarios = buildAllocBenchScenarios();

  let scenarios: AllocBenchScenario[];
  if (opts.scenarioFilters.length > 0) {
    const filters = new Set(opts.scenarioFilters);
    scenarios = allScenarios.filter(s => filters.has(s.id));
  } else if (opts.quick) {
    scenarios = allScenarios.filter(s =>
      QUICK_ALLOC_BENCH_SCENARIO_IDS.has(s.id)
    );
  } else {
    scenarios = allScenarios;
  }

  // Probe backends
  const backends: AllocBenchBackend[] = [createJsAllocBackend()];
  const wasmBackend = await probeWasmBackend();
  if (wasmBackend) backends.push(wasmBackend);

  if (opts.list) {
    console.log("Backends:");
    for (const b of backends) console.log(`  ${b.id}: ${b.label}`);
    console.log("\nScenarios:");
    for (const s of allScenarios) {
      const quick = QUICK_ALLOC_BENCH_SCENARIO_IDS.has(s.id) ? " [quick]" : "";
      console.log(`  ${s.id}: ${s.label}${quick}`);
    }
    return;
  }

  console.log(`\nAllocation Benchmark`);
  console.log(`  Backends: ${backends.map(b => b.id).join(", ")}`);
  console.log(`  Scenarios: ${scenarios.length}`);
  console.log(`  Mode: ${opts.quick ? "quick" : "full"}\n`);

  const benchTime = opts.quick ? QUICK_BENCH_TIME_MS : DEFAULT_BENCH_TIME_MS;
  const warmupTime = opts.quick ? QUICK_WARMUP_TIME_MS : DEFAULT_WARMUP_TIME_MS;

  const allResults: RunResult[] = [];
  let sinkValue = 0;

  for (const scenario of scenarios) {
    console.log(`--- ${scenario.id}: ${scenario.label} ---`);

    const bench = new Bench({
      name: scenario.id,
      time: benchTime,
      warmup: true,
      warmupTime,
      iterations: opts.quick ? undefined : scenario.defaultIterations,
      retainSamples: true,
      throws: false,
    });

    for (const backend of backends) {
      bench.add(backend.id, () => {
        sinkValue += scenario.execute(backend);
      });
    }

    await bench.run();

    for (const task of bench.tasks) {
      const backend = backends.find(b => b.id === task.name)!;
      const latency = task.result?.latency;
      const samples = latency?.samples
        ? [...latency.samples].sort((a, b) => a - b)
        : [];
      const medianMs =
        samples.length > 0 ? percentile(samples, 0.5) : undefined;
      const p95Ms = samples.length > 0 ? percentile(samples, 0.95) : undefined;

      const result: RunResult = {
        scenarioId: scenario.id,
        scenarioLabel: scenario.label,
        backendId: backend.id,
        backendLabel: backend.label,
        medianMs,
        meanMs: latency?.mean,
        minMs: latency?.min,
        maxMs: latency?.max,
        p95Ms,
        opsPerSec: task.result?.hz,
        samples: samples.length,
      };
      allResults.push(result);

      const medianStr =
        medianMs !== undefined ? `${medianMs.toFixed(3)}ms` : "N/A";
      const p95Str = p95Ms !== undefined ? `${p95Ms.toFixed(3)}ms` : "N/A";
      const hzStr = task.result?.hz ? `${task.result.hz.toFixed(0)} ops/s` : "";
      console.log(
        `  ${backend.id.padEnd(12)} median=${medianStr.padEnd(12)} p95=${p95Str.padEnd(12)} ${hzStr}`
      );
    }

    // Compute speedup
    const jsResult = allResults.find(
      r => r.scenarioId === scenario.id && r.backendId === "js"
    );
    for (const r of allResults.filter(
      r => r.scenarioId === scenario.id && r.backendId !== "js"
    )) {
      if (jsResult?.medianMs && r.medianMs) {
        const speedup = jsResult.medianMs / r.medianMs;
        console.log(`  ${r.backendId} speedup vs js: ${speedup.toFixed(2)}x`);
      }
    }
    console.log();
  }

  // Prevent DCE
  if (sinkValue === Infinity) console.log(sinkValue);

  // Output JSON
  if (opts.outputPath) {
    const dir = dirname(opts.outputPath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    writeFileSync(opts.outputPath, JSON.stringify(allResults, null, 2));
    console.log(`Results written to ${opts.outputPath}`);
  }

  // Output Markdown
  if (opts.markdownPath) {
    const lines = [
      "# Allocation Benchmark Results\n",
      "| Scenario | Backend | Median (ms) | p95 (ms) | ops/s | Samples |",
      "|----------|---------|------------|----------|-------|---------|",
    ];
    for (const r of allResults) {
      lines.push(
        `| ${r.scenarioId} | ${r.backendId} | ${r.medianMs?.toFixed(3) ?? "N/A"} | ${r.p95Ms?.toFixed(3) ?? "N/A"} | ${r.opsPerSec?.toFixed(0) ?? "N/A"} | ${r.samples ?? 0} |`
      );
    }
    const dir = dirname(opts.markdownPath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    writeFileSync(opts.markdownPath, lines.join("\n") + "\n");
    console.log(`Markdown report written to ${opts.markdownPath}`);
  }
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
