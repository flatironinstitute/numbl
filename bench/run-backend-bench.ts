#!/usr/bin/env tsx

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Bench } from "tinybench";
import {
  QUICK_BACKEND_BENCH_SCENARIO_IDS,
  buildBackendBenchScenarios,
  compareBenchValues,
  consumeBenchValue,
  digestBenchValue,
  percentileFromSamples,
  type BackendBenchScenario,
  type BenchCapability,
  type BenchKernelBridge,
  type BenchValue,
} from "../src/bench/backend-bench-core.js";
import {
  NATIVE_ADDON_EXPECTED_VERSION,
} from "../src/numbl-core/native/lapack-bridge.js";
import {
  BrowserWasmKernel,
  createBrowserWasmImportObject,
  type BrowserWasmManifest,
} from "../src/numbl-core/native/browser-wasm-kernel.js";
import { getTsLapackBridge } from "../src/numbl-core/native/ts-lapack-bridge.js";

interface BenchmarkBackend {
  id: string;
  label: string;
  type: "ts" | "native" | "wasm";
  capabilities: BenchCapability[];
  details?: string;
  bridge: BenchKernelBridge;
}

interface BackendProbe {
  id: string;
  label: string;
  type: "ts" | "native" | "wasm";
  available: boolean;
  capabilities: BenchCapability[];
  details?: string;
  reason?: string;
  bridge?: BenchKernelBridge;
}

interface ScenarioRunResult {
  scenarioId: string;
  scenarioLabel: string;
  backendId: string;
  backendLabel: string;
  backendType: "ts" | "native" | "wasm";
  status: "ok" | "skipped" | "error";
  reason?: string;
  capability: BenchCapability;
  iterations?: number;
  warmup?: number;
  medianMs?: number;
  meanMs?: number;
  minMs?: number;
  maxMs?: number;
  p95Ms?: number;
  speedupVsTs?: number;
  referenceBackendId?: string;
  maxAbsDiff?: number;
  outputDigest?: string;
}

interface CliOptions {
  list: boolean;
  quick: boolean;
  verifyOnly: boolean;
  scenarioFilters: string[];
  backendFilters: string[];
  iterationsOverride?: number;
  warmupOverride?: number;
  outputPath?: string;
  markdownPath?: string;
}

interface OrderingCheck {
  scenarioId: string;
  scenarioLabel: string;
  checked: boolean;
  passed: boolean;
  reason?: string;
  observedOrder?: string[];
}

interface WasmComparisonEntry {
  backendId: string;
  medianMs: number;
  deltaMs: number;
  slowdownVsFastest: number;
}

interface WasmScenarioComparison {
  scenarioId: string;
  scenarioLabel: string;
  fastestBackendId: string;
  fastestMedianMs: number;
  backends: WasmComparisonEntry[];
}

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ADDON_PATH = join(REPO_ROOT, "build", "Release", "numbl_addon.node");
const WASM_MANIFEST_PATH = join(REPO_ROOT, "public", "wasm-kernels", "manifest.json");
const QUICK_BENCH_TIME_MS = 250;
const QUICK_WARMUP_TIME_MS = 100;
const DEFAULT_BENCH_TIME_MS = 750;
const DEFAULT_WARMUP_TIME_MS = 200;

let sinkValue = 0;

function usage(): string {
  return `
Usage:
  npm run bench:backends -- [options]

Options:
  --list                  List detected backends and available scenarios.
  --quick                 Run a reduced scenario set with fewer iterations.
  --verify-only           Validate backend outputs without timing loops.
  --scenario <csv>        Filter scenarios by id substring or exact id.
  --backend <csv>         Filter backends by id substring or exact id.
  --warmup <n>            Override warmup iteration count.
  --iterations <n>        Override measured iteration count.
  --output <path>         Write JSON results to a file.
  --markdown <path>       Write a markdown comparison report.
  --help                  Show this help text.

Examples:
  npm run bench:backends -- --list
  npm run bench:backends -- --quick
  npm run bench:backends -- --quick --verify-only
  npm run bench:backends -- --scenario matmul-256,inv-128
  npm run bench:backends -- --backend ts,native
  npm run bench:backends -- --output bench/results/latest.json
  npm run bench:backends -- --backend wasm:blas-lapack,wasm:flame-blas-lapack --markdown bench/results/wasm-report.md
`.trim();
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    list: false,
    quick: false,
    verifyOnly: false,
    scenarioFilters: [],
    backendFilters: [],
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case "--list":
        options.list = true;
        break;
      case "--quick":
        options.quick = true;
        break;
      case "--verify-only":
        options.verifyOnly = true;
        break;
      case "--scenario":
        i++;
        if (i >= argv.length) {
          throw new Error("--scenario requires a comma-separated value");
        }
        options.scenarioFilters.push(
          ...argv[i]
            .split(",")
            .map(part => part.trim())
            .filter(Boolean)
        );
        break;
      case "--backend":
        i++;
        if (i >= argv.length) {
          throw new Error("--backend requires a comma-separated value");
        }
        options.backendFilters.push(
          ...argv[i]
            .split(",")
            .map(part => part.trim())
            .filter(Boolean)
        );
        break;
      case "--iterations":
        i++;
        if (i >= argv.length) {
          throw new Error("--iterations requires a number");
        }
        options.iterationsOverride = parsePositiveInt(argv[i], "--iterations");
        break;
      case "--warmup":
        i++;
        if (i >= argv.length) {
          throw new Error("--warmup requires a number");
        }
        options.warmupOverride = parsePositiveInt(argv[i], "--warmup");
        break;
      case "--output":
        i++;
        if (i >= argv.length) {
          throw new Error("--output requires a file path");
        }
        options.outputPath = resolve(process.cwd(), argv[i]);
        break;
      case "--markdown":
        i++;
        if (i >= argv.length) {
          throw new Error("--markdown requires a file path");
        }
        options.markdownPath = resolve(process.cwd(), argv[i]);
        break;
      case "--help":
      case "-h":
        console.log(usage());
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }

  return options;
}

function parsePositiveInt(value: string, flag: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return parsed;
}

function formatMs(value: number | undefined): string {
  if (value === undefined) return "-";
  return value >= 100 ? value.toFixed(1) : value.toFixed(3);
}

function formatSpeedup(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "-";
  return `${value.toFixed(2)}x`;
}

function pad(value: string, width: number): string {
  if (value.length >= width) return value;
  return value + " ".repeat(width - value.length);
}

function describeCapabilities(capabilities: BenchCapability[]): string {
  return capabilities.length > 0 ? capabilities.join(", ") : "(none)";
}

function collectCapabilities(bridge: BenchKernelBridge): BenchCapability[] {
  const capabilities: BenchCapability[] = [];
  if (typeof bridge.matmul === "function") capabilities.push("matmul");
  if (typeof bridge.inv === "function") capabilities.push("inv");
  if (typeof bridge.linsolve === "function") capabilities.push("linsolve");
  if (typeof bridge.fft1dComplex === "function") capabilities.push("fft1dComplex");
  if (typeof bridge.fftAlongDim === "function") capabilities.push("fftAlongDim");
  return capabilities;
}

function loadTsBackend(): BackendProbe {
  const bridge = getTsLapackBridge();
  return {
    id: "ts",
    label: "ts-lapack",
    type: "ts",
    available: true,
    capabilities: collectCapabilities(bridge),
    details: "Pure TypeScript fallback bridge",
    bridge,
  };
}

function loadNativeBackend(): BackendProbe {
  if (process.env.NUMBL_NO_NATIVE === "1") {
    return {
      id: "native",
      label: "native-addon",
      type: "native",
      available: false,
      capabilities: [],
      reason: "disabled by NUMBL_NO_NATIVE=1",
    };
  }

  if (!existsSync(ADDON_PATH)) {
    return {
      id: "native",
      label: "native-addon",
      type: "native",
      available: false,
      capabilities: [],
      reason: `missing ${ADDON_PATH}`,
    };
  }

  try {
    const req = createRequire(import.meta.url);
    const addon = req(ADDON_PATH) as BenchKernelBridge & {
      addonVersion?: () => number;
      bridgeName?: string;
    };
    const addonVersion =
      typeof addon.addonVersion === "function" ? addon.addonVersion() : 0;
    if (addonVersion !== NATIVE_ADDON_EXPECTED_VERSION) {
      return {
        id: "native",
        label: "native-addon",
        type: "native",
        available: false,
        capabilities: [],
        reason: `version mismatch (${addonVersion} != ${NATIVE_ADDON_EXPECTED_VERSION})`,
      };
    }
    return {
      id: "native",
      label: addon.bridgeName ?? "native-addon",
      type: "native",
      available: true,
      capabilities: collectCapabilities(addon),
      details: ADDON_PATH,
      bridge: addon,
    };
  } catch (error) {
    return {
      id: "native",
      label: "native-addon",
      type: "native",
      available: false,
      capabilities: [],
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

function resolveManifestWasmPath(wasmPath: string): string {
  const relativePath = wasmPath.replace(/^\/+/, "");
  if (relativePath.startsWith("wasm-kernels/")) {
    return join(REPO_ROOT, "public", relativePath);
  }
  if (isAbsolute(wasmPath)) {
    return resolve(REPO_ROOT, relativePath);
  }
  return resolve(REPO_ROOT, wasmPath);
}

async function loadWasmBackends(): Promise<BackendProbe[]> {
  if (!existsSync(WASM_MANIFEST_PATH)) {
    return [
      {
        id: "wasm",
        label: "browser-wasm",
        type: "wasm",
        available: false,
        capabilities: [],
        reason: `missing ${WASM_MANIFEST_PATH}`,
      },
    ];
  }

  const manifest = JSON.parse(
    readFileSync(WASM_MANIFEST_PATH, "utf8")
  ) as BrowserWasmManifest;
  const targets = manifest.targets ?? [];
  if (targets.length === 0) {
    return [
      {
        id: "wasm",
        label: "browser-wasm",
        type: "wasm",
        available: false,
        capabilities: [],
        reason: "manifest has no targets",
      },
    ];
  }

  const results: BackendProbe[] = [];
  for (const target of targets) {
    const wasmFile = resolveManifestWasmPath(target.wasmPath);
    if (!existsSync(wasmFile)) {
      results.push({
        id: `wasm:${target.name}`,
        label: `wasm:${target.name}`,
        type: "wasm",
        available: false,
        capabilities: [],
        reason: `missing ${wasmFile}`,
      });
      continue;
    }
    try {
      const bytes = readFileSync(wasmFile);
      const { instance } = await WebAssembly.instantiate(
        bytes,
        createBrowserWasmImportObject()
      );
      const kernel = new BrowserWasmKernel(target, instance);
      const bridge: BenchKernelBridge = {};
      if (kernel.hasExport(["numbl_matmul_f64", "_numbl_matmul_f64"])) {
        bridge.matmul = kernel.matmul.bind(kernel);
      }
      if (kernel.hasExport(["numbl_inv_f64", "_numbl_inv_f64"])) {
        bridge.inv = kernel.inv.bind(kernel);
      }
      if (kernel.hasExport(["numbl_linsolve_f64", "_numbl_linsolve_f64"])) {
        bridge.linsolve = kernel.linsolve.bind(kernel);
      }
      if (kernel.hasExport(["numbl_fft1d_f64", "_numbl_fft1d_f64"])) {
        bridge.fft1dComplex = kernel.fft1dComplex.bind(kernel);
      }
      if (kernel.hasExport(["numbl_fft_along_dim_f64", "_numbl_fft_along_dim_f64"])) {
        bridge.fftAlongDim = kernel.fftAlongDim.bind(kernel);
      }
      results.push({
        id: `wasm:${target.name}`,
        label: `wasm:${target.name}`,
        type: "wasm",
        available: true,
        capabilities: collectCapabilities(bridge),
        details: wasmFile,
        bridge,
      });
    } catch (error) {
      results.push({
        id: `wasm:${target.name}`,
        label: `wasm:${target.name}`,
        type: "wasm",
        available: false,
        capabilities: [],
        reason: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return results;
}

async function discoverBackends(): Promise<BackendProbe[]> {
  return [loadTsBackend(), loadNativeBackend(), ...(await loadWasmBackends())];
}

function matchesFilter(id: string, filters: string[]): boolean {
  if (filters.length === 0) return true;
  return filters.some(filter => id === filter || id.includes(filter));
}

function filterScenarios(
  scenarios: BackendBenchScenario[],
  options: CliOptions
): BackendBenchScenario[] {
  let filtered = scenarios;
  if (options.quick) {
    filtered = filtered.filter(scenario =>
      QUICK_BACKEND_BENCH_SCENARIO_IDS.has(scenario.id)
    );
  }
  if (options.scenarioFilters.length > 0) {
    filtered = filtered.filter(scenario =>
      matchesFilter(scenario.id, options.scenarioFilters)
    );
  }
  return filtered;
}

function selectBackends(probes: BackendProbe[], options: CliOptions): BenchmarkBackend[] {
  return probes
    .filter(probe => probe.available)
    .filter(probe => matchesFilter(probe.id, options.backendFilters))
    .map(probe => ({
      id: probe.id,
      label: probe.label,
      type: probe.type,
      capabilities: probe.capabilities,
      details: probe.details,
      bridge: probe.bridge as BenchKernelBridge,
    }));
}

function listDetectedBackends(probes: BackendProbe[]): void {
  console.log("Detected backends:");
  for (const probe of probes) {
    const status = probe.available ? "available" : "unavailable";
    const suffix = probe.available
      ? describeCapabilities(probe.capabilities)
      : probe.reason ?? "unknown reason";
    console.log(`- ${probe.id} [${status}] ${suffix}`);
    if (probe.details) {
      console.log(`  ${probe.details}`);
    }
  }
}

function listScenarios(scenarios: BackendBenchScenario[]): void {
  console.log("\nScenarios:");
  for (const scenario of scenarios) {
    console.log(
      `- ${scenario.id} [${scenario.capability}] warmup=${scenario.defaultWarmup} iterations=${scenario.defaultIterations} :: ${scenario.label}`
    );
  }
}

async function runBenchmarks(
  backends: BenchmarkBackend[],
  scenarios: BackendBenchScenario[],
  options: CliOptions
): Promise<ScenarioRunResult[]> {
  const results: ScenarioRunResult[] = [];

  for (const scenario of scenarios) {
    const supportedBackends = backends.filter(backend =>
      backend.capabilities.includes(scenario.capability)
    );
    if (supportedBackends.length === 0) {
      results.push({
        scenarioId: scenario.id,
        scenarioLabel: scenario.label,
        backendId: "-",
        backendLabel: "-",
        backendType: "ts",
        status: "skipped",
        reason: "no selected backend supports this scenario",
        capability: scenario.capability,
      });
      continue;
    }

    let reference: BenchValue | null = null;
    let referenceBackendId: string | undefined;
    const validationResults = new Map<
      string,
      {
        outputDigest: string;
        maxAbsDiff: number;
      }
    >();

    for (const backend of supportedBackends) {
      try {
        const validationOutput = scenario.execute(backend.bridge);
        sinkValue += consumeBenchValue(validationOutput);
        let maxAbsDiff = 0;
        if (reference === null) {
          reference = validationOutput;
          referenceBackendId = backend.id;
        } else {
          maxAbsDiff = compareBenchValues(reference, validationOutput, scenario.tolerance);
        }
        validationResults.set(backend.id, {
          outputDigest: digestBenchValue(validationOutput),
          maxAbsDiff,
        });
        if (options.verifyOnly) {
          results.push({
            scenarioId: scenario.id,
            scenarioLabel: scenario.label,
            backendId: backend.id,
            backendLabel: backend.label,
            backendType: backend.type,
            capability: scenario.capability,
            status: "ok",
            iterations: 0,
            warmup: 0,
            referenceBackendId,
            maxAbsDiff,
            outputDigest: digestBenchValue(validationOutput),
          });
        }
      } catch (error) {
        results.push({
          scenarioId: scenario.id,
          scenarioLabel: scenario.label,
          backendId: backend.id,
          backendLabel: backend.label,
          backendType: backend.type,
          capability: scenario.capability,
          status: "error",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    if (options.verifyOnly) {
      continue;
    }

    const timingBackends = supportedBackends.filter(backend =>
      validationResults.has(backend.id)
    );
    const bench = new Bench({
      name: scenario.id,
      time: options.quick ? QUICK_BENCH_TIME_MS : DEFAULT_BENCH_TIME_MS,
      warmup: true,
      warmupTime: options.quick ? QUICK_WARMUP_TIME_MS : DEFAULT_WARMUP_TIME_MS,
      iterations:
        options.iterationsOverride ??
        (options.quick ? Math.max(8, scenario.defaultWarmup * 2) : scenario.defaultIterations),
      warmupIterations:
        options.warmupOverride ??
        (options.quick ? Math.max(4, scenario.defaultWarmup) : scenario.defaultWarmup),
      retainSamples: true,
      throws: false,
    });

    for (const backend of timingBackends) {
      bench.add(backend.id, () => {
        sinkValue += consumeBenchValue(scenario.execute(backend.bridge));
      });
    }

    await bench.run();

    for (const backend of timingBackends) {
      const task = bench.tasks.find(candidate => candidate.name === backend.id);
      const validation = validationResults.get(backend.id);
      if (!task || !validation) continue;
      const taskResult = task.result;
      if (
        taskResult.state !== "completed" &&
        taskResult.state !== "aborted-with-statistics"
      ) {
        results.push({
          scenarioId: scenario.id,
          scenarioLabel: scenario.label,
          backendId: backend.id,
          backendLabel: backend.label,
          backendType: backend.type,
          capability: scenario.capability,
          status: "error",
          reason:
            taskResult.state === "errored"
              ? taskResult.error.message
              : `tinybench state=${taskResult.state}`,
        });
        continue;
      }

      results.push({
        scenarioId: scenario.id,
        scenarioLabel: scenario.label,
        backendId: backend.id,
        backendLabel: backend.label,
        backendType: backend.type,
        capability: scenario.capability,
        status: "ok",
        iterations: taskResult.latency.samplesCount,
        warmup: bench.warmupIterations,
        medianMs: taskResult.latency.p50,
        meanMs: taskResult.latency.mean,
        minMs: taskResult.latency.min,
        maxMs: taskResult.latency.max,
        p95Ms: percentileFromSamples(taskResult.latency.samples, 0.95),
        referenceBackendId,
        maxAbsDiff: validation.maxAbsDiff,
        outputDigest: validation.outputDigest,
      });
    }
  }

  addSpeedupMetrics(results);
  return results;
}

function addSpeedupMetrics(results: ScenarioRunResult[]): void {
  const tsBaselines = new Map<string, number>();
  for (const result of results) {
    if (
      result.status === "ok" &&
      result.backendId === "ts" &&
      typeof result.medianMs === "number"
    ) {
      tsBaselines.set(result.scenarioId, result.medianMs);
    }
  }
  for (const result of results) {
    if (result.status !== "ok" || typeof result.medianMs !== "number") continue;
    const baseline = tsBaselines.get(result.scenarioId);
    if (baseline !== undefined) {
      result.speedupVsTs = baseline / result.medianMs;
    }
  }
}

function printResults(results: ScenarioRunResult[]): void {
  const grouped = new Map<string, ScenarioRunResult[]>();
  for (const result of results) {
    const group = grouped.get(result.scenarioId);
    if (group) {
      group.push(result);
    } else {
      grouped.set(result.scenarioId, [result]);
    }
  }

  console.log("");
  for (const [scenarioId, group] of grouped) {
    const label = group[0]?.scenarioLabel ?? scenarioId;
    console.log(label);
    console.log(
      [
        pad("backend", 20),
        pad("median ms", 10),
        pad("p95 ms", 10),
        pad("iters", 7),
        pad("speedup", 9),
        "status",
      ].join(" ")
    );
    const sorted = [...group].sort((a, b) => {
      if (a.status !== b.status) return a.status === "ok" ? -1 : 1;
      return (a.medianMs ?? Number.POSITIVE_INFINITY) - (b.medianMs ?? Number.POSITIVE_INFINITY);
    });
    for (const result of sorted) {
      const backendLabel = pad(result.backendId, 20);
      if (result.status === "ok") {
        console.log(
          [
            backendLabel,
            pad(formatMs(result.medianMs), 10),
            pad(formatMs(result.p95Ms), 10),
            pad(String(result.iterations ?? "-"), 7),
            pad(formatSpeedup(result.speedupVsTs), 9),
            "ok",
          ].join(" ")
        );
      } else {
        console.log(
          [
            backendLabel,
            pad("-", 10),
            pad("-", 10),
            pad("-", 7),
            pad("-", 9),
            `${result.status}${result.reason ? `: ${result.reason}` : ""}`,
          ].join(" ")
        );
      }
    }
    console.log("");
  }
}

function buildOrderingChecks(results: ScenarioRunResult[]): OrderingCheck[] {
  const grouped = new Map<string, ScenarioRunResult[]>();
  for (const result of results) {
    const group = grouped.get(result.scenarioId);
    if (group) {
      group.push(result);
    } else {
      grouped.set(result.scenarioId, [result]);
    }
  }

  const checks: OrderingCheck[] = [];
  for (const [scenarioId, group] of grouped) {
    const scenarioLabel = group[0]?.scenarioLabel ?? scenarioId;
    const relevant = group.filter(
      result =>
        result.status === "ok" &&
        typeof result.medianMs === "number" &&
        (result.backendId === "ts" ||
          result.backendId === "native" ||
          result.backendId.startsWith("wasm:"))
    );
    const ts = relevant.find(result => result.backendId === "ts");
    const native = relevant.find(result => result.backendId === "native");
    const wasm = relevant.find(result => result.backendId.startsWith("wasm:"));

    if (!ts || !native || !wasm) {
      checks.push({
        scenarioId,
        scenarioLabel,
        checked: false,
        passed: false,
        reason: "requires timed ts/native/wasm results",
      });
      continue;
    }

    const observedOrder = [...relevant]
      .sort((a, b) => (a.medianMs ?? 0) - (b.medianMs ?? 0))
      .map(result => result.backendId);
    const passed =
      (native.medianMs ?? Number.POSITIVE_INFINITY) <=
        (wasm.medianMs ?? Number.POSITIVE_INFINITY) &&
      (wasm.medianMs ?? Number.POSITIVE_INFINITY) <=
        (ts.medianMs ?? Number.POSITIVE_INFINITY);

    checks.push({
      scenarioId,
      scenarioLabel,
      checked: true,
      passed,
      observedOrder,
      ...(passed
        ? {}
        : {
            reason: `expected native <= ${wasm.backendId} <= ts but got ${observedOrder.join(
              " < "
            )}`,
          }),
    });
  }

  return checks;
}

function printOrderingChecks(checks: OrderingCheck[]): void {
  const checked = checks.filter(check => check.checked);
  if (checked.length === 0) {
    return;
  }

  console.log("Observed dense-linalg ordering (advisory)");
  for (const check of checked) {
    const status = check.passed ? "ok" : "unexpected";
    const detail = check.observedOrder?.join(" < ") ?? check.reason ?? "-";
    console.log(`- ${check.scenarioLabel}: ${status} (${detail})`);
  }
  console.log("");
}

function buildWasmComparisons(results: ScenarioRunResult[]): WasmScenarioComparison[] {
  const grouped = new Map<string, ScenarioRunResult[]>();
  for (const result of results) {
    const group = grouped.get(result.scenarioId);
    if (group) {
      group.push(result);
    } else {
      grouped.set(result.scenarioId, [result]);
    }
  }

  const comparisons: WasmScenarioComparison[] = [];
  for (const [scenarioId, group] of grouped) {
    const wasmResults = group
      .filter(
        result =>
          result.status === "ok" &&
          result.backendId.startsWith("wasm:") &&
          typeof result.medianMs === "number"
      )
      .sort((a, b) => (a.medianMs ?? 0) - (b.medianMs ?? 0));
    if (wasmResults.length < 2) {
      continue;
    }

    const fastest = wasmResults[0];
    const fastestMedianMs = fastest.medianMs as number;
    comparisons.push({
      scenarioId,
      scenarioLabel: fastest.scenarioLabel,
      fastestBackendId: fastest.backendId,
      fastestMedianMs,
      backends: wasmResults.map(result => ({
        backendId: result.backendId,
        medianMs: result.medianMs as number,
        deltaMs: (result.medianMs as number) - fastestMedianMs,
        slowdownVsFastest: (result.medianMs as number) / fastestMedianMs,
      })),
    });
  }

  return comparisons;
}

function printWasmComparisons(comparisons: WasmScenarioComparison[]): void {
  if (comparisons.length === 0) {
    return;
  }

  console.log("Wasm backend comparisons");
  for (const comparison of comparisons) {
    console.log(comparison.scenarioLabel);
    console.log(
      [
        pad("backend", 24),
        pad("median ms", 10),
        pad("delta ms", 10),
        "slowdown",
      ].join(" ")
    );
    for (const entry of comparison.backends) {
      console.log(
        [
          pad(entry.backendId, 24),
          pad(formatMs(entry.medianMs), 10),
          pad(formatMs(entry.deltaMs), 10),
          `${entry.slowdownVsFastest.toFixed(2)}x`,
        ].join(" ")
      );
    }
    console.log("");
  }
}

function writeResultsFile(
  outputPath: string,
  probes: BackendProbe[],
  scenarios: BackendBenchScenario[],
  results: ScenarioRunResult[],
  options: CliOptions,
  orderingChecks: OrderingCheck[],
  wasmComparisons: WasmScenarioComparison[]
): void {
  mkdirSync(dirname(outputPath), { recursive: true });
  const payload = {
    generatedAt: new Date().toISOString(),
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    cwd: process.cwd(),
    repoRoot: REPO_ROOT,
    quick: options.quick,
    verifyOnly: options.verifyOnly,
    selectedScenarios: scenarios.map(scenario => scenario.id),
    selectedBackends: probes
      .filter(probe => probe.available)
      .filter(probe => matchesFilter(probe.id, options.backendFilters))
      .map(probe => probe.id),
    detectedBackends: probes.map(probe => ({
      id: probe.id,
      label: probe.label,
      type: probe.type,
      available: probe.available,
      capabilities: probe.capabilities,
      details: probe.details,
      reason: probe.reason,
    })),
    results,
    orderingChecks,
    wasmComparisons,
    sinkValue,
  };
  writeFileSync(outputPath, JSON.stringify(payload, null, 2) + "\n");
  console.log(`Wrote benchmark results to ${outputPath}`);
}

function writeMarkdownReport(
  outputPath: string,
  probes: BackendProbe[],
  scenarios: BackendBenchScenario[],
  results: ScenarioRunResult[],
  orderingChecks: OrderingCheck[],
  wasmComparisons: WasmScenarioComparison[]
): void {
  mkdirSync(dirname(outputPath), { recursive: true });
  const lines: string[] = [];
  lines.push("# Numbl Backend Benchmark Report");
  lines.push("");
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push(`Node: ${process.version}`);
  lines.push(`Platform: ${process.platform} ${process.arch}`);
  lines.push("");
  lines.push("## Detected Backends");
  lines.push("");
  for (const probe of probes) {
    const status = probe.available ? "available" : "unavailable";
    const detail = probe.available
      ? describeCapabilities(probe.capabilities)
      : probe.reason ?? "unknown";
    lines.push(`- \`${probe.id}\` (${status}): ${detail}`);
  }
  lines.push("");
  lines.push("## Scenarios");
  lines.push("");
  lines.push(scenarios.map(scenario => `- \`${scenario.id}\`: ${scenario.label}`).join("\n"));
  lines.push("");
  if (wasmComparisons.length > 0) {
    lines.push("## Wasm Comparisons");
    lines.push("");
    for (const comparison of wasmComparisons) {
      lines.push(`### ${comparison.scenarioLabel}`);
      lines.push("");
      lines.push("| backend | median ms | delta ms | slowdown vs fastest |");
      lines.push("| --- | ---: | ---: | ---: |");
      for (const entry of comparison.backends) {
        lines.push(
          `| \`${entry.backendId}\` | ${formatMs(entry.medianMs)} | ${formatMs(entry.deltaMs)} | ${entry.slowdownVsFastest.toFixed(2)}x |`
        );
      }
      lines.push("");
    }
  }
  lines.push("## Ordering Checks (Advisory)");
  lines.push("");
  for (const check of orderingChecks) {
    const label = check.checked ? (check.passed ? "ok" : "unexpected") : "not-checked";
    const detail = check.observedOrder?.join(" < ") ?? check.reason ?? "-";
    lines.push(`- ${check.scenarioLabel}: ${label} (${detail})`);
  }
  lines.push("");
  lines.push("## Raw Results");
  lines.push("");
  lines.push("| scenario | backend | median ms | p95 ms | speedup vs ts | status |");
  lines.push("| --- | --- | ---: | ---: | ---: | --- |");
  for (const result of results) {
    lines.push(
      `| ${result.scenarioId} | \`${result.backendId}\` | ${formatMs(result.medianMs)} | ${formatMs(result.p95Ms)} | ${formatSpeedup(result.speedupVsTs)} | ${result.status}${result.reason ? `: ${result.reason}` : ""} |`
    );
  }
  lines.push("");

  writeFileSync(outputPath, lines.join("\n") + "\n");
  console.log(`Wrote markdown benchmark report to ${outputPath}`);
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const probes = await discoverBackends();
  const scenarios = filterScenarios(buildBackendBenchScenarios(), options);

  if (options.list) {
    listDetectedBackends(probes);
    listScenarios(scenarios);
    return;
  }

  const backends = selectBackends(probes, options);
  if (backends.length === 0) {
    listDetectedBackends(probes);
    throw new Error("No matching available backends were found.");
  }
  if (scenarios.length === 0) {
    throw new Error("No benchmark scenarios matched the current filters.");
  }

  console.log(`Repo root: ${REPO_ROOT}`);
  console.log(
    `Running ${scenarios.length} scenario(s) across ${backends.length} backend(s).`
  );
  if (options.verifyOnly) {
    console.log("Mode: verify-only");
  }
  console.log(
    `Backends: ${backends.map(backend => backend.id).join(", ")}`
  );

  const results = await runBenchmarks(backends, scenarios, options);
  const orderingChecks = buildOrderingChecks(results);
  const wasmComparisons = buildWasmComparisons(results);
  printResults(results);
  printOrderingChecks(orderingChecks);
  printWasmComparisons(wasmComparisons);

  if (options.outputPath) {
    writeResultsFile(
      options.outputPath,
      probes,
      scenarios,
      results,
      options,
      orderingChecks,
      wasmComparisons
    );
  }
  if (options.markdownPath) {
    writeMarkdownReport(
      options.markdownPath,
      probes,
      scenarios,
      results,
      orderingChecks,
      wasmComparisons
    );
  }

  console.log(`Benchmark sink: ${sinkValue.toFixed(6)}`);
}

main().catch(error => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
