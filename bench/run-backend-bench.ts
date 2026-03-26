#!/usr/bin/env tsx

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Bench } from "tinybench";
import {
  NATIVE_ADDON_EXPECTED_VERSION,
  type LapackBridge,
} from "../src/numbl-core/native/lapack-bridge.js";
import { getTsLapackBridge } from "../src/numbl-core/native/ts-lapack-bridge.js";

type Capability =
  | "matmul"
  | "inv"
  | "linsolve"
  | "fft1dComplex"
  | "fftAlongDim";

type ComplexArray = {
  re: Float64Array;
  im: Float64Array;
};

type BenchValue = Float64Array | ComplexArray;

type KernelBridge = Pick<
  LapackBridge,
  "matmul" | "inv" | "linsolve" | "fft1dComplex" | "fftAlongDim"
>;

interface BenchmarkBackend {
  id: string;
  label: string;
  type: "ts" | "native" | "wasm";
  capabilities: Capability[];
  details?: string;
  bridge: KernelBridge;
}

interface BackendProbe {
  id: string;
  label: string;
  type: "ts" | "native" | "wasm";
  available: boolean;
  capabilities: Capability[];
  details?: string;
  reason?: string;
  bridge?: KernelBridge;
}

interface BenchmarkScenario {
  id: string;
  label: string;
  capability: Capability;
  defaultWarmup: number;
  defaultIterations: number;
  tolerance: number;
  execute: (backend: KernelBridge) => BenchValue;
}

interface ScenarioRunResult {
  scenarioId: string;
  scenarioLabel: string;
  backendId: string;
  backendLabel: string;
  backendType: "ts" | "native" | "wasm";
  status: "ok" | "skipped" | "error";
  reason?: string;
  capability: Capability;
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

interface BrowserWasmManifestTarget {
  name: string;
  wasmPath: string;
  exports?: string[];
}

interface BrowserWasmManifest {
  generatedAt?: string;
  targets?: BrowserWasmManifestTarget[];
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
const QUICK_SCENARIO_IDS = new Set([
  "matmul-128",
  "inv-64",
  "linsolve-128",
  "fft1d-4096",
  "fftAlongDim-1024x64-d0",
]);
const QUICK_BENCH_TIME_MS = 150;
const QUICK_WARMUP_TIME_MS = 50;
const DEFAULT_BENCH_TIME_MS = 400;
const DEFAULT_WARMUP_TIME_MS = 120;

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

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeDenseMatrix(m: number, n: number, seed: number): Float64Array {
  const rand = mulberry32(seed);
  const out = new Float64Array(m * n);
  for (let col = 0; col < n; col++) {
    for (let row = 0; row < m; row++) {
      const idx = row + col * m;
      const base = Math.sin((row + 1) * 0.31) + Math.cos((col + 1) * 0.17);
      out[idx] = base + (rand() - 0.5) * 0.2;
    }
  }
  return out;
}

function makeDiagonallyDominantSquare(n: number, seed: number): Float64Array {
  const rand = mulberry32(seed);
  const out = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    for (let row = 0; row < n; row++) {
      const idx = row + col * n;
      const noise = (rand() - 0.5) * 0.25;
      out[idx] = row === col ? n + 1 + noise : noise;
    }
  }
  return out;
}

function makeComplexSignal(length: number, seed: number): ComplexArray {
  const rand = mulberry32(seed);
  const re = new Float64Array(length);
  const im = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    const x = (2 * Math.PI * i) / length;
    re[i] = Math.sin(3 * x) + 0.25 * Math.cos(11 * x) + (rand() - 0.5) * 0.01;
    im[i] = Math.cos(5 * x) - 0.2 * Math.sin(7 * x) + (rand() - 0.5) * 0.01;
  }
  return { re, im };
}

function makeComplexTensor(shape: number[], seed: number): ComplexArray {
  const length = shape.reduce((acc, dim) => acc * dim, 1);
  return makeComplexSignal(length, seed);
}

function percentile(sortedValues: number[], fraction: number): number {
  if (sortedValues.length === 0) return 0;
  const index = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.ceil(sortedValues.length * fraction) - 1)
  );
  return sortedValues[index];
}

function percentileFromSamples(
  samples: readonly number[] | undefined,
  fraction: number
): number | undefined {
  if (!samples || samples.length === 0) return undefined;
  return percentile([...samples].sort((a, b) => a - b), fraction);
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

function describeCapabilities(capabilities: Capability[]): string {
  return capabilities.length > 0 ? capabilities.join(", ") : "(none)";
}

function isComplexValue(value: BenchValue): value is ComplexArray {
  return (
    typeof value === "object" &&
    "re" in value &&
    value.re instanceof Float64Array &&
    "im" in value &&
    value.im instanceof Float64Array
  );
}

function digestBenchValue(value: BenchValue): string {
  if (isComplexValue(value)) {
    return [
      `re:${digestFloat64Array(value.re)}`,
      `im:${digestFloat64Array(value.im)}`,
    ].join(" | ");
  }
  return digestFloat64Array(value);
}

function digestFloat64Array(values: Float64Array): string {
  if (values.length === 0) return "len=0";
  const sample = [0, Math.floor(values.length / 2), values.length - 1]
    .filter((index, i, all) => all.indexOf(index) === i)
    .map(index => values[index].toFixed(6))
    .join(", ");
  let checksum = 0;
  for (let i = 0; i < values.length; i += Math.max(1, Math.floor(values.length / 16))) {
    checksum += values[i];
  }
  return `len=${values.length} sample=[${sample}] checksum=${checksum.toFixed(6)}`;
}

function consumeBenchValue(value: BenchValue): number {
  if (isComplexValue(value)) {
    return consumeFloat64Array(value.re) + consumeFloat64Array(value.im);
  }
  return consumeFloat64Array(value);
}

function consumeFloat64Array(values: Float64Array): number {
  if (values.length === 0) return 0;
  const mid = Math.floor(values.length / 2);
  return values[0] + values[mid] + values[values.length - 1];
}

function compareBenchValues(
  reference: BenchValue,
  actual: BenchValue,
  tolerance: number
): number {
  if (isComplexValue(reference) !== isComplexValue(actual)) {
    throw new Error("benchmark output shape mismatch");
  }
  if (isComplexValue(reference) && isComplexValue(actual)) {
    return Math.max(
      compareFloat64Arrays(reference.re, actual.re, tolerance),
      compareFloat64Arrays(reference.im, actual.im, tolerance)
    );
  }
  if (reference instanceof Float64Array && actual instanceof Float64Array) {
    return compareFloat64Arrays(reference, actual, tolerance);
  }
  throw new Error("unsupported benchmark value type");
}

function compareFloat64Arrays(
  reference: Float64Array,
  actual: Float64Array,
  tolerance: number
): number {
  if (reference.length !== actual.length) {
    throw new Error(
      `benchmark output length mismatch (${reference.length} vs ${actual.length})`
    );
  }
  let maxAbsDiff = 0;
  for (let i = 0; i < reference.length; i++) {
    const absDiff = Math.abs(reference[i] - actual[i]);
    if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
    const scale = Math.max(1, Math.abs(reference[i]), Math.abs(actual[i]));
    if (absDiff > tolerance * scale) {
      throw new Error(
        `benchmark validation failed at index ${i}: diff=${absDiff} tolerance=${tolerance}`
      );
    }
  }
  return maxAbsDiff;
}

function collectCapabilities(bridge: KernelBridge): Capability[] {
  const capabilities: Capability[] = [];
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
    const addon = req(ADDON_PATH) as KernelBridge & {
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

function createImportObject(): WebAssembly.Imports {
  return {
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
    env: {
      emscripten_notify_memory_growth: () => {},
    },
  };
}

function byteLengthFloat64(length: number): number {
  return length * Float64Array.BYTES_PER_ELEMENT;
}

function byteLengthInt32(length: number): number {
  return length * Int32Array.BYTES_PER_ELEMENT;
}

function outSizeAlongDim(shape: number[], dim: number, n: number): number {
  let strideDim = 1;
  for (let i = 0; i < dim; i++) strideDim *= shape[i];
  let numAbove = 1;
  for (let i = dim + 1; i < shape.length; i++) numAbove *= shape[i];
  return strideDim * n * numAbove;
}

class WasmTargetBridge implements KernelBridge {
  readonly bridgeName: string;
  private readonly exports: Record<string, unknown>;
  private readonly memory: WebAssembly.Memory;
  private readonly mallocFn: (bytes: number) => number;
  private readonly freeFn: (ptr: number) => void;

  constructor(
    readonly targetName: string,
    instance: WebAssembly.Instance
  ) {
    this.bridgeName = `wasm:${targetName}`;
    this.exports = instance.exports as Record<string, unknown>;
    const initialize = this.exports["_initialize"];
    if (typeof initialize === "function") {
      (initialize as () => void)();
    }
    this.memory = this.requireExport<WebAssembly.Memory>(["memory"]);
    this.mallocFn = this.requireExport<(bytes: number) => number>([
      "malloc",
      "_malloc",
    ]);
    this.freeFn = this.requireExport<(ptr: number) => void>(["free", "_free"]);
  }

  hasExport(names: string[]): boolean {
    return names.some(name => typeof this.exports[name] === "function");
  }

  private requireExport<T>(names: string[]): T {
    for (const name of names) {
      const value = this.exports[name];
      if (value !== undefined) return value as T;
    }
    throw new Error(`${this.bridgeName}: missing export (${names.join(" or ")})`);
  }

  private allocBytes(bytes: number): number {
    return this.mallocFn(bytes);
  }

  private freeBytes(ptr: number): void {
    if (ptr !== 0) this.freeFn(ptr);
  }

  private writeF64(ptr: number, data: Float64Array): void {
    new Float64Array(this.memory.buffer, ptr, data.length).set(data);
  }

  private writeI32(ptr: number, data: Int32Array): void {
    new Int32Array(this.memory.buffer, ptr, data.length).set(data);
  }

  private readF64(ptr: number, length: number): Float64Array {
    return new Float64Array(new Float64Array(this.memory.buffer, ptr, length));
  }

  private withInputF64<T>(data: Float64Array, fn: (ptr: number) => T): T {
    const ptr = this.allocBytes(byteLengthFloat64(data.length));
    try {
      this.writeF64(ptr, data);
      return fn(ptr);
    } finally {
      this.freeBytes(ptr);
    }
  }

  private withOptionalInputF64<T>(
    data: Float64Array | null,
    fn: (ptr: number) => T
  ): T {
    if (data === null) return fn(0);
    return this.withInputF64(data, fn);
  }

  private withInputI32<T>(data: Int32Array, fn: (ptr: number) => T): T {
    const ptr = this.allocBytes(byteLengthInt32(data.length));
    try {
      this.writeI32(ptr, data);
      return fn(ptr);
    } finally {
      this.freeBytes(ptr);
    }
  }

  private callStatus(fn: (...args: number[]) => number, args: number[]): void {
    const status = fn(...args);
    if (status !== 0) {
      throw new Error(`${this.bridgeName}: wasm kernel returned status ${status}`);
    }
  }

  matmul(
    A: Float64Array,
    m: number,
    k: number,
    B: Float64Array,
    n: number
  ): Float64Array {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_matmul_f64",
      "_numbl_matmul_f64",
    ]);
    const outPtr = this.allocBytes(byteLengthFloat64(m * n));
    try {
      this.withInputF64(A, aPtr =>
        this.withInputF64(B, bPtr => {
          this.callStatus(fn, [aPtr, m, k, bPtr, n, outPtr]);
        })
      );
      return this.readF64(outPtr, m * n);
    } finally {
      this.freeBytes(outPtr);
    }
  }

  inv(data: Float64Array, n: number): Float64Array {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_inv_f64",
      "_numbl_inv_f64",
    ]);
    const outPtr = this.allocBytes(byteLengthFloat64(n * n));
    try {
      this.withInputF64(data, dataPtr => {
        this.callStatus(fn, [dataPtr, n, outPtr]);
      });
      return this.readF64(outPtr, n * n);
    } finally {
      this.freeBytes(outPtr);
    }
  }

  linsolve(
    A: Float64Array,
    m: number,
    n: number,
    B: Float64Array,
    nrhs: number
  ): Float64Array {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_linsolve_f64",
      "_numbl_linsolve_f64",
    ]);
    const outPtr = this.allocBytes(byteLengthFloat64(n * nrhs));
    try {
      this.withInputF64(A, aPtr =>
        this.withInputF64(B, bPtr => {
          this.callStatus(fn, [aPtr, m, n, bPtr, nrhs, outPtr]);
        })
      );
      return this.readF64(outPtr, n * nrhs);
    } finally {
      this.freeBytes(outPtr);
    }
  }

  fft1dComplex(
    re: Float64Array,
    im: Float64Array,
    n: number,
    inverse: boolean
  ): ComplexArray {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_fft1d_f64",
      "_numbl_fft1d_f64",
    ]);
    const outRePtr = this.allocBytes(byteLengthFloat64(n));
    const outImPtr = this.allocBytes(byteLengthFloat64(n));
    try {
      this.withInputF64(re, rePtr =>
        this.withInputF64(im, imPtr => {
          this.callStatus(fn, [
            rePtr,
            imPtr,
            n,
            inverse ? 1 : 0,
            outRePtr,
            outImPtr,
          ]);
        })
      );
      return {
        re: this.readF64(outRePtr, n),
        im: this.readF64(outImPtr, n),
      };
    } finally {
      this.freeBytes(outRePtr);
      this.freeBytes(outImPtr);
    }
  }

  fftAlongDim(
    re: Float64Array,
    im: Float64Array | null,
    shape: number[],
    dim: number,
    n: number,
    inverse: boolean
  ): ComplexArray {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_fft_along_dim_f64",
      "_numbl_fft_along_dim_f64",
    ]);
    const shapeI32 = Int32Array.from(shape);
    const outLen = outSizeAlongDim(shape, dim, n);
    const outRePtr = this.allocBytes(byteLengthFloat64(outLen));
    const outImPtr = this.allocBytes(byteLengthFloat64(outLen));
    try {
      this.withInputF64(re, rePtr =>
        this.withOptionalInputF64(im, imPtr =>
          this.withInputI32(shapeI32, shapePtr => {
            this.callStatus(fn, [
              rePtr,
              imPtr,
              shapePtr,
              shape.length,
              dim,
              n,
              inverse ? 1 : 0,
              outRePtr,
              outImPtr,
            ]);
          })
        )
      );
      return {
        re: this.readF64(outRePtr, outLen),
        im: this.readF64(outImPtr, outLen),
      };
    } finally {
      this.freeBytes(outRePtr);
      this.freeBytes(outImPtr);
    }
  }
}

function resolveManifestWasmPath(wasmPath: string): string {
  if (isAbsolute(wasmPath)) {
    const relativePath = wasmPath.replace(/^\/+/, "");
    if (relativePath.startsWith("wasm-kernels/")) {
      return join(REPO_ROOT, "public", relativePath);
    }
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
      const { instance } = await WebAssembly.instantiate(bytes, createImportObject());
      const kernel = new WasmTargetBridge(target.name, instance);
      const bridge: KernelBridge = {};
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

function buildScenarios(): BenchmarkScenario[] {
  return [
    createMatmulScenario(128, 128, 128, 4, 60),
    createMatmulScenario(256, 256, 256, 3, 18),
    createMatmulScenario(512, 512, 512, 2, 5),
    createInvScenario(64, 3, 24),
    createInvScenario(128, 2, 10),
    createInvScenario(256, 1, 4),
    createLinsolveScenario(128, 4, 3, 20),
    createLinsolveScenario(256, 4, 2, 8),
    createFft1dScenario(4096, 4, 40),
    createFft1dScenario(16384, 2, 12),
    createFftAlongDimScenario([1024, 64], 0, 1024, 2, 8),
  ];
}

function createMatmulScenario(
  m: number,
  k: number,
  n: number,
  warmup: number,
  iterations: number
): BenchmarkScenario {
  const A = makeDenseMatrix(m, k, m * 101 + k * 17 + n * 7);
  const B = makeDenseMatrix(k, n, m * 37 + k * 13 + n * 29);
  return {
    id: `matmul-${m}`,
    label: `matmul ${m}x${k} * ${k}x${n}`,
    capability: "matmul",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: 1e-10,
    execute: backend => {
      if (typeof backend.matmul !== "function") {
        throw new Error("backend does not support matmul");
      }
      return backend.matmul(A, m, k, B, n);
    },
  };
}

function createInvScenario(
  n: number,
  warmup: number,
  iterations: number
): BenchmarkScenario {
  const A = makeDiagonallyDominantSquare(n, n * 97 + 11);
  return {
    id: `inv-${n}`,
    label: `inv ${n}x${n}`,
    capability: "inv",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: 1e-9,
    execute: backend => {
      if (typeof backend.inv !== "function") {
        throw new Error("backend does not support inv");
      }
      return backend.inv(A, n);
    },
  };
}

function createLinsolveScenario(
  n: number,
  nrhs: number,
  warmup: number,
  iterations: number
): BenchmarkScenario {
  const A = makeDiagonallyDominantSquare(n, n * 71 + nrhs * 13);
  const B = makeDenseMatrix(n, nrhs, n * 53 + nrhs * 19);
  return {
    id: `linsolve-${n}`,
    label: `linsolve ${n}x${n} rhs=${nrhs}`,
    capability: "linsolve",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: 1e-9,
    execute: backend => {
      if (typeof backend.linsolve !== "function") {
        throw new Error("backend does not support linsolve");
      }
      return backend.linsolve(A, n, n, B, nrhs);
    },
  };
}

function createFft1dScenario(
  n: number,
  warmup: number,
  iterations: number
): BenchmarkScenario {
  const signal = makeComplexSignal(n, n * 19 + 5);
  return {
    id: `fft1d-${n}`,
    label: `fft1d complex n=${n}`,
    capability: "fft1dComplex",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: 1e-8,
    execute: backend => {
      if (typeof backend.fft1dComplex !== "function") {
        throw new Error("backend does not support fft1dComplex");
      }
      return backend.fft1dComplex(signal.re, signal.im, n, false);
    },
  };
}

function createFftAlongDimScenario(
  shape: number[],
  dim: number,
  n: number,
  warmup: number,
  iterations: number
): BenchmarkScenario {
  const tensor = makeComplexTensor(shape, shape.reduce((acc, value) => acc * 31 + value, 17));
  const shapeLabel = `${shape.join("x")}-d${dim}`;
  return {
    id: `fftAlongDim-${shapeLabel}`,
    label: `fftAlongDim shape=${shape.join("x")} dim=${dim} n=${n}`,
    capability: "fftAlongDim",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: 1e-8,
    execute: backend => {
      if (typeof backend.fftAlongDim !== "function") {
        throw new Error("backend does not support fftAlongDim");
      }
      return backend.fftAlongDim(tensor.re, tensor.im, shape, dim, n, false);
    },
  };
}

function matchesFilter(id: string, filters: string[]): boolean {
  if (filters.length === 0) return true;
  return filters.some(filter => id === filter || id.includes(filter));
}

function filterScenarios(
  scenarios: BenchmarkScenario[],
  options: CliOptions
): BenchmarkScenario[] {
  let filtered = scenarios;
  if (options.quick) {
    filtered = filtered.filter(scenario => QUICK_SCENARIO_IDS.has(scenario.id));
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
      bridge: probe.bridge as KernelBridge,
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

function listScenarios(scenarios: BenchmarkScenario[]): void {
  console.log("\nScenarios:");
  for (const scenario of scenarios) {
    console.log(
      `- ${scenario.id} [${scenario.capability}] warmup=${scenario.defaultWarmup} iterations=${scenario.defaultIterations} :: ${scenario.label}`
    );
  }
}

async function runBenchmarks(
  backends: BenchmarkBackend[],
  scenarios: BenchmarkScenario[],
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

  console.log("Expected ordering (native <= wasm <= ts)");
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
  scenarios: BenchmarkScenario[],
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
  scenarios: BenchmarkScenario[],
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
  lines.push("## Ordering Checks");
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
  const scenarios = filterScenarios(buildScenarios(), options);

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
