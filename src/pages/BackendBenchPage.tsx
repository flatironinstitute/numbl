import { startTransition, useState } from "react";
import {
  Box,
  Button,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { Bench } from "tinybench";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";

type Capability = "matmul" | "inv" | "linsolve" | "fft1dComplex";

type ComplexArray = {
  re: Float64Array;
  im: Float64Array;
};

type BenchValue = Float64Array | ComplexArray;

type KernelBridge = {
  matmul?: (
    A: Float64Array,
    m: number,
    k: number,
    B: Float64Array,
    n: number
  ) => Float64Array;
  inv?: (data: Float64Array, n: number) => Float64Array;
  linsolve?: (
    A: Float64Array,
    m: number,
    n: number,
    B: Float64Array,
    nrhs: number
  ) => Float64Array;
  fft1dComplex?: (
    re: Float64Array,
    im: Float64Array,
    n: number,
    inverse: boolean
  ) => ComplexArray;
};

type BrowserWasmManifestTarget = {
  name: string;
  wasmPath: string;
};

type BrowserWasmManifest = {
  targets?: BrowserWasmManifestTarget[];
};

type BenchBackend = {
  id: string;
  label: string;
  type: "ts" | "wasm";
  capabilities: Capability[];
  bridge: KernelBridge;
};

type BenchScenario = {
  id: string;
  label: string;
  capability: Capability;
  warmup: number;
  iterations: number;
  execute: (backend: KernelBridge) => BenchValue;
};

type BenchRow = {
  scenarioId: string;
  scenarioLabel: string;
  backendId: string;
  medianMs: number;
  p95Ms: number;
  slowdownVsFastest: number;
};

type BenchReport = {
  manifestTargets: string[];
  backends: string[];
  rows: BenchRow[];
};

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
      out[idx] =
        Math.sin((row + 1) * 0.31) +
        Math.cos((col + 1) * 0.17) +
        (rand() - 0.5) * 0.2;
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

function consumeBenchValue(value: BenchValue): number {
  if ("re" in value && "im" in value) {
    return value.re[0] + value.im[0];
  }
  return value[0] ?? 0;
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

class BrowserWasmBenchKernel implements KernelBridge {
  readonly bridgeName: string;
  private readonly exports: Record<string, unknown>;
  private readonly memory: WebAssembly.Memory;
  private readonly mallocFn: (bytes: number) => number;
  private readonly freeFn: (ptr: number) => void;

  constructor(targetName: string, instance: WebAssembly.Instance) {
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
}

function collectCapabilities(bridge: KernelBridge): Capability[] {
  const capabilities: Capability[] = [];
  if (typeof bridge.matmul === "function") capabilities.push("matmul");
  if (typeof bridge.inv === "function") capabilities.push("inv");
  if (typeof bridge.linsolve === "function") capabilities.push("linsolve");
  if (typeof bridge.fft1dComplex === "function") capabilities.push("fft1dComplex");
  return capabilities;
}

async function discoverBrowserBackends(): Promise<{
  manifestTargets: string[];
  backends: BenchBackend[];
}> {
  const response = await fetch("/wasm-kernels/manifest.json", {
    cache: "no-store",
  });
  const manifest = response.ok
    ? ((await response.json()) as BrowserWasmManifest)
    : { targets: [] };
  const targets = manifest.targets ?? [];
  const wasmBackends = await Promise.all(
    targets.map(async target => {
      const targetResponse = await fetch(target.wasmPath, { cache: "no-store" });
      if (!targetResponse.ok) {
        throw new Error(`${target.name}: failed to fetch ${target.wasmPath}`);
      }
      const bytes = await targetResponse.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(bytes, createImportObject());
      const kernel = new BrowserWasmBenchKernel(target.name, instance);
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
      return {
        id: `wasm:${target.name}`,
        label: `wasm:${target.name}`,
        type: "wasm" as const,
        capabilities: collectCapabilities(bridge),
        bridge,
      };
    })
  );

  const tsBridge = getTsLapackBridge();
  return {
    manifestTargets: targets.map(target => target.name),
    backends: [
      {
        id: "ts",
        label: "ts",
        type: "ts",
        capabilities: collectCapabilities(tsBridge),
        bridge: tsBridge,
      },
      ...wasmBackends,
    ],
  };
}

function buildScenarios(): BenchScenario[] {
  const matmulA = makeDenseMatrix(128, 128, 17);
  const matmulB = makeDenseMatrix(128, 128, 19);
  const invA = makeDiagonallyDominantSquare(64, 23);
  const solveA = makeDiagonallyDominantSquare(128, 29);
  const solveB = makeDenseMatrix(128, 4, 31);
  const fftSignal = makeComplexSignal(4096, 37);

  return [
    {
      id: "matmul-128",
      label: "matmul 128x128",
      capability: "matmul",
      warmup: 1,
      iterations: 5,
      execute: backend => backend.matmul!(matmulA, 128, 128, matmulB, 128),
    },
    {
      id: "inv-64",
      label: "inv 64x64",
      capability: "inv",
      warmup: 1,
      iterations: 5,
      execute: backend => backend.inv!(invA, 64),
    },
    {
      id: "linsolve-128",
      label: "linsolve 128x128 rhs=4",
      capability: "linsolve",
      warmup: 1,
      iterations: 5,
      execute: backend => backend.linsolve!(solveA, 128, 128, solveB, 4),
    },
    {
      id: "fft1d-4096",
      label: "fft1d complex n=4096",
      capability: "fft1dComplex",
      warmup: 1,
      iterations: 5,
      execute: backend =>
        backend.fft1dComplex!(fftSignal.re, fftSignal.im, 4096, false),
    },
  ];
}

function formatMs(value: number): string {
  return value >= 100 ? value.toFixed(1) : value.toFixed(3);
}

async function runBrowserBench(
  warmupScale: number,
  iterationScale: number
): Promise<BenchReport> {
  const { manifestTargets, backends } = await discoverBrowserBackends();
  const scenarios = buildScenarios();
  const rows: BenchRow[] = [];

  for (const scenario of scenarios) {
    const supported = backends.filter(backend =>
      backend.capabilities.includes(scenario.capability)
    );
    if (supported.length === 0) {
      continue;
    }

    const bench = new Bench({
      name: scenario.id,
      time: 150 * iterationScale,
      warmup: true,
      warmupTime: 50 * warmupScale,
      iterations: Math.max(8, scenario.iterations * iterationScale),
      warmupIterations: Math.max(4, scenario.warmup * warmupScale),
      retainSamples: true,
      throws: false,
    });

    for (const backend of supported) {
      bench.add(backend.id, () => {
        consumeBenchValue(scenario.execute(backend.bridge));
      });
    }

    await bench.run();

    const scenarioRows: BenchRow[] = [];
    for (const backend of supported) {
      const task = bench.tasks.find(candidate => candidate.name === backend.id);
      if (
        !task ||
        (task.result.state !== "completed" &&
          task.result.state !== "aborted-with-statistics")
      ) {
        continue;
      }

      scenarioRows.push({
        scenarioId: scenario.id,
        scenarioLabel: scenario.label,
        backendId: backend.id,
        medianMs: task.result.latency.p50,
        p95Ms:
          percentileFromSamples(task.result.latency.samples, 0.95) ??
          task.result.latency.p99,
        slowdownVsFastest: 1,
      });
    }

    const fastest = Math.min(...scenarioRows.map(row => row.medianMs));
    for (const row of scenarioRows) {
      row.slowdownVsFastest = row.medianMs / fastest;
      rows.push(row);
    }
  }

  rows.sort((a, b) => {
    if (a.scenarioLabel === b.scenarioLabel) {
      return a.medianMs - b.medianMs;
    }
    return a.scenarioLabel.localeCompare(b.scenarioLabel);
  });

  return {
    manifestTargets,
    backends: backends.map(backend => backend.id),
    rows,
  };
}

export function BackendBenchPage() {
  const [running, setRunning] = useState(false);
  const [warmupScale, setWarmupScale] = useState("1");
  const [iterationScale, setIterationScale] = useState("1");
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<BenchReport | null>(null);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const nextReport = await runBrowserBench(
        Math.max(1, Number.parseInt(warmupScale, 10) || 1),
        Math.max(1, Number.parseInt(iterationScale, 10) || 1)
      );
      startTransition(() => {
        setReport(nextReport);
      });
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
    } finally {
      setRunning(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background:
          "linear-gradient(180deg, #f6f1e8 0%, #efe7d8 50%, #e6dcc9 100%)",
        px: { xs: 2, md: 4 },
        py: { xs: 3, md: 5 },
      }}
    >
      <Stack spacing={3} sx={{ maxWidth: 1200, mx: "auto" }}>
        <Paper sx={{ p: 3 }}>
          <Stack spacing={2}>
            <Typography variant="h4">Backend Bench</Typography>
            <Typography variant="body1" color="text.secondary">
              Measures the browser-side TS and Wasm backends directly in this
              tab. Build multiple Wasm targets first if you want side-by-side
              comparisons, for example `ducc0-fft`, `blas-lapack`, and
              `flame-blas-lapack`.
            </Typography>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <TextField
                label="Warmup Scale"
                value={warmupScale}
                onChange={event => setWarmupScale(event.target.value)}
                size="small"
              />
              <TextField
                label="Iteration Scale"
                value={iterationScale}
                onChange={event => setIterationScale(event.target.value)}
                size="small"
              />
              <Button variant="contained" onClick={handleRun} disabled={running}>
                {running ? "Running..." : "Run Benchmarks"}
              </Button>
            </Stack>
            {error ? (
              <Typography color="error">{error}</Typography>
            ) : null}
            {report ? (
              <Typography variant="body2" color="text.secondary">
                Manifest targets:{" "}
                {report.manifestTargets.length > 0
                  ? report.manifestTargets.join(", ")
                  : "(none)"}
              </Typography>
            ) : null}
          </Stack>
        </Paper>

        {report ? (
          <TableContainer component={Paper}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Scenario</TableCell>
                  <TableCell>Backend</TableCell>
                  <TableCell align="right">Median ms</TableCell>
                  <TableCell align="right">P95 ms</TableCell>
                  <TableCell align="right">Slowdown</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {report.rows.map(row => (
                  <TableRow key={`${row.scenarioId}:${row.backendId}`}>
                    <TableCell>{row.scenarioLabel}</TableCell>
                    <TableCell>{row.backendId}</TableCell>
                    <TableCell align="right">{formatMs(row.medianMs)}</TableCell>
                    <TableCell align="right">{formatMs(row.p95Ms)}</TableCell>
                    <TableCell align="right">
                      {row.slowdownVsFastest.toFixed(2)}x
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : null}
      </Stack>
    </Box>
  );
}
