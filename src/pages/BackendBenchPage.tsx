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
import {
  QUICK_BACKEND_BENCH_SCENARIO_IDS,
  buildBackendBenchScenarios,
  compareBenchValues,
  consumeBenchValue,
  percentileFromSamples,
  type BackendBenchScenario,
  type BenchCapability,
  type BenchKernelBridge,
  type BenchValue,
} from "../bench/backend-bench-core.js";
import {
  BROWSER_WASM_MANIFEST_PATH,
  BrowserWasmKernel,
  createBrowserWasmImportObject,
  resolveRuntimeAssetUrl,
  type BrowserWasmManifest,
  type BrowserWasmManifestTarget,
} from "../numbl-core/native/browser-wasm-kernel.js";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";

type BenchBackend = {
  id: string;
  label: string;
  type: "ts" | "wasm";
  capabilities: BenchCapability[];
  bridge: BenchKernelBridge;
};

type BenchRow = {
  scenarioId: string;
  scenarioLabel: string;
  backendId: string;
  status: "ok" | "error";
  reason?: string;
  medianMs?: number;
  p95Ms?: number;
  samples?: number;
  maxAbsDiff?: number;
  slowdownVsFastest?: number;
};

type BenchReport = {
  manifestTargets: string[];
  backends: string[];
  rows: BenchRow[];
  loadErrors: string[];
};

function formatMs(value: number | undefined): string {
  if (value === undefined) return "-";
  return value >= 100 ? value.toFixed(1) : value.toFixed(3);
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

function createWasmBridgeFromKernel(kernel: BrowserWasmKernel): BenchKernelBridge {
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
  return bridge;
}

async function loadBrowserWasmTarget(
  target: BrowserWasmManifestTarget
): Promise<BenchBackend> {
  const url = resolveRuntimeAssetUrl(target.wasmPath);
  if (url === null) {
    throw new Error(`${target.name}: runtime location is unavailable`);
  }
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${target.name}: failed to fetch ${url} (${response.status})`);
  }
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(
    bytes,
    createBrowserWasmImportObject()
  );
  const kernel = new BrowserWasmKernel(target, instance);
  const bridge = createWasmBridgeFromKernel(kernel);
  return {
    id: `wasm:${target.name}`,
    label: `wasm:${target.name}`,
    type: "wasm",
    capabilities: collectCapabilities(bridge),
    bridge,
  };
}

async function discoverBrowserBackends(): Promise<{
  manifestTargets: string[];
  backends: BenchBackend[];
  loadErrors: string[];
}> {
  const manifestUrl = resolveRuntimeAssetUrl(BROWSER_WASM_MANIFEST_PATH);
  if (manifestUrl === null) {
    return {
      manifestTargets: [],
      backends: [
        {
          id: "ts",
          label: "ts",
          type: "ts",
          capabilities: collectCapabilities(getTsLapackBridge()),
          bridge: getTsLapackBridge(),
        },
      ],
      loadErrors: ["runtime location is unavailable"],
    };
  }

  const response = await fetch(manifestUrl, { cache: "no-store" });
  const manifest = response.ok
    ? ((await response.json()) as BrowserWasmManifest)
    : { targets: [] };
  const targets = manifest.targets ?? [];
  const settled = await Promise.allSettled(targets.map(loadBrowserWasmTarget));
  const wasmBackends: BenchBackend[] = [];
  const loadErrors: string[] = [];

  for (const result of settled) {
    if (result.status === "fulfilled") {
      wasmBackends.push(result.value);
    } else {
      loadErrors.push(
        result.reason instanceof Error ? result.reason.message : String(result.reason)
      );
    }
  }

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
    loadErrors,
  };
}

function buildBrowserScenarios(): BackendBenchScenario[] {
  return buildBackendBenchScenarios().filter(scenario =>
    QUICK_BACKEND_BENCH_SCENARIO_IDS.has(scenario.id)
  );
}

async function runBrowserBench(
  warmupScale: number,
  iterationScale: number
): Promise<BenchReport> {
  const { manifestTargets, backends, loadErrors } = await discoverBrowserBackends();
  const scenarios = buildBrowserScenarios();
  const rows: BenchRow[] = [];

  for (const scenario of scenarios) {
    const supported = backends.filter(backend =>
      backend.capabilities.includes(scenario.capability)
    );
    if (supported.length === 0) {
      continue;
    }

    let reference: BenchValue | null = null;
    const validationResults = new Map<string, { maxAbsDiff: number }>();

    for (const backend of supported) {
      try {
        const output = scenario.execute(backend.bridge);
        consumeBenchValue(output);
        if (reference === null) {
          reference = output;
          validationResults.set(backend.id, { maxAbsDiff: 0 });
          continue;
        }
        validationResults.set(backend.id, {
          maxAbsDiff: compareBenchValues(reference, output, scenario.tolerance),
        });
      } catch (error) {
        rows.push({
          scenarioId: scenario.id,
          scenarioLabel: scenario.label,
          backendId: backend.id,
          status: "error",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    const timingBackends = supported.filter(backend =>
      validationResults.has(backend.id)
    );
    if (timingBackends.length === 0) {
      continue;
    }

    const bench = new Bench({
      name: scenario.id,
      time: 250 * iterationScale,
      warmup: true,
      warmupTime: 100 * warmupScale,
      iterations: Math.max(12, scenario.defaultIterations * iterationScale),
      warmupIterations: Math.max(6, scenario.defaultWarmup * warmupScale),
      retainSamples: true,
      throws: false,
    });

    for (const backend of timingBackends) {
      bench.add(backend.id, () => {
        consumeBenchValue(scenario.execute(backend.bridge));
      });
    }

    await bench.run();

    const scenarioRows: BenchRow[] = [];
    for (const backend of timingBackends) {
      const task = bench.tasks.find(candidate => candidate.name === backend.id);
      if (
        !task ||
        (task.result.state !== "completed" &&
          task.result.state !== "aborted-with-statistics")
      ) {
        rows.push({
          scenarioId: scenario.id,
          scenarioLabel: scenario.label,
          backendId: backend.id,
          status: "error",
          reason: task
            ? `tinybench state=${task.result.state}`
            : "missing tinybench task",
        });
        continue;
      }

      scenarioRows.push({
        scenarioId: scenario.id,
        scenarioLabel: scenario.label,
        backendId: backend.id,
        status: "ok",
        medianMs: task.result.latency.p50,
        p95Ms:
          percentileFromSamples(task.result.latency.samples, 0.95) ??
          task.result.latency.p99,
        samples: task.result.latency.samplesCount,
        maxAbsDiff: validationResults.get(backend.id)?.maxAbsDiff ?? 0,
      });
    }

    const fastest = Math.min(
      ...scenarioRows
        .map(row => row.medianMs)
        .filter((value): value is number => typeof value === "number")
    );
    for (const row of scenarioRows) {
      row.slowdownVsFastest =
        typeof row.medianMs === "number" ? row.medianMs / fastest : undefined;
      rows.push(row);
    }
  }

  rows.sort((a, b) => {
    if (a.scenarioLabel === b.scenarioLabel) {
      if (a.status !== b.status) return a.status === "ok" ? -1 : 1;
      return (a.medianMs ?? Number.POSITIVE_INFINITY) - (b.medianMs ?? Number.POSITIVE_INFINITY);
    }
    return a.scenarioLabel.localeCompare(b.scenarioLabel);
  });

  return {
    manifestTargets,
    backends: backends.map(backend => backend.id),
    rows,
    loadErrors,
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
      <Stack spacing={3} sx={{ maxWidth: 1280, mx: "auto" }}>
        <Paper sx={{ p: 3 }}>
          <Stack spacing={2}>
            <Typography variant="h4">Backend Bench</Typography>
            <Typography variant="body1" color="text.secondary">
              Measures the browser-side TS and Wasm backends directly in this
              tab using the same quick scenarios as the CLI harness, with
              validation before timing. Numeric checks are machine-epsilon-based
              and condition-aware for dense linalg.
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
            {error ? <Typography color="error">{error}</Typography> : null}
            {report ? (
              <>
                <Typography variant="body2" color="text.secondary">
                  Manifest targets:{" "}
                  {report.manifestTargets.length > 0
                    ? report.manifestTargets.join(", ")
                    : "(none)"}
                </Typography>
                {report.loadErrors.map(message => (
                  <Typography key={message} variant="body2" color="warning.main">
                    {message}
                  </Typography>
                ))}
              </>
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
                  <TableCell>Status</TableCell>
                  <TableCell align="right">Median ms</TableCell>
                  <TableCell align="right">P95 ms</TableCell>
                  <TableCell align="right">Samples</TableCell>
                  <TableCell align="right">Max Diff</TableCell>
                  <TableCell align="right">Slowdown</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {report.rows.map(row => (
                  <TableRow key={`${row.scenarioId}:${row.backendId}:${row.status}`}>
                    <TableCell>{row.scenarioLabel}</TableCell>
                    <TableCell>{row.backendId}</TableCell>
                    <TableCell>
                      {row.status === "ok" ? "ok" : row.reason ?? "error"}
                    </TableCell>
                    <TableCell align="right">{formatMs(row.medianMs)}</TableCell>
                    <TableCell align="right">{formatMs(row.p95Ms)}</TableCell>
                    <TableCell align="right">
                      {row.samples !== undefined ? row.samples : "-"}
                    </TableCell>
                    <TableCell align="right">
                      {row.maxAbsDiff !== undefined
                        ? row.maxAbsDiff.toExponential(3)
                        : "-"}
                    </TableCell>
                    <TableCell align="right">
                      {row.slowdownVsFastest !== undefined
                        ? `${row.slowdownVsFastest.toFixed(2)}x`
                        : "-"}
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
