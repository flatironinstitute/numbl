import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";
import {
  buildScenarios,
  runAllScenarios,
  type BenchBackend,
} from "./linalg-bench-core.js";

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type !== "run") return;
  const { warmup = 3, iterations = 10, time = 0 } = e.data;

  try {
    const backends: BenchBackend[] = [
      {
        id: "ts-lapack",
        label: "TypeScript LAPACK (pure JS)",
        bridge: getTsLapackBridge(),
      },
    ];

    // TODO: load WASM kernels here when available

    const scenarios = buildScenarios();

    const results = await runAllScenarios(scenarios, backends, {
      warmup,
      iterations,
      time,
      onProgress: (scenarioId, backendId) => {
        self.postMessage({ type: "progress", scenarioId, backendId });
      },
    });

    self.postMessage({ type: "result", data: results });
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
