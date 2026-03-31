import { discoverBackends } from "./matmul-bench-backends.js";
import { buildScenarios, runBenchmarks } from "./matmul-bench-core.js";

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type !== "run") return;
  const { warmup = 3, iterations = 10 } = e.data;

  try {
    const backends = await discoverBackends();
    const scenarios = buildScenarios();

    const results = await runBenchmarks(scenarios, backends, {
      warmup,
      iterations,
      onProgress: msg => self.postMessage({ type: "progress", msg }),
    });

    self.postMessage({
      type: "result",
      data: results,
      backends: backends.map(b => ({ id: b.id, label: b.label })),
      scenarios: scenarios.map(s => ({ id: s.id, label: s.label })),
    });
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
