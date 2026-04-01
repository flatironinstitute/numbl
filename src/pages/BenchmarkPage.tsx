import { Box, Typography, LinearProgress, Button } from "@mui/material";
import { useEffect, useRef, useState, useCallback } from "react";

const DEFAULT_BENCHMARKS_URL =
  "https://magland.github.io/numbl-benchmarks/benchmarks";

interface BenchResult {
  name: string;
  output: string;
  status: "pending" | "running" | "done" | "error";
}

function parseOutput(
  output: string
): { median: number; times: number[] } | null {
  const medianMatch = output.match(/median=([\d.]+)/);
  const timesMatch = output.match(/times=\[([\d., ]+)\]/);
  if (!medianMatch || !timesMatch) return null;
  return {
    median: parseFloat(medianMatch[1]),
    times: timesMatch[1].split(",").map(s => parseFloat(s.trim())),
  };
}

function getBenchmarksUrl(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get("url") ?? DEFAULT_BENCHMARKS_URL;
}

function cacheBust(url: string): string {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}_=${Date.now()}`;
}

async function fetchBenchmarks(
  baseUrl: string
): Promise<{ name: string; code: string }[]> {
  const manifest = await fetch(cacheBust(`${baseUrl}/benchmarks.json`));
  if (!manifest.ok)
    throw new Error(`Failed to fetch benchmarks.json from ${baseUrl}`);
  const names: string[] = await manifest.json();
  const benchmarks = await Promise.all(
    names.map(async name => {
      const resp = await fetch(cacheBust(`${baseUrl}/${name}.m`));
      if (!resp.ok) throw new Error(`Failed to fetch ${name}.m`);
      const code = await resp.text();
      return { name, code };
    })
  );
  return benchmarks;
}

export function BenchmarkPage() {
  const [results, setResults] = useState<BenchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const benchmarksRef = useRef<{ name: string; code: string }[]>([]);
  const runningRef = useRef(false);

  const allDone =
    results.length > 0 &&
    results.every(r => r.status === "done" || r.status === "error");

  const handleCopyJson = useCallback(() => {
    const entries = results
      .filter(r => r.status === "done")
      .map(r => {
        const parsed = parseOutput(r.output);
        if (!parsed)
          return {
            benchmark: r.name,
            env: "numbl-browser",
            skipped: "parse error",
          };
        return { benchmark: r.name, env: "numbl-browser", result: parsed };
      });
    const json = JSON.stringify(
      { timestamp: new Date().toISOString(), results: entries },
      null,
      2
    );
    navigator.clipboard.writeText(json);
  }, [results]);

  const runBenchmark = useCallback((index: number): Promise<void> => {
    return new Promise(resolve => {
      const bench = benchmarksRef.current[index];
      setResults(prev =>
        prev.map((r, i) => (i === index ? { ...r, status: "running" } : r))
      );

      const worker = new Worker(
        new URL("../numbl-worker.ts", import.meta.url),
        { type: "module" }
      );

      let output = "";

      worker.onmessage = e => {
        const msg = e.data;
        if (msg.type === "output") {
          output += msg.text;
        } else if (msg.type === "done") {
          setResults(prev =>
            prev.map((r, i) =>
              i === index ? { ...r, output, status: "done" } : r
            )
          );
          worker.terminate();
          resolve();
        } else if (msg.type === "error") {
          setResults(prev =>
            prev.map((r, i) =>
              i === index ? { ...r, output: msg.message, status: "error" } : r
            )
          );
          worker.terminate();
          resolve();
        }
      };

      worker.postMessage({
        type: "run",
        code: bench.code,
        options: {
          displayResults: false,
          maxIterations: 100000000,
          optimization: 1,
        },
        workspaceFiles: [],
        mainFileName: "bench.m",
      });
    });
  }, []);

  useEffect(() => {
    if (runningRef.current) return;
    runningRef.current = true;

    const baseUrl = getBenchmarksUrl();

    (async () => {
      try {
        const benchmarks = await fetchBenchmarks(baseUrl);
        benchmarksRef.current = benchmarks;
        setResults(
          benchmarks.map(b => ({
            name: b.name,
            output: "",
            status: "pending",
          }))
        );
        setLoading(false);

        for (let i = 0; i < benchmarks.length; i++) {
          await runBenchmark(i);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      }
    })();
  }, [runBenchmark]);

  return (
    <Box sx={{ maxWidth: 800, mx: "auto", p: 3 }}>
      <Typography variant="h5" gutterBottom>
        numbl browser benchmarks
      </Typography>

      {allDone && (
        <Button
          variant="outlined"
          size="small"
          onClick={handleCopyJson}
          sx={{ mb: 2 }}
        >
          Copy results as JSON
        </Button>
      )}

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {error && (
        <Typography color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}

      {results.map(r => (
        <Box
          key={r.name}
          sx={{
            mb: 2,
            p: 2,
            border: 1,
            borderColor: "divider",
            borderRadius: 1,
          }}
        >
          <Typography variant="subtitle2">{r.name}</Typography>
          {r.status === "running" && <LinearProgress sx={{ my: 1 }} />}
          {r.status === "pending" && (
            <Typography variant="body2" color="text.secondary">
              Waiting...
            </Typography>
          )}
          {(r.status === "done" || r.status === "error") && (
            <Typography
              variant="body2"
              sx={{
                fontFamily: "monospace",
                whiteSpace: "pre-wrap",
                color: r.status === "error" ? "error.main" : "text.primary",
              }}
            >
              {r.output}
            </Typography>
          )}
        </Box>
      ))}
    </Box>
  );
}
