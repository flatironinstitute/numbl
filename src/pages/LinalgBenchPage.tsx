import { useState, useRef, useCallback } from "react";
import {
  Box,
  Button,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import type { BenchTimingResult } from "../bench/linalg-bench-core.js";

function formatMs(v: number): string {
  if (v === 0) return "-";
  return v >= 100 ? v.toFixed(1) : v.toFixed(3);
}

export function LinalgBenchPage() {
  const [results, setResults] = useState<BenchTimingResult[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [warmup, setWarmup] = useState(3);
  const [iterations, setIterations] = useState(10);
  const workerRef = useRef<Worker | null>(null);

  const run = useCallback(() => {
    setRunning(true);
    setResults([]);
    setProgress("Starting...");

    const worker = new Worker(
      new URL("../bench/linalg-bench-worker.ts", import.meta.url),
      { type: "module" }
    );
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "progress") {
        setProgress(`${msg.scenarioId} [${msg.backendId}]`);
      } else if (msg.type === "result") {
        setResults(msg.data);
        setRunning(false);
        setProgress("");
        worker.terminate();
      } else if (msg.type === "error") {
        setProgress(`Error: ${msg.message}`);
        setRunning(false);
        worker.terminate();
      }
    };

    worker.onerror = e => {
      setProgress(`Worker error: ${e.message}`);
      setRunning(false);
    };

    worker.postMessage({ type: "run", warmup, iterations });
  }, [warmup, iterations]);

  const stop = useCallback(() => {
    workerRef.current?.terminate();
    setRunning(false);
    setProgress("Cancelled");
  }, []);

  // Group results by scenario, columns = backends
  const backendIds = [...new Set(results.map(r => r.backendId))];
  const byScenario = new Map<string, Map<string, BenchTimingResult>>();
  for (const r of results) {
    if (!byScenario.has(r.scenarioId)) byScenario.set(r.scenarioId, new Map());
    byScenario.get(r.scenarioId)!.set(r.backendId, r);
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Linear Algebra Benchmarks
      </Typography>

      <Box sx={{ display: "flex", gap: 2, mb: 2, alignItems: "center" }}>
        <TextField
          label="Warmup"
          type="number"
          size="small"
          value={warmup}
          onChange={e => setWarmup(Number(e.target.value))}
          sx={{ width: 100 }}
          disabled={running}
        />
        <TextField
          label="Iterations"
          type="number"
          size="small"
          value={iterations}
          onChange={e => setIterations(Number(e.target.value))}
          sx={{ width: 100 }}
          disabled={running}
        />
        {running ? (
          <Button variant="outlined" color="error" onClick={stop}>
            Stop
          </Button>
        ) : (
          <Button variant="contained" onClick={run}>
            Run Benchmarks
          </Button>
        )}
        {running && <CircularProgress size={20} />}
        {progress && (
          <Typography variant="body2" color="text.secondary">
            {progress}
          </Typography>
        )}
      </Box>

      {results.length > 0 && (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>
                  <strong>Scenario</strong>
                </TableCell>
                {backendIds.map(bid => (
                  <TableCell key={bid} align="right">
                    <strong>{bid} (ms)</strong>
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {[...byScenario.entries()].map(([scenarioId, backendMap]) => {
                const first = backendMap.values().next().value!;
                return (
                  <TableRow key={scenarioId}>
                    <TableCell>{first.scenarioLabel}</TableCell>
                    {backendIds.map(bid => {
                      const r = backendMap.get(bid);
                      let cell = "-";
                      if (r?.error) cell = `ERR: ${r.error.slice(0, 30)}`;
                      else if (r) cell = formatMs(r.medianMs);
                      return (
                        <TableCell key={bid} align="right">
                          {cell}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
}
