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
import type { MatmulResult } from "../bench/matmul-bench-core.js";

function fmtMs(v: number): string {
  if (v === 0) return "-";
  if (v >= 1000) return (v / 1000).toFixed(2) + "s";
  if (v >= 100) return v.toFixed(1);
  if (v >= 1) return v.toFixed(3);
  return (v * 1000).toFixed(1) + "µs";
}

type BackendInfo = { id: string; label: string };
type ScenarioInfo = { id: string; label: string };

export function MatmulBenchPage() {
  const [results, setResults] = useState<MatmulResult[]>([]);
  const [backends, setBackends] = useState<BackendInfo[]>([]);
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [warmup, setWarmup] = useState(3);
  const [iterations, setIterations] = useState(10);
  const workerRef = useRef<Worker | null>(null);

  const run = useCallback(() => {
    setRunning(true);
    setResults([]);
    setProgress("Discovering backends...");

    const worker = new Worker(
      new URL("../bench/matmul-bench-worker.ts", import.meta.url),
      { type: "module" }
    );
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "progress") setProgress(msg.msg);
      else if (msg.type === "result") {
        setResults(msg.data);
        setBackends(msg.backends);
        setScenarios(msg.scenarios);
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

  const bids = backends.map(b => b.id);
  const byScenario = new Map<string, Map<string, MatmulResult>>();
  for (const r of results) {
    if (!byScenario.has(r.scenarioId)) byScenario.set(r.scenarioId, new Map());
    byScenario.get(r.scenarioId)!.set(r.backendId, r);
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1400, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Matmul (dgemm) Benchmark
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Compares pure TypeScript, FLAME-TS blocked, and WASM SIMD backends. WASM
        also enables future multi-threading via SharedArrayBuffer.
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
            Run
          </Button>
        )}
        {running && <CircularProgress size={20} />}
        {progress && (
          <Typography variant="body2" color="text.secondary">
            {progress}
          </Typography>
        )}
      </Box>

      {backends.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2">Backends discovered:</Typography>
          {backends.map(b => (
            <Typography key={b.id} variant="body2" sx={{ ml: 2 }}>
              {b.id}: {b.label}
            </Typography>
          ))}
        </Box>
      )}

      {results.length > 0 && (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>
                  <strong>Size</strong>
                </TableCell>
                {bids.map(bid => (
                  <TableCell key={bid} align="right">
                    <strong>{bid}</strong>
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {scenarios.map(s => {
                const bmap = byScenario.get(s.id);
                return (
                  <TableRow key={s.id}>
                    <TableCell>{s.label}</TableCell>
                    {bids.map(bid => {
                      const r = bmap?.get(bid);
                      let cell = "-";
                      if (r?.error) cell = "ERR";
                      else if (r) cell = fmtMs(r.medianMs);
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
