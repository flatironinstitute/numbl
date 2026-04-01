import { Box, Typography, CircularProgress, Link } from "@mui/material";
import { makeShareHash } from "../utils/shareUrl.js";
import { useEffect, useRef, useState, useCallback } from "react";
import type { PlotInstruction } from "../graphics/types.js";
import {
  figuresReducer,
  initialFiguresState,
  type FiguresState,
} from "../graphics/figuresReducer.js";
import { FigureView } from "../graphics/FigureView.js";

// Import all example .m files as raw text
const exampleModules = import.meta.glob("../../examples/plots/*.m", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

interface GalleryEntry {
  name: string;
  code: string;
}

const entries: GalleryEntry[] = Object.entries(exampleModules)
  .map(([path, code]) => {
    const filename = path.split("/").pop()!;
    const name = filename.replace(/_example\.m$/, "").replace(/\.m$/, "");
    return { name, code };
  })
  .sort((a, b) => a.name.localeCompare(b.name));

function GalleryItem({ entry }: { entry: GalleryEntry }) {
  const [figures, setFigures] = useState<FiguresState>(initialFiguresState);
  const [status, setStatus] = useState<
    "pending" | "running" | "done" | "error"
  >("running");
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  // Use a ref-based reducer to collect plot instructions
  const figuresRef = useRef<FiguresState>(initialFiguresState);
  const dispatchRef = useCallback((action: PlotInstruction) => {
    figuresRef.current = figuresReducer(figuresRef.current, action);
  }, []);

  useEffect(() => {
    const worker = new Worker(new URL("../numbl-worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;

    worker.onmessage = e => {
      const msg = e.data;
      if (msg.type === "drawnow") {
        if (msg.plotInstructions?.length) {
          for (const instr of msg.plotInstructions as PlotInstruction[]) {
            dispatchRef(instr);
          }
          setFigures({ ...figuresRef.current });
        }
      } else if (msg.type === "done") {
        if (msg.plotInstructions?.length) {
          for (const instr of msg.plotInstructions as PlotInstruction[]) {
            dispatchRef(instr);
          }
        }
        setFigures({ ...figuresRef.current });
        setStatus("done");
        worker.terminate();
      } else if (msg.type === "error") {
        setError(msg.message);
        setStatus("error");
        worker.terminate();
      }
    };

    worker.postMessage({
      type: "run",
      code: entry.code,
      options: {
        displayResults: false,
        maxIterations: 10000000,
        optimization: 1,
      },
      workspaceFiles: [],
      mainFileName: `${entry.name}.m`,
    });

    return () => {
      worker.terminate();
    };
  }, [entry, dispatchRef]);

  const handles = Object.keys(figures.figs)
    .map(Number)
    .sort((a, b) => a - b);
  const currentFig =
    handles.length > 0 ? figures.figs[handles[handles.length - 1]] : null;

  return (
    <Box
      sx={{
        border: 1,
        borderColor: "divider",
        borderRadius: 1,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Box
        sx={{
          width: "100%",
          height: 300,
          bgcolor: "#fff",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {status === "running" && <CircularProgress size={24} />}
        {status === "error" && (
          <Typography color="error" variant="caption" sx={{ p: 1 }}>
            {error}
          </Typography>
        )}
        {currentFig && <FigureView figure={currentFig} />}
      </Box>
      <Box
        sx={{
          px: 1.5,
          py: 1,
          bgcolor: "grey.50",
          borderTop: 1,
          borderColor: "divider",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <Typography variant="subtitle2" sx={{ fontFamily: "monospace" }}>
          {entry.name}
        </Typography>
        <Link
          href={`/share#${makeShareHash(entry.name, entry.code)}`}
          target="_blank"
          rel="noopener"
          sx={{ fontSize: "0.8125rem" }}
        >
          Open
        </Link>
      </Box>
    </Box>
  );
}

export function PlotGalleryPage() {
  return (
    <Box sx={{ maxWidth: 1200, mx: "auto", p: 3 }}>
      <Typography variant="h5" gutterBottom>
        numbl plot gallery
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Each plot is generated live by executing numbl scripts in your browser.
      </Typography>
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(350px, 1fr))",
          gap: 2,
        }}
      >
        {entries.map(entry => (
          <GalleryItem key={entry.name} entry={entry} />
        ))}
      </Box>
    </Box>
  );
}
