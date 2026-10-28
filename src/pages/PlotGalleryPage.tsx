import {
  Box,
  Typography,
  CircularProgress,
  Link,
  Collapse,
  IconButton,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import { makeShareHash } from "../utils/shareUrl.js";
import { usePageMeta } from "../hooks/usePageMeta";
import { useEffect, useState } from "react";
import {
  figuresReducer,
  initialFiguresState,
  type FiguresState,
} from "../numbl-core/plotting/figuresReducer.js";
import { FigureView } from "../graphics/FigureView.js";
import { useNumblWorker } from "../hooks/useNumblWorker.js";

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
  const worker = useNumblWorker(
    {
      onPlotInstructions: instructions =>
        setFigures(prev => instructions.reduce(figuresReducer, prev)),
      onDone: () => {
        setStatus("done");
        worker.terminate();
      },
      onError: msg => {
        setError(msg.message);
        setStatus("error");
        worker.terminate();
      },
    },
    { interactiveInput: false, uihtml: false }
  );

  useEffect(() => {
    worker.run({
      code: entry.code,
      options: {
        displayResults: false,
        maxIterations: 10000000,
        optimization: "1",
      },
      workspaceFiles: [],
      mainFileName: `${entry.name}.m`,
    });
  }, [entry, worker]);

  const handles = Object.keys(figures.figs)
    .map(Number)
    .sort((a, b) => a - b);
  const currentFig =
    handles.length > 0 ? figures.figs[handles[handles.length - 1]] : null;

  const [expanded, setExpanded] = useState(false);

  return (
    <Box
      data-testid="gallery-item"
      data-name={entry.name}
      data-status={status}
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
          py: 0.5,
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
        <IconButton
          size="small"
          onClick={() => setExpanded(prev => !prev)}
          sx={{ p: 0.25 }}
        >
          {expanded ? (
            <ExpandLessIcon fontSize="small" />
          ) : (
            <ExpandMoreIcon fontSize="small" />
          )}
        </IconButton>
      </Box>
      <Collapse in={expanded}>
        <Box
          sx={{
            px: 1.5,
            py: 1,
            bgcolor: "#1e1e1e",
            fontFamily: "monospace",
            fontSize: 12,
            color: "#d4d4d4",
            whiteSpace: "pre",
            overflowX: "auto",
            maxHeight: 200,
            overflowY: "auto",
            borderTop: 1,
            borderColor: "divider",
          }}
        >
          {entry.code}
        </Box>
        <Box
          sx={{
            px: 1.5,
            py: 0.75,
            bgcolor: "grey.50",
            borderTop: 1,
            borderColor: "divider",
            textAlign: "right",
          }}
        >
          <Link
            href={`/share#${makeShareHash(entry.name, entry.code)}`}
            sx={{ fontSize: "0.8125rem" }}
          >
            Open in editor
          </Link>
        </Box>
      </Collapse>
    </Box>
  );
}

export function PlotGalleryPage() {
  usePageMeta({
    title: "Plot Gallery - numbl",
    description:
      "A gallery of example plots created with numbl, each with its source code: line plots, surfaces, images, and 3-D visualizations in the browser.",
    path: "/gallery",
  });
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
