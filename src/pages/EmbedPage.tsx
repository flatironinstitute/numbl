import { createInputSAB, mainThreadRespond } from "../syncInputChannel";
import Editor, { OnMount } from "@monaco-editor/react";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import {
  Box,
  Button,
  CircularProgress,
  Link,
  Tab,
  Tabs,
  Typography,
} from "@mui/material";
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import {
  numblLanguageConfig,
  createNumblTokensProvider,
} from "../numblLanguage.js";
import { formatDiagnostic } from "../numbl-core/diagnostics";
import { useMipVfsFiles } from "../hooks/useMipVfsFiles.js";
import type { PlotInstruction } from "../graphics/types.js";
import { FigureView } from "../graphics/FigureView.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../graphics/figuresReducer.js";

const DEFAULT_SCRIPT = `% Welcome to numbl!
% Try running some MATLAB code:

x = 1:10;
y = x.^2;
disp('Hello from numbl!')
disp(['Sum of squares: ', num2str(sum(y))])
`;

function getQueryParams(): {
  script: string;
  optimization: number;
} {
  const params = new URLSearchParams(window.location.search);
  const optimization = parseInt(params.get("opt") ?? "1", 10);
  const scriptParam = params.get("script");
  let script = DEFAULT_SCRIPT;
  if (scriptParam) {
    try {
      script = atob(scriptParam);
    } catch {
      console.error("Failed to decode script parameter");
    }
  }
  return { script, optimization };
}

export function EmbedPage() {
  const { script: initialScript, optimization } = getQueryParams();
  const mipFiles = useMipVfsFiles();
  const [code, setCode] = useState<string>(initialScript);
  const [output, setOutput] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [outputTab, setOutputTab] = useState(0); // 0=Console, 1=Figure
  const workerRef = useRef<Worker | null>(null);
  const inputSAB = useRef<SharedArrayBuffer | null>(createInputSAB());
  const [figures, figuresDispatch] = useReducer(
    figuresReducer,
    initialFiguresState
  );

  const handlePlotInstruction = useCallback((instruction: PlotInstruction) => {
    figuresDispatch(instruction);

    // Switch to figure tab when plot data arrives
    if (
      instruction.type === "plot" ||
      instruction.type === "plot3" ||
      instruction.type === "surf"
    ) {
      setOutputTab(1);
    }
  }, []);

  const currentFig = useMemo(() => {
    const handles = Object.keys(figures.figs)
      .map(Number)
      .sort((a, b) => a - b);
    if (handles.length === 0) return null;
    const handle = handles[handles.length - 1];
    return figures.figs[handle];
  }, [figures.figs]);

  const hasFigures = currentFig !== null;

  const setupWorkerHandler = useCallback(
    (worker: Worker) => {
      worker.onmessage = e => {
        const msg = e.data;
        if (msg.type === "request-input") {
          const response = prompt(msg.prompt ?? "") ?? "";
          const sab = inputSAB.current;
          if (sab) mainThreadRespond(sab, response);
          return;
        }
        if (msg.type === "output") {
          setOutput(prev => prev + msg.text);
        } else if (msg.type === "drawnow") {
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions) {
              handlePlotInstruction(instr);
            }
          }
        } else if (msg.type === "done") {
          setIsRunning(false);
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions) {
              handlePlotInstruction(instr);
            }
          }
        } else if (msg.type === "error") {
          setIsRunning(false);
          setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
        }
      };
    },
    [handlePlotInstruction]
  );

  // Initialize worker
  useEffect(() => {
    workerRef.current = new Worker(
      new URL("../numbl-worker.ts", import.meta.url),
      { type: "module" }
    );
    setupWorkerHandler(workerRef.current);

    return () => {
      workerRef.current?.terminate();
    };
  }, [setupWorkerHandler]);

  const handleRun = useCallback(async () => {
    if (!workerRef.current) return;
    setIsRunning(true);
    setOutput("");
    figuresDispatch({ type: "clear" });

    workerRef.current.postMessage({
      type: "run",
      code,
      options: {
        displayResults: true,
        maxIterations: 10000000,
        optimization,
      },
      workspaceFiles: mipFiles.workspaceFiles,
      vfsFiles: mipFiles.vfsFiles,
      mainFileName: "script.m",
      inputSAB: inputSAB.current ?? undefined,
    });
  }, [code, optimization, mipFiles]);

  const handleStop = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = new Worker(
        new URL("../numbl-worker.ts", import.meta.url),
        { type: "module" }
      );
      setupWorkerHandler(workerRef.current);
      setIsRunning(false);
      setOutput(prev => prev + "\n--- Execution stopped ---\n");
    }
  }, [setupWorkerHandler]);

  const handleEditorMount: OnMount = (_, monaco) => {
    if (
      !monaco.languages
        .getLanguages()
        .some((l: { id: string }) => l.id === "matlab")
    ) {
      monaco.languages.register({ id: "matlab" });
      monaco.languages.setLanguageConfiguration("matlab", numblLanguageConfig);
      monaco.languages.setMonarchTokensProvider(
        "matlab",
        createNumblTokensProvider()
      );
    }
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          px: 1,
          py: 0.5,
          bgcolor: "grey.100",
          borderBottom: 1,
          borderColor: "divider",
          minHeight: 36,
        }}
      >
        <Button
          variant="contained"
          size="small"
          color={isRunning ? "error" : "success"}
          startIcon={isRunning ? <StopIcon /> : <PlayArrowIcon />}
          onClick={isRunning ? handleStop : handleRun}
          sx={{ textTransform: "none", fontSize: "0.75rem", py: 0.25 }}
        >
          {isRunning ? "Stop" : "Run"}
        </Button>
      </Box>

      {/* Editor (top half) */}
      <Box sx={{ flex: 1, overflow: "hidden", minHeight: 0 }}>
        <Editor
          height="100%"
          defaultLanguage="matlab"
          value={code}
          onChange={value => setCode(value || "")}
          theme="vs-dark"
          onMount={handleEditorMount}
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            wordWrap: "on",
            automaticLayout: true,
          }}
        />
      </Box>

      {/* Output panel (bottom half) */}
      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          minHeight: 0,
          borderTop: 1,
          borderColor: "divider",
        }}
      >
        <Tabs
          value={outputTab}
          onChange={(_, v) => setOutputTab(v)}
          sx={{ borderBottom: 1, borderColor: "divider", minHeight: 32 }}
        >
          <Tab label="Console" sx={{ minHeight: 32, py: 0 }} />
          {hasFigures && <Tab label="Figure" sx={{ minHeight: 32, py: 0 }} />}
        </Tabs>

        {/* Console */}
        <Box
          sx={{
            flexGrow: 1,
            overflow: "auto",
            display: outputTab === 0 ? "block" : "none",
            p: 1,
            bgcolor: "#1e1e1e",
            fontFamily: "monospace",
            fontSize: 13,
            color: "#d4d4d4",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {isRunning && !output && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <CircularProgress size={14} sx={{ color: "#d4d4d4" }} />
              <span>Running...</span>
            </Box>
          )}
          {output ||
            (!isRunning && (
              <span style={{ opacity: 0.5 }}>Press Run to execute</span>
            ))}
        </Box>

        {/* Figure */}
        {hasFigures && (
          <Box
            sx={{
              flexGrow: 1,
              overflow: "hidden",
              display: outputTab === 1 ? "flex" : "none",
              justifyContent: "center",
              alignItems: "center",
              bgcolor: "#fff",
            }}
          >
            {currentFig && <FigureView figure={currentFig} />}
          </Box>
        )}
      </Box>

      {/* Footer */}
      <Box
        sx={{
          px: 1,
          py: 0.5,
          bgcolor: "grey.50",
          borderTop: 1,
          borderColor: "divider",
          textAlign: "center",
        }}
      >
        <Typography variant="caption" color="text.secondary">
          Powered by{" "}
          <Link href="https://numbl.org" target="_blank" rel="noopener">
            numbl
          </Link>
        </Typography>
      </Box>
    </Box>
  );
}
