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
import { useSystemFiles } from "../hooks/useSystemFiles.js";
import {
  useMipCorePackage,
  markSystemActivity,
} from "../hooks/useMipCorePackage.js";
import { syncSystemVfsChanges } from "../vfs/syncVfsChanges.js";
import type { PlotInstruction } from "../graphics/types.js";
import { FigureView } from "../graphics/FigureView.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../graphics/figuresReducer.js";

const DEFAULT_SCRIPT = `% Welcome to numbl!
% Try running some code:

x = 1:10;
y = x.^2;
disp('Hello from numbl!')
disp(['Sum of squares: ', num2str(sum(y))])
`;

// Decode a base64 script parameter as UTF-8. The encoder (numbl-embed.js)
// base64-encodes the UTF-8 bytes so scripts may contain non-ASCII characters
// (em dashes, Greek letters, …). For pure-ASCII input this matches a plain
// atob(), so older base64 links still decode correctly.
function base64ToUtf8(b64: string): string {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

function getQueryScript(): string {
  const params = new URLSearchParams(window.location.search);
  const scriptParam = params.get("script");
  if (scriptParam) {
    try {
      return base64ToUtf8(scriptParam);
    } catch {
      console.error("Failed to decode script parameter");
    }
  }
  return DEFAULT_SCRIPT;
}

// Optional setup code that runs before the visible script on every Run. Kept
// out of the editor; its output is hidden behind a "Preparing…" message unless
// it errors. Used by docs to factor out boilerplate like `mip load --install`.
function getQueryPreamble(): string {
  const params = new URLSearchParams(window.location.search);
  const preambleParam = params.get("preamble");
  if (preambleParam) {
    try {
      return base64ToUtf8(preambleParam);
    } catch {
      console.error("Failed to decode preamble parameter");
    }
  }
  return "";
}

// Message shown while the preamble runs. Defaults to "Preparing...", but a
// page can override it (e.g. "Installing...") via the `preparing` param.
const DEFAULT_PREPARING_LABEL = "Preparing...";
function getQueryPreparingLabel(): string {
  const params = new URLSearchParams(window.location.search);
  const labelParam = params.get("preparing");
  if (labelParam) {
    try {
      return base64ToUtf8(labelParam);
    } catch {
      console.error("Failed to decode preparing parameter");
    }
  }
  return DEFAULT_PREPARING_LABEL;
}

export function EmbedPage() {
  const initialScript = getQueryScript();
  const preamble = useMemo(() => getQueryPreamble(), []);
  const preparingLabel = useMemo(() => getQueryPreparingLabel(), []);
  const { reloadSystemFiles, getSystemVfsFiles, getSystemWorkspaceFiles } =
    useSystemFiles();
  useMipCorePackage(reloadSystemFiles);
  const [code, setCode] = useState<string>(initialScript);
  const [output, setOutput] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  // While a run is in progress, which phase: the hidden preamble ("Preparing…")
  // or the visible main script ("Running…"). null when idle.
  const [runPhase, setRunPhase] = useState<"preamble" | "main" | null>(null);
  const [outputTab, setOutputTab] = useState(0); // 0=Console, 1=Figure
  const workerRef = useRef<Worker | null>(null);
  const inputSAB = useRef<SharedArrayBuffer | null>(createInputSAB());
  const [figures, figuresDispatch] = useReducer(
    figuresReducer,
    initialFiguresState
  );

  const handlePlotInstruction = useCallback((instruction: PlotInstruction) => {
    figuresDispatch(instruction);

    // Embed-specific: switch to the Figure tab as soon as a run draws anything.
    // A drawing instruction is any that isn't a control/setter (set_*) or a
    // clear/close. Keying on that — rather than a fixed plot/plot3/surf list —
    // covers every plot type (surf, surface, quiver, contour, …), which is why
    // surfacefun figures previously failed to bring the tab forward.
    const t = instruction.type;
    const isControl =
      t.startsWith("set_") ||
      t === "cla" ||
      t === "clf" ||
      t === "close" ||
      t === "close_all";
    if (!isControl) {
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
        } else if (msg.type === "preamble_done") {
          // Preamble succeeded; the visible script is now running.
          setRunPhase("main");
        } else if (msg.type === "preamble_error") {
          // Preamble failed — reveal its (otherwise hidden) output and error.
          setIsRunning(false);
          setRunPhase(null);
          setOutputTab(0);
          const preText = msg.text ? `${msg.text}\n` : "";
          setOutput(
            `${preText}Error while preparing this example:\n${formatDiagnostic(msg)}\n`
          );
        } else if (msg.type === "drawnow") {
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions) {
              handlePlotInstruction(instr);
            }
          }
        } else if (msg.type === "done") {
          setIsRunning(false);
          setRunPhase(null);
          // Persist anything the run installed under /system/ (e.g. a package
          // a preamble's `mip load --install` downloaded) to the durable system
          // directory, so later runs and other embeds reuse it instead of
          // re-downloading.
          if (msg.vfsChanges) {
            void syncSystemVfsChanges(msg.vfsChanges).then(changed => {
              if (changed) reloadSystemFiles();
            });
          }
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions) {
              handlePlotInstruction(instr);
            }
          }
        } else if (msg.type === "error") {
          setIsRunning(false);
          setRunPhase(null);
          setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
        }
      };
    },
    [handlePlotInstruction, reloadSystemFiles]
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
    setRunPhase(preamble.trim() ? "preamble" : "main");
    setOutputTab(0); // start on Console; switch to Figure when one is drawn
    figuresDispatch({ type: "clear" });

    // Keep the persistent system directory from being wiped out from under an
    // actively-used embed (see useMipCorePackage's inactivity policy).
    markSystemActivity();

    const [wsFiles, vfsFiles] = await Promise.all([
      getSystemWorkspaceFiles(),
      getSystemVfsFiles(),
    ]);

    workerRef.current.postMessage({
      type: "run",
      code,
      preamble: preamble.trim() ? preamble : undefined,
      options: {
        displayResults: true,
        maxIterations: 10000000,
        optimization: 1,
      },
      workspaceFiles: wsFiles,
      vfsFiles,
      mainFileName: "script.m",
      inputSAB: inputSAB.current ?? undefined,
    });
  }, [code, preamble, getSystemWorkspaceFiles, getSystemVfsFiles]);

  const handleStop = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = new Worker(
        new URL("../numbl-worker.ts", import.meta.url),
        { type: "module" }
      );
      setupWorkerHandler(workerRef.current);
      setIsRunning(false);
      setRunPhase(null);
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
              <span>
                {runPhase === "preamble" ? preparingLabel : "Running..."}
              </span>
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
