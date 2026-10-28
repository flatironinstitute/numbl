import { Box, Link, Typography } from "@mui/material";
import { useNavigate } from "react-router-dom";
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import { ReplView } from "../components/ReplView.js";
import { FigureView } from "../graphics/FigureView.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../numbl-core/plotting/figuresReducer.js";
import { useSystemFiles } from "../hooks/useSystemFiles.js";
import { useMipCorePackage } from "../hooks/useMipCorePackage.js";
import { syncSystemVfsChanges } from "../vfs/syncVfsChanges.js";
import { useNumblWorker } from "../hooks/useNumblWorker.js";
import { buildWorkerFiles } from "../buildWorkerFiles.js";
import { isOptLevel, type OptLevel } from "../numbl-core/executors/plugins.js";

interface TerminalMethods {
  writeOutput: (text: string, isError?: boolean) => void;
  writePrompt: () => void;
  clearTerminal: () => void;
  fit: () => void;
}

function getOptimizationParam(): OptLevel {
  const opt = new URLSearchParams(window.location.search).get("opt") ?? "1";
  return isOptLevel(opt) ? opt : "1";
}

export function EmbedReplPage() {
  const optimization = getOptimizationParam();
  const {
    systemFiles,
    reloadSystemFiles,
    getSystemVfsFiles,
    getSystemWorkspaceFiles,
  } = useSystemFiles();
  useMipCorePackage(reloadSystemFiles);
  const navigate = useNavigate();

  const [isReplExecuting, setIsReplExecuting] = useState(false);
  const [figures, figuresDispatch] = useReducer(
    figuresReducer,
    initialFiguresState
  );
  const replTerminalRef = useRef<TerminalMethods | null>(null);

  const currentFig = useMemo(() => {
    const handles = Object.keys(figures.figs)
      .map(Number)
      .sort((a, b) => a - b);
    if (handles.length === 0) return null;
    const handle = handles[handles.length - 1];
    return figures.figs[handle];
  }, [figures.figs]);

  const hasFigures = currentFig !== null;

  const worker = useNumblWorker(
    {
      onOutput: text => replTerminalRef.current?.writeOutput(text, false),
      onPlotInstructions: instructions => {
        for (const instr of instructions) figuresDispatch(instr);
      },
      onResult: msg => {
        const term = replTerminalRef.current;
        if (!msg.success && msg.error) term?.writeOutput(msg.error, true);
        term?.writePrompt();
        setIsReplExecuting(false);
      },
      // Sync system VFS changes to IndexedDB, then reload system files.
      onVfsChanges: changes => {
        void syncSystemVfsChanges(changes).then(changed => {
          if (changed) reloadSystemFiles();
        });
      },
      onWorkerError: e => {
        replTerminalRef.current?.writeOutput(
          `Worker error: ${e.message}`,
          true
        );
        setIsReplExecuting(false);
      },
    },
    { optimization }
  );

  // Track that workspace needs updating when system files change
  const workspaceStale = useRef(true);
  useEffect(() => {
    workspaceStale.current = true;
  }, [systemFiles]);

  const handleReplExecute = useCallback(
    async (command: string) => {
      const trimmed = command.trim();
      if (trimmed === "exit" || trimmed === "quit") {
        navigate("/");
        return;
      }
      if (isReplExecuting) return;
      setIsReplExecuting(true);

      // Send latest workspace files if stale
      if (workspaceStale.current) {
        const { workspaceFiles, vfsFiles } = await buildWorkerFiles({
          getSystemVfsFiles,
          getSystemWorkspaceFiles,
        });
        worker.updateWorkspace(workspaceFiles, vfsFiles);
        workspaceStale.current = false;
      }

      worker.execute(command);
    },
    [
      isReplExecuting,
      getSystemWorkspaceFiles,
      getSystemVfsFiles,
      navigate,
      worker,
    ]
  );

  const handleReplClear = useCallback(() => {
    if (isReplExecuting) return;
    worker.clear();
  }, [isReplExecuting, worker]);

  const handleTerminalReady = useCallback((methods: TerminalMethods) => {
    replTerminalRef.current = methods;
  }, []);

  // Re-fit terminal when figure area appears/disappears
  useEffect(() => {
    // Small delay to let the layout update before re-fitting
    const timer = setTimeout(() => {
      replTerminalRef.current?.fit();
    }, 50);
    return () => clearTimeout(timer);
  }, [hasFigures]);

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      {/* REPL terminal */}
      <Box sx={{ flex: 1, overflow: "hidden", minHeight: 0 }}>
        <ReplView
          onExecute={handleReplExecute}
          onClear={handleReplClear}
          isExecuting={isReplExecuting}
          onTerminalReady={handleTerminalReady}
          title="numbl"
        />
      </Box>

      {/* Figure area */}
      {hasFigures && (
        <Box
          sx={{
            height: "40%",
            borderTop: 1,
            borderColor: "divider",
            overflow: "hidden",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            bgcolor: "#fff",
          }}
        >
          {currentFig && <FigureView figure={currentFig} />}
        </Box>
      )}

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
