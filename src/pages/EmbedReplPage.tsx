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
} from "../graphics/figuresReducer.js";
import type { PlotInstruction } from "../graphics/types.js";
import { createInputSAB, mainThreadRespond } from "../syncInputChannel";
import { useSystemFiles } from "../hooks/useSystemFiles.js";
import { useMipCorePackage } from "../hooks/useMipCorePackage.js";
import { syncSystemVfsChanges } from "../vfs/syncVfsChanges.js";
import type { VfsChanges } from "../vfs/VirtualFileSystem.js";

interface TerminalMethods {
  writeOutput: (text: string, isError?: boolean) => void;
  writePrompt: () => void;
  clearTerminal: () => void;
  fit: () => void;
}

function useOptimizationParam(): number {
  const params = new URLSearchParams(window.location.search);
  return parseInt(params.get("opt") ?? "1", 10);
}

export function EmbedReplPage() {
  const optimization = useOptimizationParam();
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
  const replWorkerRef = useRef<Worker | null>(null);
  const inputSAB = useRef<SharedArrayBuffer | null>(createInputSAB());

  const handlePlotInstruction = useCallback((instruction: PlotInstruction) => {
    figuresDispatch(instruction);
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

  /** Sync system VFS changes to IndexedDB, then reload system files. */
  const handleVfsChanges = useCallback(
    async (changes: VfsChanges | undefined) => {
      if (!changes) return;
      const changed = await syncSystemVfsChanges(changes);
      if (changed) {
        reloadSystemFiles();
      }
    },
    [reloadSystemFiles]
  );

  // Initialize REPL worker
  useEffect(() => {
    const worker = new Worker(new URL("../numbl-worker.ts", import.meta.url), {
      type: "module",
    });
    replWorkerRef.current = worker;

    if (inputSAB.current) {
      worker.postMessage({
        type: "set_input_sab",
        inputSAB: inputSAB.current,
      });
    }

    worker.postMessage({
      type: "set_optimization",
      optimization,
    });

    worker.onmessage = e => {
      const msg = e.data;
      const term = replTerminalRef.current;

      if (msg.type === "request-input") {
        const response = prompt(msg.prompt ?? "") ?? "";
        const sab = inputSAB.current;
        if (sab) mainThreadRespond(sab, response);
        return;
      }

      switch (msg.type) {
        case "output":
          if (term?.writeOutput) {
            term.writeOutput(msg.text, false);
          }
          break;

        case "drawnow":
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions as PlotInstruction[]) {
              handlePlotInstruction(instr);
            }
          }
          break;

        case "result":
          if (msg.success) {
            if (msg.plotInstructions?.length) {
              for (const instr of msg.plotInstructions as PlotInstruction[]) {
                handlePlotInstruction(instr);
              }
            }
          } else if (msg.error && term?.writeOutput) {
            term.writeOutput(msg.error, true);
          }
          handleVfsChanges(msg.vfsChanges);
          if (term?.writePrompt) {
            term.writePrompt();
          }
          setIsReplExecuting(false);
          break;

        case "cleared":
          break;
      }
    };

    worker.onerror = (e: ErrorEvent) => {
      const term = replTerminalRef.current;
      if (term?.writeOutput) {
        term.writeOutput(`Worker error: ${e.message}`, true);
      }
      setIsReplExecuting(false);
    };

    return () => {
      worker.terminate();
    };
  }, [handlePlotInstruction, handleVfsChanges, optimization]);

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
      if (workspaceStale.current && replWorkerRef.current) {
        const [wsFiles, vfsFiles] = await Promise.all([
          getSystemWorkspaceFiles(),
          getSystemVfsFiles(),
        ]);
        replWorkerRef.current.postMessage({
          type: "update_workspace",
          workspaceFiles: wsFiles,
          vfsFiles,
        });
        workspaceStale.current = false;
      }

      replWorkerRef.current?.postMessage({
        type: "execute",
        code: command,
      });
    },
    [isReplExecuting, getSystemWorkspaceFiles, getSystemVfsFiles, navigate]
  );

  const handleReplClear = useCallback(() => {
    if (isReplExecuting) return;
    replWorkerRef.current?.postMessage({ type: "clear" });
  }, [isReplExecuting]);

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
