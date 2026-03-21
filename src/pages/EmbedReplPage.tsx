import { Box, Link, Typography } from "@mui/material";
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
import type { PlotInstruction } from "../numbl-core/executor/types.js";
import { extractMipDirectives } from "../mip-directives-core";
import { loadMipPackageBrowser } from "../mip/browser-backend";

interface TerminalMethods {
  writeOutput: (text: string, isError?: boolean) => void;
  writePrompt: () => void;
  clearTerminal: () => void;
}

function useInterpretParam(): boolean {
  const params = new URLSearchParams(window.location.search);
  return params.get("interpret") === "true" || params.get("interpret") === "1";
}

function useOptimizationParam(): number {
  const params = new URLSearchParams(window.location.search);
  return parseInt(params.get("opt") ?? "0", 10) || 0;
}

export function EmbedReplPage() {
  const interpret = useInterpretParam();
  const optimization = useOptimizationParam();
  const [isReplExecuting, setIsReplExecuting] = useState(false);
  const [figures, figuresDispatch] = useReducer(
    figuresReducer,
    initialFiguresState
  );
  const replTerminalRef = useRef<TerminalMethods | null>(null);
  const replWorkerRef = useRef<Worker | null>(null);
  const replMipFilesRef = useRef<{ name: string; source: string }[]>([]);
  const replMipSearchPathsRef = useRef<string[]>([]);

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

  // Initialize REPL worker
  useEffect(() => {
    const worker = new Worker(
      new URL("../numbl-repl-worker.ts", import.meta.url),
      { type: "module" }
    );
    replWorkerRef.current = worker;

    // Send interpret mode and optimization level to worker
    if (interpret || optimization > 0) {
      worker.postMessage({
        type: "set_interpret",
        interpret,
        optimization,
      });
    }

    worker.onmessage = e => {
      const msg = e.data;
      const term = replTerminalRef.current;

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
  }, [handlePlotInstruction, interpret]);

  const handleReplExecute = useCallback(
    async (command: string) => {
      if (isReplExecuting) return;
      setIsReplExecuting(true);

      const term = replTerminalRef.current;

      // Handle mip directives
      let codeToRun = command;
      try {
        const { directives, cleanedSource } = extractMipDirectives(
          command,
          "repl"
        );
        if (directives.length > 0) {
          codeToRun = cleanedSource;
          for (const d of directives) {
            if (d.type === "load") {
              term?.writeOutput?.(
                `Loading mip package: ${d.packageName}...\n`,
                false
              );
              const result = await loadMipPackageBrowser(d.packageName, msg => {
                term?.writeOutput?.(`  ${msg}\n`, false);
              });
              replMipFilesRef.current.push(...result.workspaceFiles);
              replMipSearchPathsRef.current.push(...result.searchPaths);
            }
          }
          // Update worker with mip packages
          replWorkerRef.current?.postMessage({
            type: "update_workspace",
            workspaceFiles: replMipFilesRef.current,
            searchPaths: replMipSearchPathsRef.current,
          });
        }
      } catch (error) {
        term?.writeOutput?.(
          `MIP load error: ${error instanceof Error ? error.message : "Unknown error"}\n`,
          true
        );
        term?.writePrompt?.();
        setIsReplExecuting(false);
        return;
      }

      // If only mip directives and no code, just show prompt
      if (codeToRun.trim().length === 0) {
        term?.writePrompt?.();
        setIsReplExecuting(false);
        return;
      }

      replWorkerRef.current?.postMessage({
        type: "execute",
        code: codeToRun,
      });
    },
    [isReplExecuting]
  );

  const handleReplClear = useCallback(() => {
    if (isReplExecuting) return;
    replWorkerRef.current?.postMessage({ type: "clear" });
  }, [isReplExecuting]);

  const handleTerminalReady = useCallback((methods: TerminalMethods) => {
    replTerminalRef.current = methods;
  }, []);

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
