/* eslint-disable @typescript-eslint/no-explicit-any */
import Editor, { OnMount } from "@monaco-editor/react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import js from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import { githubGist } from "react-syntax-highlighter/dist/esm/styles/hljs";

SyntaxHighlighter.registerLanguage("javascript", js);
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import CloudIcon from "@mui/icons-material/Cloud";
import ComputerIcon from "@mui/icons-material/Computer";
import DnsIcon from "@mui/icons-material/Dns";
import MenuIcon from "@mui/icons-material/Menu";
import {
  Box,
  Button,
  Drawer,
  Tab,
  Tabs,
  Typography,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  CircularProgress,
  useMediaQuery,
} from "@mui/material";
import {
  useEffect,
  useState,
  useRef,
  useCallback,
  useReducer,
  useMemo,
  type ReactNode,
} from "react";
import {
  numblLanguageConfig,
  createNumblTokensProvider,
} from "../numblLanguage.js";
import { parseMFile } from "../numbl-core/parser/index.js";
import type { PlotInstruction } from "../numbl-core/executor/types.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../graphics/figuresReducer.js";
import { formatDiagnostic } from "../numbl-core/diagnostics";
import { Splitter } from "./Splitter";
import { FileBrowser } from "./FileBrowser";
import { FigureView } from "../graphics/FigureView.js";
import { ReplView } from "./ReplView";
import { TreeViewer } from "./TreeViewer";
import { MipPackageManager } from "./MipPackageManager";
import type { WorkspaceFile } from "../hooks/useProjectFiles";
import {
  isRemoteExecutionEnabled,
  setRemoteExecutionEnabled,
  executeRemoteStream,
  getRemoteServiceUrl,
  setRemoteServiceUrl as saveRemoteServiceUrl,
  checkRemoteServiceHealth,
  DEFAULT_REMOTE_SERVICE_URL,
} from "../utils/remoteExecution";
import { extractMipDirectives } from "../mip-directives-core";
import { loadMipPackageBrowser } from "../mip/browser-backend";

export interface IDEWorkspaceProps {
  files: WorkspaceFile[];
  activeFileId: string;
  setActiveFileId: (id: string) => void;
  updateFileContent: (content: string) => void;
  addFile: (folderPath?: string) => Promise<string>;
  addFolder: (parentPath?: string) => Promise<string>;
  deleteFile: (fileId: string) => void;
  deleteFolder: (folderPath: string) => void;
  renameFile: (fileId: string, newName: string) => void;
  renameFolder: (oldPath: string, newName: string) => void;
  moveFile: (fileId: string, targetFolder: string | null) => void;
  uploadFiles: (
    entries: { path: string; content: string }[],
    targetFolder?: string
  ) => Promise<void>;
  headerContent: ReactNode;
}

export function IDEWorkspace({
  files,
  activeFileId,
  setActiveFileId,
  updateFileContent,
  addFile,
  addFolder,
  deleteFile,
  deleteFolder,
  renameFile,
  renameFolder,
  moveFile,
  uploadFiles,
  headerContent,
}: IDEWorkspaceProps) {
  const interpret = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return (
      params.get("interpret") === "true" || params.get("interpret") === "1"
    );
  }, []);
  const optimization = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return parseInt(params.get("opt") ?? "0", 10) || 0;
  }, []);
  const [output, setOutput] = useState("");
  const [dispatchUnknownCounts, setDispatchUnknownCounts] = useState<Record<
    string,
    number
  > | null>(null);
  const [generatedJS, setGeneratedJS] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [figures, figuresDispatch] = useReducer(
    figuresReducer,
    initialFiguresState
  );
  const [outputTab, setOutputTab] = useState(0);
  const [internalsSubTab, setInternalsSubTab] = useState<
    "js" | "ast" | "dispatch"
  >("js");
  const [allFilesRep, setAllFilesRep] = useState<
    { name: string; ast: unknown; irProgram: unknown }[]
  >([]);

  const extractAllFilesRep = useCallback(
    (wsRep: any): { name: string; ast: unknown; irProgram: unknown }[] => {
      const reps: { name: string; ast: unknown; irProgram: unknown }[] = [];
      if (wsRep?.mainFile) {
        reps.push({
          name: wsRep.mainFile.name,
          ast: wsRep.mainFile.ast,
          irProgram: wsRep.mainFile.irProgram,
        });
      }
      for (const f of wsRep?.workspaceFiles ?? []) {
        reps.push({ name: f.name, ast: f.ast, irProgram: f.irProgram });
      }
      return reps;
    },
    []
  );

  const astData = useMemo(() => {
    if (allFilesRep.length === 0) return null;
    if (allFilesRep.length === 1) return allFilesRep[0].ast;
    const obj: Record<string, unknown> = {};
    for (const f of allFilesRep) obj[f.name] = f.ast;
    return obj;
  }, [allFilesRep]);

  const [fileSources, setFileSources] = useState<Map<string, string> | null>(
    null
  );
  const [figureTab, setFigureTab] = useState(0);
  const [triggerRenameId, setTriggerRenameId] = useState<string | undefined>();

  // REPL state
  const [editorTab, setEditorTab] = useState(0);
  const [isReplExecuting, setIsReplExecuting] = useState(false);
  const replWorkerRef = useRef<Worker | null>(null);
  const replTerminalRef = useRef<any>(null);
  const replMipFilesRef = useRef<{ name: string; source: string }[]>([]);
  const replMipSearchPathsRef = useRef<string[]>([]);

  // Mobile layout
  const isMobile = useMediaQuery("(max-width:768px)");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mobileOutputTab, setMobileOutputTab] = useState(0);

  // Remote execution state
  const [useRemoteExecution, setUseRemoteExecution] = useState(
    isRemoteExecutionEnabled()
  );
  const [remoteServiceUrl, setRemoteServiceUrl] = useState(
    getRemoteServiceUrl()
  );
  const [remoteSettingsOpen, setRemoteSettingsOpen] = useState(false);
  const [remoteUrlDraft, setRemoteUrlDraft] = useState(getRemoteServiceUrl());
  const [updateSecret, setUpdateSecret] = useState(
    () => localStorage.getItem("numbl_update_secret") || ""
  );
  const [rebuildOutput, setRebuildOutput] = useState("");
  const [isRebuilding, setIsRebuilding] = useState(false);

  const executionMode: "browser" | "localhost" | "cloud" = useMemo(() => {
    if (!useRemoteExecution) return "browser";
    try {
      const url = new URL(remoteServiceUrl);
      if (url.hostname === "localhost" || url.hostname === "127.0.0.1") {
        return "localhost";
      }
    } catch {
      // invalid URL
    }
    return "cloud";
  }, [useRemoteExecution, remoteServiceUrl]);

  const [remoteNativeAddon, setRemoteNativeAddon] = useState<boolean | null>(
    null
  );
  useEffect(() => {
    if (!useRemoteExecution) {
      setRemoteNativeAddon(null);
      return;
    }
    let cancelled = false;
    checkRemoteServiceHealth(remoteServiceUrl).then(health => {
      if (!cancelled && health) {
        setRemoteNativeAddon(health.nativeAddon === true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [useRemoteExecution, remoteServiceUrl]);

  // Script worker
  const scriptWorkerRef = useRef<Worker | null>(null);
  const remoteAbortRef = useRef<AbortController | null>(null);
  const editorRef = useRef<any>(null);

  // Panel sizing
  const initialSidebarWidth = window.innerWidth >= 1200 ? 260 : 200;
  const [sidebarWidth, setSidebarWidth] = useState(initialSidebarWidth);
  const [editorWidth, setEditorWidth] = useState(
    (window.innerWidth - initialSidebarWidth) / 2
  );
  const [outputHeight, setOutputHeight] = useState(window.innerHeight / 2);

  const activeFile = useMemo(
    () => files.find(f => f.id === activeFileId),
    [files, activeFileId]
  );

  const sortedFigureHandles = useMemo(() => {
    return Object.keys(figures.figs)
      .map(k => Number(k))
      .sort((a, b) => a - b);
  }, [figures.figs]);

  useEffect(() => {
    const handles = Object.keys(figures.figs)
      .map(Number)
      .sort((a, b) => a - b);
    if (handles.length > 0) {
      const idx = handles.indexOf(figures.currentHandle);
      if (idx >= 0) {
        setFigureTab(idx);
      }
    }
  }, [figures.figs, figures.currentHandle]);

  const handlePlotInstruction = useCallback((instruction: PlotInstruction) => {
    figuresDispatch(instruction);
  }, []);

  // Initialize script worker
  useEffect(() => {
    scriptWorkerRef.current = new Worker(
      new URL("../numbl-worker.ts", import.meta.url),
      {
        type: "module",
      }
    );

    scriptWorkerRef.current.onmessage = e => {
      const msg = e.data;
      if (msg.type === "output") {
        setOutput(prev => prev + msg.text);
      } else if (msg.type === "drawnow") {
        if (msg.plotInstructions?.length) {
          for (const instr of msg.plotInstructions) {
            handlePlotInstruction(instr);
          }
        }
      } else if (msg.type === "done") {
        setGeneratedJS(msg.generatedJS || "");
        setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
        setFileSources(msg.workspaceRep?.fileSources ?? null);
        setDispatchUnknownCounts(msg.dispatchUnknownCounts ?? null);
        setIsRunning(false);
        if (msg.plotInstructions?.length) {
          for (const instr of msg.plotInstructions) {
            handlePlotInstruction(instr);
          }
        }
      } else if (msg.type === "error") {
        if (msg.generatedJS) {
          setGeneratedJS(msg.generatedJS);
        }
        setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
        setFileSources(msg.workspaceRep?.fileSources ?? null);
        setIsRunning(false);
        setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
      }
    };

    return () => {
      scriptWorkerRef.current?.terminate();
    };
  }, [handlePlotInstruction, extractAllFilesRep]);

  // Initialize REPL worker
  useEffect(() => {
    const worker = new Worker(
      new URL("../numbl-repl-worker.ts", import.meta.url),
      { type: "module" }
    );
    replWorkerRef.current = worker;

    // Send interpret mode flag to worker
    if (interpret) {
      worker.postMessage({ type: "set_interpret", interpret: true });
    }

    if (files.length > 0) {
      worker.postMessage({
        type: "update_workspace",
        workspaceFiles: files.map(f => ({ name: f.name, source: f.content })),
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
  }, [handlePlotInstruction]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update workspace in REPL worker when files change
  useEffect(() => {
    if (replWorkerRef.current && files.length > 0) {
      replWorkerRef.current.postMessage({
        type: "update_workspace",
        workspaceFiles: files.map(f => ({ name: f.name, source: f.content })),
      });
    }
  }, [files]);

  // Clear triggerRenameId after a short delay
  useEffect(() => {
    if (triggerRenameId) {
      const timer = setTimeout(() => {
        setTriggerRenameId(undefined);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [triggerRenameId]);

  const executeCode = useCallback(async () => {
    if (!activeFile) return;

    setIsRunning(true);
    setOutput("");
    setDispatchUnknownCounts(null);
    setAllFilesRep([]);
    setFileSources(null);
    figuresDispatch({ type: "clear" });

    const workspaceFiles = files
      .filter(f => f.id !== activeFileId)
      .map(f => ({ name: f.name, source: f.content }));

    // Extract mip directives and load packages before execution
    let codeToRun = activeFile.content;
    const mipWorkspaceFiles: { name: string; source: string }[] = [];
    const mipSearchPaths: string[] = [];
    try {
      const { directives, cleanedSource } = extractMipDirectives(
        activeFile.content,
        activeFile.name
      );
      if (directives.length > 0) {
        codeToRun = cleanedSource;
        for (const d of directives) {
          if (d.type === "load") {
            setOutput(
              prev => prev + `Loading mip package: ${d.packageName}...\n`
            );
            const result = await loadMipPackageBrowser(d.packageName, msg => {
              setOutput(prev => prev + `  ${msg}\n`);
            });
            mipWorkspaceFiles.push(...result.workspaceFiles);
            mipSearchPaths.push(...result.searchPaths);
          }
        }
      }
    } catch (error) {
      setOutput(
        prev =>
          prev +
          `\nMIP load error: ${error instanceof Error ? error.message : "Unknown error"}\n`
      );
      setIsRunning(false);
      return;
    }

    const combinedWorkspaceFiles = [...workspaceFiles, ...mipWorkspaceFiles];

    try {
      const ast = parseMFile(codeToRun);
      console.info("AST:", ast);
    } catch {
      console.info("Parse check skipped - error will be handled by worker");
    }

    if (useRemoteExecution) {
      const abortController = new AbortController();
      remoteAbortRef.current = abortController;
      try {
        const allFiles = [
          ...combinedWorkspaceFiles.map(f => ({
            name: f.name,
            content: f.source,
          })),
          { name: activeFile.name, content: codeToRun },
        ];

        const result = await executeRemoteStream(
          {
            files: allFiles,
            mainScript: activeFile.name,
          },
          {
            onOutput: (text: string) => {
              setOutput(prev => prev + text);
            },
            onDrawnow: plotInstructions => {
              for (const instr of plotInstructions) {
                handlePlotInstruction(instr);
              }
            },
          },
          remoteServiceUrl,
          abortController.signal
        );

        if (!result.success) {
          setOutput(
            prev =>
              prev +
              `\nError: ${result.error}${result.timedOut ? " (timed out)" : ""}\n`
          );
        }
      } catch (error) {
        setOutput(
          prev =>
            prev +
            `\nRemote execution failed: ${error instanceof Error ? error.message : "Unknown error"}\n`
        );
      } finally {
        remoteAbortRef.current = null;
        setIsRunning(false);
      }
      return;
    }

    if (!scriptWorkerRef.current) return;

    scriptWorkerRef.current.postMessage({
      type: "run",
      code: codeToRun,
      workspaceFiles: combinedWorkspaceFiles,
      mainFileName: activeFile.name,
      options: {
        displayResults: true,
        maxIterations: 10000000,
        interpret,
        optimization,
      },
      searchPaths: mipSearchPaths.length > 0 ? mipSearchPaths : undefined,
    });
  }, [
    activeFile,
    activeFileId,
    files,
    useRemoteExecution,
    remoteServiceUrl,
    handlePlotInstruction,
    interpret,
  ]);

  const stopExecution = useCallback(() => {
    if (remoteAbortRef.current) {
      remoteAbortRef.current.abort();
      remoteAbortRef.current = null;
      setIsRunning(false);
      setOutput(prev => prev + "\n--- Execution stopped ---\n");
      return;
    }

    if (scriptWorkerRef.current) {
      scriptWorkerRef.current.terminate();

      scriptWorkerRef.current = new Worker(
        new URL("../numbl-worker.ts", import.meta.url),
        {
          type: "module",
        }
      );

      scriptWorkerRef.current.onmessage = e => {
        const msg = e.data;
        if (msg.type === "output") {
          setOutput(prev => prev + msg.text);
        } else if (msg.type === "drawnow") {
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions) {
              handlePlotInstruction(instr);
            }
          }
        } else if (msg.type === "done") {
          setGeneratedJS(msg.generatedJS || "");
          setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
          setFileSources(msg.workspaceRep?.fileSources ?? null);
          setDispatchUnknownCounts(msg.dispatchUnknownCounts ?? null);
          setIsRunning(false);
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions) {
              handlePlotInstruction(instr);
            }
          }
        } else if (msg.type === "error") {
          if (msg.generatedJS) {
            setGeneratedJS(msg.generatedJS);
          }
          setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
          setFileSources(msg.workspaceRep?.fileSources ?? null);
          setIsRunning(false);
          setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
        }
      };

      setIsRunning(false);
      setOutput(prev => prev + "\n--- Execution stopped ---\n");
    }
  }, [handlePlotInstruction, extractAllFilesRep]);

  const handleExecutionModeChange = useCallback(
    (
      _: React.MouseEvent<HTMLElement>,
      newMode: "browser" | "remote" | null
    ) => {
      if (newMode !== null) {
        const isRemote = newMode === "remote";
        setUseRemoteExecution(isRemote);
        setRemoteExecutionEnabled(isRemote);
      }
    },
    []
  );

  const handleRemoteSettingsSave = useCallback(() => {
    setRemoteServiceUrl(remoteUrlDraft);
    saveRemoteServiceUrl(remoteUrlDraft);
    setRemoteSettingsOpen(false);
  }, [remoteUrlDraft]);

  const triggerRebuild = useCallback(async () => {
    setIsRebuilding(true);
    setRebuildOutput("Running update...\n");
    try {
      const response = await fetch(`${remoteServiceUrl}/update`, {
        method: "POST",
        headers: { Authorization: `Bearer ${updateSecret}` },
      });
      const data = await response.json();
      if (data.success) {
        setRebuildOutput(data.output || "Done.");
      } else {
        setRebuildOutput(`Error: ${data.error}`);
      }
    } catch (e) {
      setRebuildOutput(
        `Connection failed: ${e instanceof Error ? e.message : String(e)}`
      );
    } finally {
      setIsRebuilding(false);
    }
  }, [remoteServiceUrl, updateSecret]);

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
          // Update worker with new workspace files including mip packages
          const userFiles = files.map(f => ({
            name: f.name,
            source: f.content,
          }));
          replWorkerRef.current?.postMessage({
            type: "update_workspace",
            workspaceFiles: [...userFiles, ...replMipFilesRef.current],
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
    [isReplExecuting, files]
  );

  const handleReplClear = useCallback(() => {
    if (isReplExecuting) return;
    replWorkerRef.current?.postMessage({ type: "clear" });
  }, [isReplExecuting]);

  const handleTerminalReady = useCallback((methods: any) => {
    replTerminalRef.current = methods;
  }, []);

  const lastActiveFileId = useRef<string>("");
  useEffect(() => {
    if (!editorRef.current) return;
    if (!activeFile) return;
    if (activeFileId === lastActiveFileId.current) {
      return;
    }
    lastActiveFileId.current = activeFileId;
    if (editorRef.current.value !== activeFile.content) {
      editorRef.current.setValue(activeFile.content);
    }
  }, [activeFileId, activeFile]);

  const handleEditorDidMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    lastActiveFileId.current = activeFileId;

    monaco.languages.register({ id: "numbl" });
    monaco.languages.setLanguageConfiguration("numbl", numblLanguageConfig);
    monaco.languages.setMonarchTokensProvider(
      "numbl",
      createNumblTokensProvider()
    );

    monaco.editor.defineTheme("numbl-dark", {
      base: "vs-dark",
      inherit: true,
      rules: [
        { token: "comment", foreground: "6A9955" },
        { token: "keyword", foreground: "C586C0" },
        { token: "number", foreground: "B5CEA8" },
        { token: "string", foreground: "CE9178" },
        { token: "operator", foreground: "D4D4D4" },
      ],
      colors: {},
    });

    monaco.editor.setTheme("numbl-dark");
  };

  // Render IDE

  const editor = (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Tabs
        value={editorTab}
        onChange={(_, newValue) => setEditorTab(newValue)}
        sx={{ borderBottom: 1, borderColor: "divider", minHeight: 32 }}
      >
        <Tab label="Script" sx={{ minHeight: 32, py: 0, fontSize: "0.8rem" }} />
        <Tab label="REPL" sx={{ minHeight: 32, py: 0, fontSize: "0.8rem" }} />
      </Tabs>

      <Box sx={{ flexGrow: 1, overflow: "hidden" }}>
        {editorTab === 0 ? (
          <Box
            sx={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Box
              sx={{
                px: 1,
                py: 0.5,
                borderBottom: 1,
                borderColor: "divider",
                display: "flex",
                alignItems: "center",
                gap: 1,
              }}
            >
              <Button
                variant="contained"
                color={isRunning ? "error" : "primary"}
                size="small"
                startIcon={isRunning ? <StopIcon /> : <PlayArrowIcon />}
                onClick={isRunning ? stopExecution : executeCode}
                sx={{ py: 0.25, fontSize: "0.8rem" }}
              >
                {isRunning ? "Stop" : "Run"}
              </Button>
              <Tooltip
                title={
                  (executionMode === "browser"
                    ? "Executing in browser"
                    : executionMode === "localhost"
                      ? "Executing on localhost"
                      : "Executing on remote cloud") +
                  (remoteNativeAddon === true
                    ? " (native)"
                    : remoteNativeAddon === false
                      ? " (no native addon)"
                      : "")
                }
              >
                <IconButton
                  size="small"
                  onClick={() => {
                    setRemoteUrlDraft(remoteServiceUrl);
                    setRebuildOutput("");
                    setRemoteSettingsOpen(true);
                  }}
                  sx={{ opacity: 0.5, "&:hover": { opacity: 1 } }}
                >
                  {executionMode === "browser" ? (
                    <ComputerIcon sx={{ fontSize: "0.9rem" }} />
                  ) : executionMode === "localhost" ? (
                    <DnsIcon sx={{ fontSize: "0.9rem" }} />
                  ) : (
                    <CloudIcon sx={{ fontSize: "0.9rem" }} />
                  )}
                </IconButton>
              </Tooltip>
              {remoteNativeAddon !== null && (
                <Typography
                  variant="caption"
                  sx={{
                    fontSize: "0.65rem",
                    opacity: 0.5,
                    color: remoteNativeAddon ? "success.main" : "text.disabled",
                  }}
                >
                  {remoteNativeAddon ? "native" : "no native"}
                </Typography>
              )}
              {activeFile && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontFamily: "monospace" }}
                >
                  {activeFile.name}
                </Typography>
              )}
            </Box>

            <Box sx={{ flexGrow: 1 }}>
              <Editor
                height="100%"
                language={
                  activeFile?.name.endsWith(".js") ? "javascript" : "numbl"
                }
                defaultValue={activeFile?.content || ""}
                onChange={value => updateFileContent(value || "")}
                onMount={handleEditorDidMount}
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: "on",
                  scrollBeyondLastLine: false,
                }}
              />
            </Box>
          </Box>
        ) : (
          <ReplView
            onExecute={handleReplExecute}
            onClear={handleReplClear}
            isExecuting={isReplExecuting}
            onTerminalReady={handleTerminalReady}
          />
        )}
      </Box>
    </Box>
  );

  const outputPanel = (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Tabs
        value={outputTab}
        onChange={(_, newValue) => setOutputTab(newValue)}
        sx={{ borderBottom: 1, borderColor: "divider", minHeight: 32 }}
      >
        <Tab label="Output" sx={{ minHeight: 32, py: 0, fontSize: "0.8rem" }} />
        <Tab
          label="Packages"
          sx={{ minHeight: 32, py: 0, fontSize: "0.8rem" }}
        />
        <Tab
          label="Internals"
          sx={{ minHeight: 32, py: 0, fontSize: "0.8rem" }}
        />
      </Tabs>
      <Box sx={{ flexGrow: 1, overflow: "hidden" }}>
        {outputTab === 0 && (
          <Box sx={{ height: "100%", overflow: "auto", p: 1 }}>
            <pre
              style={{
                margin: 0,
                fontFamily: "monospace",
                fontSize: "14px",
                whiteSpace: "pre-wrap",
              }}
            >
              {output}
            </pre>
          </Box>
        )}
        {outputTab === 1 && <MipPackageManager />}
        {outputTab === 2 && (
          <Box
            sx={{ height: "100%", display: "flex", flexDirection: "column" }}
          >
            <Box sx={{ display: "flex", alignItems: "center", px: 1, py: 0.5 }}>
              <ToggleButtonGroup
                value={internalsSubTab}
                exclusive
                onChange={(_, v) => {
                  if (v) setInternalsSubTab(v);
                }}
                size="small"
              >
                <ToggleButton
                  value="js"
                  sx={{
                    py: 0,
                    px: 1,
                    fontSize: "0.75rem",
                    textTransform: "none",
                  }}
                >
                  JavaScript
                </ToggleButton>
                <ToggleButton
                  value="ast"
                  sx={{
                    py: 0,
                    px: 1,
                    fontSize: "0.75rem",
                    textTransform: "none",
                  }}
                >
                  AST
                </ToggleButton>
                <ToggleButton
                  value="dispatch"
                  sx={{
                    py: 0,
                    px: 1,
                    fontSize: "0.75rem",
                    textTransform: "none",
                  }}
                >
                  Dispatch
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
            <Box sx={{ flexGrow: 1, overflow: "hidden" }}>
              {internalsSubTab === "js" && (
                <Box sx={{ height: "100%", overflow: "auto" }}>
                  <SyntaxHighlighter
                    language="javascript"
                    style={githubGist}
                    customStyle={{ margin: 0, fontSize: 12 }}
                  >
                    {generatedJS || ""}
                  </SyntaxHighlighter>
                </Box>
              )}
              {internalsSubTab === "ast" && (
                <TreeViewer
                  data={astData}
                  label="ast"
                  fileSources={fileSources}
                />
              )}
              {internalsSubTab === "dispatch" && (
                <Box sx={{ height: "100%", overflow: "auto", p: 1 }}>
                  {dispatchUnknownCounts &&
                  Object.keys(dispatchUnknownCounts).length > 0 ? (
                    <pre
                      style={{
                        margin: 0,
                        fontFamily: "monospace",
                        fontSize: "13px",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {Object.entries(dispatchUnknownCounts)
                        .sort(([, a], [, b]) => b - a)
                        .map(([name, count]) => `${name}: ${count}`)
                        .join("\n")}
                    </pre>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No dispatchUnknown calls
                    </Typography>
                  )}
                </Box>
              )}
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  );

  const figuresPanel = (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Tabs
        value={figureTab}
        onChange={(_, newValue) => setFigureTab(newValue)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ borderBottom: 1, borderColor: "divider", minHeight: 32 }}
      >
        {sortedFigureHandles.map(h => (
          <Tab
            key={h}
            label={`Figure ${h}`}
            sx={{ minHeight: 32, py: 0, fontSize: "0.8rem" }}
          />
        ))}
      </Tabs>
      <Box sx={{ flexGrow: 1, overflow: "auto", p: 1 }}>
        {sortedFigureHandles.length > 0 ? (
          <FigureView figure={figures.figs[sortedFigureHandles[figureTab]]} />
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
            No figures
          </Typography>
        )}
      </Box>
    </Box>
  );

  const fileBrowserContent = (
    <FileBrowser
      files={files}
      activeFileId={activeFileId}
      onSelectFile={id => {
        setActiveFileId(id);
        if (isMobile) setDrawerOpen(false);
      }}
      onAddFile={async folderPath => {
        const id = await addFile(folderPath);
        if (id) setTriggerRenameId(id);
      }}
      onAddFolder={async parentPath => {
        const folderPath = await addFolder(parentPath);
        if (folderPath) setTriggerRenameId(`folder:${folderPath}`);
      }}
      onDeleteFile={deleteFile}
      onDeleteFolder={deleteFolder}
      onRenameFile={renameFile}
      onRenameFolder={renameFolder}
      onMoveFile={moveFile}
      onUploadFiles={uploadFiles}
      fileCount={files.length}
      triggerRenameId={triggerRenameId}
    />
  );

  return (
    <>
      <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
        {/* Top bar */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            px: 1,
            py: 0.5,
            borderBottom: 1,
            borderColor: "divider",
            bgcolor: "background.paper",
            minHeight: 36,
            flexShrink: 0,
          }}
        >
          {isMobile && (
            <IconButton
              size="small"
              onClick={() => setDrawerOpen(true)}
              sx={{ mr: 0.5 }}
            >
              <MenuIcon fontSize="small" />
            </IconButton>
          )}
          {headerContent}
        </Box>
        {isMobile ? (
          <>
            <Drawer
              open={drawerOpen}
              onClose={() => setDrawerOpen(false)}
              PaperProps={{ sx: { width: 260 } }}
            >
              {fileBrowserContent}
            </Drawer>
            <Splitter
              direction="horizontal"
              initialSize={Math.round(window.innerHeight * 0.55)}
              minSize={100}
            >
              {editor}
              <Box
                sx={{
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <Tabs
                  value={mobileOutputTab}
                  onChange={(_, v) => setMobileOutputTab(v)}
                  variant="scrollable"
                  scrollButtons="auto"
                  sx={{
                    borderBottom: 1,
                    borderColor: "divider",
                    minHeight: 32,
                  }}
                >
                  <Tab
                    label="Output"
                    sx={{
                      minHeight: 32,
                      py: 0,
                      fontSize: "0.75rem",
                      minWidth: 0,
                      px: 1,
                    }}
                  />
                  <Tab
                    label="JS"
                    sx={{
                      minHeight: 32,
                      py: 0,
                      fontSize: "0.75rem",
                      minWidth: 0,
                      px: 1,
                    }}
                  />
                  <Tab
                    label="AST"
                    sx={{
                      minHeight: 32,
                      py: 0,
                      fontSize: "0.75rem",
                      minWidth: 0,
                      px: 1,
                    }}
                  />
                  <Tab
                    label="Figures"
                    sx={{
                      minHeight: 32,
                      py: 0,
                      fontSize: "0.75rem",
                      minWidth: 0,
                      px: 1,
                    }}
                  />
                  <Tab
                    label="Packages"
                    sx={{
                      minHeight: 32,
                      py: 0,
                      fontSize: "0.75rem",
                      minWidth: 0,
                      px: 1,
                    }}
                  />
                </Tabs>
                <Box sx={{ flexGrow: 1, overflow: "hidden" }}>
                  {mobileOutputTab === 0 && (
                    <Box sx={{ height: "100%", overflow: "auto", p: 1 }}>
                      <pre
                        style={{
                          margin: 0,
                          fontFamily: "monospace",
                          fontSize: "13px",
                          whiteSpace: "pre-wrap",
                        }}
                      >
                        {output}
                      </pre>
                    </Box>
                  )}
                  {mobileOutputTab === 1 && (
                    <Box sx={{ height: "100%", overflow: "auto" }}>
                      <SyntaxHighlighter
                        language="javascript"
                        style={githubGist}
                        customStyle={{ margin: 0, fontSize: 12 }}
                      >
                        {generatedJS || ""}
                      </SyntaxHighlighter>
                    </Box>
                  )}
                  {mobileOutputTab === 2 && (
                    <TreeViewer
                      data={astData}
                      label="ast"
                      fileSources={fileSources}
                    />
                  )}
                  {mobileOutputTab === 3 && (
                    <Box sx={{ height: "100%", overflow: "auto", p: 1 }}>
                      {sortedFigureHandles.length > 0 ? (
                        <>
                          {sortedFigureHandles.length > 1 && (
                            <Tabs
                              value={figureTab}
                              onChange={(_, v) => setFigureTab(v)}
                              variant="scrollable"
                              scrollButtons="auto"
                              sx={{ minHeight: 28, mb: 0.5 }}
                            >
                              {sortedFigureHandles.map(h => (
                                <Tab
                                  key={h}
                                  label={`Fig ${h}`}
                                  sx={{
                                    minHeight: 28,
                                    py: 0,
                                    fontSize: "0.7rem",
                                  }}
                                />
                              ))}
                            </Tabs>
                          )}
                          <FigureView
                            figure={
                              figures.figs[sortedFigureHandles[figureTab]]
                            }
                          />
                        </>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          No figures
                        </Typography>
                      )}
                    </Box>
                  )}
                  {mobileOutputTab === 4 && <MipPackageManager />}
                </Box>
              </Box>
            </Splitter>
          </>
        ) : (
          <Splitter
            direction="vertical"
            initialSize={sidebarWidth}
            minSize={150}
            maxSize={400}
            onSizeChange={setSidebarWidth}
          >
            {fileBrowserContent}
            <Splitter
              direction="vertical"
              initialSize={editorWidth}
              onSizeChange={setEditorWidth}
            >
              {editor}
              <Splitter
                direction="horizontal"
                initialSize={outputHeight}
                minSize={150}
                onSizeChange={setOutputHeight}
              >
                {outputPanel}
                {figuresPanel}
              </Splitter>
            </Splitter>
          </Splitter>
        )}
      </Box>

      <Dialog
        open={remoteSettingsOpen}
        onClose={() => setRemoteSettingsOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Execution Settings</DialogTitle>
        <DialogContent
          sx={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            pt: 3,
            overflow: "visible",
          }}
        >
          <ToggleButtonGroup
            value={useRemoteExecution ? "remote" : "browser"}
            exclusive
            onChange={handleExecutionModeChange}
            size="small"
          >
            <ToggleButton value="browser" sx={{ px: 2 }}>
              <ComputerIcon sx={{ fontSize: "1rem", mr: 0.5 }} />
              In browser
            </ToggleButton>
            <ToggleButton value="remote" sx={{ px: 2 }}>
              <CloudIcon sx={{ fontSize: "1rem", mr: 0.5 }} />
              Remote
            </ToggleButton>
          </ToggleButtonGroup>
          {useRemoteExecution && (
            <>
              <TextField
                label="Service URL"
                value={remoteUrlDraft}
                onChange={e => setRemoteUrlDraft(e.target.value)}
                size="small"
                fullWidth
                slotProps={{
                  input: {
                    style: { fontFamily: "monospace", fontSize: "0.85rem" },
                  },
                }}
                helperText={
                  <span>
                    <span
                      style={{ cursor: "pointer", textDecoration: "underline" }}
                      onClick={() =>
                        setRemoteUrlDraft(DEFAULT_REMOTE_SERVICE_URL)
                      }
                    >
                      {DEFAULT_REMOTE_SERVICE_URL}
                    </span>
                    {" · "}
                    <span
                      style={{ cursor: "pointer", textDecoration: "underline" }}
                      onClick={() => setRemoteUrlDraft("http://localhost:3001")}
                    >
                      http://localhost:3001
                    </span>
                  </span>
                }
              />
              <TextField
                label="Update secret"
                value={updateSecret}
                onChange={e => {
                  setUpdateSecret(e.target.value);
                  localStorage.setItem("numbl_update_secret", e.target.value);
                }}
                size="small"
                fullWidth
                type="password"
                helperText="Required to trigger remote rebuild (NUMBL_UPDATE_SECRET on server)"
              />
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={triggerRebuild}
                  disabled={isRebuilding || !updateSecret}
                >
                  {isRebuilding ? "Rebuilding..." : "Trigger rebuild"}
                </Button>
                {isRebuilding && <CircularProgress size={16} />}
              </Box>
              {rebuildOutput && (
                <Box
                  component="pre"
                  sx={{
                    background: "#1e1e1e",
                    color: "#ccc",
                    p: 1,
                    borderRadius: 1,
                    fontSize: "0.75rem",
                    maxHeight: 200,
                    overflow: "auto",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-all",
                    m: 0,
                  }}
                >
                  {rebuildOutput}
                </Box>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRemoteSettingsOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleRemoteSettingsSave}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
