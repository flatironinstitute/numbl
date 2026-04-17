/* eslint-disable @typescript-eslint/no-explicit-any */
import { createInputSAB, mainThreadRespond } from "../syncInputChannel";
import Editor, { OnMount } from "@monaco-editor/react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import js from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import c from "react-syntax-highlighter/dist/esm/languages/hljs/c";
import { githubGist } from "react-syntax-highlighter/dist/esm/styles/hljs";

SyntaxHighlighter.registerLanguage("javascript", js);
SyntaxHighlighter.registerLanguage("c", c);
const _textEncoder = new TextEncoder();
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import ComputerIcon from "@mui/icons-material/Computer";
import DnsIcon from "@mui/icons-material/Dns";
import MenuIcon from "@mui/icons-material/Menu";
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Drawer,
  Tab,
  Tabs,
  TextField,
  Typography,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
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
import type { PlotInstruction } from "../graphics/types.js";
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
import {
  fileText,
  isBinaryData,
  type WorkspaceFile,
} from "../hooks/useProjectFiles";
import {
  executeRemoteStream,
  getRemoteServiceUrl,
  setRemoteServiceUrl as saveRemoteServiceUrl,
  checkRemoteServiceHealth,
} from "../utils/remoteExecution";
import {
  syncVfsChangesToProject,
  syncSystemVfsChanges,
} from "../vfs/syncVfsChanges";
import type { VfsChanges } from "../vfs/VirtualFileSystem";
import { useSystemFiles } from "../hooks/useSystemFiles";
import { useMipCorePackage } from "../hooks/useMipCorePackage";

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
  projectName?: string; // For VFS sync back to IndexedDB
  /** Load content for a single project file (cached). */
  loadFileContent: (fileId: string) => Promise<Uint8Array>;
  /** Load all project file contents. */
  loadAllContents: () => Promise<Map<string, Uint8Array>>;
  /** Content cache ref for project files. */
  contentCache: React.RefObject<Map<string, Uint8Array>>;
  mergeVfsChanges?: (result: {
    addedFiles: WorkspaceFile[];
    modifiedFiles: { path: string; data: Uint8Array }[];
    deletedPaths: string[];
  }) => void;
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
  projectName,
  loadFileContent,
  loadAllContents,
  contentCache,
  mergeVfsChanges,
}: IDEWorkspaceProps) {
  const {
    systemFiles,
    reloadSystemFiles,
    updateSystemFileContent,
    addSystemFile,
    addSystemFolder,
    deleteSystemFile,
    deleteSystemFolder,
    renameSystemFile,
    renameSystemFolder,
    moveSystemFile,
    loadSystemFileContent,
    getSystemVfsFiles,
    getSystemWorkspaceFiles,
    loadAllSystemContents,
  } = useSystemFiles();

  useMipCorePackage(reloadSystemFiles);

  const isSystemPath = useCallback(
    (name: string) => name === "system" || name.startsWith("system/"),
    []
  );
  const isSystemFileId = useCallback(
    (fileId: string) => systemFiles.some(f => f.id === fileId),
    [systemFiles]
  );

  // Merged file list: project files + system files (metadata only — no content blobs)
  const allFiles = useMemo(
    () => [...files, ...systemFiles],
    [files, systemFiles]
  );
  const [optimization, setOptimization] = useState(1);
  const [fuse, setFuse] = useState(false);
  const [output, setOutput] = useState("");
  const [dispatchUnknownCounts, setDispatchUnknownCounts] = useState<Record<
    string,
    number
  > | null>(null);
  const [generatedJS, setGeneratedJS] = useState("");
  const [generatedC, setGeneratedC] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [figures, figuresDispatch] = useReducer(
    figuresReducer,
    initialFiguresState
  );
  const [outputTab, setOutputTab] = useState(0);
  const [internalsSubTab, setInternalsSubTab] = useState<
    "js" | "c" | "ast" | "dispatch"
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
  const replTerminalRef = useRef<any>(null);

  // Mobile layout
  const isMobile = useMediaQuery("(max-width:768px)");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mobileOutputTab, setMobileOutputTab] = useState(0);

  // Remote execution state
  const [useRemoteExecution, setUseRemoteExecution] = useState(false);
  const [remoteServiceUrl, setRemoteServiceUrl] = useState(
    getRemoteServiceUrl()
  );
  const [localServerSettingsOpen, setLocalServerSettingsOpen] = useState(false);
  const [localServerUrlDraft, setLocalServerUrlDraft] = useState(
    getRemoteServiceUrl()
  );
  const generatePasskey = useCallback(() => {
    const key = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
    sessionStorage.setItem("numbl_passkey", key);
    return key;
  }, []);
  const [passkey, setPasskey] = useState(() => {
    const stored = sessionStorage.getItem("numbl_passkey");
    if (stored) return stored;
    return generatePasskey();
  });
  // Keep a ref so network calls always use the latest passkey
  const passkeyRef = useRef(passkey);
  passkeyRef.current = passkey;

  const executionMode: "browser" | "localhost" = useRemoteExecution
    ? "localhost"
    : "browser";

  const [remoteNativeAddon, setRemoteNativeAddon] = useState<boolean | null>(
    null
  );
  const [remoteServerStatus, setRemoteServerStatus] = useState<
    "unknown" | "connected" | "disconnected"
  >("unknown");
  useEffect(() => {
    if (!useRemoteExecution) {
      setRemoteNativeAddon(null);
      setRemoteServerStatus("unknown");
      return;
    }
    let cancelled = false;
    checkRemoteServiceHealth(remoteServiceUrl, passkeyRef.current).then(
      health => {
        if (cancelled) return;
        if (health) {
          setRemoteNativeAddon(health.nativeAddon === true);
          setRemoteServerStatus("connected");
        } else {
          setRemoteServerStatus("disconnected");
        }
      }
    );
    return () => {
      cancelled = true;
    };
  }, [useRemoteExecution, remoteServiceUrl, passkey]);

  // Unified worker (handles both script runs and REPL)
  const workerRef = useRef<Worker | null>(null);
  const workerInputSAB = useRef<SharedArrayBuffer | null>(createInputSAB());
  const cancelSAB = useRef<SharedArrayBuffer | null>(
    typeof SharedArrayBuffer !== "undefined" ? new SharedArrayBuffer(4) : null
  );
  // Tracks which execution mode is active for routing streaming messages
  const activeExecutionMode = useRef<"script" | "repl" | null>(null);
  const remoteAbortRef = useRef<AbortController | null>(null);
  const editorRef = useRef<any>(null);

  // Persistent workspace toggle
  const [persistWorkspace, setPersistWorkspace] = useState(false);

  // Sync optimization level to worker when toggled
  useEffect(() => {
    workerRef.current?.postMessage({
      type: "set_optimization",
      optimization,
    });
  }, [optimization]);

  // Sync fuse flag to worker when toggled
  useEffect(() => {
    workerRef.current?.postMessage({ type: "set_fuse", fuse });
  }, [fuse]);

  // Panel sizing
  const initialSidebarWidth = window.innerWidth >= 1200 ? 260 : 200;
  const [sidebarWidth, setSidebarWidth] = useState(initialSidebarWidth);
  const [editorWidth, setEditorWidth] = useState(
    (window.innerWidth - initialSidebarWidth) / 2
  );
  const [outputHeight, setOutputHeight] = useState(window.innerHeight / 2);

  const activeFile = useMemo(
    () => allFiles.find(f => f.id === activeFileId),
    [allFiles, activeFileId]
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

  /** Build VFS + workspace text files for sending to workers. Loads all content from DB. */
  const buildWorkerFiles = useCallback(
    async (wsFiles: WorkspaceFile[], excludeFileId?: string) => {
      const [projectContents, systemVfs, systemWs] = await Promise.all([
        loadAllContents(),
        getSystemVfsFiles(),
        getSystemWorkspaceFiles(),
      ]);
      const systemContents = await loadAllSystemContents();

      const vfsFiles = [
        ...wsFiles.map(f => ({
          path: f.name,
          content: projectContents.get(f.id) ?? new Uint8Array(0),
        })),
        ...systemVfs,
      ];

      const decoder = new TextDecoder("utf-8");
      // Only .m files are sent as text workspace files. Binary blobs (.wasm)
      // and .numbl.js mex-like files reach the worker via vfsFiles and are
      // discovered through scanDirectory once their containing directory is
      // on the search path. See useSystemFiles.getSystemWorkspaceFiles for
      // the reasoning.
      const workspaceFiles = [
        ...wsFiles
          .filter(f => !excludeFileId || f.id !== excludeFileId)
          .filter(f => f.name.endsWith(".m"))
          .map(f => ({
            name: f.name,
            source: decoder.decode(
              projectContents.get(f.id) ?? new Uint8Array(0)
            ),
          })),
        ...systemWs,
      ];

      return { vfsFiles, workspaceFiles, projectContents, systemContents };
    },
    [
      loadAllContents,
      getSystemVfsFiles,
      getSystemWorkspaceFiles,
      loadAllSystemContents,
    ]
  );

  /** Handle VFS changes from worker execution. */
  const handleVfsChanges = useCallback(
    async (changes: VfsChanges | undefined) => {
      if (!changes) return;
      const { created, modified, deleted } = changes;
      if (created.length === 0 && modified.length === 0 && deleted.length === 0)
        return;
      if (projectName) {
        const { projectResult, systemResult } = await syncVfsChangesToProject(
          projectName,
          changes
        );
        if (projectResult && mergeVfsChanges) {
          mergeVfsChanges(projectResult);
        }
        if (systemResult) {
          reloadSystemFiles();
        }
      } else {
        // No project (e.g. share route) — still sync system file changes
        const synced = await syncSystemVfsChanges(changes);
        if (synced) {
          reloadSystemFiles();
        }
      }
    },
    [projectName, mergeVfsChanges, reloadSystemFiles]
  );

  // Set up the unified worker's onmessage handler
  const setupWorkerHandler = useCallback(
    (worker: Worker) => {
      worker.onmessage = e => {
        const msg = e.data;

        // Input request — shared by both modes
        if (msg.type === "request-input") {
          const response = prompt(msg.prompt ?? "") ?? "";
          const sab = workerInputSAB.current;
          if (sab) mainThreadRespond(sab, response);
          return;
        }

        // Streaming output — route based on active execution mode
        if (msg.type === "output") {
          if (activeExecutionMode.current === "repl") {
            const term = replTerminalRef.current;
            if (term?.writeOutput) term.writeOutput(msg.text, false);
          } else {
            setOutput(prev => prev + msg.text);
          }
          return;
        }

        if (msg.type === "drawnow") {
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions) {
              handlePlotInstruction(instr);
            }
          }
          return;
        }

        // Script completion messages
        if (msg.type === "done") {
          activeExecutionMode.current = null;
          setGeneratedJS(msg.generatedJS || "");
          setGeneratedC(msg.generatedC || "");
          setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
          setFileSources(msg.workspaceRep?.fileSources ?? null);
          setDispatchUnknownCounts(msg.dispatchUnknownCounts ?? null);
          setIsRunning(false);
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions) {
              handlePlotInstruction(instr);
            }
          }
          handleVfsChanges(msg.vfsChanges);
          return;
        }

        if (msg.type === "error") {
          activeExecutionMode.current = null;
          if (msg.generatedJS) {
            setGeneratedJS(msg.generatedJS);
          }
          setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
          setFileSources(msg.workspaceRep?.fileSources ?? null);
          setIsRunning(false);
          setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
          handleVfsChanges(msg.vfsChanges);
          return;
        }

        // REPL completion messages
        if (msg.type === "result") {
          activeExecutionMode.current = null;
          const term = replTerminalRef.current;
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
          return;
        }

        // "cleared" — no action needed
      };

      worker.onerror = (ev: ErrorEvent) => {
        activeExecutionMode.current = null;
        const term = replTerminalRef.current;
        if (term?.writeOutput) {
          term.writeOutput(`Worker error: ${ev.message}`, true);
        }
        setIsRunning(false);
        setIsReplExecuting(false);
      };
    },
    [handlePlotInstruction, extractAllFilesRep, handleVfsChanges]
  );

  // Initialize unified worker
  useEffect(() => {
    const worker = new Worker(new URL("../numbl-worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;

    if (workerInputSAB.current) {
      worker.postMessage({
        type: "set_input_sab",
        inputSAB: workerInputSAB.current,
      });
    }

    setupWorkerHandler(worker);

    return () => {
      worker.terminate();
    };
  }, [setupWorkerHandler]);

  // Track that workspace needs updating when files change
  const workspaceStale = useRef(true);
  useEffect(() => {
    workspaceStale.current = true;
  }, [files, systemFiles]);

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

    // Load all file contents from DB
    const { vfsFiles, workspaceFiles, projectContents } =
      await buildWorkerFiles(files, activeFileId);

    const activeData =
      projectContents.get(activeFileId) ??
      contentCache.current.get(activeFileId) ??
      new Uint8Array(0);
    const codeToRun = fileText(activeData);

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
        // Send all project files (excluding system/ directory) to the server
        const decoder = new TextDecoder("utf-8");
        const remoteFiles: { name: string; content: string }[] = [];
        for (const f of files) {
          if (f.name === "system" || f.name.startsWith("system/")) continue;
          const data =
            projectContents.get(f.id) ??
            contentCache.current.get(f.id) ??
            new Uint8Array(0);
          if (isBinaryData(data)) continue;
          remoteFiles.push({ name: f.name, content: decoder.decode(data) });
        }
        // Ensure the active file is included with the latest content
        if (!remoteFiles.some(f => f.name === activeFile.name)) {
          remoteFiles.push({ name: activeFile.name, content: codeToRun });
        } else {
          const idx = remoteFiles.findIndex(f => f.name === activeFile.name);
          remoteFiles[idx] = { name: activeFile.name, content: codeToRun };
        }

        const result = await executeRemoteStream(
          {
            files: remoteFiles,
            mainScript: activeFile.name,
            optimization,
            fuse,
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
          abortController.signal,
          passkeyRef.current
        );

        setGeneratedJS(result.generatedJS || "");
        setGeneratedC(result.generatedC || "");

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

    if (!workerRef.current) return;

    // Send workspace update before run
    workerRef.current.postMessage({
      type: "update_workspace",
      workspaceFiles,
      vfsFiles,
    });
    workspaceStale.current = false;

    // Reset cancel flag
    if (cancelSAB.current) {
      const flag = new Int32Array(cancelSAB.current);
      Atomics.store(flag, 0, 0);
    }

    activeExecutionMode.current = "script";
    workerRef.current.postMessage({
      type: "run",
      code: codeToRun,
      workspaceFiles,
      mainFileName: activeFile.name,
      options: {
        displayResults: true,
        maxIterations: 10000000,
        optimization,
        fuse,
      },
      vfsFiles,
      persistent: persistWorkspace,
      cancelSAB: cancelSAB.current ?? undefined,
    });
  }, [
    activeFile,
    activeFileId,
    files,
    useRemoteExecution,
    remoteServiceUrl,
    handlePlotInstruction,
    optimization,
    fuse,
    buildWorkerFiles,
    contentCache,
    persistWorkspace,
  ]);

  const stopExecution = useCallback(() => {
    if (remoteAbortRef.current) {
      remoteAbortRef.current.abort();
      remoteAbortRef.current = null;
      setIsRunning(false);
      setOutput(prev => prev + "\n--- Execution stopped ---\n");
      return;
    }

    if (!workerRef.current) return;

    if (persistWorkspace) {
      // Persistent mode: cooperative cancellation (preserve worker state)
      if (cancelSAB.current) {
        const flag = new Int32Array(cancelSAB.current);
        Atomics.store(flag, 0, 1);
      }
      // The worker will catch CancellationError and send "done"
      // which will clear isRunning via the handler
      setOutput(prev => prev + "\n--- Cancelling execution... ---\n");
    } else {
      // Non-persistent mode: terminate and recreate
      workerRef.current.terminate();
      activeExecutionMode.current = null;

      const worker = new Worker(
        new URL("../numbl-worker.ts", import.meta.url),
        { type: "module" }
      );
      workerRef.current = worker;

      if (workerInputSAB.current) {
        worker.postMessage({
          type: "set_input_sab",
          inputSAB: workerInputSAB.current,
        });
      }

      setupWorkerHandler(worker);
      workspaceStale.current = true;

      setIsRunning(false);
      setOutput(prev => prev + "\n--- Execution stopped ---\n");
    }
  }, [persistWorkspace, setupWorkerHandler]);

  const handleLocalServerSettingsSave = useCallback(() => {
    setRemoteServiceUrl(localServerUrlDraft);
    saveRemoteServiceUrl(localServerUrlDraft);
    setLocalServerSettingsOpen(false);
  }, [localServerUrlDraft]);

  const handleExecutionModeChange = useCallback(
    (_: React.MouseEvent<HTMLElement>, newMode: "browser" | "local" | null) => {
      if (newMode !== null) {
        const isLocal = newMode === "local";
        setUseRemoteExecution(isLocal);
        // C-JIT is only available on the local server
        if (!isLocal) {
          setOptimization(o => (o > 1 ? 1 : o));
        }
      }
    },
    []
  );

  const handleReplExecute = useCallback(
    async (command: string) => {
      if (isReplExecuting || isRunning) return;
      setIsReplExecuting(true);

      // Send latest workspace files if stale
      if (workspaceStale.current && workerRef.current) {
        const { vfsFiles, workspaceFiles } = await buildWorkerFiles(files);
        workerRef.current.postMessage({
          type: "update_workspace",
          workspaceFiles,
          vfsFiles,
        });
        workspaceStale.current = false;
      }

      // Reset cancel flag
      if (cancelSAB.current) {
        const flag = new Int32Array(cancelSAB.current);
        Atomics.store(flag, 0, 0);
      }

      activeExecutionMode.current = "repl";
      workerRef.current?.postMessage({
        type: "execute",
        code: command,
        cancelSAB: cancelSAB.current ?? undefined,
      });
    },
    [isReplExecuting, isRunning, files, buildWorkerFiles]
  );

  const handleReplClear = useCallback(() => {
    if (isReplExecuting) return;
    workerRef.current?.postMessage({ type: "clear" });
  }, [isReplExecuting]);

  const handleTerminalReady = useCallback((methods: any) => {
    replTerminalRef.current = methods;
  }, []);

  const lastActiveFileId = useRef<string>("");
  const [activeFileData, setActiveFileData] = useState<Uint8Array | null>(null);
  const activeFileDataIdRef = useRef<string>("");
  const activeFileIsBinary = activeFileData
    ? isBinaryData(activeFileData)
    : false;

  // Load active file content from DB when file changes
  useEffect(() => {
    if (!activeFileId) {
      setActiveFileData(null);
      activeFileDataIdRef.current = "";
      return;
    }
    setActiveFileData(null);
    activeFileDataIdRef.current = "";
    let cancelled = false;
    const isSystem = systemFiles.some(f => f.id === activeFileId);
    const loader = isSystem ? loadSystemFileContent : loadFileContent;
    loader(activeFileId).then(data => {
      if (!cancelled) {
        activeFileDataIdRef.current = activeFileId;
        setActiveFileData(data);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [activeFileId, systemFiles, loadSystemFileContent, loadFileContent]);

  useEffect(() => {
    if (!editorRef.current) return;
    if (!activeFile) return;
    if (activeFileId === lastActiveFileId.current) return;
    if (!activeFileData || activeFileDataIdRef.current !== activeFileId) return;
    lastActiveFileId.current = activeFileId;
    const text = isBinaryData(activeFileData) ? "" : fileText(activeFileData);
    if (editorRef.current.value !== text) {
      editorRef.current.setValue(text);
    }
  }, [activeFileId, activeFile, activeFileData]);

  const handleEditorDidMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    // Only mark as synced if file data is already loaded; otherwise the sync
    // effect will set the content once the async load completes.
    if (activeFileData && activeFileDataIdRef.current === activeFileId) {
      lastActiveFileId.current = activeFileId;
    }

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
        value={useRemoteExecution ? 0 : editorTab}
        onChange={(_, newValue) => setEditorTab(newValue)}
        sx={{
          borderBottom: 1,
          borderColor: "divider",
          minHeight: 34,
          "& .MuiTab-root": {
            textTransform: "none",
            fontWeight: 500,
            fontSize: "0.8rem",
            minHeight: 34,
            py: 0,
          },
        }}
      >
        <Tab label="Script" />
        {!useRemoteExecution && <Tab label="REPL" />}
      </Tabs>

      <Box sx={{ flexGrow: 1, overflow: "hidden" }}>
        {editorTab === 0 || useRemoteExecution ? (
          <Box
            sx={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Box
              sx={{
                px: 1.5,
                py: 0.5,
                borderBottom: 1,
                borderColor: "divider",
                display: "flex",
                alignItems: "center",
                gap: 1,
                bgcolor: "background.default",
              }}
            >
              <Button
                variant="contained"
                color={isRunning ? "error" : "success"}
                size="small"
                startIcon={isRunning ? <StopIcon /> : <PlayArrowIcon />}
                onClick={isRunning ? stopExecution : executeCode}
                sx={{
                  py: 0.25,
                  px: 2,
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  textTransform: "none",
                  borderRadius: 1.5,
                  boxShadow: "none",
                  "&:hover": { boxShadow: "none" },
                }}
              >
                {isRunning ? "Stop" : "Run"}
              </Button>
              <Tooltip
                title={
                  executionMode === "browser"
                    ? "Executing in browser"
                    : "Executing on local server" +
                      (remoteServerStatus === "disconnected"
                        ? " (not connected)"
                        : "") +
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
                    setLocalServerUrlDraft(remoteServiceUrl);
                    setLocalServerSettingsOpen(true);
                  }}
                  sx={{ opacity: 0.5, "&:hover": { opacity: 1 } }}
                >
                  {executionMode === "browser" ? (
                    <ComputerIcon sx={{ fontSize: "0.9rem" }} />
                  ) : (
                    <DnsIcon
                      sx={{
                        fontSize: "0.9rem",
                        color: "error.main",
                      }}
                    />
                  )}
                </IconButton>
              </Tooltip>
              {executionMode === "localhost" && (
                <Typography
                  variant="caption"
                  sx={{
                    fontSize: "0.65rem",
                    opacity: 0.5,
                    color: "error.main",
                  }}
                >
                  {remoteServerStatus === "disconnected"
                    ? "no server"
                    : remoteNativeAddon
                      ? "native"
                      : remoteNativeAddon === false
                        ? "no native"
                        : "local"}
                </Typography>
              )}
              {!useRemoteExecution && (
                <Tooltip
                  title={
                    persistWorkspace
                      ? "Workspace persists across runs and REPL (click for fresh each run)"
                      : "Fresh workspace each run (click to persist across runs and REPL)"
                  }
                >
                  <Typography
                    variant="caption"
                    onClick={() => setPersistWorkspace(p => !p)}
                    sx={{
                      cursor: "pointer",
                      fontSize: "0.7rem",
                      px: 0.5,
                      py: 0.1,
                      borderRadius: 0.5,
                      bgcolor: persistWorkspace
                        ? "action.selected"
                        : "transparent",
                      opacity: persistWorkspace ? 1 : 0.5,
                      "&:hover": { opacity: 1 },
                      userSelect: "none",
                    }}
                  >
                    {persistWorkspace ? "persist" : "1x"}
                  </Typography>
                </Tooltip>
              )}
              <Tooltip
                title={
                  optimization === 0
                    ? "Interpreter only (click for JS-JIT)"
                    : optimization === 1
                      ? useRemoteExecution
                        ? "JS-JIT (click for C-JIT)"
                        : "JS-JIT (click to disable)"
                      : "C-JIT (click to disable)"
                }
              >
                <Typography
                  variant="caption"
                  onClick={() =>
                    setOptimization(o => {
                      if (o === 0) return 1;
                      if (o === 1 && useRemoteExecution) return 2;
                      return 0;
                    })
                  }
                  sx={{
                    cursor: "pointer",
                    fontSize: "0.7rem",
                    px: 0.5,
                    py: 0.1,
                    borderRadius: 0.5,
                    bgcolor:
                      optimization >= 1 ? "action.selected" : "transparent",
                    opacity: optimization >= 1 ? 1 : 0.5,
                    "&:hover": { opacity: 1 },
                    userSelect: "none",
                  }}
                >
                  {optimization === 0
                    ? "no jit"
                    : optimization === 1
                      ? "jit"
                      : "jit-c"}
                </Typography>
              </Tooltip>
              <Tooltip
                title={
                  fuse
                    ? "Tensor fusion enabled (click to disable)"
                    : "Tensor fusion disabled (click to enable)"
                }
              >
                <Typography
                  variant="caption"
                  onClick={() => setFuse(f => !f)}
                  sx={{
                    cursor: "pointer",
                    fontSize: "0.7rem",
                    px: 0.5,
                    py: 0.1,
                    borderRadius: 0.5,
                    bgcolor: fuse ? "action.selected" : "transparent",
                    opacity: fuse ? 1 : 0.5,
                    "&:hover": { opacity: 1 },
                    userSelect: "none",
                  }}
                >
                  {fuse ? "fuse" : "no fuse"}
                </Typography>
              </Tooltip>
              {activeFile && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontFamily: "monospace", fontWeight: "bold" }}
                >
                  {activeFile.name}
                </Typography>
              )}
            </Box>

            <Box sx={{ flexGrow: 1 }}>
              {activeFileIsBinary ? (
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    color: "text.secondary",
                  }}
                >
                  <Typography variant="body2">
                    Binary file ({activeFileData?.length ?? 0} bytes)
                  </Typography>
                </Box>
              ) : (
                <Editor
                  height="100%"
                  language={
                    activeFile?.name.endsWith(".js") ? "javascript" : "numbl"
                  }
                  defaultValue={activeFileData ? fileText(activeFileData) : ""}
                  onChange={value => {
                    const encoded = _textEncoder.encode(value || "");
                    setActiveFileData(encoded);
                    workspaceStale.current = true;
                    if (isSystemFileId(activeFileId)) {
                      updateSystemFileContent(activeFileId, encoded);
                    } else {
                      updateFileContent(value || "");
                    }
                  }}
                  onMount={handleEditorDidMount}
                  options={{
                    minimap: { enabled: false },
                    fontSize: 14,
                    lineNumbers: "on",
                    scrollBeyondLastLine: false,
                  }}
                />
              )}
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
        sx={{
          borderBottom: 1,
          borderColor: "divider",
          minHeight: 34,
          "& .MuiTab-root": {
            textTransform: "none",
            fontWeight: 500,
            fontSize: "0.8rem",
            minHeight: 34,
            py: 0,
          },
        }}
      >
        <Tab label="Output" />
        <Tab label="Internals" />
      </Tabs>
      <Box sx={{ flexGrow: 1, overflow: "hidden" }}>
        {outputTab === 0 && (
          <Box sx={{ height: "100%", overflow: "auto", p: 1.5 }}>
            {output ? (
              <pre
                style={{
                  margin: 0,
                  fontFamily: 'Menlo, Monaco, "Courier New", monospace',
                  fontSize: "13px",
                  lineHeight: 1.5,
                  whiteSpace: "pre-wrap",
                }}
              >
                {output}
              </pre>
            ) : (
              <Typography
                variant="body2"
                color="text.disabled"
                sx={{ fontStyle: "italic", mt: 2, textAlign: "center" }}
              >
                Run a script to see output here
              </Typography>
            )}
          </Box>
        )}
        {outputTab === 1 && (
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
                {generatedC && (
                  <ToggleButton
                    value="c"
                    sx={{
                      py: 0,
                      px: 1,
                      fontSize: "0.75rem",
                      textTransform: "none",
                    }}
                  >
                    C
                  </ToggleButton>
                )}
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
              {internalsSubTab === "c" && (
                <Box sx={{ height: "100%", overflow: "auto" }}>
                  <SyntaxHighlighter
                    language="c"
                    style={githubGist}
                    customStyle={{ margin: 0, fontSize: 12 }}
                  >
                    {generatedC || ""}
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
        sx={{
          borderBottom: 1,
          borderColor: "divider",
          minHeight: 34,
          "& .MuiTab-root": {
            textTransform: "none",
            fontWeight: 500,
            fontSize: "0.8rem",
            minHeight: 34,
            py: 0,
          },
        }}
      >
        {sortedFigureHandles.map(h => (
          <Tab key={h} label={`Figure ${h}`} />
        ))}
      </Tabs>
      <Box sx={{ flexGrow: 1, overflow: "auto", p: 1 }}>
        {sortedFigureHandles.length > 0 ? (
          <FigureView figure={figures.figs[sortedFigureHandles[figureTab]]} />
        ) : (
          <Typography
            variant="body2"
            color="text.disabled"
            sx={{ p: 2, textAlign: "center", fontStyle: "italic" }}
          >
            No figures
          </Typography>
        )}
      </Box>
    </Box>
  );

  const fileBrowserContent = (
    <FileBrowser
      files={allFiles}
      activeFileId={activeFileId}
      onSelectFile={id => {
        setActiveFileId(id);
        if (isMobile) setDrawerOpen(false);
      }}
      onAddFile={async folderPath => {
        if (folderPath && isSystemPath(folderPath)) {
          const id = await addSystemFile(folderPath);
          if (id) setTriggerRenameId(id);
        } else {
          const id = await addFile(folderPath);
          if (id) setTriggerRenameId(id);
        }
      }}
      onAddFolder={async parentPath => {
        if (parentPath && isSystemPath(parentPath)) {
          const folderPath = await addSystemFolder(parentPath);
          if (folderPath) setTriggerRenameId(`folder:${folderPath}`);
        } else {
          const folderPath = await addFolder(parentPath);
          if (folderPath) setTriggerRenameId(`folder:${folderPath}`);
        }
      }}
      onDeleteFile={fileId =>
        isSystemFileId(fileId) ? deleteSystemFile(fileId) : deleteFile(fileId)
      }
      onDeleteFolder={folderPath =>
        isSystemPath(folderPath)
          ? deleteSystemFolder(folderPath)
          : deleteFolder(folderPath)
      }
      onRenameFile={(fileId, newName) =>
        isSystemFileId(fileId)
          ? renameSystemFile(fileId, newName)
          : renameFile(fileId, newName)
      }
      onRenameFolder={(oldPath, newName) =>
        isSystemPath(oldPath)
          ? renameSystemFolder(oldPath, newName)
          : renameFolder(oldPath, newName)
      }
      onMoveFile={(fileId, targetFolder) =>
        isSystemFileId(fileId)
          ? moveSystemFile(fileId, targetFolder)
          : moveFile(fileId, targetFolder)
      }
      onUploadFiles={uploadFiles}
      fileCount={allFiles.length}
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
            px: 1.5,
            py: 0.5,
            borderBottom: 1,
            borderColor: "divider",
            bgcolor: "#1e1e2e",
            color: "#cdd6f4",
            minHeight: 40,
            flexShrink: 0,
            "& .MuiIconButton-root": { color: "#cdd6f4" },
            "& .MuiTypography-root": { color: "#cdd6f4" },
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
              slotProps={{ paper: { sx: { width: 260 } } }}
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
        open={localServerSettingsOpen}
        onClose={() => setLocalServerSettingsOpen(false)}
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
            value={useRemoteExecution ? "local" : "browser"}
            exclusive
            onChange={handleExecutionModeChange}
            size="small"
          >
            <ToggleButton value="browser" sx={{ px: 2 }}>
              <ComputerIcon sx={{ fontSize: "1rem", mr: 0.5 }} />
              In browser
            </ToggleButton>
            <ToggleButton value="local" sx={{ px: 2 }}>
              <DnsIcon sx={{ fontSize: "1rem", mr: 0.5 }} />
              Local server
            </ToggleButton>
          </ToggleButtonGroup>
          <TextField
            label="Server URL"
            value={localServerUrlDraft}
            onChange={e => setLocalServerUrlDraft(e.target.value)}
            size="small"
            fullWidth
            disabled={!useRemoteExecution}
            slotProps={{
              input: {
                style: { fontFamily: "monospace", fontSize: "0.85rem" },
              },
            }}
            helperText="Default: http://localhost:3001"
          />
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Typography
              variant="body2"
              color={
                !useRemoteExecution
                  ? "text.disabled"
                  : remoteServerStatus === "connected"
                    ? "success.main"
                    : remoteServerStatus === "disconnected"
                      ? "warning.main"
                      : "text.secondary"
              }
            >
              {!useRemoteExecution
                ? "Server status: n/a"
                : remoteServerStatus === "connected"
                  ? "Connected" +
                    (remoteNativeAddon === true
                      ? " (native addon available)"
                      : remoteNativeAddon === false
                        ? " (no native addon)"
                        : "")
                  : remoteServerStatus === "disconnected"
                    ? "Not connected"
                    : "Checking..."}
            </Typography>
            <Button
              size="small"
              disabled={!useRemoteExecution}
              onClick={() => {
                setRemoteServerStatus("unknown");
                checkRemoteServiceHealth(
                  localServerUrlDraft,
                  passkeyRef.current
                ).then(health => {
                  if (health) {
                    setRemoteNativeAddon(health.nativeAddon === true);
                    setRemoteServerStatus("connected");
                  } else {
                    setRemoteServerStatus("disconnected");
                  }
                });
              }}
              sx={{ minWidth: 0, textTransform: "none" }}
            >
              Check
            </Button>
          </Box>
          <Box
            sx={{
              opacity: useRemoteExecution ? 1 : 0.4,
            }}
          >
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Start a local server in a terminal:
            </Typography>
            <Box
              component="pre"
              sx={{
                bgcolor: "#1e1e1e",
                color: "#ccc",
                p: 1.5,
                borderRadius: 1,
                fontSize: "0.8rem",
                m: 0,
                whiteSpace: "pre-wrap",
              }}
            >
              {`# If installed globally:\nnpx numbl serve --passkey ${passkey}\n\n# Or from the repo (dev mode):\nnpx tsx src/cli.ts serve --passkey ${passkey}`}
            </Box>
            <Typography
              variant="caption"
              color="text.disabled"
              onClick={() => {
                setPasskey(generatePasskey());
                setRemoteServerStatus("disconnected");
              }}
              sx={{
                cursor: "pointer",
                mt: 0.5,
                "&:hover": { color: "text.secondary" },
              }}
            >
              regenerate passkey
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLocalServerSettingsOpen(false)}>
            Cancel
          </Button>
          <Button variant="contained" onClick={handleLocalServerSettingsSave}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
