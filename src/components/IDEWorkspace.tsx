/* eslint-disable @typescript-eslint/no-explicit-any */
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
import NoteAddIcon from "@mui/icons-material/NoteAdd";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Drawer,
  Link,
  Snackbar,
  Tab,
  Tabs,
  TextField,
  Typography,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Switch,
  FormControlLabel,
  Divider,
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
import type { PlotInstruction } from "../numbl-core/plotting/types.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../numbl-core/plotting/figuresReducer.js";
import { formatDiagnostic } from "../numbl-core/diagnostics";
import { Splitter } from "./Splitter";
import { FileBrowser } from "./FileBrowser";
import { FigureView } from "../graphics/FigureView.js";
import { downloadFigureHdf5 } from "../graphics/exportFigureHdf5.js";
import { buildFigureViewerLink } from "../graphics/openInFigureViewer.js";
import { uploadFigureForViewer } from "../graphics/figureUpload.js";
import { ReplView } from "./ReplView";
import { TreeViewer } from "./TreeViewer";
import { MarkdownView } from "./MarkdownView";
import { fileText, isBinaryData } from "../hooks/useProjectFiles";
import type { FileEntry } from "../hooks/projectFilesState";
import {
  executeRemoteStream,
  getRemoteServiceUrl,
  setRemoteServiceUrl as saveRemoteServiceUrl,
  checkRemoteServiceHealth,
} from "../utils/remoteExecution";
import {
  effectiveWasmBridgeUrl,
  getWasmBridgeUrl,
  setWasmBridgeUrl,
  isWasmBridgeEnabled,
  setWasmBridgeEnabled,
} from "../utils/wasmLapackBridge";
import {
  syncVfsChangesToProject,
  syncSystemVfsChanges,
} from "../vfs/syncVfsChanges";
import type { VfsChanges } from "../vfs/VirtualFileSystem";
import { useSystemFiles } from "../hooks/useSystemFiles";
import { useMipCorePackage } from "../hooks/useMipCorePackage";
import { useNumblWorker } from "../hooks/useNumblWorker";
import { buildWorkerFiles } from "../buildWorkerFiles";
import type { WorkspaceRep } from "../numblWorkerProtocol";

export interface IDEWorkspaceProps {
  files: FileEntry[];
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
  duplicateFile: (fileId: string) => Promise<string>;
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
  /** Cap (px) on the initial output-panel height (desktop layout). When set,
   *  the figure panel below it starts with more room. */
  maxInitialOutputPanelHeight?: number;
  mergeVfsChanges?: (result: {
    addedFiles: FileEntry[];
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
  duplicateFile,
  uploadFiles,
  headerContent,
  projectName,
  loadFileContent,
  loadAllContents,
  contentCache,
  maxInitialOutputPanelHeight,
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
    duplicateSystemFile,
    loadSystemFileContent,
    getSystemVfsFiles,
    getSystemWorkspaceFiles,
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
  const [optimization, setOptimization] =
    useState<import("../numbl-core/executors/plugins.js").OptLevel>("1");
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
    (
      wsRep: WorkspaceRep | null
    ): { name: string; ast: unknown; irProgram: unknown }[] => {
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
  // When true, the figures panel overlays the whole window for a larger,
  // distraction-free view of the figures (toggle in the panel header / Escape).
  const [figuresExpanded, setFiguresExpanded] = useState(false);
  const [triggerRenameId, setTriggerRenameId] = useState<string | undefined>();

  // Markdown view mode for .md files
  const [mdViewMode, setMdViewMode] = useState<"rendered" | "source">(
    "rendered"
  );

  // REPL state
  const [editorTab, setEditorTab] = useState(0);
  const [isReplExecuting, setIsReplExecuting] = useState(false);
  const replTerminalRef = useRef<any>(null);

  // Mobile layout
  const isMobile = useMediaQuery("(max-width:768px)");
  // Keep the latest value in a ref so the stable handlePlotInstruction callback
  // can read it without being recreated (which would churn the worker).
  const isMobileRef = useRef(isMobile);
  isMobileRef.current = isMobile;
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
  // WASM matrix-acceleration settings (drafts edited in the dialog).
  const [wasmBridgeEnabledDraft, setWasmBridgeEnabledDraft] =
    useState(isWasmBridgeEnabled);
  const [wasmBridgeUrlDraft, setWasmBridgeUrlDraft] =
    useState(getWasmBridgeUrl);
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

  // Tracks which execution mode is active for routing streaming messages
  const activeExecutionMode = useRef<"script" | "repl" | null>(null);
  const remoteAbortRef = useRef<AbortController | null>(null);
  const editorRef = useRef<any>(null);

  // Persistent workspace toggle
  const [persistWorkspace, setPersistWorkspace] = useState(false);

  // Panel sizing
  const initialSidebarWidth = window.innerWidth >= 1200 ? 260 : 200;
  const [sidebarWidth, setSidebarWidth] = useState(initialSidebarWidth);
  const [editorWidth, setEditorWidth] = useState(
    (window.innerWidth - initialSidebarWidth) / 2
  );
  // Output panel starts at half the window height, optionally capped so the
  // figure panel below it starts larger (deployed sites can set this).
  const [outputHeight, setOutputHeight] = useState(() =>
    maxInitialOutputPanelHeight != null
      ? Math.min(window.innerHeight / 2, maxInitialOutputPanelHeight)
      : window.innerHeight / 2
  );
  // Let the splitter shrink to the requested initial height when it's below the
  // usual 150px floor, so a small cap is actually honored.
  const outputPanelMinSize =
    maxInitialOutputPanelHeight != null
      ? Math.min(150, maxInitialOutputPanelHeight)
      : 150;

  const activeFile = useMemo(
    () => allFiles.find(f => f.id === activeFileId),
    [allFiles, activeFileId]
  );

  // Open the workspace file targeted by a relative Markdown link (e.g. a
  // README linking to a script). Matches on the full relative path first,
  // then on the file's base name as a fallback.
  const handleMarkdownFileLink = useCallback(
    (href: string) => {
      let path = href.split(/[?#]/)[0];
      try {
        path = decodeURIComponent(path);
      } catch {
        /* keep raw path if it isn't valid percent-encoding */
      }
      path = path.replace(/^\.\//, "");
      if (!path) return;
      const base = (s: string) => s.split("/").pop() ?? s;
      const match =
        allFiles.find(f => f.name === path) ??
        allFiles.find(f => base(f.name) === base(path));
      if (match) {
        setActiveFileId(match.id);
        if (isMobile) setDrawerOpen(false);
      }
    },
    [allFiles, setActiveFileId, isMobile]
  );

  const sortedFigureHandles = useMemo(() => {
    return Object.keys(figures.figs)
      .map(k => Number(k))
      .sort((a, b) => a - b);
  }, [figures.figs]);

  const [downloadingFigure, setDownloadingFigure] = useState(false);
  const handleDownloadFigure = useCallback(async () => {
    const handle = sortedFigureHandles[figureTab];
    const fig = figures.figs[handle];
    if (!fig) return;
    setDownloadingFigure(true);
    try {
      await downloadFigureHdf5(fig, handle);
    } catch (e) {
      console.error("Figure HDF5 export failed:", e);
    } finally {
      setDownloadingFigure(false);
    }
  }, [figures.figs, sortedFigureHandles, figureTab]);

  // Hand-off banner state for figures too large to open via a direct link.
  const [viewerHandoff, setViewerHandoff] = useState<
    | { kind: "uploading" }
    | { kind: "ready"; url: string }
    | { kind: "failed"; baseUrl: string }
    | null
  >(null);
  const handleOpenFigureInViewer = useCallback(async () => {
    const fig = figures.figs[sortedFigureHandles[figureTab]];
    if (!fig) return;
    const link = buildFigureViewerLink(fig);
    if (link.url) {
      window.open(link.url, "_blank");
      setViewerHandoff(null);
      return;
    }
    setViewerHandoff({ kind: "uploading" });
    try {
      const url = await uploadFigureForViewer(fig);
      setViewerHandoff({ kind: "ready", url });
    } catch {
      setViewerHandoff({ kind: "failed", baseUrl: link.baseUrl });
    }
  }, [figures.figs, sortedFigureHandles, figureTab]);

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

    // On the mobile/compact layout, figures live behind a tab (desktop shows
    // them in a dedicated side panel). Bring that tab forward as soon as a run
    // draws something — same behavior as the embed widget. A drawing
    // instruction is any that isn't a control/setter (set_*) or a clear/close,
    // so this covers every plot type (surf, quiver, contour, …), not just a
    // fixed plot/plot3/surf list.
    if (!isMobileRef.current) return;
    const t = instruction.type;
    const isControl =
      t.startsWith("set_") ||
      t === "cla" ||
      t === "clf" ||
      t === "close" ||
      t === "close_all";
    if (!isControl) setMobileOutputTab(1);
  }, []);

  // Escape collapses the expanded (full-window) figures panel.
  useEffect(() => {
    if (!figuresExpanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFiguresExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [figuresExpanded]);

  /** Handle VFS changes from worker execution. */
  const handleVfsChanges = useCallback(
    async (changes: VfsChanges) => {
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

  // Unified worker (handles both script runs and REPL). Streaming output is
  // routed by which mode is active.
  const worker = useNumblWorker(
    {
      onOutput: text => {
        if (activeExecutionMode.current === "repl") {
          replTerminalRef.current?.writeOutput?.(text, false);
        } else {
          setOutput(prev => prev + text);
        }
      },
      onPlotInstructions: instructions => {
        for (const instr of instructions) handlePlotInstruction(instr);
      },
      onDone: msg => {
        activeExecutionMode.current = null;
        setGeneratedJS(msg.generatedJS || "");
        setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
        setFileSources(msg.workspaceRep?.fileSources ?? null);
        setDispatchUnknownCounts(msg.dispatchUnknownCounts ?? null);
        setIsRunning(false);
      },
      onError: msg => {
        activeExecutionMode.current = null;
        if (msg.generatedJS) setGeneratedJS(msg.generatedJS);
        setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
        setFileSources(msg.workspaceRep.fileSources);
        setIsRunning(false);
        setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
      },
      onResult: msg => {
        activeExecutionMode.current = null;
        const term = replTerminalRef.current;
        if (!msg.success && msg.error) term?.writeOutput?.(msg.error, true);
        term?.writePrompt?.();
        setIsReplExecuting(false);
      },
      onVfsChanges: changes => void handleVfsChanges(changes),
      onWorkerError: ev => {
        activeExecutionMode.current = null;
        replTerminalRef.current?.writeOutput?.(
          `Worker error: ${ev.message}`,
          true
        );
        setIsRunning(false);
        setIsReplExecuting(false);
      },
    },
    { optimization, wasmBridgeUrl: effectiveWasmBridgeUrl }
  );

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
    // Only .m files are runnable.
    if (!activeFile.name.toLowerCase().endsWith(".m")) return;

    setIsRunning(true);
    setOutput("");
    setDispatchUnknownCounts(null);
    setAllFilesRep([]);
    setFileSources(null);
    figuresDispatch({ type: "clear" });
    // On mobile, start on the Output tab; handlePlotInstruction switches to
    // Figures when the run draws one (matches the embed widget, and avoids
    // stranding the user on an empty Figures tab when a re-run draws nothing).
    setMobileOutputTab(0);

    // Load all file contents from DB
    const { vfsFiles, workspaceFiles, projectContents } =
      await buildWorkerFiles(
        {
          projectFiles: files,
          loadAllContents,
          getSystemVfsFiles,
          getSystemWorkspaceFiles,
        },
        activeFileId
      );

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

    // Send workspace update before run
    worker.updateWorkspace(workspaceFiles, vfsFiles);
    workspaceStale.current = false;

    activeExecutionMode.current = "script";
    worker.run({
      code: codeToRun,
      workspaceFiles,
      mainFileName: activeFile.name,
      options: {
        displayResults: true,
        maxIterations: 10000000,
        optimization,
      },
      vfsFiles,
      persistent: persistWorkspace,
    });
  }, [
    activeFile,
    activeFileId,
    files,
    useRemoteExecution,
    remoteServiceUrl,
    handlePlotInstruction,
    optimization,
    loadAllContents,
    getSystemVfsFiles,
    getSystemWorkspaceFiles,
    contentCache,
    persistWorkspace,
    worker,
  ]);

  const stopExecution = useCallback(() => {
    if (remoteAbortRef.current) {
      remoteAbortRef.current.abort();
      remoteAbortRef.current = null;
      setIsRunning(false);
      setOutput(prev => prev + "\n--- Execution stopped ---\n");
      return;
    }

    if (persistWorkspace) {
      // Persistent mode: cooperative cancellation (preserve worker state).
      // The worker reports "done", which clears isRunning.
      worker.cancel();
      setOutput(prev => prev + "\n--- Cancelling execution... ---\n");
    } else {
      // Non-persistent mode: terminate and recreate
      worker.restart();
      activeExecutionMode.current = null;
      workspaceStale.current = true;

      setIsRunning(false);
      setOutput(prev => prev + "\n--- Execution stopped ---\n");
    }
  }, [persistWorkspace, worker]);

  const handleLocalServerSettingsSave = useCallback(() => {
    setRemoteServiceUrl(localServerUrlDraft);
    saveRemoteServiceUrl(localServerUrlDraft);
    // Persist WASM accelerator settings and push the new effective URL to
    // the worker (null ⇒ uninstall) so it takes effect without a reload.
    setWasmBridgeEnabled(wasmBridgeEnabledDraft);
    setWasmBridgeUrl(wasmBridgeUrlDraft);
    worker.setWasmBridge(effectiveWasmBridgeUrl());
    setLocalServerSettingsOpen(false);
  }, [localServerUrlDraft, wasmBridgeEnabledDraft, wasmBridgeUrlDraft, worker]);

  const handleExecutionModeChange = useCallback(
    (_: React.MouseEvent<HTMLElement>, newMode: "browser" | "local" | null) => {
      if (newMode !== null) {
        const isLocal = newMode === "local";
        setUseRemoteExecution(isLocal);
      }
    },
    []
  );

  const handleReplExecute = useCallback(
    async (command: string) => {
      if (isReplExecuting || isRunning) return;
      setIsReplExecuting(true);

      // Send latest workspace files if stale
      if (workspaceStale.current) {
        const { vfsFiles, workspaceFiles } = await buildWorkerFiles({
          projectFiles: files,
          loadAllContents,
          getSystemVfsFiles,
          getSystemWorkspaceFiles,
        });
        worker.updateWorkspace(workspaceFiles, vfsFiles);
        workspaceStale.current = false;
      }

      activeExecutionMode.current = "repl";
      worker.execute(command);
    },
    [
      isReplExecuting,
      isRunning,
      files,
      loadAllContents,
      getSystemVfsFiles,
      getSystemWorkspaceFiles,
      worker,
    ]
  );

  const handleReplClear = useCallback(() => {
    if (isReplExecuting) return;
    worker.clear();
  }, [isReplExecuting, worker]);

  const handleTerminalReady = useCallback((methods: any) => {
    replTerminalRef.current = methods;
  }, []);

  const lastActiveFileId = useRef<string>("");
  const [activeFileData, setActiveFileData] = useState<Uint8Array | null>(null);
  const activeFileDataIdRef = useRef<string>("");
  const activeFileIsBinary = activeFileData
    ? isBinaryData(activeFileData)
    : false;
  const isMarkdown = !!activeFile?.name.toLowerCase().endsWith(".md");
  // Only .m files can be executed; non-.m files hide the run controls.
  const isRunnable = !!activeFile?.name.toLowerCase().endsWith(".m");

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

  // The Monaco editor unmounts when a binary file or the rendered-Markdown
  // view is shown. Drop the stale ref so the content-sync effect below doesn't
  // call setValue on a disposed editor; it remounts (with the right
  // defaultValue) when the editor is shown again.
  const editorHidden =
    activeFileIsBinary || (isMarkdown && mdViewMode === "rendered");
  useEffect(() => {
    if (editorHidden) {
      editorRef.current = null;
      lastActiveFileId.current = "";
    }
  }, [editorHidden]);

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
        {(editorTab === 0 || useRemoteExecution) && !activeFile ? (
          <Box
            sx={{
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexDirection: "column",
              gap: 2,
              p: 3,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              No file open
            </Typography>
            <Button
              variant="contained"
              size="small"
              startIcon={<NoteAddIcon />}
              onClick={() => addFile()}
              sx={{ textTransform: "none" }}
            >
              New file
            </Button>
          </Box>
        ) : editorTab === 0 || useRemoteExecution ? (
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
              {isRunnable && (
                <>
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
                      optimization === "0"
                        ? "Interpreter only (click for JIT)"
                        : "JIT (click to disable)"
                    }
                  >
                    <Typography
                      variant="caption"
                      onClick={() =>
                        setOptimization(o => (o === "0" ? "1" : "0"))
                      }
                      sx={{
                        cursor: "pointer",
                        fontSize: "0.7rem",
                        px: 0.5,
                        py: 0.1,
                        borderRadius: 0.5,
                        bgcolor:
                          optimization !== "0"
                            ? "action.selected"
                            : "transparent",
                        opacity: optimization !== "0" ? 1 : 0.5,
                        "&:hover": { opacity: 1 },
                        userSelect: "none",
                      }}
                    >
                      {optimization === "0" ? "no jit" : "jit"}
                    </Typography>
                  </Tooltip>
                </>
              )}
              {isMarkdown && (
                <ToggleButtonGroup
                  value={mdViewMode}
                  exclusive
                  size="small"
                  onChange={(_, v) => {
                    if (v) setMdViewMode(v);
                  }}
                  sx={{
                    "& .MuiToggleButton-root": {
                      py: 0,
                      px: 1,
                      fontSize: "0.7rem",
                      textTransform: "none",
                      lineHeight: 1.6,
                    },
                  }}
                >
                  <ToggleButton value="rendered">Rendered</ToggleButton>
                  <ToggleButton value="source">Source</ToggleButton>
                </ToggleButtonGroup>
              )}
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

            <Box sx={{ flexGrow: 1, minHeight: 0 }}>
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
              ) : isMarkdown && mdViewMode === "rendered" ? (
                <Box sx={{ height: "100%", overflow: "auto", p: 3 }}>
                  <MarkdownView
                    source={activeFileData ? fileText(activeFileData) : ""}
                    onFileLink={handleMarkdownFileLink}
                  />
                </Box>
              ) : (
                <Editor
                  height="100%"
                  language={
                    activeFile?.name.endsWith(".js")
                      ? "javascript"
                      : isMarkdown
                        ? "markdown"
                        : "numbl"
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
    <>
      <Box
        sx={
          figuresExpanded
            ? {
                position: "fixed",
                inset: 0,
                zIndex: 1250,
                bgcolor: "background.paper",
                display: "flex",
                flexDirection: "column",
              }
            : { height: "100%", display: "flex", flexDirection: "column" }
        }
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            borderBottom: 1,
            borderColor: "divider",
          }}
        >
          <Tabs
            value={figureTab}
            onChange={(_, newValue) => setFigureTab(newValue)}
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              flexGrow: 1,
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
          <Tooltip title="Open this figure in the numbl figure viewer (new tab)">
            <span>
              <IconButton
                size="small"
                onClick={handleOpenFigureInViewer}
                disabled={sortedFigureHandles.length === 0}
                sx={{ mx: 0.5, flexShrink: 0 }}
              >
                <OpenInNewIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Download this figure's data as HDF5 (.h5)">
            <span>
              <IconButton
                size="small"
                onClick={handleDownloadFigure}
                disabled={downloadingFigure || sortedFigureHandles.length === 0}
                sx={{ mx: 0.5, flexShrink: 0 }}
              >
                <SaveAltIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip
            title={
              figuresExpanded
                ? "Collapse figures (Esc)"
                : "Expand figures to full window"
            }
          >
            <IconButton
              size="small"
              onClick={() => setFiguresExpanded(v => !v)}
              sx={{ mx: 0.5, flexShrink: 0 }}
            >
              {figuresExpanded ? (
                <FullscreenExitIcon fontSize="small" />
              ) : (
                <FullscreenIcon fontSize="small" />
              )}
            </IconButton>
          </Tooltip>
        </Box>
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
      <Snackbar
        open={viewerHandoff !== null}
        onClose={() => setViewerHandoff(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          severity={viewerHandoff?.kind === "failed" ? "warning" : "info"}
          onClose={() => setViewerHandoff(null)}
          sx={{ width: "100%" }}
        >
          {viewerHandoff?.kind === "uploading" &&
            "Preparing a temporary link to the figure viewer…"}
          {viewerHandoff?.kind === "ready" && (
            <>
              Figure too large for a direct link — uploaded (encrypted,
              temporary).{" "}
              <Link href={viewerHandoff.url} target="_blank" rel="noreferrer">
                Open in figure viewer
              </Link>
            </>
          )}
          {viewerHandoff?.kind === "failed" && (
            <>
              This figure is too large to open through a link, and the temporary
              upload failed. Download it (
              <SaveAltIcon
                fontSize="inherit"
                sx={{ verticalAlign: "middle" }}
              />
              ) and{" "}
              <Link
                href={viewerHandoff.baseUrl}
                target="_blank"
                rel="noreferrer"
              >
                open the figure viewer
              </Link>
              , then load the file.
            </>
          )}
        </Alert>
      </Snackbar>
    </>
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
      onDuplicateFile={async fileId => {
        const id = isSystemFileId(fileId)
          ? await duplicateSystemFile(fileId)
          : await duplicateFile(fileId);
        if (id) setTriggerRenameId(id);
      }}
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
                  {mobileOutputTab === 2 && (
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
                  {mobileOutputTab === 3 && (
                    <TreeViewer
                      data={astData}
                      label="ast"
                      fileSources={fileSources}
                    />
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
                minSize={outputPanelMinSize}
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
          <Divider />
          <Box>
            <FormControlLabel
              control={
                <Switch
                  checked={wasmBridgeEnabledDraft}
                  onChange={e => setWasmBridgeEnabledDraft(e.target.checked)}
                  size="small"
                />
              }
              label="Accelerate matrix math (WASM)"
            />
            <Typography variant="caption" color="text.secondary" component="p">
              Loads an external WebAssembly LAPACK/BLAS bridge to speed up
              matrix multiplication in the browser. Falls back to the built-in
              pure-JS implementation when off or unavailable. Turning it off is
              temporary — it re-enables on page reload.
            </Typography>
            <TextField
              label="Bridge URL"
              value={wasmBridgeUrlDraft}
              onChange={e => setWasmBridgeUrlDraft(e.target.value)}
              size="small"
              fullWidth
              disabled={!wasmBridgeEnabledDraft}
              sx={{ mt: 1 }}
              slotProps={{
                input: {
                  style: { fontFamily: "monospace", fontSize: "0.85rem" },
                },
              }}
              helperText="Base URL serving numbl-bridge.json"
            />
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
