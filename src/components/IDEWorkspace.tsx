/* eslint-disable @typescript-eslint/no-explicit-any */
import { createInputSAB, mainThreadRespond } from "../syncInputChannel";
import Editor, { OnMount } from "@monaco-editor/react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import js from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import { githubGist } from "react-syntax-highlighter/dist/esm/styles/hljs";

SyntaxHighlighter.registerLanguage("javascript", js);
const _textEncoder = new TextEncoder();
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
  isRemoteExecutionEnabled,
  setRemoteExecutionEnabled,
  executeRemoteStream,
  getRemoteServiceUrl,
  setRemoteServiceUrl as saveRemoteServiceUrl,
  checkRemoteServiceHealth,
  DEFAULT_REMOTE_SERVICE_URL,
} from "../utils/remoteExecution";
import {
  syncVfsChangesToProject,
  syncHomeVfsChanges,
} from "../vfs/syncVfsChanges";
import type { VfsChanges } from "../vfs/VirtualFileSystem";
import { useHomeFiles } from "../hooks/useHomeFiles";
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
    homeFiles,
    reloadHomeFiles,
    updateHomeFileContent,
    addHomeFile,
    addHomeFolder,
    deleteHomeFile,
    deleteHomeFolder,
    renameHomeFile,
    renameHomeFolder,
    moveHomeFile,
    loadHomeFileContent,
    getHomeVfsFiles,
    getHomeWorkspaceFiles,
    loadAllHomeContents,
  } = useHomeFiles();

  useMipCorePackage(reloadHomeFiles);

  const isHomePath = useCallback(
    (name: string) => name === "~" || name.startsWith("~/"),
    []
  );
  const isHomeFileId = useCallback(
    (fileId: string) => homeFiles.some(f => f.id === fileId),
    [homeFiles]
  );

  // Merged file list: project files + home files (metadata only — no content blobs)
  const allFiles = useMemo(() => [...files, ...homeFiles], [files, homeFiles]);
  const optimization = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return parseInt(params.get("opt") ?? "1", 10);
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
  const scriptInputSAB = useRef<SharedArrayBuffer | null>(createInputSAB());
  const replInputSAB = useRef<SharedArrayBuffer | null>(createInputSAB());
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
      const [projectContents, homeVfs, homeWs] = await Promise.all([
        loadAllContents(),
        getHomeVfsFiles(),
        getHomeWorkspaceFiles(),
      ]);
      const homeContents = await loadAllHomeContents();

      const vfsFiles = [
        ...wsFiles.map(f => ({
          path: f.name,
          content: projectContents.get(f.id) ?? new Uint8Array(0),
        })),
        ...homeVfs,
      ];

      const decoder = new TextDecoder("utf-8");
      const workspaceFiles = [
        ...wsFiles
          .filter(f => !excludeFileId || f.id !== excludeFileId)
          .map(f => ({
            name: f.name,
            source: decoder.decode(
              projectContents.get(f.id) ?? new Uint8Array(0)
            ),
          })),
        ...homeWs,
      ];

      return { vfsFiles, workspaceFiles, projectContents, homeContents };
    },
    [
      loadAllContents,
      getHomeVfsFiles,
      getHomeWorkspaceFiles,
      loadAllHomeContents,
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
        const { projectResult, homeResult } = await syncVfsChangesToProject(
          projectName,
          changes
        );
        if (projectResult && mergeVfsChanges) {
          mergeVfsChanges(projectResult);
        }
        if (homeResult) {
          reloadHomeFiles();
        }
      } else {
        // No project (e.g. share route) — still sync home file changes
        const synced = await syncHomeVfsChanges(changes);
        if (synced) {
          reloadHomeFiles();
        }
      }
    },
    [projectName, mergeVfsChanges, reloadHomeFiles]
  );

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
      if (msg.type === "request-input") {
        const response = prompt(msg.prompt ?? "") ?? "";
        const sab = scriptInputSAB.current;
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
        handleVfsChanges(msg.vfsChanges);
      } else if (msg.type === "error") {
        if (msg.generatedJS) {
          setGeneratedJS(msg.generatedJS);
        }
        setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
        setFileSources(msg.workspaceRep?.fileSources ?? null);
        setIsRunning(false);
        setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
        handleVfsChanges(msg.vfsChanges);
      }
    };

    return () => {
      scriptWorkerRef.current?.terminate();
    };
  }, [handlePlotInstruction, extractAllFilesRep, handleVfsChanges]);

  // Initialize REPL worker
  useEffect(() => {
    const worker = new Worker(
      new URL("../numbl-repl-worker.ts", import.meta.url),
      { type: "module" }
    );
    replWorkerRef.current = worker;

    if (replInputSAB.current) {
      worker.postMessage({
        type: "set_input_sab",
        inputSAB: replInputSAB.current,
      });
    }

    // Don't send workspace files on init — content is loaded lazily
    // and will be sent before the first REPL execute or script run.

    worker.onmessage = e => {
      const msg = e.data;
      const term = replTerminalRef.current;

      if (msg.type === "request-input") {
        const response = prompt(msg.prompt ?? "") ?? "";
        const sab = replInputSAB.current;
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
  }, [handlePlotInstruction, handleVfsChanges]);

  // Track that REPL workspace needs updating when files change
  const replWorkspaceStale = useRef(true);
  useEffect(() => {
    replWorkspaceStale.current = true;
  }, [files, homeFiles]);

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
        const remoteFiles = [
          ...workspaceFiles.map(f => ({
            name: f.name,
            content: f.source,
          })),
          { name: activeFile.name, content: codeToRun },
        ];

        const result = await executeRemoteStream(
          {
            files: remoteFiles,
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
      workspaceFiles,
      mainFileName: activeFile.name,
      options: {
        displayResults: true,
        maxIterations: 10000000,
        optimization,
      },
      vfsFiles,
      inputSAB: scriptInputSAB.current ?? undefined,
    });
  }, [
    activeFile,
    activeFileId,
    files,
    useRemoteExecution,
    remoteServiceUrl,
    handlePlotInstruction,
    optimization,
    buildWorkerFiles,
    contentCache,
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
        if (msg.type === "request-input") {
          const response = prompt(msg.prompt ?? "") ?? "";
          const sab = scriptInputSAB.current;
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
          handleVfsChanges(msg.vfsChanges);
        } else if (msg.type === "error") {
          if (msg.generatedJS) {
            setGeneratedJS(msg.generatedJS);
          }
          setAllFilesRep(extractAllFilesRep(msg.workspaceRep));
          setFileSources(msg.workspaceRep?.fileSources ?? null);
          setIsRunning(false);
          setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
          handleVfsChanges(msg.vfsChanges);
        }
      };

      setIsRunning(false);
      setOutput(prev => prev + "\n--- Execution stopped ---\n");
    }
  }, [handlePlotInstruction, extractAllFilesRep, handleVfsChanges]);

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

      // Send latest workspace files if stale
      if (replWorkspaceStale.current && replWorkerRef.current) {
        const { vfsFiles, workspaceFiles } = await buildWorkerFiles(files);
        replWorkerRef.current.postMessage({
          type: "update_workspace",
          workspaceFiles,
          vfsFiles,
        });
        replWorkspaceStale.current = false;
      }

      replWorkerRef.current?.postMessage({
        type: "execute",
        code: command,
      });
    },
    [isReplExecuting, files, buildWorkerFiles]
  );

  const handleReplClear = useCallback(() => {
    if (isReplExecuting) return;
    replWorkerRef.current?.postMessage({ type: "clear" });
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
    const isHome = homeFiles.some(f => f.id === activeFileId);
    const loader = isHome ? loadHomeFileContent : loadFileContent;
    loader(activeFileId).then(data => {
      if (!cancelled) {
        activeFileDataIdRef.current = activeFileId;
        setActiveFileData(data);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [activeFileId, homeFiles, loadHomeFileContent, loadFileContent]);

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
        value={editorTab}
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
        <Tab label="REPL" />
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
                    replWorkspaceStale.current = true;
                    if (isHomeFileId(activeFileId)) {
                      updateHomeFileContent(activeFileId, encoded);
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
        if (folderPath && isHomePath(folderPath)) {
          const id = await addHomeFile(folderPath);
          if (id) setTriggerRenameId(id);
        } else {
          const id = await addFile(folderPath);
          if (id) setTriggerRenameId(id);
        }
      }}
      onAddFolder={async parentPath => {
        if (parentPath && isHomePath(parentPath)) {
          const folderPath = await addHomeFolder(parentPath);
          if (folderPath) setTriggerRenameId(`folder:${folderPath}`);
        } else {
          const folderPath = await addFolder(parentPath);
          if (folderPath) setTriggerRenameId(`folder:${folderPath}`);
        }
      }}
      onDeleteFile={fileId =>
        isHomeFileId(fileId) ? deleteHomeFile(fileId) : deleteFile(fileId)
      }
      onDeleteFolder={folderPath =>
        isHomePath(folderPath)
          ? deleteHomeFolder(folderPath)
          : deleteFolder(folderPath)
      }
      onRenameFile={(fileId, newName) =>
        isHomeFileId(fileId)
          ? renameHomeFile(fileId, newName)
          : renameFile(fileId, newName)
      }
      onRenameFolder={(oldPath, newName) =>
        isHomePath(oldPath)
          ? renameHomeFolder(oldPath, newName)
          : renameFolder(oldPath, newName)
      }
      onMoveFile={(fileId, targetFolder) =>
        isHomeFileId(fileId)
          ? moveHomeFile(fileId, targetFolder)
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
