import MenuIcon from "@mui/icons-material/Menu";
import { Box, Drawer, IconButton, useMediaQuery } from "@mui/material";
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { OptLevel } from "../numbl-core/executors/plugins.js";
import type { PlotInstruction } from "../numbl-core/plotting/types.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../numbl-core/plotting/figuresReducer.js";
import { formatDiagnostic } from "../numbl-core/diagnostics";
import { Splitter } from "./Splitter";
import { ReplView, type TerminalMethods } from "./ReplView";
import { fileText } from "../hooks/useProjectFiles";
import type { FileEntry } from "../hooks/projectFilesState";
import { effectiveWasmBridgeUrl } from "../utils/wasmLapackBridge";
import {
  syncVfsChangesToProject,
  syncSystemVfsChanges,
} from "../vfs/syncVfsChanges";
import type { VfsChanges } from "../vfs/VirtualFileSystem";
import { useSystemFiles } from "../hooks/useSystemFiles";
import { useMipCorePackage } from "../hooks/useMipCorePackage";
import { useExecutionBackends } from "../hooks/useExecutionBackends";
import { useRemoteExecution } from "../hooks/useRemoteExecution";
import { useActiveFileContent } from "../hooks/useActiveFileContent";
import { buildWorkerFiles } from "../buildWorkerFiles";
import type { WorkspaceRep } from "../numblWorkerProtocol";
import type {
  ExecutionBackend,
  ExecutionEvents,
} from "../execution/ExecutionBackend";
import { EditorPanel, RunControls, ScriptPane } from "./ide/EditorPanel";
import { OutputPanel } from "./ide/OutputPanel";
import { FiguresPanel } from "./ide/FiguresPanel";
import { sortedHandles } from "./ide/figureHandles";
import { MobileOutputPanel } from "./ide/MobileOutputPanel";
import {
  WorkspaceFileBrowser,
  type ProjectFileOps,
} from "./ide/WorkspaceFileBrowser";
import { ExecutionSettingsDialog } from "./ide/ExecutionSettingsDialog";

const textEncoder = new TextEncoder();

export interface IDEWorkspaceProps extends ProjectFileOps {
  files: FileEntry[];
  activeFileId: string;
  setActiveFileId: (id: string) => void;
  updateFileContent: (content: string) => void;
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

export function IDEWorkspace(props: IDEWorkspaceProps) {
  const {
    files,
    activeFileId,
    setActiveFileId,
    updateFileContent,
    headerContent,
    projectName,
    loadFileContent,
    loadAllContents,
    contentCache,
    maxInitialOutputPanelHeight,
    mergeVfsChanges,
  } = props;
  const system = useSystemFiles();
  const {
    systemFiles,
    reloadSystemFiles,
    updateSystemFileContent,
    loadSystemFileContent,
    getSystemVfsFiles,
    getSystemWorkspaceFiles,
  } = system;

  useMipCorePackage(reloadSystemFiles);

  // Merged file list: project files + system files (metadata only — no content blobs)
  const allFiles = useMemo(
    () => [...files, ...systemFiles],
    [files, systemFiles]
  );
  const activeFile = useMemo(
    () => allFiles.find(f => f.id === activeFileId),
    [allFiles, activeFileId]
  );

  const [optimization, setOptimization] = useState<OptLevel>("1");
  const [output, setOutput] = useState("");
  const [dispatchUnknownCounts, setDispatchUnknownCounts] = useState<Record<
    string,
    number
  > | null>(null);
  const [generatedJS, setGeneratedJS] = useState("");
  const [workspaceRep, setWorkspaceRep] = useState<WorkspaceRep | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [figures, figuresDispatch] = useReducer(
    figuresReducer,
    initialFiguresState
  );
  const [figureTab, setFigureTab] = useState(0);
  const [mdViewMode, setMdViewMode] = useState<"rendered" | "source">(
    "rendered"
  );

  // REPL state
  const [editorTab, setEditorTab] = useState(0);
  const [isReplExecuting, setIsReplExecuting] = useState(false);
  const replTerminalRef = useRef<TerminalMethods | null>(null);

  // Mobile layout
  const isMobile = useMediaQuery("(max-width:768px)");
  // Keep the latest value in a ref so the stable handlePlotInstruction callback
  // can read it without being recreated (which would churn the worker).
  const isMobileRef = useRef(isMobile);
  useEffect(() => {
    isMobileRef.current = isMobile;
  }, [isMobile]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mobileOutputTab, setMobileOutputTab] = useState(0);

  const remote = useRemoteExecution();
  const [settingsOpen, setSettingsOpen] = useState(false);

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

  // Select the current figure's tab when the script switches figures.
  useEffect(() => {
    const idx = sortedHandles(figures.figs).indexOf(figures.currentHandle);
    // The tab stays user-selectable, so it is state rather than derived.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (idx >= 0) setFigureTab(idx);
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

  // Runs go to the browser worker or, when enabled, the execution server; the
  // REPL always uses the worker.
  const { worker, active: backend } = useExecutionBackends(remote, {
    optimization,
    wasmBridgeUrl: effectiveWasmBridgeUrl,
  });
  // The backend running the current script, which Stop must reach even if
  // the execution target changed meanwhile.
  const runningBackend = useRef<ExecutionBackend | null>(null);

  // Track that workspace needs updating when files change
  const workspaceStale = useRef(true);
  useEffect(() => {
    workspaceStale.current = true;
  }, [files, systemFiles]);

  const executeCode = useCallback(async () => {
    if (!activeFile) return;
    // Only .m files are runnable.
    if (!activeFile.name.toLowerCase().endsWith(".m")) return;

    setIsRunning(true);
    setOutput("");
    setDispatchUnknownCounts(null);
    setWorkspaceRep(null);
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

    const events: ExecutionEvents = {
      onOutput: text => setOutput(prev => prev + text),
      onPlotInstructions: instructions => {
        for (const instr of instructions) handlePlotInstruction(instr);
      },
      onError: diagnostic =>
        setOutput(prev => prev + `\n${formatDiagnostic(diagnostic)}\n`),
      onVfsChanges: changes => void handleVfsChanges(changes),
      onDone: result => {
        runningBackend.current = null;
        setGeneratedJS(result.generatedJS || "");
        setWorkspaceRep(result.workspaceRep ?? null);
        setDispatchUnknownCounts(result.dispatchUnknownCounts ?? null);
        setIsRunning(false);
      },
    };

    runningBackend.current = backend;
    backend.run(
      {
        code: fileText(activeData),
        mainFileName: activeFile.name,
        vfsFiles,
        workspaceFiles,
        optimization,
        persistent: persistWorkspace,
      },
      events
    );
    // The worker backend also hands the files to the REPL.
    if (backend === worker) workspaceStale.current = false;
  }, [
    activeFile,
    activeFileId,
    files,
    backend,
    worker,
    handlePlotInstruction,
    handleVfsChanges,
    optimization,
    loadAllContents,
    getSystemVfsFiles,
    getSystemWorkspaceFiles,
    contentCache,
    persistWorkspace,
  ]);

  const stopExecution = useCallback(() => {
    const running = runningBackend.current ?? backend;
    if (running.stop() === "stopping") {
      // Cooperative cancellation keeps the worker's state; the run's onDone
      // clears isRunning.
      setOutput(prev => prev + "\n--- Cancelling execution... ---\n");
      return;
    }
    // A replaced worker has lost its workspace files.
    if (running === worker) workspaceStale.current = true;
    setIsRunning(false);
    setOutput(prev => prev + "\n--- Execution stopped ---\n");
  }, [backend, worker]);

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
        worker.client.updateWorkspace(workspaceFiles, vfsFiles);
        workspaceStale.current = false;
      }

      const term = () => replTerminalRef.current;
      worker.execute(command, {
        onOutput: text => term()?.writeOutput(text, false),
        onPlotInstructions: instructions => {
          for (const instr of instructions) handlePlotInstruction(instr);
        },
        onVfsChanges: changes => void handleVfsChanges(changes),
        onResult: result => {
          if (!result.success && result.error) {
            term()?.writeOutput(result.error, true);
          }
          term()?.writePrompt();
          setIsReplExecuting(false);
        },
      });
    },
    [
      isReplExecuting,
      isRunning,
      files,
      loadAllContents,
      getSystemVfsFiles,
      getSystemWorkspaceFiles,
      worker,
      handlePlotInstruction,
      handleVfsChanges,
    ]
  );

  const handleReplClear = useCallback(() => {
    if (isReplExecuting) return;
    worker.clear();
  }, [isReplExecuting, worker]);

  const handleTerminalReady = useCallback((methods: TerminalMethods) => {
    replTerminalRef.current = methods;
  }, []);

  // Active file content, reloaded when the file or the system files change.
  const loadActiveFile = useCallback(
    (fileId: string) =>
      systemFiles.some(f => f.id === fileId)
        ? loadSystemFileContent(fileId)
        : loadFileContent(fileId),
    [systemFiles, loadSystemFileContent, loadFileContent]
  );
  const [activeFileData, setActiveFileData] = useActiveFileContent(
    activeFileId,
    loadActiveFile
  );

  const handleEditorChange = useCallback(
    (value: string) => {
      const encoded = textEncoder.encode(value);
      setActiveFileData(encoded);
      workspaceStale.current = true;
      if (systemFiles.some(f => f.id === activeFileId)) {
        updateSystemFileContent(activeFileId, encoded);
      } else {
        updateFileContent(value);
      }
    },
    [
      setActiveFileData,
      systemFiles,
      activeFileId,
      updateSystemFileContent,
      updateFileContent,
    ]
  );

  // Render IDE

  const editor = (
    <EditorPanel
      showRepl={backend.capabilities.repl}
      tab={editorTab}
      onTabChange={setEditorTab}
      script={
        <ScriptPane
          file={activeFile}
          content={activeFileData}
          runControls={
            <RunControls
              isRunning={isRunning}
              onRun={executeCode}
              onStop={stopExecution}
              remote={remote}
              onOpenSettings={() => setSettingsOpen(true)}
              persistWorkspace={persistWorkspace}
              onTogglePersist={() => setPersistWorkspace(p => !p)}
              optimization={optimization}
              onToggleOptimization={() =>
                setOptimization(o => (o === "0" ? "1" : "0"))
              }
            />
          }
          mdViewMode={mdViewMode}
          onMdViewModeChange={setMdViewMode}
          onChange={handleEditorChange}
          onMarkdownFileLink={handleMarkdownFileLink}
          onAddFile={() => props.addFile()}
        />
      }
      repl={
        <ReplView
          onExecute={handleReplExecute}
          onClear={handleReplClear}
          isExecuting={isReplExecuting}
          onTerminalReady={handleTerminalReady}
        />
      }
    />
  );

  const fileBrowser = (
    <WorkspaceFileBrowser
      files={allFiles}
      activeFileId={activeFileId}
      onSelectFile={id => {
        setActiveFileId(id);
        if (isMobile) setDrawerOpen(false);
      }}
      project={props}
      system={system}
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
              {fileBrowser}
            </Drawer>
            <Splitter
              direction="horizontal"
              initialSize={Math.round(window.innerHeight * 0.55)}
              minSize={100}
            >
              {editor}
              <MobileOutputPanel
                tab={mobileOutputTab}
                onTabChange={setMobileOutputTab}
                output={output}
                figures={figures}
                figureTab={figureTab}
                onFigureTabChange={setFigureTab}
                generatedJS={generatedJS}
                workspaceRep={workspaceRep}
              />
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
            {fileBrowser}
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
                <OutputPanel
                  output={output}
                  generatedJS={generatedJS}
                  workspaceRep={workspaceRep}
                  dispatchUnknownCounts={dispatchUnknownCounts}
                />
                <FiguresPanel
                  figures={figures}
                  figureTab={figureTab}
                  onFigureTabChange={setFigureTab}
                />
              </Splitter>
            </Splitter>
          </Splitter>
        )}
      </Box>

      <ExecutionSettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        remote={remote}
        onWasmBridgeChange={url => worker.client.setWasmBridge(url)}
      />
    </>
  );
}
