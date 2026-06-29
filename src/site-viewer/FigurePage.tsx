import GitHubIcon from "@mui/icons-material/GitHub";
import CodeIcon from "@mui/icons-material/Code";
import {
  Box,
  Button,
  CircularProgress,
  IconButton,
  Link,
  Tooltip,
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
import { createInputSAB, mainThreadRespond } from "../syncInputChannel";
import { useStaticProjectFiles } from "../hooks/useStaticProjectFiles";
import type { FigureSpec } from "../hooks/useStaticProjectFiles";
import { useSystemFiles } from "../hooks/useSystemFiles";
import {
  useMipCorePackage,
  markSystemActivity,
} from "../hooks/useMipCorePackage";
import { syncSystemVfsChanges } from "../vfs/syncVfsChanges";
import { formatDiagnostic } from "../numbl-core/diagnostics";
import type { PlotInstruction } from "../graphics/types.js";
import { FigureView } from "../graphics/FigureView.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../graphics/figuresReducer.js";

const NUMBL_ORIGIN = "https://numbl.org";

/**
 * Resolve which declared figure this route refers to. `#figure/<id>` selects by
 * slug; `#figure` (no slug) and an unknown slug both fall back to the first
 * declared figure (with a notice in the unknown case).
 */
function resolveFigure(
  figures: FigureSpec[],
  figureId: string | undefined
): { spec: FigureSpec | null; unknownSlug: boolean } {
  if (figures.length === 0) return { spec: null, unknownSlug: false };
  if (!figureId) return { spec: figures[0], unknownSlug: false };
  const match = figures.find(f => f.id === figureId);
  if (match) return { spec: match, unknownSlug: false };
  return { spec: figures[0], unknownSlug: true };
}

/**
 * Standalone, editor-less "figure view" for a statically-deployed numbl
 * project. Auto-runs a manifest-declared script in the same browser worker the
 * IDE uses, streams its console output, then shows the resulting figure. The
 * MATLAB ↔ JS `uihtml` bridge is wired both ways and the worker runs in
 * persistent mode, so an interactive widget's own controls stay live.
 *
 * Reached at `#figure` / `#figure/<id>`; the full IDE remains at `/`.
 */
export function FigurePage({ figureId }: { figureId?: string }) {
  const {
    files,
    loading,
    title,
    repository,
    figures,
    loadError,
    loadAllContents,
  } = useStaticProjectFiles();
  const { reloadSystemFiles, getSystemVfsFiles, getSystemWorkspaceFiles } =
    useSystemFiles();
  useMipCorePackage(reloadSystemFiles);

  const { spec, unknownSlug } = useMemo(
    () => resolveFigure(figures, figureId),
    [figures, figureId]
  );

  const [output, setOutput] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState(false);
  const [showOutput, setShowOutput] = useState(false);
  const [figuresState, figuresDispatch] = useReducer(
    figuresReducer,
    initialFiguresState
  );
  const workerRef = useRef<Worker | null>(null);
  const inputSAB = useRef<SharedArrayBuffer | null>(createInputSAB());
  // Guard so the auto-run fires once per resolved figure, not on every render.
  // Reset whenever the worker is (re)created so a fresh worker can re-run — e.g.
  // under React StrictMode's dev mount/unmount/remount.
  const startedRef = useRef<string | null>(null);

  const figureLabel = spec?.name || title || "Figure";

  useEffect(() => {
    document.title = figureLabel;
  }, [figureLabel]);

  const currentFig = useMemo(() => {
    const handles = Object.keys(figuresState.figs)
      .map(Number)
      .sort((a, b) => a - b);
    if (handles.length === 0) return null;
    return figuresState.figs[handles[handles.length - 1]];
  }, [figuresState.figs]);
  const hasFigure = currentFig !== null;

  const handlePlotInstruction = useCallback((instruction: PlotInstruction) => {
    figuresDispatch(instruction);
  }, []);

  const setupWorkerHandler = useCallback(
    (worker: Worker) => {
      worker.onmessage = e => {
        const msg = e.data;
        if (msg.type === "request-input") {
          const response = prompt(msg.prompt ?? "") ?? "";
          if (inputSAB.current) mainThreadRespond(inputSAB.current, response);
          return;
        }
        if (msg.type === "output") {
          setOutput(prev => prev + msg.text);
          return;
        }
        if (msg.type === "drawnow") {
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions)
              handlePlotInstruction(instr);
          }
          return;
        }
        // uihtml reverse channel: MATLAB -> JS (sendEventToHTMLSource). Relay
        // into the matching uihtml iframe rendered by FigureView; its bootstrap
        // filters by compId and fires the page's addEventListener(name) hooks.
        if (msg.type === "html_source_event") {
          const frames = document.querySelectorAll<HTMLIFrameElement>(
            `iframe[title="uihtml-${msg.compId}"]`
          );
          frames.forEach(f =>
            f.contentWindow?.postMessage(
              {
                source: "numbl-host",
                compId: msg.compId,
                name: msg.name,
                dataJson: msg.dataJson,
              },
              "*"
            )
          );
          return;
        }
        if (msg.type === "done") {
          setIsRunning(false);
          // Persist anything the run installed under /system/ (e.g. a package a
          // figure's `mip load --install` downloaded) for reuse on later runs.
          if (msg.vfsChanges) {
            void syncSystemVfsChanges(msg.vfsChanges).then(changed => {
              if (changed) reloadSystemFiles();
            });
          }
          if (msg.plotInstructions?.length) {
            for (const instr of msg.plotInstructions)
              handlePlotInstruction(instr);
          }
          return;
        }
        if (msg.type === "error") {
          setIsRunning(false);
          setRunError(true);
          setShowOutput(true);
          setOutput(prev => prev + `\n${formatDiagnostic(msg)}\n`);
          return;
        }
      };
    },
    [handlePlotInstruction, reloadSystemFiles]
  );

  // Initialize the worker once.
  useEffect(() => {
    const worker = new Worker(new URL("../numbl-worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;
    setupWorkerHandler(worker);
    // A fresh worker has no run yet — allow the auto-run effect to fire again.
    startedRef.current = null;
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [setupWorkerHandler]);

  // uihtml forward channel: a uihtml iframe posts an event to the window
  // (JS sendToMATLAB / Data setter). Forward it to the worker, which re-enters
  // the live (persistent) interpreter and fires HTMLEventReceivedFcn.
  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      const d = e.data;
      if (!d || d.source !== "numbl-uihtml") return;
      workerRef.current?.postMessage({
        type: "html_event",
        compId: d.compId,
        kind: d.kind,
        name: d.name,
        data: d.data,
      });
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  const runFigure = useCallback(async () => {
    const worker = workerRef.current;
    if (!worker || !spec) return;
    const entryFile = files.find(f => f.name === spec.entry);
    if (!entryFile) {
      setRunError(true);
      setShowOutput(true);
      setOutput(
        `Figure script not found in the project bundle: ${spec.entry}\n`
      );
      return;
    }

    setIsRunning(true);
    setRunError(false);
    setOutput("");
    figuresDispatch({ type: "clear" });
    // Keep the persistent system directory alive while this view is in use.
    markSystemActivity();

    const decoder = new TextDecoder("utf-8");
    const [projectContents, systemVfs, systemWs] = await Promise.all([
      loadAllContents(),
      getSystemVfsFiles(),
      getSystemWorkspaceFiles(),
    ]);

    const vfsFiles = [
      ...files.map(f => ({
        path: f.name,
        content: projectContents.get(f.id) ?? new Uint8Array(0),
      })),
      ...systemVfs,
    ];
    // Only .m files are sent as text workspace files; other assets reach the
    // worker via vfsFiles (mirrors IDEWorkspace.buildWorkerFiles).
    const workspaceFiles = [
      ...files
        .filter(f => f.name.endsWith(".m"))
        .map(f => ({
          name: f.name,
          source: decoder.decode(
            projectContents.get(f.id) ?? new Uint8Array(0)
          ),
        })),
      ...systemWs,
    ];
    const code = decoder.decode(
      projectContents.get(entryFile.id) ?? new Uint8Array(0)
    );

    worker.postMessage({ type: "update_workspace", workspaceFiles, vfsFiles });
    worker.postMessage({
      type: "run",
      code,
      options: {
        displayResults: true,
        maxIterations: 10000000,
        optimization: 1,
      },
      workspaceFiles,
      vfsFiles,
      mainFileName: entryFile.name,
      // Persistent so the figure's uihtml callbacks keep re-entering after the
      // script's top-level run completes.
      persistent: true,
      inputSAB: inputSAB.current ?? undefined,
    });
  }, [
    spec,
    files,
    loadAllContents,
    getSystemVfsFiles,
    getSystemWorkspaceFiles,
  ]);

  // Auto-run once the bundle has loaded and the figure is resolved. This kicks
  // off the imperative worker run (which updates state) on becoming ready — the
  // intended "synchronize with an external system" use of an effect.
  useEffect(() => {
    if (loading || !spec || !workerRef.current) return;
    if (startedRef.current === spec.id) return;
    startedRef.current = spec.id;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void runFigure();
  }, [loading, spec, runFigure]);

  const ideUrl = useMemo(() => {
    // Drop the hash to return to the IDE route at the same deploy path.
    return window.location.pathname + window.location.search;
  }, []);

  if (loading) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (!spec) {
    return (
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          gap: 1,
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
          textAlign: "center",
          px: 2,
        }}
      >
        <Typography variant="h6">No figure view configured</Typography>
        <Typography variant="body2" color="text.secondary">
          {loadError
            ? "The project bundle failed to load."
            : 'This project declares no "figures" in numbl-project.json.'}
        </Typography>
        <Link href={ideUrl} underline="hover">
          Open the full IDE
        </Link>
      </Box>
    );
  }

  const consoleVisible = showOutput || !hasFigure;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1,
          px: 1.5,
          py: 0.5,
          bgcolor: "grey.100",
          borderBottom: 1,
          borderColor: "divider",
          minHeight: 40,
        }}
      >
        <Typography variant="body2" fontWeight="medium" noWrap>
          {figureLabel}
        </Typography>
        {isRunning && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <CircularProgress size={12} />
            <Typography variant="caption" color="text.secondary">
              Running…
            </Typography>
          </Box>
        )}
        <Box sx={{ flexGrow: 1 }} />
        {hasFigure && (
          <Button
            size="small"
            onClick={() => setShowOutput(v => !v)}
            sx={{ textTransform: "none", fontSize: "0.75rem" }}
          >
            {showOutput ? "Hide output" : "Show output"}
          </Button>
        )}
        <Tooltip title="View / edit the code in the full IDE">
          <Button
            size="small"
            href={ideUrl}
            startIcon={<CodeIcon fontSize="small" />}
            sx={{ textTransform: "none", fontSize: "0.75rem" }}
          >
            Code
          </Button>
        </Tooltip>
        {repository && (
          <Tooltip title="View source repository">
            <IconButton
              size="small"
              component="a"
              href={repository}
              target="_blank"
              rel="noopener noreferrer"
            >
              <GitHubIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
        <Link
          href={NUMBL_ORIGIN}
          target="_blank"
          rel="noopener noreferrer"
          variant="caption"
          underline="hover"
          sx={{ opacity: 0.7, "&:hover": { opacity: 1 } }}
        >
          powered by numbl
        </Link>
      </Box>

      {unknownSlug && (
        <Box
          sx={{
            px: 1.5,
            py: 0.5,
            bgcolor: "#fff8e1",
            borderBottom: "1px solid #f0e0a0",
            fontSize: 13,
            color: "#5c4a00",
          }}
        >
          Unknown figure “{figureId}”. Showing “{figureLabel}” instead.
        </Box>
      )}

      {/* Body */}
      <Box sx={{ flexGrow: 1, position: "relative", minHeight: 0 }}>
        {/* Figure fills the body once available. */}
        {hasFigure && (
          <Box
            sx={{
              position: "absolute",
              inset: 0,
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              bgcolor: "#fff",
            }}
          >
            {currentFig && <FigureView figure={currentFig} />}
          </Box>
        )}

        {/* Console: full-body while waiting for the first figure, then a
            dismissible bottom overlay when the user asks to see output. */}
        {consoleVisible && (
          <Box
            sx={{
              position: "absolute",
              left: 0,
              right: 0,
              bottom: 0,
              top: hasFigure ? "auto" : 0,
              maxHeight: hasFigure ? "45%" : "100%",
              overflow: "auto",
              p: 1.5,
              bgcolor: "#1e1e1e",
              fontFamily: "monospace",
              fontSize: 13,
              color: runError ? "#f48771" : "#d4d4d4",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              borderTop: hasFigure ? "1px solid #333" : undefined,
            }}
          >
            {isRunning && !output && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <CircularProgress size={14} sx={{ color: "#d4d4d4" }} />
                <span>Running…</span>
              </Box>
            )}
            {output || (!isRunning && <span style={{ opacity: 0.5 }}>—</span>)}
          </Box>
        )}
      </Box>
    </Box>
  );
}
