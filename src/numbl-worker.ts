/**
 * Web Worker behind the web app: script runs ("run") and REPL commands
 * ("execute"), which share state through an ExecSession. Message types are in
 * numblWorkerProtocol.ts; the main-thread side is NumblWorkerClient.
 */

import { executeCode } from "./numbl-core/executeCode.js";
import { ExecSession } from "./numbl-core/execSession.js";
import { SessionState } from "./numbl-core/sessionState.js";
import { parseMFile } from "./numbl-core/parser/index.js";
import { CancellationError } from "./numbl-core/runtime/index.js";
import type { WorkspaceFile } from "./numbl-core/workspace/index.js";
import type { OptLevel } from "./numbl-core/executors/plugins.js";
import { diagnoseErrors } from "./numbl-core/diagnostics";
import { VirtualFileSystem } from "./vfs/VirtualFileSystem.js";
import { BrowserFileIOAdapter } from "./vfs/BrowserFileIOAdapter.js";
import { BrowserSystemAdapter } from "./vfs/BrowserSystemAdapter.js";
import { channelInput } from "./inputChannel.js";
import { formatExecuteError } from "./workerErrorFormat.js";
import { ensureQhullBackend } from "./numbl-core/native/qhull-browser.js";
import { ensureWasmLapackBridge } from "./numbl-core/native/wasm-lapack-browser.js";
import { IdbWasmCache } from "./utils/wasmByteCache.js";
import { startGpuHelper } from "./gpuHelper.js";
import { setJsGpuHelper } from "./numbl-core/jsUserFunctions.js";
import type {
  FromNumblWorker,
  RunMessage,
  ToNumblWorker,
  WorkerDiagnostic,
  WorkerVfsFile,
  WorkspaceRep,
} from "./numblWorkerProtocol.js";

function post(msg: FromNumblWorker): void {
  self.postMessage(msg);
}

// Start loading the qhull Delaunay backend as soon as the worker spins up, so
// it is installed well before any user-triggered run. Fire-and-forget: it
// resolves on its own, so the synchronous message handler below never needs to
// await it. A delaunay/delaunayn call issued before it finishes loading (or if
// loading fails) throws "backend not initialized".
void ensureQhullBackend();

// Spawn the WebGPU helper now: a worker created during a (synchronous) run
// would not start until the run ends. See gpuHelper.ts.
setJsGpuHelper(startGpuHelper());

// Byte cache for the optional WASM LAPACK accelerator. The accelerator itself
// is loaded lazily when the main thread sends `set_wasm_bridge` with the
// configured URL (workers can't read localStorage).
const wasmBridgeCache = new IdbWasmCache();

// ── Persistent state (used by REPL execute and persistent script runs) ──

const MIP_SEARCH_PATH = "/system/mip/packages/gh/mip-org/core/mip/mip";

// Workspace state carried across REPL commands and persistent runs. Its figure
// handles are updated after every run too: the IDE clears its figures at the
// start of each script run, so runs start from none, while REPL commands
// continue from here, so a bare `figure` opens a new figure.
const session = new ExecSession();
session.extraSearchPaths = [MIP_SEARCH_PATH];
let optimizationLevel: OptLevel = "1";
let vfs: VirtualFileSystem | null = null;
// Whether the host answers input() requests (see inputChannel.ts).
let inputEnabled = false;
const systemAdapter = new BrowserSystemAdapter();

// ── uihtml reverse channel (HTML → MATLAB) ──────────────────────────────
// When a run registers uihtml callbacks, the session keeps its UihtmlSession
// (and the live interpreter behind it) so iframe events can re-enter and fire
// the callbacks. A new run/execute or clear disposes it first (one armed
// runtime at a time).

/** ExecOptions hook: forward a `sendEventToHTMLSource` call to the host, which
 *  relays it into the target iframe. */
function onHtmlSourceEvent(
  compId: string,
  name: string,
  dataJson: string
): void {
  post({ type: "html_source_event", compId, name, dataJson });
}

// ── Script run helpers ──────────────────────────────────────────────────

function writeVfsFiles(vfs: VirtualFileSystem, files: WorkerVfsFile[]): void {
  for (const f of files) vfs.writeFile(f.path, f.content);
  vfs.clearChangeTracking();
}

function workspaceRepFor(code: string, fileName: string): WorkspaceRep {
  let ast: unknown = null;
  try {
    ast = parseMFile(code, fileName);
  } catch {
    // parsing may fail for partial code; ignore
  }
  return {
    mainFile: { name: fileName, ast, irProgram: null },
    workspaceFiles: [],
    fileSources: new Map<string, string>([[fileName, code]]),
  };
}

/** "done" for a cancelled run: no output, no AST. */
function postCancelled(adapter: BrowserFileIOAdapter): void {
  post({
    type: "done",
    generatedJS: undefined,
    outputCount: 0,
    workspaceRep: null,
    plotInstructions: [],
    vfsChanges: adapter.getChanges(),
  });
}

function workerDiagnostics(
  error: unknown,
  code: string,
  fileName: string,
  wsFiles: WorkspaceFile[]
): WorkerDiagnostic[] {
  return diagnoseErrors(error, code, fileName, wsFiles).map(d => ({
    message: d.message,
    errorType: d.errorType,
    identifier: d.identifier || undefined,
    file: d.file,
    line: d.line,
    snippet: d.snippet,
    callStack: d.callStack ?? null,
  }));
}

// ── Script execution ("run") ────────────────────────────────────────────

function handleRun(msg: RunMessage): void {
  const {
    code,
    preamble,
    options,
    workspaceFiles: wsFiles,
    mainFileName,
    searchPaths,
    vfsFiles,
    persistent,
    cancelSAB,
  } = msg;

  // A new run supersedes any armed uihtml session from a prior run.
  session.disposeUihtmlSession();

  // Choose VFS/adapter/session state based on persistent flag
  let sysAdapter: BrowserSystemAdapter;
  let runVfs: VirtualFileSystem;
  let runState: SessionState;
  let useHoldState: boolean | undefined;
  let useWorkspaceFiles: WorkspaceFile[];
  let useSearchPaths: string[] | undefined;

  if (persistent) {
    // Persistent mode: share state with REPL
    if (!vfs) {
      vfs = new VirtualFileSystem();
      systemAdapter.setVfs(vfs);
    }
    // Merge in any vfsFiles provided with the run message
    if (vfsFiles) writeVfsFiles(vfs, vfsFiles);
    sysAdapter = systemAdapter;
    runVfs = vfs;
    runState = session.state;
    useHoldState = session.holdState;
    useWorkspaceFiles = wsFiles.length > 0 ? wsFiles : session.workspaceFiles;
    useSearchPaths = searchPaths ?? session.searchPaths;
  } else {
    // Non-persistent mode: fresh state
    runVfs = new VirtualFileSystem();
    if (vfsFiles) writeVfsFiles(runVfs, vfsFiles);
    sysAdapter = new BrowserSystemAdapter(runVfs);
    runState = new SessionState();
    useHoldState = undefined;
    useWorkspaceFiles = wsFiles;
    useSearchPaths = searchPaths;
  }
  const adapter = new BrowserFileIOAdapter(runVfs);

  // Treat the directory of the script being run as the current working
  // directory, mirroring the CLI `run` command (which chdir's into
  // dirname(filepath)). Project files were written relative to the VFS
  // root, so resolving the script's project-relative name against the
  // current cwd (the root at this point) yields its absolute path; chdir
  // into its parent. A root-level script resolves to the root, leaving the
  // cwd unchanged. The directory then becomes the first-priority implicit
  // search path (see executeCode), so sibling functions and relative file
  // I/O resolve against the script's folder.
  const mainAbsPath = runVfs.normalizePath(mainFileName ?? "script.m");
  const lastSlash = mainAbsPath.lastIndexOf("/");
  runVfs.setCwd(lastSlash > 0 ? mainAbsPath.slice(0, lastSlash) : "/");

  // Run under the script's absolute VFS path so `mfilename('fullpath')` and
  // `fileparts(mfilename('fullpath'))` resolve to the project directory (e.g.
  // /project/refine_demo). Otherwise a bare name yields an empty directory.
  const activeFileName = mainAbsPath;

  const onInput = inputEnabled ? channelInput(cancelSAB) : undefined;

  // ── Preamble phase ──────────────────────────────────────────────────
  // Optional setup code (e.g. `mip load --install ...`) that runs before the
  // visible script, sharing the SAME VFS so anything it installs is available
  // to the main run. Its console output is captured (not forwarded) and is
  // surfaced only if the preamble itself fails — the embed page shows a
  // "Preparing…" message in the meantime. Shares the main run's session
  // state, and threads its search paths / workspace files into the main run.
  if (typeof preamble === "string" && preamble.trim()) {
    const preambleOutput: string[] = [];
    try {
      const pre = executeCode(
        preamble,
        {
          onOutput: (text: string) => {
            preambleOutput.push(text);
          },
          onDrawnow: () => {},
          displayResults: false,
          maxIterations: options?.maxIterations ?? 10000000,
          optimization: options?.optimization ?? optimizationLevel,
          session: runState,
          fileIO: adapter,
          system: sysAdapter,
          onInput,
          cancelSAB,
        },
        useWorkspaceFiles,
        runVfs.normalizePath("preamble.m"),
        [...(useSearchPaths ?? []), MIP_SEARCH_PATH]
      );
      // Thread the preamble's resulting paths into the main run.
      if (pre.searchPaths) useSearchPaths = pre.searchPaths;
      if (pre.workspaceFiles) useWorkspaceFiles = pre.workspaceFiles;
      // Note: change tracking is intentionally NOT cleared here — files the
      // preamble installs under /system/ flow through to `done`'s vfsChanges
      // so the host can persist them to the durable system directory.
      post({ type: "preamble_done" });
    } catch (error: unknown) {
      if (error instanceof CancellationError) {
        postCancelled(adapter);
        return;
      }
      const first = workerDiagnostics(
        error,
        preamble,
        "preamble.m",
        wsFiles
      )[0];
      post({
        type: "preamble_error",
        text: preambleOutput.join(""),
        message: first?.message ?? String(error),
        errorType: first?.errorType ?? "unknown",
        file: first?.file ?? null,
        line: first?.line ?? null,
        snippet: first?.snippet ?? null,
        callStack: first?.callStack ?? null,
      });
      return;
    }
  }

  try {
    const result = executeCode(
      code,
      {
        onOutput: (text: string) => {
          post({ type: "output", text });
        },
        onDrawnow: plotInstructions => {
          post({ type: "drawnow", plotInstructions });
        },
        displayResults: options?.displayResults ?? true,
        maxIterations: options?.maxIterations ?? 10000000,
        optimization: options?.optimization ?? optimizationLevel,
        session: runState,
        initialHoldState: useHoldState,
        fileIO: adapter,
        system: sysAdapter,
        onInput,
        onHtmlSourceEvent,
        cancelSAB,
      },
      useWorkspaceFiles,
      activeFileName,
      [...(useSearchPaths ?? []), MIP_SEARCH_PATH]
    );

    // A persistent run carries its state into the REPL. Either way, keep
    // the uihtml session alive if the run registered callbacks, so iframe
    // events can re-enter the interpreter (see "html_event").
    if (persistent) {
      session.adopt(result);
    } else {
      session.uihtmlSession = result.uihtmlSession ?? null;
      session.figureHandles = result.figureHandles;
    }

    post({
      type: "done",
      generatedJS: result.generatedJS,
      outputCount: result.output.length,
      workspaceRep: workspaceRepFor(code, activeFileName),
      plotInstructions: result.plotInstructions,
      dispatchUnknownCounts: result.dispatchUnknownCounts,
      vfsChanges: adapter.getChanges(),
    });
  } catch (error: unknown) {
    if (error instanceof CancellationError) {
      postCancelled(adapter);
      return;
    }

    const diags = workerDiagnostics(error, code, activeFileName, wsFiles);
    const generatedJS = (error as { generatedJS?: string } | null)?.generatedJS;
    const workspaceRep = workspaceRepFor(code, activeFileName);
    const vfsChanges = adapter.getChanges();
    diags.forEach((diag, i) => {
      const last = i === diags.length - 1;
      post({
        type: "error",
        ...diag,
        last,
        generatedJS,
        workspaceRep,
        vfsChanges: last ? vfsChanges : undefined,
      });
    });
  }
}

// ── REPL execution ("execute") ──────────────────────────────────────────

function handleExecute(code: string, cancelSAB?: SharedArrayBuffer): void {
  // Create adapter from persistent VFS
  if (!vfs) {
    vfs = new VirtualFileSystem();
    systemAdapter.setVfs(vfs);
  }
  const adapter = new BrowserFileIOAdapter(vfs);

  try {
    // Supersedes any armed uihtml session; keeps the new one if this
    // command registered uihtml callbacks.
    const result = session.execute(code, {
      onOutput: (text: string) => {
        post({ type: "output", text });
      },
      onDrawnow: plotInstructions => {
        post({ type: "drawnow", plotInstructions });
      },
      displayResults: true,
      maxIterations: 10000000,
      optimization: optimizationLevel,
      fileIO: adapter,
      system: systemAdapter,
      onInput: inputEnabled ? channelInput(cancelSAB) : undefined,
      onHtmlSourceEvent,
      cancelSAB,
    });

    post({
      type: "result",
      success: true,
      output: result.output.join(""),
      plotInstructions: result.plotInstructions,
      vfsChanges: adapter.getChanges(),
    });
  } catch (error: unknown) {
    // On error, what the command did before failing stays (MATLAB)
    const { message } = formatExecuteError(error, code);
    post({
      type: "result",
      success: false,
      error: message,
      vfsChanges: adapter.getChanges(),
    });
  }
}

// ── Worker message handler ──────────────────────────────────────────────

self.onmessage = (e: MessageEvent<ToNumblWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case "clear":
      session.clearWorkspace();
      vfs = null;
      post({ type: "cleared" });
      return;

    // uihtml reverse channel: an iframe event re-enters the live runtime.
    case "html_event": {
      const liveUihtmlSession = session.uihtmlSession;
      if (!liveUihtmlSession) return;
      const eventType =
        msg.kind === "dataChanged" ? "DataChanged" : "HTMLEventReceived";
      try {
        liveUihtmlSession.dispatchEvent(msg.compId, eventType, {
          name: msg.name,
          data: msg.data,
        });
      } catch (err) {
        post({
          type: "output",
          text: `Error in uihtml callback: ${
            err instanceof Error ? err.message : String(err)
          }\n`,
        });
      }
      return;
    }

    case "set_input":
      inputEnabled = msg.enabled;
      return;

    case "set_optimization":
      optimizationLevel = msg.optimization ?? optimizationLevel;
      return;

    case "set_wasm_bridge":
      // Best-effort: loads/instantiates off the message loop and installs the
      // accelerator when ready. A null/empty url uninstalls it.
      void ensureWasmLapackBridge(msg.url ?? null, { cache: wasmBridgeCache });
      return;

    case "update_workspace":
      session.workspaceFiles = msg.workspaceFiles || [];
      if (msg.searchPaths !== undefined) session.searchPaths = msg.searchPaths;
      if (msg.vfsFiles) {
        vfs = new VirtualFileSystem();
        writeVfsFiles(vfs, msg.vfsFiles);
        systemAdapter.setVfs(vfs);
      }
      return;

    case "run":
      handleRun(msg);
      return;

    case "execute":
      handleExecute(msg.code, msg.cancelSAB);
      return;
  }
};
