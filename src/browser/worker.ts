/**
 * The NumblSession worker: owns the VFS, bootstraps mip, restores/persists
 * the /system directory, optionally runs the main script, executes further
 * code incrementally against the persistent workspace, and keeps the latest
 * uihtml session live so dispatched events re-enter the interpreter.
 *
 * This file is bundled standalone at build time and inlined into
 * dist-browser/browser.js as a Blob-URL worker, so consumers need no
 * bundler support for dependency workers.
 */
import { executeCode } from "../numbl-core/executeCode.js";
import type { ExecResult, UihtmlSession } from "../numbl-core/executeCode.js";
import type { RuntimeValue } from "../numbl-core/runtime/index.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";
import { VirtualFileSystem } from "../vfs/VirtualFileSystem.js";
import { BrowserFileIOAdapter } from "../vfs/BrowserFileIOAdapter.js";
import { BrowserSystemAdapter } from "../vfs/BrowserSystemAdapter.js";
import type { PlotInstruction } from "../graphics/types.js";
import { formatExecuteError } from "../workerErrorFormat.js";
import { fetchMipCoreFiles, MIP_MARKER_PATH, MIP_SEARCH_PATH } from "./mip.js";
import { SystemStore } from "./system-store.js";
import type {
  BootMessage,
  FromWorker,
  ToWorker,
  UihtmlComponent,
} from "./protocol.js";

const post = (msg: FromWorker) => self.postMessage(msg);

let vfs: VirtualFileSystem | null = null;
let store: SystemStore | null = null;
let session: UihtmlSession | null = null;
const components: UihtmlComponent[] = [];
// Files written before boot finishes are stashed and applied once the VFS exists.
const pendingWrites: { path: string; content: string | Uint8Array }[] = [];

// Boot options reused by `execute`.
let mipEnabled = false;
let optimization: "0" | "1" = "1";
let maxIterations = 1e9;

// Workspace state carried across `execute` calls (REPL semantics).
let variableValues: Record<string, RuntimeValue> = {};
let holdState = false;
let persistentWorkspaceFiles: WorkspaceFile[] = [];
let persistentSearchPaths: string[] | undefined;
let implicitCwdPath: string | null | undefined;

const enc = new TextEncoder();

function toBytes(content: string | Uint8Array): Uint8Array {
  return typeof content === "string" ? enc.encode(content) : content;
}

/** Project-relative paths land under /project/. */
function projectPath(path: string): string {
  return path.startsWith("/") ? path : "/project/" + path;
}

function fileExists(fs: VirtualFileSystem, path: string): boolean {
  try {
    fs.readFile(path);
    return true;
  } catch {
    return false;
  }
}

function trackUihtml(instructions: PlotInstruction[]) {
  for (const pi of instructions) {
    if (pi.type !== "uihtml") continue;
    const comp = { compId: pi.id, html: pi.html, dataJson: pi.data ?? "" };
    const existing = components.findIndex(c => c.compId === comp.compId);
    if (existing >= 0) components[existing] = comp;
    else components.push(comp);
    post({
      type: "uihtml",
      compId: comp.compId,
      html: comp.html,
      dataJson: comp.dataJson,
    });
  }
}

/** Carry workspace state forward after a successful executeCode run. */
function adoptResultState(result: ExecResult) {
  variableValues = result.variableValues;
  holdState = result.holdState;
  if (result.searchPaths) {
    persistentSearchPaths = result.searchPaths;
    persistentWorkspaceFiles = result.workspaceFiles ?? [];
  }
  if (result.implicitCwdPath !== undefined) {
    implicitCwdPath = result.implicitCwdPath;
  }
}

function executeSearchPaths(): string[] {
  const paths = [...(persistentSearchPaths ?? [])];
  if (mipEnabled && !paths.includes(MIP_SEARCH_PATH)) {
    paths.push(MIP_SEARCH_PATH);
  }
  return paths;
}

async function persistSystemChanges() {
  if (!vfs) return;
  const changes = vfs.getChanges();
  vfs.clearChangeTracking();
  if (store) await store.applyChanges(changes);
}

async function boot(msg: BootMessage) {
  vfs = new VirtualFileSystem();
  mipEnabled = msg.mip;
  optimization = msg.optimization;
  maxIterations = msg.maxIterations;

  if (msg.persistSystem) {
    store = new SystemStore();
    const restored = await store.loadValid(msg.systemInactivityMs);
    for (const f of restored) vfs.writeFile(f.path, f.content);
    if (restored.length > 0) {
      post({
        type: "progress",
        message: `Restored ${restored.length} cached system files`,
      });
    }
  }

  for (const f of msg.files)
    vfs.writeFile(projectPath(f.path), toBytes(f.content));
  for (const w of pendingWrites)
    vfs.writeFile(projectPath(w.path), toBytes(w.content));
  pendingWrites.length = 0;

  const mainAbs = msg.mainFile
    ? vfs.normalizePath(projectPath(msg.mainFile))
    : null;
  if (mainAbs) {
    const lastSlash = mainAbs.lastIndexOf("/");
    vfs.setCwd(lastSlash > 0 ? mainAbs.slice(0, lastSlash) : "/");
  } else {
    vfs.setCwd("/project");
  }

  // Only changes made from here on (e.g. packages mip installs) persist.
  vfs.clearChangeTracking();

  if (msg.mip && !fileExists(vfs, MIP_MARKER_PATH)) {
    post({ type: "progress", message: "Fetching mip package manager…" });
    for (const f of await fetchMipCoreFiles()) vfs.writeFile(f.path, f.content);
  }

  const decoder = new TextDecoder("utf-8");
  persistentWorkspaceFiles = msg.files
    .filter(f => f.path.endsWith(".m"))
    .map(f => ({
      name: f.path,
      source:
        typeof f.content === "string" ? f.content : decoder.decode(f.content),
    }));

  if (mainAbs && msg.mainFile) {
    post({ type: "progress", message: "Running main script…" });
    const mainSource =
      persistentWorkspaceFiles.find(f => f.name === msg.mainFile)?.source ??
      decoder.decode(vfs.readFile(mainAbs));

    const result = executeCode(
      mainSource,
      {
        onOutput: text => post({ type: "output", text }),
        onDrawnow: instructions => trackUihtml(instructions),
        displayResults: msg.displayResults,
        maxIterations: msg.maxIterations,
        optimization: msg.optimization,
        fileIO: new BrowserFileIOAdapter(vfs),
        system: new BrowserSystemAdapter(vfs),
        onHtmlSourceEvent: (compId, name, dataJson) =>
          post({ type: "htmlSourceEvent", compId, name, dataJson }),
      },
      persistentWorkspaceFiles,
      mainAbs,
      msg.mip ? [MIP_SEARCH_PATH] : []
    );
    trackUihtml(result.plotInstructions);
    session = result.uihtmlSession ?? null;
    adoptResultState(result);
  }

  await persistSystemChanges();
  post({
    type: "ready",
    hasUihtmlSession: session !== null,
    components: [...components],
  });
}

function execute(id: number, code: string) {
  if (!vfs) {
    post({
      type: "executeResult",
      id,
      result: {
        ok: false,
        output: "",
        plotInstructions: [],
        error: "session not booted",
      },
    });
    return;
  }

  // A new execution supersedes any armed uihtml session.
  if (session) {
    session.dispose();
    session = null;
  }

  try {
    const result = executeCode(
      code,
      {
        onOutput: text => post({ type: "output", text }),
        onDrawnow: instructions => trackUihtml(instructions),
        displayResults: true,
        maxIterations,
        optimization,
        initialVariableValues: variableValues,
        initialHoldState: holdState,
        fileIO: new BrowserFileIOAdapter(vfs),
        system: new BrowserSystemAdapter(vfs),
        onHtmlSourceEvent: (compId, name, dataJson) =>
          post({ type: "htmlSourceEvent", compId, name, dataJson }),
        implicitCwdPath,
      },
      persistentWorkspaceFiles,
      "repl",
      executeSearchPaths()
    );
    trackUihtml(result.plotInstructions);
    session = result.uihtmlSession ?? null;
    adoptResultState(result);
    post({
      type: "executeResult",
      id,
      result: {
        ok: true,
        output: result.output.join(""),
        plotInstructions: result.plotInstructions,
      },
    });
  } catch (error: unknown) {
    // On error the workspace state is left unchanged.
    const { message } = formatExecuteError(error, code);
    post({
      type: "executeResult",
      id,
      result: { ok: false, output: "", plotInstructions: [], error: message },
    });
  }

  void persistSystemChanges();
  void store?.markActivity();
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;

  if (msg.type === "boot") {
    boot(msg).catch((err: unknown) => {
      post({
        type: "bootError",
        message: err instanceof Error ? err.message : String(err),
      });
    });
    return;
  }

  if (msg.type === "writeFile") {
    // The interpreter reads through the adapter on demand, so a VFS write
    // between events is visible to the next dispatched event.
    if (vfs) vfs.writeFile(projectPath(msg.path), toBytes(msg.content));
    else pendingWrites.push(msg);
    return;
  }

  if (msg.type === "readFile") {
    try {
      if (!vfs) throw new Error("session not booted");
      const content = vfs.readFile(projectPath(msg.path));
      post({ type: "readFileResult", id: msg.id, ok: true, content });
    } catch (err) {
      post({
        type: "readFileResult",
        id: msg.id,
        ok: false,
        message: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (msg.type === "execute") {
    execute(msg.id, msg.code);
    return;
  }

  if (msg.type === "dispatch") {
    if (!session) {
      post({
        type: "dispatchResult",
        id: msg.id,
        ok: false,
        message: "no live uihtml session",
      });
      return;
    }
    try {
      session.dispatchEvent(msg.compId, "HTMLEventReceived", {
        name: msg.name,
        data: msg.data,
      });
      post({ type: "dispatchResult", id: msg.id, ok: true });
    } catch (err) {
      post({
        type: "dispatchResult",
        id: msg.id,
        ok: false,
        message: err instanceof Error ? err.message : String(err),
      });
    }
    void persistSystemChanges();
    void store?.markActivity();
  }
};
