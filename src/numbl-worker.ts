/**
 * Unified Web Worker for both script execution and REPL.
 *
 * Supports two execution modes:
 *   - "run": Script execution (stateless by default, or persistent if flagged)
 *   - "execute": REPL execution (always uses persistent state)
 *
 * Protocol:
 *   Main -> Worker:  { type: "run", code, preamble?, options, workspaceFiles, mainFileName, searchPaths, vfsFiles, inputSAB, persistent?, persistVfs?, cancelSAB? }
 *   Main -> Worker:  { type: "execute", code, cancelSAB? }
 *   Main -> Worker:  { type: "set_optimization", optimization }
 *   Main -> Worker:  { type: "update_workspace", workspaceFiles, vfsFiles, searchPaths? }
 *   Main -> Worker:  { type: "set_input_sab", inputSAB }
 *   Main -> Worker:  { type: "clear" }
 *
 *   Worker -> Main:  { type: "output", text }
 *   Worker -> Main:  { type: "preamble_done" } (preamble ran OK; main run starting)
 *   Worker -> Main:  { type: "preamble_error", text, message, errorType, file, line, snippet, callStack }
 *   Worker -> Main:  { type: "drawnow", plotInstructions }
 *   Worker -> Main:  { type: "done", generatedJS, outputCount, workspaceRep?, plotInstructions?, dispatchUnknownCounts?, vfsChanges? }
 *   Worker -> Main:  { type: "error", message, errorType, file, line, snippet, callStack?, generatedJS?, workspaceRep?, vfsChanges? }
 *   Worker -> Main:  { type: "result", success, output?, error?, plotInstructions?, vfsChanges? }
 *   Worker -> Main:  { type: "cleared" }
 */

import { executeCode } from "./numbl-core/executeCode.js";
import { parseMFile } from "./numbl-core/parser/index.js";
import { SemanticError } from "./numbl-core/lowering/errors.js";
import {
  offsetToColumn,
  RuntimeError,
  CancellationError,
} from "./numbl-core/runtime/index.js";
import { SyntaxError } from "./numbl-core/parser/index.js";
import type { RuntimeValue } from "./numbl-core/runtime/index.js";
import type { WorkspaceFile } from "./numbl-core/workspace/index.js";
import { diagnoseErrors } from "./numbl-core/diagnostics";
import { VirtualFileSystem } from "./vfs/VirtualFileSystem.js";
import { BrowserFileIOAdapter } from "./vfs/BrowserFileIOAdapter.js";
import { BrowserSystemAdapter } from "./vfs/BrowserSystemAdapter.js";
import { workerOnInput } from "./syncInputChannel.js";

// ── Persistent state (used by REPL execute and persistent script runs) ──

let variableValues: Record<string, RuntimeValue> = {};
let holdState = false;
let persistentWorkspaceFiles: WorkspaceFile[] = [];
let persistentSearchPaths: string[] | undefined;
let implicitCwdPath: string | null | undefined;
let optimizationLevel: import("./numbl-core/executors/plugins.js").OptLevel =
  "1";
let vfs: VirtualFileSystem | null = null;
// VFS that persists across runs for `persistVfs` runs (the embed). Lets a
// preamble install (e.g. `mip load --install ...`) survive between runs so the
// install only downloads once per worker session — mip skips the download when
// the package directory already exists. Variables are NOT persisted; each run
// starts clean (see the persistVfs branch in the "run" handler).
let embedFsVfs: VirtualFileSystem | null = null;
let inputSAB: SharedArrayBuffer | null = null;
const systemAdapter = new BrowserSystemAdapter();

// ── uihtml reverse channel (HTML → MATLAB) ──────────────────────────────
// When a run registers uihtml callbacks, we keep its session (and the live
// interpreter behind it) so iframe events can re-enter and fire the callbacks.
// A new run/execute or clear disposes it first (one armed runtime at a time).
let liveUihtmlSession:
  | import("./numbl-core/executeCode.js").UihtmlSession
  | null = null;

function disposeUihtmlSession(): void {
  if (liveUihtmlSession) {
    liveUihtmlSession.dispose();
    liveUihtmlSession = null;
  }
}

/** ExecOptions hook: forward a `sendEventToHTMLSource` call to the host, which
 *  relays it into the target iframe. */
function onHtmlSourceEvent(
  compId: string,
  name: string,
  dataJson: string
): void {
  self.postMessage({ type: "html_source_event", compId, name, dataJson });
}

// ── Snippet helpers (used by REPL error formatting) ─────────────────────

function extractSnippetByLine(
  source: string,
  lineNumber: number,
  contextLines = 2,
  column?: number
): string | null {
  if (lineNumber < 1) return null;
  const lines = source.split("\n");
  if (lineNumber > lines.length) return null;

  const startLine = Math.max(1, lineNumber - contextLines);
  const endLine = Math.min(lines.length, lineNumber + contextLines);

  const gutterWidth = 6;
  const result: string[] = [];
  for (let i = startLine; i <= endLine; i++) {
    const num = i.toString().padStart(4, " ");
    const marker = i === lineNumber ? ">" : " ";
    result.push(`${marker}${num} | ${lines[i - 1]}`);
    if (i === lineNumber && column && column >= 1) {
      result.push(" ".repeat(gutterWidth) + " ".repeat(column - 1) + "^");
    }
  }
  return result.join("\n");
}

function formatError(
  message: string,
  errorType: "syntax" | "semantic" | "runtime" | "unknown",
  line: number | null,
  snippet: string | null
): string {
  const parts: string[] = [];
  const errorKind =
    errorType === "syntax"
      ? "Syntax error"
      : errorType === "semantic"
        ? "Semantic error"
        : errorType === "runtime"
          ? "Runtime error"
          : "Error";
  const loc = line !== null ? `at line ${line}` : null;
  parts.push(loc ? `${errorKind} ${loc}:` : `${errorKind}:`);
  parts.push(`  ${message}`);
  if (snippet) {
    parts.push("");
    parts.push(snippet);
  }
  return parts.join("\n");
}

// ── Script run helpers ──────────────────────────────────────────────────

function postError(
  message: string,
  errorType: "syntax" | "semantic" | "runtime" | "unknown",
  file: string | null,
  line: number | null,
  snippet: string | null,
  callStack: unknown,
  generatedJS: string | undefined,
  workspaceRep?: unknown,
  vfsChanges?: unknown
): void {
  self.postMessage({
    type: "error",
    message,
    errorType,
    file,
    line,
    snippet,
    callStack,
    generatedJS,
    workspaceRep,
    vfsChanges,
  });
}

// ── Worker message handler ──────────────────────────────────────────────

self.onmessage = (e: MessageEvent) => {
  const { type } = e.data;

  // ── Configuration messages ──────────────────────────────────────────

  if (type === "clear") {
    disposeUihtmlSession();
    variableValues = {};
    holdState = false;
    vfs = null;
    embedFsVfs = null;
    self.postMessage({ type: "cleared" });
    return;
  }

  // ── uihtml reverse channel: an iframe event re-enters the live runtime ──
  if (type === "html_event") {
    if (!liveUihtmlSession) return;
    const { compId, kind, name, data } = e.data;
    const eventType =
      kind === "dataChanged" ? "DataChanged" : "HTMLEventReceived";
    try {
      liveUihtmlSession.dispatchEvent(compId, eventType, { name, data });
    } catch (err) {
      self.postMessage({
        type: "output",
        text: `Error in uihtml callback: ${
          err instanceof Error ? err.message : String(err)
        }\n`,
      });
    }
    return;
  }

  if (type === "set_input_sab") {
    inputSAB = e.data.inputSAB ?? null;
    return;
  }

  if (type === "set_optimization") {
    optimizationLevel = e.data.optimization ?? optimizationLevel;
    return;
  }

  if (type === "update_workspace") {
    persistentWorkspaceFiles = e.data.workspaceFiles || [];
    if (e.data.searchPaths !== undefined) {
      persistentSearchPaths = e.data.searchPaths;
    }
    if (e.data.vfsFiles) {
      vfs = new VirtualFileSystem();
      for (const f of e.data.vfsFiles as {
        path: string;
        content: Uint8Array;
      }[]) {
        vfs.writeFile(f.path, f.content);
      }
      vfs.clearChangeTracking();
      systemAdapter.setVfs(vfs);
    }
    return;
  }

  // ── Script execution ("run") ────────────────────────────────────────

  if (type === "run") {
    const {
      code,
      preamble,
      options,
      workspaceFiles,
      mainFileName,
      searchPaths,
      vfsFiles,
      persistent,
      persistVfs,
      cancelSAB,
    } = e.data;

    // A new run supersedes any armed uihtml session from a prior run.
    disposeUihtmlSession();

    const wsFiles: WorkspaceFile[] = workspaceFiles;
    let activeFileName: string = mainFileName ?? "script.m";
    let generatedJS: string | undefined;

    // Choose VFS/adapter/variables based on persistent flag
    let adapter: BrowserFileIOAdapter | undefined;
    let sysAdapter: BrowserSystemAdapter;
    let runVfs: VirtualFileSystem;
    let useVariableValues: Record<string, RuntimeValue>;
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
      if (vfsFiles) {
        for (const f of vfsFiles as {
          path: string;
          content: Uint8Array;
        }[]) {
          vfs.writeFile(f.path, f.content);
        }
        vfs.clearChangeTracking();
      }
      adapter = new BrowserFileIOAdapter(vfs);
      sysAdapter = systemAdapter;
      runVfs = vfs;
      useVariableValues = variableValues;
      useHoldState = holdState;
      useWorkspaceFiles =
        wsFiles.length > 0 ? wsFiles : persistentWorkspaceFiles;
      useSearchPaths = searchPaths ?? persistentSearchPaths;
    } else if (persistVfs) {
      // Persistent-VFS mode (the embed): the file system survives across runs
      // so a preamble install caches, but variables/exec state are fresh each
      // run. Incoming vfsFiles (the system/mip-core files) are merged in,
      // overwriting their own paths but leaving anything installed by an
      // earlier run (e.g. a downloaded package) in place.
      if (!embedFsVfs) embedFsVfs = new VirtualFileSystem();
      if (vfsFiles) {
        for (const f of vfsFiles as {
          path: string;
          content: Uint8Array;
        }[]) {
          embedFsVfs.writeFile(f.path, f.content);
        }
      }
      embedFsVfs.clearChangeTracking();
      adapter = new BrowserFileIOAdapter(embedFsVfs);
      sysAdapter = new BrowserSystemAdapter(embedFsVfs);
      runVfs = embedFsVfs;
      useVariableValues = {};
      useHoldState = undefined;
      useWorkspaceFiles = wsFiles;
      useSearchPaths = searchPaths;
    } else {
      // Non-persistent mode: fresh state
      const freshVfs = new VirtualFileSystem();
      if (vfsFiles) {
        for (const f of vfsFiles as {
          path: string;
          content: Uint8Array;
        }[]) {
          freshVfs.writeFile(f.path, f.content);
        }
        freshVfs.clearChangeTracking();
      }
      adapter = new BrowserFileIOAdapter(freshVfs);
      sysAdapter = new BrowserSystemAdapter(freshVfs);
      runVfs = freshVfs;
      useVariableValues = {};
      useHoldState = undefined;
      useWorkspaceFiles = wsFiles;
      useSearchPaths = searchPaths;
    }

    // Treat the directory of the script being run as the current working
    // directory, mirroring the CLI `run` command (which chdir's into
    // dirname(filepath)). Project files were written relative to the VFS
    // root, so resolving the script's project-relative name against the
    // current cwd (the root at this point) yields its absolute path; chdir
    // into its parent. A root-level script resolves to the root, leaving the
    // cwd unchanged. The directory then becomes the first-priority implicit
    // search path (see executeCode), so sibling functions and relative file
    // I/O resolve against the script's folder.
    const mainAbsPath = runVfs.normalizePath(activeFileName);
    const lastSlash = mainAbsPath.lastIndexOf("/");
    runVfs.setCwd(lastSlash > 0 ? mainAbsPath.slice(0, lastSlash) : "/");

    // Run under the script's absolute VFS path so `mfilename('fullpath')` and
    // `fileparts(mfilename('fullpath'))` resolve to the project directory (e.g.
    // /project/refine_demo). Otherwise a bare name yields an empty directory.
    activeFileName = mainAbsPath;

    const runInputSAB = e.data.inputSAB ?? inputSAB;

    // ── Preamble phase ──────────────────────────────────────────────────
    // Optional setup code (e.g. `mip load --install ...`) that runs before the
    // visible script, sharing the SAME VFS so anything it installs is available
    // to the main run. Its console output is captured (not forwarded) and is
    // surfaced only if the preamble itself fails — the embed page shows a
    // "Preparing…" message in the meantime. Threads the preamble's resulting
    // variables / search paths / workspace files into the main run.
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
            initialVariableValues: useVariableValues,
            fileIO: adapter,
            system: sysAdapter,
            onInput: runInputSAB ? workerOnInput(runInputSAB) : undefined,
            cancelSAB,
          },
          useWorkspaceFiles,
          runVfs.normalizePath("preamble.m"),
          [
            ...(useSearchPaths ?? []),
            "/system/mip/packages/gh/mip-org/core/mip/mip",
          ]
        );
        // Thread the preamble's resulting state into the main run.
        useVariableValues = pre.variableValues;
        if (pre.searchPaths) useSearchPaths = pre.searchPaths;
        if (pre.workspaceFiles) useWorkspaceFiles = pre.workspaceFiles;
        // Don't report the package install as user-visible VFS changes.
        runVfs.clearChangeTracking();
        self.postMessage({ type: "preamble_done" });
      } catch (error: unknown) {
        if (error instanceof CancellationError) {
          self.postMessage({
            type: "done",
            generatedJS: undefined,
            outputCount: 0,
            workspaceRep: null,
            plotInstructions: [],
            vfsChanges: adapter?.getChanges(),
          });
          return;
        }
        const diags = diagnoseErrors(error, preamble, "preamble.m", wsFiles);
        const first = diags[0];
        self.postMessage({
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
            self.postMessage({ type: "output", text });
          },
          onDrawnow: plotInstructions => {
            self.postMessage({ type: "drawnow", plotInstructions });
          },
          displayResults: options?.displayResults ?? true,
          maxIterations: options?.maxIterations ?? 10000000,
          optimization: options?.optimization ?? optimizationLevel,
          initialVariableValues: useVariableValues,
          initialHoldState: useHoldState,
          fileIO: adapter,
          system: sysAdapter,
          onInput: runInputSAB ? workerOnInput(runInputSAB) : undefined,
          onHtmlSourceEvent,
          cancelSAB,
        },
        useWorkspaceFiles,
        activeFileName,
        [
          ...(useSearchPaths ?? []),
          "/system/mip/packages/gh/mip-org/core/mip/mip",
        ]
      );

      // Keep the session alive if the run registered uihtml callbacks, so
      // iframe events can re-enter the interpreter (see "html_event").
      liveUihtmlSession = result.uihtmlSession ?? null;

      // Update persistent state if in persistent mode
      if (persistent) {
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

      // Build workspaceRep for AST display
      let ast: unknown = null;
      try {
        ast = parseMFile(code, activeFileName);
      } catch {
        // parsing may fail for partial code; ignore
      }
      const workspaceRep = {
        mainFile: { name: activeFileName, ast, irProgram: null },
        workspaceFiles: [] as {
          name: string;
          ast: unknown;
          irProgram: unknown;
        }[],
        fileSources: new Map<string, string>([[activeFileName, code]]),
      };

      const vfsChanges = adapter?.getChanges();
      self.postMessage({
        type: "done",
        generatedJS: result.generatedJS,
        outputCount: result.output.length,
        workspaceRep,
        plotInstructions: result.plotInstructions,
        dispatchUnknownCounts: result.dispatchUnknownCounts,
        vfsChanges,
      });
    } catch (error: unknown) {
      if (error instanceof CancellationError) {
        // Cancellation: report as done with no output
        self.postMessage({
          type: "done",
          generatedJS: undefined,
          outputCount: 0,
          workspaceRep: null,
          plotInstructions: [],
          vfsChanges: adapter?.getChanges(),
        });
        return;
      }

      const diags = diagnoseErrors(error, code, activeFileName, wsFiles);
      const errObj = error as Record<string, unknown> | null;
      const errGeneratedJS =
        (errObj?.generatedJS as string | undefined) ?? generatedJS;
      let errAst: unknown = null;
      try {
        errAst = parseMFile(code, activeFileName);
      } catch {
        // ignore parse errors here
      }
      const errWorkspaceRep = {
        mainFile: { name: activeFileName, ast: errAst, irProgram: null },
        workspaceFiles: [] as {
          name: string;
          ast: unknown;
          irProgram: unknown;
        }[],
        fileSources: new Map<string, string>([[activeFileName, code]]),
      };
      const errVfsChanges = adapter?.getChanges();
      for (const diag of diags) {
        postError(
          diag.message,
          diag.errorType,
          diag.file,
          diag.line,
          diag.snippet,
          diag.callStack,
          errGeneratedJS,
          errWorkspaceRep,
          errVfsChanges
        );
      }
    }
    return;
  }

  // ── REPL execution ("execute") ──────────────────────────────────────

  if (type !== "execute") return;

  const { code, cancelSAB } = e.data;

  // A new REPL command supersedes any armed uihtml session.
  disposeUihtmlSession();

  // Create adapter from persistent VFS
  if (!vfs) {
    vfs = new VirtualFileSystem();
    systemAdapter.setVfs(vfs);
  }
  const adapter = new BrowserFileIOAdapter(vfs);

  try {
    const result = executeCode(
      code,
      {
        onOutput: (text: string) => {
          self.postMessage({ type: "output", text });
        },
        onDrawnow: plotInstructions => {
          self.postMessage({ type: "drawnow", plotInstructions });
        },
        displayResults: true,
        maxIterations: 10000000,
        optimization: optimizationLevel,
        initialVariableValues: variableValues,
        initialHoldState: holdState,
        fileIO: adapter,
        system: systemAdapter,
        onInput: inputSAB ? workerOnInput(inputSAB) : undefined,
        onHtmlSourceEvent,
        implicitCwdPath,
        cancelSAB,
      },
      persistentWorkspaceFiles,
      "repl",
      [
        ...(persistentSearchPaths ?? []),
        "/system/mip/packages/gh/mip-org/core/mip/mip",
      ]
    );

    // Keep the session if this command registered uihtml callbacks.
    liveUihtmlSession = result.uihtmlSession ?? null;

    // Update persistent state on success
    variableValues = result.variableValues;
    holdState = result.holdState;
    if (result.searchPaths) {
      persistentSearchPaths = result.searchPaths;
      persistentWorkspaceFiles = result.workspaceFiles ?? [];
    }
    if (result.implicitCwdPath !== undefined) {
      implicitCwdPath = result.implicitCwdPath;
    }

    const vfsChanges = adapter.getChanges();
    self.postMessage({
      type: "result",
      success: true,
      output: result.output.join(""),
      plotInstructions: result.plotInstructions,
      vfsChanges,
    });
  } catch (error: unknown) {
    // On error, variableValues remains unchanged
    const errVfsChanges = adapter.getChanges();

    if (error instanceof CancellationError) {
      self.postMessage({
        type: "result",
        success: false,
        error: "Execution cancelled",
        vfsChanges: errVfsChanges,
      });
      return;
    }

    if (error instanceof RuntimeError) {
      const snippet =
        error.line !== null
          ? extractSnippetByLine(code, error.line, 2, error.column ?? undefined)
          : null;
      const errorMsg = formatError(
        error.message,
        "runtime",
        error.line,
        snippet
      );
      self.postMessage({
        type: "result",
        success: false,
        error: errorMsg,
        vfsChanges: errVfsChanges,
      });
    } else if (error instanceof SyntaxError) {
      const col = error.column ?? offsetToColumn(code, error.position);
      const snippet =
        error.line !== null
          ? extractSnippetByLine(code, error.line, 2, col)
          : null;
      const errorMsg = formatError(
        error.message,
        "syntax",
        error.line,
        snippet
      );
      self.postMessage({
        type: "result",
        success: false,
        error: errorMsg,
        vfsChanges: errVfsChanges,
      });
    } else if (error instanceof SemanticError && error.span !== null) {
      const snippet =
        error.line !== null
          ? extractSnippetByLine(code, error.line, 2, error.column)
          : null;
      const errorMsg = formatError(
        error.message,
        "semantic",
        error.line,
        snippet
      );
      self.postMessage({
        type: "result",
        success: false,
        error: errorMsg,
        vfsChanges: errVfsChanges,
      });
    } else if (error instanceof SemanticError) {
      const errorMsg = formatError(error.message, "semantic", null, null);
      self.postMessage({
        type: "result",
        success: false,
        error: errorMsg,
        vfsChanges: errVfsChanges,
      });
    } else if (error instanceof Error) {
      const errorMsg = formatError(error.message, "unknown", null, null);
      self.postMessage({
        type: "result",
        success: false,
        error: errorMsg,
        vfsChanges: errVfsChanges,
      });
    } else {
      const errorMsg = formatError(String(error), "unknown", null, null);
      self.postMessage({
        type: "result",
        success: false,
        error: errorMsg,
        vfsChanges: errVfsChanges,
      });
    }
  }
};
