/**
 * Unified Web Worker for both script execution and REPL.
 *
 * Supports two execution modes:
 *   - "run": Script execution (stateless by default, or persistent if flagged)
 *   - "execute": REPL execution (always uses persistent state)
 *
 * Protocol:
 *   Main -> Worker:  { type: "run", code, options, workspaceFiles, mainFileName, searchPaths, vfsFiles, inputSAB, persistent?, cancelSAB? }
 *   Main -> Worker:  { type: "execute", code, cancelSAB? }
 *   Main -> Worker:  { type: "set_optimization", optimization }
 *   Main -> Worker:  { type: "set_fuse", fuse }
 *   Main -> Worker:  { type: "update_workspace", workspaceFiles, vfsFiles, searchPaths? }
 *   Main -> Worker:  { type: "set_input_sab", inputSAB }
 *   Main -> Worker:  { type: "clear" }
 *
 *   Worker -> Main:  { type: "output", text }
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
let optimizationLevel = 1;
let fuseEnabled = false;
let vfs: VirtualFileSystem | null = null;
let inputSAB: SharedArrayBuffer | null = null;
const systemAdapter = new BrowserSystemAdapter();

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
    variableValues = {};
    holdState = false;
    vfs = null;
    self.postMessage({ type: "cleared" });
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

  if (type === "set_fuse") {
    fuseEnabled = e.data.fuse ?? fuseEnabled;
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
      options,
      workspaceFiles,
      mainFileName,
      searchPaths,
      vfsFiles,
      persistent,
      cancelSAB,
    } = e.data;

    const wsFiles: WorkspaceFile[] = workspaceFiles;
    const activeFileName: string = mainFileName ?? "script.m";
    let generatedJS: string | undefined;

    // Choose VFS/adapter/variables based on persistent flag
    let adapter: BrowserFileIOAdapter | undefined;
    let sysAdapter: BrowserSystemAdapter;
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
      useVariableValues = variableValues;
      useHoldState = holdState;
      useWorkspaceFiles =
        wsFiles.length > 0 ? wsFiles : persistentWorkspaceFiles;
      useSearchPaths = searchPaths ?? persistentSearchPaths;
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
      useVariableValues = {};
      useHoldState = undefined;
      useWorkspaceFiles = wsFiles;
      useSearchPaths = searchPaths;
    }

    const runInputSAB = e.data.inputSAB ?? inputSAB;

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
          fuse: options?.fuse ?? fuseEnabled,
          initialVariableValues: useVariableValues,
          initialHoldState: useHoldState,
          fileIO: adapter,
          system: sysAdapter,
          onInput: runInputSAB ? workerOnInput(runInputSAB) : undefined,
          cancelSAB,
        },
        useWorkspaceFiles,
        activeFileName,
        [
          ...(useSearchPaths ?? []),
          "/system/.mip/packages/mip-org/core/mip/mip",
        ]
      );

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
        generatedC: result.generatedC,
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
        fuse: fuseEnabled,
        initialVariableValues: variableValues,
        initialHoldState: holdState,
        fileIO: adapter,
        system: systemAdapter,
        onInput: inputSAB ? workerOnInput(inputSAB) : undefined,
        implicitCwdPath,
        cancelSAB,
      },
      persistentWorkspaceFiles,
      "repl",
      [
        ...(persistentSearchPaths ?? []),
        "/system/.mip/packages/mip-org/core/mip/mip",
      ]
    );

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
