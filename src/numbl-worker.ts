/**
 * Web Worker that runs numbl script execution off the main thread.
 *
 * Protocol:
 *   Main → Worker:  { type: "run", code: string, options: { displayResults, maxIterations } }
 *   Worker → Main:  { type: "output", text: string }
 *   Worker → Main:  { type: "done", generatedJS: string, outputCount: number, irProgram?: object }
 *   Worker → Main:  { type: "error", message: string, line: number | null, file: string | null,
 *                      errorType: "syntax"|"semantic"|"runtime"|"unknown",
 *                      snippet: string | null, callStack?: CallFrame[] | null, generatedJS?: string }
 */

import { executeCode } from "./numbl-core/executeCode.js";
import { parseMFile } from "./numbl-core/parser/index.js";
import type { WorkspaceFile } from "./numbl-core/workspace/index.js";
import { diagnoseErrors } from "./numbl-core/diagnostics";
import { VirtualFileSystem } from "./vfs/VirtualFileSystem.js";
import { BrowserFileIOAdapter } from "./vfs/BrowserFileIOAdapter.js";
import { BrowserSystemAdapter } from "./vfs/BrowserSystemAdapter.js";
import { workerOnInput } from "./syncInputChannel.js";

/** Post a structured error message back to the main thread. */
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

// ── Worker message handler ───────────────────────────────────────────────────

self.onmessage = (e: MessageEvent) => {
  const {
    type,
    code,
    options,
    workspaceFiles,
    mainFileName,
    searchPaths,
    vfsFiles,
    inputSAB,
  } = e.data;
  if (type !== "run") return;

  const wsFiles: WorkspaceFile[] = workspaceFiles;
  const activeFileName: string = mainFileName ?? "script.m";
  let generatedJS: string | undefined;

  // Set up VFS if files were provided
  let adapter: BrowserFileIOAdapter | undefined;
  if (vfsFiles) {
    const vfs = new VirtualFileSystem();
    for (const f of vfsFiles as { path: string; content: Uint8Array }[]) {
      vfs.writeFile(f.path, f.content);
    }
    vfs.clearChangeTracking();
    adapter = new BrowserFileIOAdapter(vfs);
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
        optimization: options?.optimization ?? 1,
        initialVariableValues: {},
        fileIO: adapter,
        system: new BrowserSystemAdapter(),
        onInput: inputSAB ? workerOnInput(inputSAB) : undefined,
      },
      wsFiles,
      activeFileName,
      [...(searchPaths ?? []), "~/.mip/packages/mip-org/core/mip/mip"]
    );

    // Build a workspaceRep-like object for AST display
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
    const diags = diagnoseErrors(error, code, activeFileName, wsFiles);
    // Extract compilation artifacts attached to the error by the executor
    const errObj = error as Record<string, unknown> | null;
    const errGeneratedJS =
      (errObj?.generatedJS as string | undefined) ?? generatedJS;
    // Build AST for error display
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
};
