/**
 * Persistent Web Worker for REPL execution.
 *
 * Unlike the main script worker, this worker persists across executions
 * and maintains a mutable variable values object that carries over between
 * each code snippet execution.
 *
 * Protocol:
 *   Main → Worker:  { type: "execute", code: string }
 *   Main → Worker:  { type: "set_optimization", optimization: number }
 *   Main → Worker:  { type: "update_workspace", workspaceFiles: WorkspaceFile[] }
 *   Worker → Main:  { type: "output", text: string }
 *   Worker → Main:  { type: "drawnow", plotInstructions: object }
 *   Worker → Main:  { type: "result", success: boolean, output: string, error?: string, plotInstructions?: object }
 *   Main → Worker:  { type: "clear" }
 *   Worker → Main:  { type: "cleared" }
 */

import { executeCode } from "./numbl-core/executeCode.js";
import { SemanticError } from "./numbl-core/lowering/errors.js";
import { offsetToColumn, RuntimeError } from "./numbl-core/runtime/index.js";
import { SyntaxError } from "./numbl-core/parser/index.js";
import type { RuntimeValue } from "./numbl-core/runtime/index.js";
import type { WorkspaceFile } from "./numbl-core/workspace/index.js";
import { VirtualFileSystem } from "./vfs/VirtualFileSystem.js";
import { BrowserFileIOAdapter } from "./vfs/BrowserFileIOAdapter.js";

// ── Persistent state ─────────────────────────────────────────────────────────

let variableValues: Record<string, RuntimeValue> = {};
let holdState = false;
let workspaceFiles: WorkspaceFile[] = [];
let searchPaths: string[] | undefined;
let optimizationLevel = 1;
let vfs: VirtualFileSystem | null = null;

// ── Snippet helpers ──────────────────────────────────────────────────────────

/** Extract a code snippet (with context lines and pointer) around a 1-based line number. */
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

  const gutterWidth = 6; // "  NNNN | " prefix width
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

/** Format an error message for display */
function formatError(
  message: string,
  errorType: "syntax" | "semantic" | "runtime" | "unknown",
  line: number | null,
  snippet: string | null
): string {
  const parts: string[] = [];

  // Header: error type and location
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

  // Error message indented
  parts.push(`  ${message}`);

  // Code snippet if available
  if (snippet) {
    parts.push("");
    parts.push(snippet);
  }

  return parts.join("\n");
}

// ── Worker message handler ───────────────────────────────────────────────────

self.onmessage = (e: MessageEvent) => {
  const { type, code } = e.data;

  if (type === "clear") {
    variableValues = {};
    holdState = false;
    vfs = null;
    self.postMessage({ type: "cleared" });
    return;
  }

  if (type === "set_optimization") {
    optimizationLevel = e.data.optimization ?? optimizationLevel;
    return;
  }

  if (type === "update_workspace") {
    workspaceFiles = e.data.workspaceFiles || [];
    if (e.data.searchPaths !== undefined) {
      searchPaths = e.data.searchPaths;
    }
    // Update VFS with new workspace files
    if (e.data.vfsFiles) {
      vfs = new VirtualFileSystem();
      for (const f of e.data.vfsFiles as {
        path: string;
        content: Uint8Array;
      }[]) {
        vfs.writeFile(f.path, f.content);
      }
      vfs.clearChangeTracking();
    }
    return;
  }

  if (type !== "execute") return;

  // Create adapter from persistent VFS (or a fresh one if no VFS yet)
  if (!vfs) {
    vfs = new VirtualFileSystem();
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
      },
      workspaceFiles,
      "repl",
      searchPaths
    );

    // Update persistent state with results from this execution
    // IMPORTANT: Only update on success - if an error occurs, we skip this line
    // and preserve the existing state
    variableValues = result.variableValues;
    holdState = result.holdState;

    const vfsChanges = adapter.getChanges();
    self.postMessage({
      type: "result",
      success: true,
      output: result.output.join(""),
      plotInstructions: result.plotInstructions,
      vfsChanges,
    });
  } catch (error: unknown) {
    // On error, variableValues remains unchanged - variables are preserved
    const errVfsChanges = adapter.getChanges();

    // ── RuntimeError ─────────────────────────────────────────────────────────
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

      // ── SyntaxError ──────────────────────────────────────────────────────────
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

      // ── SemanticError (with span) ─────────────────────────────────────────────
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

      // ── SemanticError (without span) ─────────────────────────────────────────
    } else if (error instanceof SemanticError) {
      const errorMsg = formatError(error.message, "semantic", null, null);
      self.postMessage({
        type: "result",
        success: false,
        error: errorMsg,
        vfsChanges: errVfsChanges,
      });

      // ── Unexpected errors ─────────────────────────────────────────────────────
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
