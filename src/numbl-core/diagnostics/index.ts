/**
 * Shared error diagnostic utilities for extracting structured error info
 * with source context (snippets). Used by both the CLI and the web worker.
 */

import {
  RuntimeError,
  offsetToColumn,
  offsetToLine,
  type CallFrame,
} from "../runtime/error.js";
import { SyntaxError } from "../parser/errors.js";
import { SemanticError } from "../lowering/errors.js";
import type { WorkspaceFile } from "../workspace/index.js";
import { CompilationErrors } from "./errors.js";

interface DiagnosticInfo {
  message: string;
  errorType: "syntax" | "semantic" | "runtime" | "unknown";
  file: string | null;
  line: number | null;
  snippet: string | null;
  callStack?: CallFrame[] | null;
}

/** Look up source text for a given filename. */
export function getSourceForFile(
  file: string,
  mainFileName: string,
  mainSource: string,
  wsFiles: WorkspaceFile[] | undefined
): string | null {
  if (file === mainFileName) return mainSource;
  return wsFiles?.find(f => f.name === file)?.source ?? null;
}

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

/**
 * Extract structured diagnostic info from any error, returning an array.
 * For CompilationErrors, returns one DiagnosticInfo per inner error.
 * For all other errors, returns a single-element array.
 */
export function diagnoseErrors(
  error: unknown,
  mainSource: string,
  mainFileName: string,
  wsFiles?: WorkspaceFile[]
): DiagnosticInfo[] {
  if (error instanceof CompilationErrors) {
    return error.errors.map(e =>
      diagnoseError(e, mainSource, mainFileName, wsFiles)
    );
  }
  return [diagnoseError(error, mainSource, mainFileName, wsFiles)];
}

/** Extract structured diagnostic info from a single error thrown during compilation/execution. */
function diagnoseError(
  error: unknown,
  mainSource: string,
  mainFileName: string,
  wsFiles?: WorkspaceFile[]
): DiagnosticInfo {
  // ── RuntimeError ─────────────────────────────────────────────────────────
  if (error instanceof RuntimeError) {
    let snippet: string | null = null;
    if (error.line !== null) {
      const file = error.file ?? mainFileName;
      const src = getSourceForFile(file, mainFileName, mainSource, wsFiles);
      snippet = src
        ? extractSnippetByLine(src, error.line, 2, error.column ?? undefined)
        : null;
    }
    // Enrich call frames with source line text so downstream renderers
    // (e.g. the IDE worker) don't need access to file sources.
    // Each frame's callerSourceLine is the trimmed source at its own callerFile:callerLine.
    const callStack = error.callStack;
    if (callStack && callStack.length > 0) {
      for (const frame of callStack) {
        if (frame.callerFile && frame.callerLine > 0) {
          const src = getSourceForFile(
            frame.callerFile,
            mainFileName,
            mainSource,
            wsFiles
          );
          if (src) {
            const lines = src.split("\n");
            if (frame.callerLine >= 1 && frame.callerLine <= lines.length) {
              frame.callerSourceLine =
                lines[frame.callerLine - 1].trim() || undefined;
            }
          }
        }
      }
    }

    return {
      message: error.message,
      errorType: "runtime",
      file: error.file,
      line: error.line,
      snippet,
      callStack,
    };
  }

  // ── SyntaxError ──────────────────────────────────────────────────────────
  if (error instanceof SyntaxError) {
    const src = error.file
      ? getSourceForFile(error.file, mainFileName, mainSource, wsFiles)
      : null;
    const col =
      error.column ?? (src ? offsetToColumn(src, error.position) : undefined);
    const snippet =
      src && error.line !== null
        ? extractSnippetByLine(src, error.line, 2, col)
        : null;
    return {
      message: error.message,
      errorType: "syntax",
      file: error.file,
      line: error.line,
      snippet,
      callStack: null,
    };
  }

  // ── SemanticError (with span) ─────────────────────────────────────────────
  if (error instanceof SemanticError && error.span !== null) {
    // Resolve file/line/column from span if not already set
    const file = error.file ?? error.span.file;
    const src = file
      ? getSourceForFile(file, mainFileName, mainSource, wsFiles)
      : null;
    const line =
      error.line ?? (src ? offsetToLine(src, error.span.start) : null);
    const column =
      error.column ?? (src ? offsetToColumn(src, error.span.start) : undefined);
    const snippet =
      src && line !== null ? extractSnippetByLine(src, line, 2, column) : null;
    return {
      message: error.message,
      errorType: "semantic",
      file,
      line,
      snippet,
      callStack: null,
    };
  }

  // ── SemanticError (without span) ─────────────────────────────────────────
  if (error instanceof SemanticError) {
    return {
      message: error.message,
      errorType: "semantic",
      file: null,
      line: null,
      snippet: null,
      callStack: null,
    };
  }

  // ── Unexpected errors ─────────────────────────────────────────────────────
  const message = error instanceof Error ? error.message : String(error);
  return {
    message,
    errorType: "unknown",
    file: null,
    line: null,
    snippet: null,
    callStack: null,
  };
}

/** Format multiple DiagnosticInfo as a human-readable string for console output. */
export function formatDiagnostics(
  infos: DiagnosticInfo[],
  getSource?: (file: string) => string | null
): string {
  return infos.map(i => formatDiagnostic(i, getSource)).join("\n\n");
}

/** Format a DiagnosticInfo as a human-readable string for console output. */
export function formatDiagnostic(
  info: DiagnosticInfo,
  getSource?: (file: string) => string | null
): string {
  // When there's a call stack, use MATLAB-style format
  if (info.callStack != null && info.callStack.length > 0) {
    const stack = info.callStack;
    const N = stack.length;
    const parts: string[] = [];

    // Innermost frame: where the error occurred
    const innerName = stack[N - 1].name;
    const innerFile = info.file;
    const innerLine = info.line ?? 0;
    if (innerFile && innerLine > 0) {
      parts.push(`Error using ${innerName} (${innerFile}:${innerLine})`);
    } else if (innerLine > 0) {
      parts.push(`Error using ${innerName} (line ${innerLine})`);
    } else {
      parts.push(`Error using ${innerName}`);
    }
    parts.push(info.message);

    // Outer frames
    for (let i = N - 2; i >= 0; i--) {
      const name = stack[i].name;
      const callerFrame = stack[i + 1];
      const file = callerFrame.callerFile;
      const line = callerFrame.callerLine;
      parts.push("");
      if (file && line > 0) {
        parts.push(`Error in ${name} (${file}:${line})`);
        const srcLine = getSource
          ? getSourceLine(getSource, file, line)
          : (callerFrame.callerSourceLine ?? null);
        if (srcLine) parts.push(`    ${srcLine}`);
      } else if (line > 0) {
        parts.push(`Error in ${name} (line ${line})`);
      } else {
        parts.push(`Error in ${name}`);
      }
    }

    // Outermost caller: the script/entry point
    const outermost = stack[0];
    if (outermost.callerFile && outermost.callerLine > 0) {
      parts.push("");
      parts.push(`Error in ${outermost.callerFile}:${outermost.callerLine}`);
      const srcLine = getSource
        ? getSourceLine(getSource, outermost.callerFile, outermost.callerLine)
        : (outermost.callerSourceLine ?? null);
      if (srcLine) parts.push(`    ${srcLine}`);
    }

    return parts.join("\n");
  }

  // No call stack — simple format
  const labels: Record<DiagnosticInfo["errorType"], string> = {
    syntax: "SyntaxError",
    semantic: "SemanticError",
    runtime: "RuntimeError",
    unknown: "Error",
  };

  let result = labels[info.errorType];

  if (info.file && info.line !== null) {
    result += ` at ${info.file}:${info.line}`;
  } else if (info.line !== null) {
    result += ` at line ${info.line}`;
  }

  result += `: ${info.message}`;

  if (info.snippet) {
    result += `\n${info.snippet}`;
  }

  return result;
}

/** Get a single trimmed source line by file and 1-based line number. */
function getSourceLine(
  getSource: (file: string) => string | null,
  file: string,
  line: number
): string | null {
  const src = getSource(file);
  if (!src) return null;
  const lines = src.split("\n");
  if (line < 1 || line > lines.length) return null;
  return lines[line - 1].trim() || null;
}
