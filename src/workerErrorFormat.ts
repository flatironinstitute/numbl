/**
 * Error formatting shared by the worker REPL/execute paths: classify an error
 * thrown by executeCode and render a message with a source snippet.
 */
import {
  offsetToColumn,
  RuntimeError,
  CancellationError,
} from "./numbl-core/runtime/index.js";
import { SyntaxError } from "./numbl-core/parser/index.js";
import { SemanticError } from "./numbl-core/lowering/errors.js";

export function extractSnippetByLine(
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

export function formatError(
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

/** Format any error thrown by executeCode against the code that raised it. */
export function formatExecuteError(
  error: unknown,
  code: string
): { message: string; cancelled: boolean } {
  if (error instanceof CancellationError) {
    return { message: "Execution cancelled", cancelled: true };
  }
  if (error instanceof RuntimeError) {
    const snippet =
      error.line !== null
        ? extractSnippetByLine(code, error.line, 2, error.column ?? undefined)
        : null;
    return {
      message: formatError(error.message, "runtime", error.line, snippet),
      cancelled: false,
    };
  }
  if (error instanceof SyntaxError) {
    const col = error.column ?? offsetToColumn(code, error.position);
    const snippet =
      error.line !== null
        ? extractSnippetByLine(code, error.line, 2, col)
        : null;
    return {
      message: formatError(error.message, "syntax", error.line, snippet),
      cancelled: false,
    };
  }
  if (error instanceof SemanticError && error.span !== null) {
    const snippet =
      error.line !== null
        ? extractSnippetByLine(code, error.line, 2, error.column)
        : null;
    return {
      message: formatError(error.message, "semantic", error.line, snippet),
      cancelled: false,
    };
  }
  if (error instanceof SemanticError) {
    return {
      message: formatError(error.message, "semantic", null, null),
      cancelled: false,
    };
  }
  const msg = error instanceof Error ? error.message : String(error);
  return { message: formatError(msg, "unknown", null, null), cancelled: false };
}
