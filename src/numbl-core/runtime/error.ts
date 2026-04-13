/**
 * Runtime error type and source location utilities.
 */

export interface Span {
  file: string;
  start: number;
  end: number;
}

export type CallFrame = {
  name: string;
  callerFile: string | null;
  callerLine: number;
  /** Trimmed source text at callerLine (populated during error enrichment) */
  callerSourceLine?: string;
};

export class CancellationError extends Error {
  constructor() {
    super("Execution cancelled");
    this.name = "CancellationError";
  }
}

export class RuntimeError extends Error {
  /** Source span where the error occurred (carries file name + per-file offsets) */
  span: Span | null;
  /** 1-based line number (set when source is available) */
  line: number | null;
  /** 1-based column number (set when source is available) */
  column: number | null;
  /** Source file name (set from span or by error handler) */
  file: string | null;
  /** Code snippet showing error context */
  snippet: string | null;
  /** Underlying cause (for chained errors) */
  cause?: Error;
  /** Call stack at the time of the error (null = not yet captured) */
  callStack: CallFrame[] | null;
  /** Error identifier (e.g. 'myid:myerr') */
  identifier: string;
  /** File sources for resolving source lines in call stack display */
  fileSources: Map<string, string> | null;

  constructor(message: string, span?: Span | null) {
    super(message);
    this.name = "RuntimeError";
    this.span = span ?? null;
    this.line = null;
    this.column = null;
    this.file = span?.file ?? null;
    this.snippet = null;
    this.callStack = null;
    this.identifier = "";
    this.fileSources = null;
  }

  /**
   * Attach source location span to this error.
   */
  withSpan(span: Span): RuntimeError {
    this.span = span;
    this.file = span.file;
    return this;
  }

  /**
   * Enrich error with line/column/snippet from the file's source text.
   * The fileSources map provides filename → source text lookups.
   */
  withContext(fileSources: Map<string, string>): RuntimeError {
    if (this.span) {
      const source = fileSources.get(this.span.file);
      if (source) {
        this.file = this.span.file;
        this.line = offsetToLine(source, this.span.start);
        this.column = offsetToColumn(source, this.span.start);
        this.snippet = extractSnippet(source, this.span.start);
      }
    }
    return this;
  }

  /**
   * Format error with location and snippet context.
   */
  toString(): string {
    if (this.callStack != null && this.callStack.length > 0) {
      return this._formatWithCallStack();
    }

    // No call stack — simple format
    let result = this.name;

    if (this.file && this.line !== null) {
      result += ` at ${this.file}:${this.line}`;
      if (this.column !== null) {
        result += `:${this.column}`;
      }
    } else if (this.line !== null) {
      result += ` at line ${this.line}`;
    }

    result += `: ${this.message}`;

    if (this.snippet) {
      result += `\n${this.snippet}`;
    }

    return result;
  }

  /** Format error with MATLAB-style call stack. */
  private _formatWithCallStack(): string {
    const stack = this.callStack!;
    const N = stack.length;
    const parts: string[] = [];

    // Innermost frame: where the error occurred
    const innerName = stack[N - 1].name;
    const innerFile = this.file;
    const innerLine = this.line ?? 0;
    if (innerFile && innerLine > 0) {
      parts.push(`Error using ${innerName} (${innerFile}:${innerLine})`);
    } else if (innerLine > 0) {
      parts.push(`Error using ${innerName} (line ${innerLine})`);
    } else {
      parts.push(`Error using ${innerName}`);
    }
    parts.push(this.message);

    // Outer frames: each function in the call chain
    for (let i = N - 2; i >= 0; i--) {
      const name = stack[i].name;
      const callerFrame = stack[i + 1];
      const file = callerFrame.callerFile;
      const line = callerFrame.callerLine;
      parts.push("");
      if (file && line > 0) {
        parts.push(`Error in ${name} (${file}:${line})`);
        const srcLine = this._getSourceLine(file, line);
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
      const srcLine = this._getSourceLine(
        outermost.callerFile,
        outermost.callerLine
      );
      if (srcLine) parts.push(`    ${srcLine}`);
    }

    return parts.join("\n");
  }

  /** Look up a single trimmed source line from fileSources. */
  private _getSourceLine(file: string, line: number): string | null {
    if (!this.fileSources) return null;
    const src = this.fileSources.get(file);
    if (!src) return null;
    const lines = src.split("\n");
    if (line < 1 || line > lines.length) return null;
    return lines[line - 1].trim() || null;
  }
}

/**
 * Compute 1-based line number from a character offset in source text.
 */
export function offsetToLine(source: string, offset: number): number {
  let line = 1;
  for (let i = 0; i < offset && i < source.length; i++) {
    if (source[i] === "\n") line++;
  }
  return line;
}

/**
 * Build an array of newline character positions for fast offset→line lookup.
 * Returns a sorted array where lineBreaks[i] is the offset of the (i+1)th '\n'.
 */
export function buildLineTable(source: string): number[] {
  const breaks: number[] = [];
  for (let i = 0; i < source.length; i++) {
    if (source[i] === "\n") breaks.push(i);
  }
  return breaks;
}

/**
 * Compute 1-based line number from a character offset using a pre-built line table.
 * Uses binary search — O(log n) instead of O(n).
 */
export function offsetToLineFast(lineBreaks: number[], offset: number): number {
  // Binary search for the number of newlines before `offset`
  let lo = 0;
  let hi = lineBreaks.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (lineBreaks[mid] < offset) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo + 1; // 1-based
}

/**
 * Compute 1-based column number from a character offset in source text.
 */
export function offsetToColumn(source: string, offset: number): number {
  let column = 1;
  for (let i = offset - 1; i >= 0; i--) {
    if (source[i] === "\n") break;
    column++;
  }
  return column;
}

/**
 * Extract code snippet with context lines around the error location.
 */
export function extractSnippet(
  source: string,
  offset: number,
  contextLines: number = 2
): string {
  const lines = source.split("\n");
  const errorLine = offsetToLine(source, offset);
  const errorCol = offsetToColumn(source, offset);

  const startLine = Math.max(1, errorLine - contextLines);
  const endLine = Math.min(lines.length, errorLine + contextLines);

  const snippetLines: string[] = [];
  for (let i = startLine; i <= endLine; i++) {
    const lineNum = i.toString().padStart(4, " ");
    const prefix = i === errorLine ? "> " : "  ";
    snippetLines.push(`${prefix}${lineNum} | ${lines[i - 1]}`);

    // Add error indicator on the error line
    if (i === errorLine) {
      const indent = "  " + lineNum + " | ";
      const pointer = " ".repeat(errorCol - 1) + "^";
      snippetLines.push(indent.replace(/./g, " ") + pointer);
    }
  }

  return snippetLines.join("\n");
}
