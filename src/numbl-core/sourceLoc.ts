/**
 * Source-location helpers: map a character offset in source text to a line
 * and column. A leaf module (no imports) so the parser, the runtime and the
 * JIT can share it without import cycles.
 */

/** Compute 1-based line number from a character offset in source text. */
export function offsetToLine(source: string, offset: number): number {
  let line = 1;
  for (let i = 0; i < offset && i < source.length; i++) {
    if (source[i] === "\n") line++;
  }
  return line;
}

/** Compute 1-based column number from a character offset in source text. */
export function offsetToColumn(source: string, offset: number): number {
  let column = 1;
  for (let i = offset - 1; i >= 0; i--) {
    if (source[i] === "\n") break;
    column++;
  }
  return column;
}

/** Compute 1-based {line, column} from a character offset in source text,
 *  clamping the offset to the source length. Column counts UTF-16 code
 *  units, which matches what Monaco expects. */
export function offsetToLineCol(
  source: string,
  offset: number
): { line: number; column: number } {
  const limit = Math.min(offset, source.length);
  let line = 1;
  let lineStart = 0;
  for (let i = 0; i < limit; i++) {
    if (source[i] === "\n") {
      line++;
      lineStart = i + 1;
    }
  }
  return { line, column: limit - lineStart + 1 };
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
 * Compute 1-based line number from a character offset using a pre-built line
 * table (binary search).
 */
export function offsetToLineFast(lineBreaks: number[], offset: number): number {
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
  return lo + 1;
}
