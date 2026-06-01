/**
 * Source location helpers used by the parser. Inlined from numbl's
 * runtime/error.ts so the parser has no cross-module dependencies.
 */

/** Compute 1-based line number from a character offset in source text. */
export function offsetToLine(source: string, offset: number): number {
  return offsetToLineCol(source, offset).line;
}

/** Compute 1-based {line, column} from a character offset in source text.
 *  Column counts UTF-16 code units, which matches what Monaco expects. */
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
