/**
 * mtoc2 lowering errors.
 *
 * Most errors carry a `Span` so the CLI can point at the offending
 * source location. Builtin authors (whose `transfer`/`emit` hooks
 * don't see a span) throw with `span` undefined; the framework
 * backfills via `withSpan` at the call-site boundary.
 */

import type { Span } from "../parser/index.js";

export class UnsupportedConstruct extends Error {
  span?: Span;
  constructor(message: string, span?: Span) {
    super(message);
    this.name = "UnsupportedConstruct";
    this.span = span;
  }
}

export class TypeError extends Error {
  span?: Span;
  constructor(message: string, span?: Span) {
    super(message);
    this.name = "TypeError";
    this.span = span;
  }
}

/** Raised for ordinary user-facing runtime errors — things like
 *  "Undefined function or variable", "Too many output arguments", or
 *  "Output argument 'x' not assigned". Distinct from
 *  `UnsupportedConstruct` (which means "this language feature isn't
 *  implemented in mtoc2 yet"); using one for the other muddles the
 *  user-vs-tool signal. Mirrors numbl's `RuntimeError` so error class
 *  matches across the two runners. */
export class RuntimeError extends Error {
  span?: Span;
  constructor(message: string, span?: Span) {
    super(message);
    this.name = "RuntimeError";
    this.span = span;
  }
}

/** Run `fn` and, if it throws an `UnsupportedConstruct`, `TypeError`,
 *  or `RuntimeError` with no span attached, backfill `span`. Used at
 *  framework→builtin call boundaries so builtin authors can throw
 *  without carrying a span around in every signature. Pre-existing
 *  spans (from nested framework calls) are preserved. */
export function withSpan<T>(span: Span, fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    if (
      e instanceof UnsupportedConstruct ||
      e instanceof TypeError ||
      e instanceof RuntimeError
    ) {
      if (e.span === undefined) e.span = span;
    }
    throw e;
  }
}

export function formatError(
  e: UnsupportedConstruct | TypeError | RuntimeError,
  source: string
): string {
  if (e.span === undefined) {
    return `<unknown>: ${e.name}: ${e.message}`;
  }
  const { line, col } = offsetToLineCol(source, e.span.start);
  return `${e.span.file}:${line}:${col}: ${e.name}: ${e.message}`;
}

function offsetToLineCol(
  source: string,
  offset: number
): { line: number; col: number } {
  let line = 1;
  let col = 1;
  for (let i = 0; i < offset && i < source.length; i++) {
    if (source[i] === "\n") {
      line++;
      col = 1;
    } else {
      col++;
    }
  }
  return { line, col };
}
