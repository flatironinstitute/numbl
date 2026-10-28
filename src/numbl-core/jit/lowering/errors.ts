/**
 * JIT lowering errors.
 *
 * Most errors carry a `Span` so the CLI can point at the offending
 * source location. Builtin authors (whose `transfer`/`emit` hooks
 * don't see a span) throw with `span` undefined; the framework
 * backfills via `withSpan` at the call-site boundary.
 */

import type { Span } from "../numbl/index.js";

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

/** Run `fn` and, if it throws an `UnsupportedConstruct` or `TypeError`
 *  with no span attached, backfill `span`. Used at
 *  framework→builtin call boundaries so builtin authors can throw
 *  without carrying a span around in every signature. Pre-existing
 *  spans (from nested framework calls) are preserved. */
export function withSpan<T>(span: Span, fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    if (e instanceof UnsupportedConstruct || e instanceof TypeError) {
      if (e.span === undefined) e.span = span;
    }
    throw e;
  }
}
