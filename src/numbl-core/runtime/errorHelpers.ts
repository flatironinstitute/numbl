/**
 * Error helper utilities for creating context-aware RuntimeErrors.
 */

import { RuntimeError } from "./error.js";
import type { Span } from "../parser/types.js";

/**
 * Create a RuntimeError with optional span context.
 */
export function runtimeError(
  message: string,
  span?: Span | null
): RuntimeError {
  return new RuntimeError(message, span ?? undefined);
}

/**
 * Format a RuntimeError for display with file:line and snippet.
 * This is a convenience wrapper around RuntimeError.toString().
 */
export function formatError(
  err: RuntimeError,
  fileSources?: Map<string, string>
): string {
  // If fileSources is provided and error has no context yet, enrich it
  if (fileSources && err.span && !err.snippet) {
    err.withContext(fileSources);
  }
  return err.toString();
}
