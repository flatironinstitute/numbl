/**
 * The NDJSON protocol of `numbl run|eval --stream`, one JSON object per line
 * on stdout. The execution server (`numbl serve`) relays each line as a
 * server-sent event and appends a `close` message; the IDE's remote backend
 * decodes them. Environment-neutral: imported by the CLI, the server and the
 * web app.
 *
 * Every encoded message carries `v` (STREAM_PROTOCOL_VERSION). Version 1
 * extends the earlier unversioned format only by adding fields and message
 * types, so decoders accept messages without `v`.
 */

import type { PlotInstruction } from "./numbl-core/plotting/types.js";
import type { CallFrame } from "./numbl-core/runtime/error.js";

export const STREAM_PROTOCOL_VERSION = 1;

/** Structured error information, one per diagnostic of a failed run. */
export interface StreamDiagnostic {
  message: string;
  errorType: "syntax" | "runtime" | "unknown";
  /** MATLAB error identifier (e.g. "MATLAB:undefinedFunction"), if any. */
  identifier?: string;
  file: string | null;
  line: number | null;
  /** Source lines around the error, with a `>` marker on the error line. */
  snippet: string | null;
  /** Call stack of a runtime error, innermost frame last. */
  callStack?: CallFrame[] | null;
}

export type StreamMessage =
  /** Program output (disp, fprintf, auto-display). */
  | { type: "output"; text: string }
  /** Plot instructions flushed by drawnow or at the end of the run. */
  | { type: "drawnow"; plotInstructions: PlotInstruction[] }
  /** The program called input(); the reply is read as one line on stdin. */
  | { type: "input_request"; prompt: string }
  /** One per diagnostic; `done` with success false follows. */
  | ({ type: "error"; timedOut?: boolean } & StreamDiagnostic)
  /** Last message from the CLI. The process exits 0 on success, 1 on error. */
  | {
      type: "done";
      success: boolean;
      generatedJS?: string;
      generatedC?: string;
    }
  /** Appended by the execution server once the CLI process has exited. */
  | { type: "close"; success: boolean };

/** Serialize a message as one NDJSON line (without the trailing newline). */
export function encodeStreamMessage(msg: StreamMessage): string {
  return JSON.stringify({ v: STREAM_PROTOCOL_VERSION, ...msg });
}

const isStr = (x: unknown): x is string => typeof x === "string";
const isStrOrNull = (x: unknown) => x === null || typeof x === "string";
const isNumOrNull = (x: unknown) => x === null || typeof x === "number";

/**
 * Parse one line (or server-sent event payload). Returns null for anything
 * that is not a well-formed message of a known type, including messages
 * from a newer protocol version that this decoder does not understand.
 */
export function decodeStreamMessage(line: string): StreamMessage | null {
  let obj: unknown;
  try {
    obj = JSON.parse(line);
  } catch {
    return null;
  }
  if (typeof obj !== "object" || obj === null) return null;
  const m = obj as Record<string, unknown>;
  if (m.v !== undefined && m.v !== STREAM_PROTOCOL_VERSION) return null;
  switch (m.type) {
    case "output":
      return isStr(m.text) ? { type: "output", text: m.text } : null;
    case "drawnow":
      return Array.isArray(m.plotInstructions)
        ? {
            type: "drawnow",
            plotInstructions: m.plotInstructions as PlotInstruction[],
          }
        : null;
    case "input_request":
      return isStr(m.prompt)
        ? { type: "input_request", prompt: m.prompt }
        : null;
    case "error": {
      if (!isStr(m.message)) return null;
      const errorType =
        m.errorType === "syntax" || m.errorType === "runtime"
          ? m.errorType
          : "unknown";
      return {
        type: "error",
        message: m.message,
        errorType,
        ...(isStr(m.identifier) && m.identifier
          ? { identifier: m.identifier }
          : {}),
        file: isStrOrNull(m.file) ? (m.file as string | null) : null,
        line: isNumOrNull(m.line) ? (m.line as number | null) : null,
        snippet: isStrOrNull(m.snippet) ? (m.snippet as string | null) : null,
        ...(Array.isArray(m.callStack)
          ? { callStack: m.callStack as CallFrame[] }
          : {}),
        ...(m.timedOut === true ? { timedOut: true } : {}),
      };
    }
    case "done":
      return {
        type: "done",
        // Unversioned CLIs sent `done` only after a successful run.
        success: m.success !== false,
        ...(isStr(m.generatedJS) ? { generatedJS: m.generatedJS } : {}),
        ...(isStr(m.generatedC) ? { generatedC: m.generatedC } : {}),
      };
    case "close":
      return typeof m.success === "boolean"
        ? { type: "close", success: m.success }
        : null;
    default:
      return null;
  }
}
