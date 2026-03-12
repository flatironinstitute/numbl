/**
 * Type definitions for the executor module.
 */

import type { RuntimeValue } from "../runtime/index.js";
import type { PlotTrace, Plot3Trace, SurfTrace } from "../runtime/plotUtils.js";
import type { FileIOAdapter } from "../fileIOAdapter.js";

// ── Plot Instructions ───────────────────────────────────────────────────

export type { PlotTrace, Plot3Trace, SurfTrace } from "../runtime/plotUtils.js";

export type PlotInstruction =
  | { type: "set_figure_handle"; handle: number }
  | { type: "plot"; traces: PlotTrace[] }
  | { type: "plot3"; traces: Plot3Trace[] }
  | { type: "surf"; trace: SurfTrace }
  | { type: "set_hold"; value: boolean }
  | { type: "set_title"; text: string }
  | { type: "set_xlabel"; text: string }
  | { type: "set_ylabel"; text: string }
  | { type: "close" }
  | { type: "close_all" }
  | { type: "clf" };

// ── Exec Options ────────────────────────────────────────────────────────

export interface ExecOptions {
  onOutput?: (text: string) => void;
  onDrawnow?: (plotInstructions: PlotInstruction[]) => void;
  displayResults?: boolean;
  maxIterations?: number;
  initialVariableValues?: Record<string, RuntimeValue>;
  /** Optional callback for verbose compilation logging. */
  log?: (message: string) => void;
  /** Enable profiling of builtin function calls. */
  profile?: boolean;
  /** Called each time a JIT function is compiled, with a description and the generated JS. */
  onJitCompile?: (description: string, jsCode: string) => void;
  /** Skip $rt.$file / $rt.$line tracking in generated code. */
  noLineTracking?: boolean;
  /** Initial hold state for plotting (persisted across REPL executions). */
  initialHoldState?: boolean;
  /** Override or add builtins for this execution only. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  customBuiltins?: Record<string, (nargout: number, args: any[]) => any>;
  /** Platform-specific file I/O adapter (e.g. Node.js fs). */
  fileIO?: FileIOAdapter;
}

export interface BuiltinProfileEntry {
  totalTimeMs: number;
  callCount: number;
}

export interface ProfileData {
  codegenTimeMs: number;
  codegenBreakdown: {
    parseMainMs: number;
    parseWorkspaceMs: number;
    loadJsUserFunctionsMs: number;
    registrationMs: number;
    buildFunctionIndexMs: number;
    lowerMainMs: number;
    codegenMs: number;
  };
  executionTimeMs: number;
  jitCompileTimeMs: number;
  builtins: Record<string, BuiltinProfileEntry>;
  dispatches: Record<string, BuiltinProfileEntry>;
}

export interface ExecResult {
  output: string[];
  generatedJS: string;
  plotInstructions: PlotInstruction[];
  returnValue: RuntimeValue;
  variableValues: Record<string, RuntimeValue>;
  holdState: boolean;
  profileData?: ProfileData;
}

// ── Sentinel values ─────────────────────────────────────────────────────

export const COLON_SENTINEL = Symbol("COLON");
export const END_SENTINEL = Symbol("END");
