/**
 * Type definitions for the executor module.
 */

import type { RuntimeValue } from "../runtime/index.js";
import type { FileIOAdapter } from "../fileIOAdapter.js";

// Re-export all shared graphics types from the single source of truth.
export type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
  ImagescTrace,
  ContourTrace,
  PlotInstruction,
} from "../../graphics/types.js";

import type { PlotInstruction } from "../../graphics/types.js";

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
  /** Initial hold state for plotting (persisted across REPL executions). */
  initialHoldState?: boolean;
  /** Override or add builtins for this execution only. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  customBuiltins?: Record<string, (nargout: number, args: any[]) => any>;
  /** Platform-specific file I/O adapter (e.g. Node.js fs). */
  fileIO?: FileIOAdapter;
  /** Optimization level for interpreter (0 = none, >=1 = JIT scalar functions). */
  optimization?: number;
}

export interface BuiltinProfileEntry {
  totalTimeMs: number;
  callCount: number;
}

export interface BuiltinProfileBreakdown {
  /** Calls from the registry fallback (rt.builtins). */
  fallback: BuiltinProfileEntry;
  /** Calls from the interpreter (IBuiltin.resolve → apply). */
  interp: BuiltinProfileEntry;
  /** Calls from JIT-compiled code (ib_* helpers). */
  jit: BuiltinProfileEntry;
}

export interface ProfileData {
  executionTimeMs: number;
  jitCompileTimeMs: number;
  builtins: Record<string, BuiltinProfileBreakdown>;
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
  dispatchUnknownCounts?: Record<string, number>;
}

// ── Sentinel values ─────────────────────────────────────────────────────

export const COLON_SENTINEL = Symbol("COLON");
export const END_SENTINEL = Symbol("END");
