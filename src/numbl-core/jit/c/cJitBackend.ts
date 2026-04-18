/**
 * Registrable hook for the C-JIT backend.
 *
 * The C-JIT's implementation pulls in Node-only modules (`child_process`,
 * `fs`, `path`, ...) that would pollute the browser bundle if imported
 * statically from [jit/index.ts](./index.ts). This module defines a
 * thin interface + a module-level slot; the real backend is installed
 * from the CLI entry point via [cJitInstall.ts](./cJitInstall.ts),
 * which is only pulled into the Node-targeted build.
 *
 * When no backend is registered (e.g. browser, or someone running the
 * library API directly) `getCJitBackend()` returns null and
 * `tryJitCall` silently falls through to JS-JIT.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { FunctionDef } from "../../interpreter/types.js";
import type { JitStmt, JitType } from "../jitTypes.js";
import type { GeneratedFn } from "../jitLower.js";

/**
 * Outcome of a C-JIT compile attempt.
 *
 * - `ok: true`: compilation succeeded; `fn` is callable.
 * - `ok: false, kind: "infeasible"`: the lowered IR contains a construct
 *   the C-JIT doesn't handle (JS-JIT probably would). Carries a `reason`
 *   and optional `line` from the feasibility checker; `--check-c-jit-parity`
 *   treats this as a hard error.
 * - `ok: false, kind: "env"`: the environment couldn't support C-JIT
 *   (no C compiler, compile/link failed, header missing, etc.). Also a
 *   hard error under `--check-c-jit-parity` because the user explicitly
 *   asked for C-JIT.
 */
export type CJitCompileResult =
  | { ok: true; fn: (...args: unknown[]) => unknown }
  | { ok: false; kind: "infeasible" | "env"; reason: string; line?: number };

export interface CJitBackend {
  /**
   * Attempt to emit + compile + load a C specialization for the given
   * lowered IR. Returns a structured result: callers distinguish
   * `infeasible` (parity gap with JS-JIT) from `env` (missing compiler)
   * when `--check-c-jit-parity` is on.
   */
  tryCompile(
    interp: Interpreter,
    fn: FunctionDef,
    body: JitStmt[],
    outputNames: string[],
    localVars: Set<string>,
    outputType: JitType | null,
    outputTypes: JitType[],
    argTypes: JitType[],
    nargout: number,
    generatedIRBodies: Map<string, GeneratedFn>
  ): CJitCompileResult;
}

let _backend: CJitBackend | null = null;

export function registerCJitBackend(b: CJitBackend): void {
  _backend = b;
}

export function getCJitBackend(): CJitBackend | null {
  return _backend;
}
