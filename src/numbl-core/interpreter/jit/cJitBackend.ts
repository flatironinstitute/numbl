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

import type { Interpreter } from "../interpreter.js";
import type { FunctionDef } from "../types.js";
import type { JitStmt, JitType } from "./jitTypes.js";

export interface CJitBackend {
  /**
   * Attempt to emit + compile + load a C specialization for the given
   * lowered IR. Returns a callable wrapper on success, or null on any
   * bail (infeasible IR, compile failure, Node API headers missing,
   * etc.) — the caller falls back to JS-JIT in either case.
   */
  tryCompile(
    interp: Interpreter,
    fn: FunctionDef,
    body: JitStmt[],
    outputNames: string[],
    localVars: Set<string>,
    outputType: JitType | null,
    argTypes: JitType[],
    nargout: number
  ): ((...args: unknown[]) => unknown) | null;
}

let _backend: CJitBackend | null = null;

export function registerCJitBackend(b: CJitBackend): void {
  _backend = b;
}

export function getCJitBackend(): CJitBackend | null {
  return _backend;
}
