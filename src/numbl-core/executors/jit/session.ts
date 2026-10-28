/**
 * Per-run JIT state shared between the JIT executors (both backends). One
 * JIT `Workspace` (wrapping the interpreter's `CodeIndex`) and one
 * `Lowerer` per run, so the `Lowerer.specializations` cache persists
 * across whole-scope + per-call dispatches.
 *
 * Keyed on the run's CodeIndex (WeakMap-cleared with it), not on the
 * `SessionState`: each run of a session parses its files afresh, and the
 * compiled artifacts bind that run's Runtime (host helpers), so neither
 * this nor the registry's compile cache can carry over to the next run.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import { Workspace, Lowerer, type WorkspaceFile } from "../../jit/index.js";

export interface SessionState {
  workspace: Workspace;
  lowerer: Lowerer;
}

const sessionStateByCode = new WeakMap<object, SessionState>();

export function getOrCreateSession(interp: Interpreter): SessionState {
  const key = interp.code;
  let s = sessionStateByCode.get(key);
  if (!s) {
    // The JIT's resolver delegates to numbl's `resolveFunction`, but
    // the JIT keeps its own `Workspace.files` map keyed by file name
    // and looks up the *main file's AST* through it (not through
    // `code.fileASTCache`) for main-file local function resolution.
    // Mirror every AST numbl has cached so the JIT can find them.
    const files: WorkspaceFile[] = [];
    for (const [name, ast] of interp.code.fileASTCache) {
      files.push({ name, source: "", ast });
    }
    const workspace = Workspace.fromCodeIndex(
      interp.code,
      interp.code.mainFileName,
      files
    );
    s = { workspace, lowerer: new Lowerer(workspace) };
    sessionStateByCode.set(key, s);
  }
  return s;
}
