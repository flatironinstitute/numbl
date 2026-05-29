/**
 * Per-LoweringContext mtoc2 session state shared between
 * `jitCallExecutor` and `jitTopLevelExecutor`. One `Workspace`
 * (built from numbl's existing `LoweringContext`) and one `Lowerer`
 * per execution session, so the `Lowerer.specializations` cache
 * persists across whole-scope + per-call dispatches.
 *
 * The session is keyed on the LoweringContext object — WeakMap-cleared
 * automatically when the context goes out of scope.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import { Workspace, Lowerer, type WorkspaceFile } from "../../jit/index.js";

export interface SessionState {
  workspace: Workspace;
  lowerer: Lowerer;
}

const sessionStateByCtx = new WeakMap<object, SessionState>();

export function getOrCreateSession(interp: Interpreter): SessionState {
  const key = interp.ctx;
  let s = sessionStateByCtx.get(key);
  if (!s) {
    // Mtoc2's resolver delegates to numbl's `resolveFunction`, but
    // mtoc2 keeps its own `Workspace.files` map keyed by file name
    // and looks up the *main file's AST* through it (not through
    // `ctx.fileASTCache`) for main-file local function resolution.
    // Mirror every AST numbl has cached so mtoc2 can find them.
    const files: WorkspaceFile[] = [];
    for (const [name, ast] of interp.ctx.fileASTCache) {
      files.push({ name, source: "", ast });
    }
    const workspace = Workspace.fromExistingContext(
      interp.ctx,
      interp.ctx.mainFileName,
      files
    );
    s = { workspace, lowerer: new Lowerer(workspace) };
    sessionStateByCtx.set(key, s);
  }
  return s;
}
