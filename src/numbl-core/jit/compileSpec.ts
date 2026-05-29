/**
 * JIT entry point for sibling consumers (e.g. numbl's executor
 * registry) that want to compile a single user-function
 * specialization on demand instead of running mtoc2 as a whole-program
 * AOT translator.
 *
 * Bypasses the script-level driver (`translate.ts` / `lowerProject`)
 * and produces an emitted JS module that returns a factory
 * `($h) => specFn`. The caller evaluates the source with
 * `new Function(source)()`, invokes the factory once per spec to bind
 * its I/O hooks, then calls `specFn(...args)` per dispatch. Per-call
 * dispatch has no I/O setup cost.
 *
 * The `lowerer` argument is the JIT spec cache: passing the same
 * `Lowerer` instance across calls means `Lowerer.specializations`
 * accumulates compiled IR across the session, so repeated calls to the
 * same `(funcDecl, argTypes, nargout)` triple skip the lowering cost.
 *
 * Caller responsibilities:
 *
 * - **Strip `exact` before calling.** Runtime-observed scalars should
 *   arrive as opaque-sign types (`scalarDouble("unknown")`,
 *   `tensorDouble(shape)` without exact data) so that `f(1); f(2);
 *   f(3)` doesn't shard the spec cache by value. `compileSpec` runs
 *   `withoutExact` as a defensive second pass.
 * - **Treat lowering errors as JIT bailouts.** `UnsupportedConstruct`
 *   and `TypeError` from the lowerer propagate to the caller as the
 *   "decline this JIT proposal" signal — the caller falls back to its
 *   interpreter for that dispatch.
 *
 * The emitted module includes every spec currently in
 * `lowerer.specializations` (not just the entry spec), because the
 * entry spec's body references its transitively-called specs by their
 * mangled cNames. Dead specs from prior compileSpec calls are bounded
 * by `MTOC2_MAX_SPECS_PER_FUNCTION` and don't affect correctness; if
 * profiling shows the re-emit cost dominates, a future change can do
 * a reachability walk over the entry's IR body.
 */

import type { Stmt } from "./parser/index.js";
import type { Lowerer } from "./lowering/lower.js";
import type { Workspace } from "./workspace/workspace.js";
import { specializeUserFunction } from "./lowering/specialize.js";
import { withoutExact, type Type } from "./lowering/types.js";
import { emitJsProgram } from "./codegen/emitJs.js";
import type { IRFunc, IRProgram, IRStmt } from "./lowering/ir.js";
import { UnsupportedConstruct } from "./lowering/errors.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

export interface CompileSpecArgs {
  /** Workspace whose files are pre-registered. Safe to share across
   *  many `compileSpec` calls — `Workspace.finalize()` is idempotent
   *  and nothing in the workspace mutates during lowering. */
  workspace: Workspace;
  /** Lowerer whose `.specializations` map persists across calls. Pass
   *  the same instance throughout the JIT session so cached lowerings
   *  are reused. */
  lowerer: Lowerer;
  /** The function AST to specialize. Caller fetches via
   *  `workspace.ctx.fileASTCache` or `workspace.resolve(...)`. */
  funcDecl: FuncStmt;
  /** Argument types for this specialization. Caller should already
   *  have stripped `exact`; `compileSpec` runs `withoutExact` again
   *  defensively. */
  argTypes: Type[];
  /** Number of outputs the call site requests. Salts the spec key. */
  nargout: number;
}

export interface CompileSpecResult {
  /** Mangled spec name (the IRFunc cName). Useful for caller-side
   *  cache keys and tracing. */
  cName: string;
  /** Full module source. Use:
   *
   *  ```ts
   *  const factory = new Function(source)();
   *  const specFn = factory({ write: (s) => ... });
   *  const out = specFn(...args);
   *  ```
   *
   *  `$h.write` must be a synchronous string sink that consumes
   *  `disp` / `fprintf` output. */
  source: string;
  /** Names of runtime snippets activated by emit. Diagnostic only. */
  activatedSnippets: ReadonlyArray<string>;
}

export function compileSpec(args: CompileSpecArgs): CompileSpecResult {
  const { workspace, lowerer, funcDecl, argTypes, nargout } = args;
  workspace.finalize();
  const widenedArgTypes = argTypes.map(withoutExact);
  const spec = specializeUserFunction.call(
    lowerer,
    funcDecl,
    widenedArgTypes,
    undefined,
    undefined,
    undefined,
    nargout,
    undefined
  );
  assertNoNonVoidBareExprStmts(spec);
  const prog: IRProgram = {
    topLevelStmts: [],
    functions: new Map(lowerer.specializations),
  };
  const { source, activatedSnippets } = emitJsProgram(prog, {
    workspace,
    exposeSpec: spec.cName,
  });
  return {
    cName: spec.cName,
    source,
    activatedSnippets,
  };
}

/** Bare expression statements with a non-Void value (e.g. `sin(i);`)
 *  are dialect-bound: numbl assigns each such value to the host-level
 *  `ans` variable. mtoc2 has no `ans` protocol with its hosts, so any
 *  emitted spec that contains such a statement would silently fail to
 *  update `ans` and produce wrong output for callers that read it
 *  later. Decline the spec instead — numbl's executor will catch the
 *  `UnsupportedConstruct` and route the call through its interpreter,
 *  which handles `ans` natively. */
function assertNoNonVoidBareExprStmts(spec: IRFunc): void {
  const walk = (stmts: IRStmt[]): void => {
    for (const s of stmts) {
      if (s.kind === "ExprStmt" && s.expr.ty.kind !== "Void") {
        throw new UnsupportedConstruct(
          `bare expression with a non-void value is not supported at the ` +
            `JIT boundary (numbl's 'ans' binding can't be modeled here)`,
          s.span
        );
      }
      if (s.kind === "If") {
        walk(s.thenBody);
        walk(s.elseBody);
      } else if (s.kind === "While" || s.kind === "For") {
        walk(s.body);
      }
    }
  };
  walk(spec.body);
}
