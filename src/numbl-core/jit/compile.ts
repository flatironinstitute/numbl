/**
 * The JIT compile pipeline, shared by both backends: specialize one user
 * function (or synthetic loop / script wrapper) on its argument types,
 * check it, collect the specs it reaches, run the target's IR passes, and
 * emit. `compileSpec` (JS) and `compileSpecC` (C) are this pipeline with
 * their `CompileTarget`.
 *
 * The `lowerer` is the session's spec cache: `Lowerer.specializations`
 * accumulates lowered specs across calls (from both backends), so a repeat
 * of the same `(funcDecl, argTypes, nargout)` skips lowering. The emitted
 * module holds only the entry and the specs it transitively calls.
 *
 * Caller responsibilities:
 *
 * - **Strip `exact` before calling.** Runtime-observed scalars should
 *   arrive as opaque-sign types so that `f(1); f(2); f(3)` does not shard
 *   the spec cache by value. `withoutExact` runs again here defensively.
 * - **Treat lowering errors as declines.** `UnsupportedConstruct` and
 *   `TypeError` propagate to the caller, which falls back to the
 *   interpreter.
 */

import type { Stmt } from "./numbl/index.js";
import type { Lowerer } from "./lowering/lower.js";
import type { Workspace } from "./workspace/workspace.js";
import { specializeUserFunction } from "./lowering/specialize.js";
import { withoutExact, type Type } from "./lowering/types.js";
import type { IRFunc, IRProgram, IRStmt } from "./lowering/ir.js";
import { UnsupportedConstruct } from "./lowering/errors.js";
import { assertDefiniteAssignment } from "./lowering/definiteAssign.js";
import { isRecursive, reachableCallees } from "./lowering/callGraph.js";
import {
  addAssignedFlags,
  type AssignedFlag,
} from "./lowering/assignedFlags.js";
import { inlinePass } from "./codegen/inlinePass.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

export interface CompileSpecArgs {
  /** Workspace whose files are pre-registered. Safe to share across
   *  compiles: `Workspace.finalize()` is idempotent and nothing in the
   *  workspace mutates during lowering. */
  workspace: Workspace;
  /** Lowerer whose `.specializations` map persists across calls. Pass the
   *  same instance throughout the session so lowerings are reused. */
  lowerer: Lowerer;
  /** The function AST to specialize. */
  funcDecl: FuncStmt;
  /** Argument types for this specialization (`exact` already stripped). */
  argTypes: Type[];
  /** Number of outputs the call site requests. Salts the spec key. */
  nargout: number;
  /** True when `funcDecl` is a synthetic wrapper (loop or script body),
   *  whose declared outputs include the loop variable and other locals
   *  that are legitimately unassigned on a zero-iteration run. Relaxes the
   *  entry's output-assignment check (`definiteAssignment` targets). */
  entrySynthetic?: boolean;
}

/** What a target emits: the module source plus emit-time facts. */
export interface EmitOutput {
  source: string;
  /** Runtime snippets the module uses. */
  activatedSnippets: ReadonlyArray<string>;
  /** Files named by the statement locations a raise reports, by index
   *  (decode with `decodeLoc`). */
  locFiles: ReadonlyArray<string>;
}

/** A backend's half of the pipeline: what it needs from the shared IR
 *  passes, and its emitter. */
export interface CompileTarget<E extends EmitOutput> {
  /** The target cannot represent an unassigned local (C predeclares every
   *  local to 0): decline a spec that may read a local or return an output
   *  before assigning it, and flag a synthetic entry's outputs that may
   *  stay unassigned (`assignedFlags.ts`). The JS backend models
   *  unassigned as `undefined` and checks reads at run time. */
  readonly definiteAssignment: boolean;
  /** Fold single-use element-wise temps into their consumers before emit,
   *  so chains fuse into one loop (`inlinePass.ts`). */
  readonly fusesElementwise: boolean;
  /** Emit `prog`, exposing the spec named `exposeSpec`. */
  emit(prog: IRProgram, exposeSpec: string): E;
}

export interface CompiledSpec<E extends EmitOutput> {
  /** Mangled name of the entry spec. */
  cName: string;
  /** The entry spec as emitted (with any assigned-flag outputs). */
  entry: IRFunc;
  /** Synthetic entries of `definiteAssignment` targets: outputs that may
   *  be left unassigned, each paired with a trailing flag output that is 0
   *  when it was. The host skips writing such an output back. */
  assignedFlags: ReadonlyArray<AssignedFlag>;
  /** The entry spec can call itself (directly or through its callees). */
  recursive: boolean;
  emitted: E;
}

export function compileWith<E extends EmitOutput>(
  target: CompileTarget<E>,
  args: CompileSpecArgs
): CompiledSpec<E> {
  const { workspace, lowerer, funcDecl, argTypes, nargout } = args;
  workspace.finalize();
  // Everything this call adds to the shared spec cache, so a failure can be
  // rolled back: a spec that lowers cleanly but fails a check or emit is
  // not kept for later compiles to trip over.
  const before = new Set(lowerer.specializations.keys());
  try {
    const spec = specializeUserFunction.call(
      lowerer,
      funcDecl,
      argTypes.map(withoutExact),
      undefined,
      undefined,
      undefined,
      nargout,
      undefined
    );
    assertNoNonVoidBareExprStmts(spec);
    const prog = reachableProgram(lowerer, spec);
    const recursive = isRecursive(prog, spec);
    let entry = spec;
    let assignedFlags: AssignedFlag[] = [];
    if (target.definiteAssignment) {
      ({ entry, assignedFlags } = checkDefiniteAssignment(
        prog,
        spec,
        args.entrySynthetic === true
      ));
      prog.functions.set(entry.cName, entry);
    }
    // `inlinePass` returns a copy; the cached specs are untouched.
    const emitted = target.emit(
      target.fusesElementwise ? inlinePass(prog) : prog,
      spec.cName
    );
    return { cName: spec.cName, entry, assignedFlags, recursive, emitted };
  } catch (err) {
    for (const key of lowerer.specializations.keys()) {
      if (!before.has(key)) lowerer.specializations.delete(key);
    }
    throw err;
  }
}

/** `entry` and the specs it transitively calls, in the cache's order. The
 *  cache also holds every earlier entry of the session (other functions,
 *  and synthetic loop / script wrappers), which this module never calls. */
function reachableProgram(lowerer: Lowerer, entry: IRFunc): IRProgram {
  const all: IRProgram = { functions: lowerer.specializations };
  const reach = new Set<string>([entry.cName]);
  for (const fn of reachableCallees(all, entry)) reach.add(fn.cName);
  const functions = new Map<string, IRFunc>();
  for (const [key, fn] of lowerer.specializations) {
    if (reach.has(key)) functions.set(key, fn);
  }
  return { functions };
}

/** Decline when any spec may read a local before it is definitely
 *  assigned: C predeclares locals to 0, so it would silently use that
 *  instead of raising MATLAB's undefined-variable error as --opt 0/1 do
 *  (see `definiteAssign.ts`). A synthetic entry's outputs are not checked
 *  but flagged instead: those not definitely assigned get an assigned
 *  flag. A body with a `return` is flagged wholesale, as the fall-through
 *  analysis does not see the early exit. */
function checkDefiniteAssignment(
  prog: IRProgram,
  spec: IRFunc,
  synthetic: boolean
): { entry: IRFunc; assignedFlags: AssignedFlag[] } {
  const definite = assertDefiniteAssignment(spec, {
    checkOutputs: !synthetic,
  });
  for (const fn of prog.functions.values()) {
    if (fn !== spec) assertDefiniteAssignment(fn, { checkOutputs: true });
  }
  if (!synthetic) return { entry: spec, assignedFlags: [] };
  const hasReturn = containsReturn(spec.body);
  const maybe = new Set(
    spec.outputs.filter(
      o => !spec.params.includes(o) && (hasReturn || !definite.has(o))
    )
  );
  const { fn, flags } = addAssignedFlags(spec, maybe);
  return { entry: fn, assignedFlags: flags };
}

function containsReturn(stmts: IRStmt[]): boolean {
  return stmts.some(
    s =>
      s.kind === "ReturnFromFunction" ||
      (s.kind === "If" &&
        (containsReturn(s.thenBody) || containsReturn(s.elseBody))) ||
      ((s.kind === "While" || s.kind === "For") && containsReturn(s.body))
  );
}

/** Bare expression statements with a non-Void value (e.g. `sin(i);`) bind
 *  `ans` in the interpreter. The JIT has no `ans` protocol with its hosts,
 *  so a spec containing one would silently leave `ans` stale: decline it
 *  and let the interpreter run the unit. */
function assertNoNonVoidBareExprStmts(spec: IRFunc): void {
  const walk = (stmts: IRStmt[]): void => {
    for (const s of stmts) {
      if (
        s.kind === "ExprStmt" &&
        s.expr.ty.kind !== "Void" &&
        s.discard !== true
      ) {
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
