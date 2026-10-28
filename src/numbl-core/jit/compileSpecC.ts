/**
 * JIT entry point that emits a single user-function specialization as
 * a C module ready for `cc -shared -fPIC` and `dlopen`.
 *
 * Sibling of `compileSpec.ts` (the JS-emit JIT entry); shares the same
 * `specializeUserFunction` flow + lowerer-cache contract. The output
 * differs only in the emitted language: C instead of JS, with the
 * entry spec marked `extern` so the host (numbl's koffi bridge) can
 * resolve it after `dlopen`. Transitively-called specs stay `static`.
 *
 * Caller responsibilities:
 *
 * - **Strip `exact` before calling.** Same as `compileSpec` — runtime-
 *   observed scalars should arrive as opaque-sign types so repeated
 *   calls converge on a single spec instead of sharding by value.
 *   `compileSpecC` runs `withoutExact` defensively.
 * - **Treat lowering errors as JIT bailouts.** `UnsupportedConstruct`
 *   and `TypeError` from the lowerer propagate to the caller as the
 *   "decline this JIT proposal" signal.
 *
 * Returns a `signature` describing the C function the host needs to
 * declare to koffi. The signature uses JIT `Type` values directly —
 * the host adapter translates each to a koffi type spec and assembles
 * the prototype string.
 */

import type { Stmt } from "./parser/index.js";
import type { Lowerer } from "./lowering/lower.js";
import type { Workspace } from "./workspace/workspace.js";
import { specializeUserFunction } from "./lowering/specialize.js";
import { withoutExact, type Type } from "./lowering/types.js";
import { emitProgram } from "./codegen/emit.js";
import { inlinePass } from "./codegen/inlinePass.js";
import type { IRFunc, IRProgram, IRStmt } from "./lowering/ir.js";
import { UnsupportedConstruct } from "./lowering/errors.js";
import { assertDefiniteAssignment } from "./lowering/definiteAssign.js";
import { isRecursive, reachableCallees } from "./lowering/callGraph.js";
import {
  addAssignedFlags,
  type AssignedFlag,
} from "./lowering/assignedFlags.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

/** Describes a single C parameter of the JIT-emitted entry spec. */
export interface SpecCParam {
  /** Identifier the C source uses for the param. */
  readonly cName: string;
  /** JIT type — host translates to koffi type + marshals values. */
  readonly ty: Type;
}

/** Describes a single C return slot of the JIT-emitted entry spec.
 *  For `nargout === 1`: a single by-value return slot whose `cName` is
 *  unused (the function uses C `return`).
 *  For `nargout >= 2`: each output is appended as an out-pointer param
 *  named `_jit_o<i>` AFTER the user params — see `emitStmt.ts`
 *  `fnParamList` for the convention.
 *  For `nargout === 0`: no return slot. */
export interface SpecCOutput {
  readonly cName: string;
  readonly ty: Type;
}

export interface SpecCSignature {
  /** The exported host-entry wrapper around the entry spec (the symbol
   *  koffi resolves after `dlopen`; see `emitHostEntry`). */
  readonly cName: string;
  /** Param shape, in the order the C function expects. */
  readonly params: ReadonlyArray<SpecCParam>;
  /** Output shape. Convention:
   *   - 0 outputs ⇒ empty array; C return type is `void`.
   *   - 1 output ⇒ single entry; C return type is `cTypeFor(out.ty)`.
   *   - ≥2 outputs ⇒ N entries; C return type is `void`, the entries
   *     describe the out-pointer params appended to the call. */
  readonly outputs: ReadonlyArray<SpecCOutput>;
}

export interface CompileSpecCArgs {
  /** Workspace whose files are pre-registered. Safe to share across
   *  many `compileSpecC` calls — `Workspace.finalize()` is idempotent. */
  workspace: Workspace;
  /** Lowerer whose `.specializations` map persists across calls. */
  lowerer: Lowerer;
  /** The function AST to specialize. */
  funcDecl: FuncStmt;
  /** Argument types for this specialization. Caller should have
   *  stripped `exact`; `compileSpecC` runs `withoutExact` again
   *  defensively. */
  argTypes: Type[];
  /** Number of outputs the call site requests. Salts the spec key. */
  nargout: number;
  /** True when the entry `funcDecl` is a synthetic wrapper (loop or
   *  top-level body), whose declared outputs may include the loop variable
   *  and other loop-body locals that are legitimately unassigned on a
   *  zero-iteration run. Suppresses the output-definite-assignment check on
   *  the entry spec (callees are still fully checked). Defaults to false
   *  (a real user-function entry, e.g. the call executor). */
  entrySynthetic?: boolean;
}

export interface CompileSpecCResult {
  /** Mangled spec name. Same as `signature.cName`. */
  cName: string;
  /** Full C source — ready for `cc -shared -fPIC`. Includes all
   *  activated runtime helpers + every spec in `lowerer.specializations`
   *  (the entry spec references its transitive callees by their
   *  mangled cNames). */
  source: string;
  /** Names of runtime snippets activated by emit. Diagnostic only. */
  activatedSnippets: ReadonlyArray<string>;
  /** Signature description so the host can build the koffi prototype. */
  signature: SpecCSignature;
  /** Synthetic entries only: outputs that may be left unassigned, each
   *  paired with a trailing flag output that is 0 when it was. The host
   *  skips writing such an output back. */
  assignedFlags: ReadonlyArray<AssignedFlag>;
  /** Files named by the statement locations a raise reports, by index
   *  (decode with `decodeLoc`). */
  locFiles: ReadonlyArray<string>;
  /** The entry spec can call itself (directly or through its callees). */
  recursive: boolean;
}

export function compileSpecC(args: CompileSpecCArgs): CompileSpecCResult {
  const { workspace, lowerer, funcDecl, argTypes, nargout } = args;
  workspace.finalize();
  const widenedArgTypes = argTypes.map(withoutExact);
  // Roll back on failure, for the reason `compileSpec` (JS) documents: the
  // emitted program contains every spec in `lowerer.specializations`, and the
  // checks below run over all of them, so a spec that lowers but does not
  // survive emit would otherwise make every later compile in the session fail
  // with its error.
  const before = new Set(lowerer.specializations.keys());
  try {
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
    // Same boundary check the JS path uses: bare non-void ExprStmts
    // would bind to numbl's `ans`, which the JIT has no protocol for.
    assertNoNonVoidBareExprStmts(spec);
    // Decline (fall back to JS-JIT / interpreter) when any function may read a
    // local before it is definitely assigned. C predeclares locals to 0, so it
    // would silently return that default instead of raising MATLAB's
    // undefined-variable error like --opt 0/1 do. See definiteAssign.ts.
    const prog: IRProgram = {
      functions: new Map(lowerer.specializations),
    };
    // The entry `spec` is a synthetic wrapper for the loop/top-level
    // executors (its "outputs" include the loop variable, legitimately
    // unassigned on a zero-iteration run), so only run the output-assignment
    // check on it for a real user-function entry (the call executor).
    const checkEntryOutputs = args.entrySynthetic !== true;
    const definite = assertDefiniteAssignment(spec, {
      checkOutputs: checkEntryOutputs,
    });
    // Only the entry's transitive callees: the shared spec map also holds
    // earlier synthetic loop/top-level wrappers (from either backend), whose
    // pseudo-outputs would fail the output check and decline every later
    // compile in the session.
    for (const fn of reachableCallees(prog, spec)) {
      assertDefiniteAssignment(fn, { checkOutputs: true });
    }
    // Flag synthetic outputs that may stay unassigned (C would hand back
    // its predeclared 0). A body with a `return` is flagged wholesale, as
    // the fall-through analysis doesn't see the early exit.
    let entry = spec;
    let assignedFlags: AssignedFlag[] = [];
    if (args.entrySynthetic === true) {
      const hasReturn = containsReturn(spec.body);
      const maybe = new Set(
        spec.outputs.filter(
          o => !spec.params.includes(o) && (hasReturn || !definite.has(o))
        )
      );
      ({ fn: entry, flags: assignedFlags } = addAssignedFlags(spec, maybe));
      prog.functions.set(entry.cName, entry);
    }
    // Fold single-use element-wise temps into their consumers so chains
    // fuse into one C loop. Returns a copy; the cached specs (shared
    // with the JS backend) are untouched.
    const { source, hostEntry, locFiles } = emitProgram(inlinePass(prog), {
      exposeSpec: spec.cName,
    });
    return {
      cName: spec.cName,
      source,
      // The C emit doesn't currently expose its activated-snippet set
      // through emitProgram's return; the .so build doesn't need it.
      // Diagnostic field kept for symmetry with `compileSpec` (JS).
      activatedSnippets: [],
      signature: { ...buildSignature(entry), cName: hostEntry },
      assignedFlags,
      locFiles,
      recursive: isRecursive(prog, spec),
    };
  } catch (err) {
    for (const key of lowerer.specializations.keys()) {
      if (!before.has(key)) lowerer.specializations.delete(key);
    }
    throw err;
  }
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

function buildSignature(spec: IRFunc): SpecCSignature {
  const params: SpecCParam[] = spec.cParams.map((cName, i) => ({
    cName,
    ty: spec.paramTypes[i],
  }));
  const outputs: SpecCOutput[] = spec.cOutputs.map((cName, i) => ({
    cName,
    ty: spec.outputTypes[i],
  }));
  return { cName: spec.cName, params, outputs };
}

/** Mirror of the same-named helper in `compileSpec.ts`. */
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
