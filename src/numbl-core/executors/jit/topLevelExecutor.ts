/**
 * jit-top-level — compiles a script body via mtoc2's `compileSpec`
 * with the body wrapped as a synthetic user function whose params are
 * the variables the body reads (env inputs) and whose outputs are the
 * variables it assigns (env outputs).
 *
 * The interpreter calls `Registry.tryRunWholeScope` once at script
 * start; we register this via `registerWholeScope`. On compile
 * success the entire script runs in mtoc2-emitted JS; on
 * `UnsupportedConstruct` / `TypeError` we decline and the dispatcher
 * falls through to per-stmt dispatch (where jit-call still handles
 * individual user-function calls).
 *
 * Top-level loops *inside* the script execute as part of the
 * compiled artifact, so `loopDepth` only gates dispatch-level calls,
 * not this whole-scope attempt.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import { jitTypeKey, type JitType } from "../../jitTypes.js";
import type { Stmt, Span } from "../../parser/index.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { RuntimeValue } from "../../runtime/types.js";
import {
  compileSpec,
  UnsupportedConstruct,
  JitTypeError,
  type Type as CompilerType,
} from "../../jit/index.js";
import { jitTypeToCompilerType } from "./typeAdapter.js";
import { numblToJit, jitToNumbl, isGrowBail } from "./valueAdapter.js";
import { getOrCreateSession } from "./session.js";
import { buildHostHelpers, type JitHostHelpers } from "./hostHelpers.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

const TOP_LEVEL_COST = { compileMs: 100, perCallNs: 1000, runNs: 1000 };

interface JitTopLevelData {
  readonly stmts: readonly Stmt[];
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly compilerInputTypes: readonly CompilerType[];
  readonly currentFile: string;
  readonly cacheKey: string;
}

interface CompiledArtifact {
  readonly specFn: (...args: unknown[]) => unknown;
  readonly nargout: number;
}

/** True when every top-level stmt is suppressed (or is a kind that
 *  doesn't display regardless of suppression — function definitions,
 *  control flow, etc.). Top-level JIT is gated on this because
 *  numbl's `displayAssign` / `displayResult` hooks fire only at the
 *  interpreter and mtoc2's emit has no equivalent. */
function isAllSuppressed(stmts: readonly Stmt[]): boolean {
  for (const s of stmts) {
    // Unsuppressed bare expressions are NOT declined here. A void call
    // (`disp(...)`, `fprintf(...)`, `assert(...)`, `error/warning`)
    // performs its own output via the emit and has no value to echo, so
    // it JITs fine. A value-yielding bare expr (`1 + 2`) would trigger
    // numbl's `displayResult` ("ans = 3") which the emit doesn't
    // replicate — but `compileSpec`'s `assertNoNonVoidBareExprStmts`
    // rejects exactly those (non-Void bare exprs), declining the spec so
    // the interpreter handles the `ans` echo. So the compiler is the
    // backstop; we only gate on display-*assigns* here.
    if (s.type === "Assign" && !s.suppressed) return false;
    if (s.type === "MultiAssign" && !s.suppressed) return false;
  }
  return true;
}

/** Build a synthetic `FuncStmt` whose body is the script's
 *  top-level statements. Used as the input to `compileSpec`. */
function synthesizeTopLevelFuncStmt(
  stmts: readonly Stmt[],
  inputs: readonly string[],
  outputs: readonly string[],
  fileName: string
): FuncStmt {
  // Span.file disambiguates the spec cache key for scripts that
  // share the same compiled-once shape across re-runs (each
  // execution session has its own Lowerer, so this only matters for
  // diagnostics).
  const span: Span = { file: fileName, start: 0, end: 0 };
  return {
    type: "Function",
    name: "$top",
    functionId: "$top",
    params: [...inputs],
    outputs: [...outputs],
    body: [...stmts] as Stmt[],
    argumentsBlocks: [],
    span,
  };
}

export const jitTopLevelExecutor: Executor<
  JitTopLevelData,
  CompiledArtifact | null
> = {
  name: "jit-top-level",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<JitTopLevelData> | null {
    if (lowered.kind !== "top-level") return null;
    // Defensive: whole-scope only fires at script entry, before any
    // loop body has run.
    if (ctx.interp.loopDepth > 0) return null;
    const classification = lowered.classification;
    // `%!numbl:assert_jit c` requires C-JIT at --opt 2 — decline the JS
    // path so the script C-JITs or falls through to the interpreter
    // (which raises). Plain `assert_jit` only requires JS-JIT at --opt 1.
    if (ctx.interp.optimization === "2" && classification.assertsCJit) {
      return null;
    }
    // Top-level scripts with `return` are interpreter-only — mtoc2's
    // user-function body model doesn't accept a top-level return.
    if (classification.hasReturn) return null;
    // Unsuppressed top-level Assign / ExprStmt trigger numbl's
    // `displayAssign` / `displayResult` hooks (the "x = 10" /
    // "ans = ..." prints) which mtoc2's emit doesn't know about.
    // Decline so the interpreter handles those scripts; suppressed-
    // only scripts (everything ending in `;` or pure function calls
    // like `disp(...)`) can still JIT.
    if (!isAllSuppressed(classification.stmts)) return null;

    const compilerInputTypes: CompilerType[] = [];
    for (const jt of classification.inputTypes) {
      const mt = jitTypeToCompilerType(jt);
      if (mt === null) return null;
      compilerInputTypes.push(mt);
    }

    return {
      data: {
        stmts: classification.stmts,
        inputs: classification.inputs,
        outputs: classification.outputs,
        inputTypes: classification.inputTypes,
        compilerInputTypes,
        currentFile: classification.currentFile,
        cacheKey: classification.cacheKey,
      },
      cost: TOP_LEVEL_COST,
      bailRisk: false,
    };
  },

  cacheKey(d): string {
    return d.cacheKey;
  },

  compile(d, ctx: DispatchContext): CompiledArtifact | null {
    const interp = ctx.interp as Interpreter;
    const { workspace, lowerer } = getOrCreateSession(interp);
    const funcDecl = synthesizeTopLevelFuncStmt(
      d.stmts,
      d.inputs,
      d.outputs,
      d.currentFile
    );
    const nargout = d.outputs.length;
    try {
      const { source, cName } = compileSpec({
        workspace,
        lowerer,
        funcDecl,
        argTypes: d.compilerInputTypes as CompilerType[],
        nargout,
      });
      const typeDesc = d.inputTypes.map(jitTypeKey).join(", ");
      interp.onJitCompile?.(
        `jit-top-level:${cName}(${typeDesc}) -> outputs=${d.outputs.length}`,
        source,
        "js"
      );
      const factory = new Function(source)() as (
        $h: JitHostHelpers
      ) => (...args: unknown[]) => unknown;
      const specFn = factory(buildHostHelpers(interp.rt));
      return { specFn, nargout };
    } catch (e) {
      if (e instanceof UnsupportedConstruct || e instanceof JitTypeError) {
        return null;
      }
      throw e;
    }
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "jit-top-level: codegen declined" } };
    }
    const interp = ctx.interp as Interpreter;
    try {
      // Gather inputs from numbl's env. A read-before-write variable
      // may be undefined when the script starts (the analysis only
      // tracks textual references); pass undefined through and let
      // mtoc2's spec body fail at the first use if so.
      const inputValues: unknown[] = [];
      for (let i = 0; i < d.inputs.length; i++) {
        const rv = interp.env.get(d.inputs[i]);
        inputValues.push(
          rv === undefined ? undefined : numblToJit(rv, d.compilerInputTypes[i])
        );
      }
      const result = compiled.specFn(...inputValues);
      // Write outputs back to numbl's env. The shape mirrors
      // mtoc2's nargout convention: 0 outputs => undefined; 1 => bare
      // value; >= 2 => array of values.
      if (d.outputs.length === 0) {
        // No-op (rare — script with no top-level assignments).
      } else if (d.outputs.length === 1) {
        if (result !== undefined) {
          const rv = ensureRuntimeValue(jitToNumbl(result)) as RuntimeValue;
          interp.env.set(d.outputs[0], rv);
        }
      } else {
        if (!Array.isArray(result)) {
          throw new Error(
            `jit-top-level: expected array for outputs.length=${d.outputs.length}`
          );
        }
        for (let i = 0; i < d.outputs.length; i++) {
          const elt = result[i];
          if (elt !== undefined) {
            const rv = ensureRuntimeValue(jitToNumbl(elt)) as RuntimeValue;
            interp.env.set(d.outputs[i], rv);
          }
        }
      }
      return { ok: true };
    } catch (e) {
      if (isGrowBail(e)) {
        interp.onJitBail?.(
          "jit-top-level: indexed-store array growth; falling back to interpreter"
        );
      }
      return {
        bail: {
          message: `jit-top-level: runtime error: ${e instanceof Error ? e.message : String(e)}`,
        },
      };
    }
  },
};
