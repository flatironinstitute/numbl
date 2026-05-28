/**
 * mtoc2-top-level — compiles a script body via mtoc2's `compileSpec`
 * with the body wrapped as a synthetic user function whose params are
 * the variables the body reads (env inputs) and whose outputs are the
 * variables it assigns (env outputs).
 *
 * The interpreter calls `Registry.tryRunWholeScope` once at script
 * start; we register this via `registerWholeScope`. On compile
 * success the entire script runs in mtoc2-emitted JS; on
 * `UnsupportedConstruct` / `TypeError` we decline and the dispatcher
 * falls through to per-stmt dispatch (where mtoc2-call still handles
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
  Mtoc2TypeError,
  type Type as Mtoc2Type,
} from "../../mtoc2/index.js";
import { jitTypeToMtoc2Type } from "./typeAdapter.js";
import { numblToMtoc2, mtoc2ToNumbl } from "./valueAdapter.js";
import { getOrCreateSession } from "./session.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

const TOP_LEVEL_COST = { compileMs: 100, perCallNs: 1000, runNs: 1000 };

interface Mtoc2TopLevelData {
  readonly stmts: readonly Stmt[];
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly mtoc2InputTypes: readonly Mtoc2Type[];
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
    if (s.type === "ExprStmt" && !s.suppressed) {
      // `disp(...)` / `fprintf(...)` work either way — the function
      // call itself performs the display via mtoc2's emit. But a
      // value-yielding bare expr like `1 + 2` would trigger numbl's
      // `displayResult` ("ans = 3") which mtoc2 doesn't replicate.
      // Distinguishing is non-trivial at the AST level; be
      // conservative and decline any unsuppressed ExprStmt.
      return false;
    }
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

export const mtoc2TopLevelExecutor: Executor<
  Mtoc2TopLevelData,
  CompiledArtifact | null
> = {
  name: "mtoc2-top-level",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<Mtoc2TopLevelData> | null {
    if (lowered.kind !== "top-level") return null;
    // Defensive: whole-scope only fires at script entry, before any
    // loop body has run.
    if (ctx.interp.loopDepth > 0) return null;
    const classification = lowered.classification;
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

    const mtoc2InputTypes: Mtoc2Type[] = [];
    for (const jt of classification.inputTypes) {
      const mt = jitTypeToMtoc2Type(jt);
      if (mt === null) return null;
      mtoc2InputTypes.push(mt);
    }

    return {
      data: {
        stmts: classification.stmts,
        inputs: classification.inputs,
        outputs: classification.outputs,
        inputTypes: classification.inputTypes,
        mtoc2InputTypes,
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
        argTypes: d.mtoc2InputTypes as Mtoc2Type[],
        nargout,
      });
      const typeDesc = d.inputTypes.map(jitTypeKey).join(", ");
      interp.onJitCompile?.(
        `mtoc2-top-level:${cName}(${typeDesc}) -> outputs=${d.outputs.length}`,
        source
      );
      const factory = new Function(source)() as ($h: {
        write: (s: string) => void;
      }) => (...args: unknown[]) => unknown;
      const rt = interp.rt;
      const specFn = factory({ write: (s: string) => rt.output(s) });
      return { specFn, nargout };
    } catch (e) {
      if (e instanceof UnsupportedConstruct || e instanceof Mtoc2TypeError) {
        return null;
      }
      throw e;
    }
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "mtoc2-top-level: codegen declined" } };
    }
    const interp = ctx.interp as Interpreter;
    try {
      // Gather inputs from numbl's env. A read-before-write variable
      // may be undefined when the script starts (the analysis only
      // tracks textual references); pass undefined through and let
      // mtoc2's spec body fail at the first use if so.
      const inputValues: unknown[] = [];
      for (const name of d.inputs) {
        const rv = interp.env.get(name);
        inputValues.push(rv === undefined ? undefined : numblToMtoc2(rv));
      }
      const result = compiled.specFn(...inputValues);
      // Write outputs back to numbl's env. The shape mirrors
      // mtoc2's nargout convention: 0 outputs => undefined; 1 => bare
      // value; >= 2 => array of values.
      if (d.outputs.length === 0) {
        // No-op (rare — script with no top-level assignments).
      } else if (d.outputs.length === 1) {
        if (result !== undefined) {
          const rv = ensureRuntimeValue(mtoc2ToNumbl(result)) as RuntimeValue;
          interp.env.set(d.outputs[0], rv);
        }
      } else {
        if (!Array.isArray(result)) {
          throw new Error(
            `mtoc2-top-level: expected array for outputs.length=${d.outputs.length}`
          );
        }
        for (let i = 0; i < d.outputs.length; i++) {
          const elt = result[i];
          if (elt !== undefined) {
            const rv = ensureRuntimeValue(mtoc2ToNumbl(elt)) as RuntimeValue;
            interp.env.set(d.outputs[i], rv);
          }
        }
      }
      return { ok: true };
    } catch (e) {
      return {
        bail: {
          message: `mtoc2-top-level: runtime error: ${e instanceof Error ? e.message : String(e)}`,
        },
      };
    }
  },
};
