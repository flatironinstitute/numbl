/**
 * jit-loop — compiles a single For/While stmt via mtoc2's
 * `compileSpec` with the loop wrapped as a synthetic user function
 * whose params are the variables the loop reads (env inputs) and
 * whose outputs are the variables it assigns that are live after the
 * loop (filtered in classification).
 *
 * Mirrors `topLevelExecutor.ts` — the only structural difference is
 * that the synthetic body is a single-element `[loopStmt]` instead of
 * a whole script body.
 *
 * Loop-depth gate: fires when `interp.loopDepth === 0`, or when every
 * enclosing loop has been declined by the JIT. A loop dispatched while the
 * interpreter iterates an enclosing loop-body would pay per-iter
 * propose/cache overhead with no speedup, because the *outer* loop is the
 * natural JIT entry point (it captures the inner loop's body and runs it
 * natively in mtoc2-emitted code) — but that only holds when the outer loop
 * actually compiles. If it declined, the interpreter is walking its body and
 * this loop is the outermost thing still compilable, so the gate lifts
 * (`Interpreter.jitDeclinedLoopDepth`).
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
import { recordJitDecline } from "../../jitDeclineDiagnostics.js";
import { jitTypeToCompilerType } from "./typeAdapter.js";
import { numblToJit, jitToNumbl, isGrowBail } from "./valueAdapter.js";
import type { ConstHandle } from "../classification.js";
import { getOrCreateSession } from "./session.js";
import { buildHostHelpers, type JitHostHelpers } from "./hostHelpers.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

const LOOP_COST = { compileMs: 50, perCallNs: 500, runNs: 500 };

interface JitLoopData {
  readonly loopStmt: Stmt & { type: "For" | "While" };
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly compilerInputTypes: readonly CompilerType[];
  readonly constHandles: readonly ConstHandle[];
  readonly currentFile: string;
  readonly cacheKey: string;
}

interface CompiledArtifact {
  readonly specFn: (...args: unknown[]) => unknown;
  /** The synthetic spec's output types, so a struct written back into the
   *  env is rebuilt as a struct (the emitted value is an untagged object). */
  readonly outputTypes: ReadonlyArray<CompilerType>;
}

/** Build a synthetic `FuncStmt` whose body is a single loop stmt.
 *  Used as the input to `compileSpec`. The name embeds the loop's
 *  source offset so two same-shape loops in the same file don't
 *  collide in mtoc2's `(name, file, argTypes, nargout)`-keyed
 *  specialization cache. */
function synthesizeLoopFuncStmt(
  loopStmt: Stmt & { type: "For" | "While" },
  inputs: readonly string[],
  outputs: readonly string[],
  constHandles: readonly ConstHandle[],
  fileName: string
): FuncStmt {
  const offset = loopStmt.span?.start ?? 0;
  const name = `$loop_${offset}`;
  const span: Span = { file: fileName, start: 0, end: 0 };
  // Capture-free handle inputs are inlined as in-scope `<name> = @...`
  // assignments at the top of the body (not params), reducing the
  // boundary case to the supported in-scope handle case.
  const handleDefs: Stmt[] = constHandles.map(h => ({
    type: "Assign",
    name: h.name,
    expr: h.expr,
    suppressed: true,
    span,
  }));
  return {
    type: "Function",
    name,
    functionId: name,
    params: [...inputs],
    outputs: [...outputs],
    body: [...handleDefs, loopStmt],
    argumentsBlocks: [],
    span,
  };
}

export const jitLoopExecutor: Executor<JitLoopData, CompiledArtifact | null> = {
  name: "jit-loop",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<JitLoopData> | null {
    if (lowered.kind !== "loop") return null;
    // Outer-loop only — unless every enclosing loop was itself declined,
    // in which case nothing above will capture this one and it gets its own
    // chance (see Interpreter.jitDeclinedLoopDepth).
    if (ctx.interp.loopDepth > 0 && ctx.interp.jitDeclinedLoopDepth === 0)
      return null;
    const classification = lowered.classification;
    // `%!numbl:assert_jit c` requires C-JIT at --opt 2 — decline the JS
    // path so the loop C-JITs or falls through to the interpreter (which
    // raises). Plain `assert_jit` only requires JS-JIT at --opt 1.
    if (ctx.interp.optimization === "2" && classification.assertsCJit) {
      return null;
    }
    // mtoc2's user-function body model has no place to put a
    // top-of-function `return`. If the body contains one, decline so
    // the interpreter handles control flow correctly.
    if (classification.hasReturn) return null;

    const compilerInputTypes: CompilerType[] = [];
    for (const jt of classification.inputTypes) {
      const mt = jitTypeToCompilerType(jt);
      if (mt === null) return null;
      compilerInputTypes.push(mt);
    }

    return {
      data: {
        loopStmt: classification.stmt,
        inputs: classification.inputs,
        outputs: classification.outputs,
        inputTypes: classification.inputTypes,
        compilerInputTypes,
        constHandles: classification.constHandles,
        currentFile: classification.currentFile,
        cacheKey: classification.cacheKey,
      },
      cost: LOOP_COST,
      bailRisk: false,
    };
  },

  cacheKey(d): string {
    return d.cacheKey;
  },

  compile(d, ctx: DispatchContext): CompiledArtifact | null {
    const interp = ctx.interp as Interpreter;
    const { workspace, lowerer } = getOrCreateSession(interp);
    const funcDecl = synthesizeLoopFuncStmt(
      d.loopStmt,
      d.inputs,
      d.outputs,
      d.constHandles,
      d.currentFile
    );
    const nargout = d.outputs.length;
    try {
      const { source, cName, outputTypes } = compileSpec({
        workspace,
        lowerer,
        funcDecl,
        argTypes: d.compilerInputTypes as CompilerType[],
        nargout,
      });
      const typeDesc = d.inputTypes.map(jitTypeKey).join(", ");
      interp.onJitCompile?.(
        `jit-loop:${cName}(${typeDesc}) -> outputs=${d.outputs.length}`,
        source,
        "js"
      );
      const factory = new Function(source)() as (
        $h: JitHostHelpers
      ) => (...args: unknown[]) => unknown;
      const specFn = factory(buildHostHelpers(interp.rt));
      return { specFn, outputTypes };
    } catch (e) {
      if (e instanceof UnsupportedConstruct || e instanceof JitTypeError) {
        recordJitDecline({
          message: e.message,
          kind: e.constructor.name,
          at: e.span ? { file: e.span.file, start: e.span.start } : undefined,
          where: "jit-loop",
        });
        return null;
      }
      throw e;
    }
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "jit-loop: codegen declined" } };
    }
    const interp = ctx.interp as Interpreter;
    try {
      const inputValues: unknown[] = [];
      for (let i = 0; i < d.inputs.length; i++) {
        const rv = interp.env.get(d.inputs[i]);
        inputValues.push(
          rv === undefined ? undefined : numblToJit(rv, d.compilerInputTypes[i])
        );
      }
      const result = compiled.specFn(...inputValues);
      // Writeback shape mirrors mtoc2's nargout convention.
      if (d.outputs.length === 0) {
        // No live-after-loop assigns — nothing to write back.
      } else if (d.outputs.length === 1) {
        if (result !== undefined) {
          const rv = ensureRuntimeValue(
            jitToNumbl(result, compiled.outputTypes[0])
          ) as RuntimeValue;
          interp.env.set(d.outputs[0], rv);
        }
      } else {
        if (!Array.isArray(result)) {
          throw new Error(
            `jit-loop: expected array for outputs.length=${d.outputs.length}`
          );
        }
        for (let i = 0; i < d.outputs.length; i++) {
          const elt = result[i];
          if (elt !== undefined) {
            const rv = ensureRuntimeValue(
              jitToNumbl(elt, compiled.outputTypes[i])
            ) as RuntimeValue;
            interp.env.set(d.outputs[i], rv);
          }
        }
      }
      return { ok: true };
    } catch (e) {
      // A grow-store sentinel (`v(k) = x` past the runtime extent) means
      // the array would grow — unsupported in the JIT — so warn and bail
      // to the interpreter. Other runtime errors also bail (so the
      // interpreter reproduces numbl's canonical error/behavior), but
      // silently — they're not a JIT limitation.
      if (isGrowBail(e)) {
        interp.onJitBail?.(
          "jit-loop: indexed-store array growth; falling back to interpreter"
        );
      }
      return {
        bail: {
          message: `jit-loop: runtime error: ${e instanceof Error ? e.message : String(e)}`,
        },
      };
    }
  },
};
