/**
 * mtoc2-loop — compiles a single For/While stmt via mtoc2's
 * `compileSpec` with the loop wrapped as a synthetic user function
 * whose params are the variables the loop reads (env inputs) and
 * whose outputs are the variables it assigns that are live after the
 * loop (filtered in classification).
 *
 * Mirrors `topLevelExecutor.ts` — the only structural difference is
 * that the synthetic body is a single-element `[loopStmt]` instead of
 * a whole script body.
 *
 * Loop-depth gate: only fires when `interp.loopDepth === 0`. A loop
 * dispatched while the interpreter is iterating an enclosing
 * loop-body would pay per-iter propose/cache overhead with no
 * speedup, because the *outer* loop is the natural JIT entry point
 * (it captures the inner loop's body and runs it natively in mtoc2-
 * emitted code). When the outer loop fires its own JIT, the inner
 * loop never reaches the interpreter dispatcher at all.
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
import { buildHostHelpers, type Mtoc2HostHelpers } from "./hostHelpers.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

const LOOP_COST = { compileMs: 50, perCallNs: 500, runNs: 500 };

interface Mtoc2LoopData {
  readonly loopStmt: Stmt & { type: "For" | "While" };
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly mtoc2InputTypes: readonly Mtoc2Type[];
  readonly currentFile: string;
  readonly cacheKey: string;
}

interface CompiledArtifact {
  readonly specFn: (...args: unknown[]) => unknown;
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
  fileName: string
): FuncStmt {
  const offset = loopStmt.span?.start ?? 0;
  const name = `$loop_${offset}`;
  const span: Span = { file: fileName, start: 0, end: 0 };
  return {
    type: "Function",
    name,
    functionId: name,
    params: [...inputs],
    outputs: [...outputs],
    body: [loopStmt],
    argumentsBlocks: [],
    span,
  };
}

export const mtoc2LoopExecutor: Executor<
  Mtoc2LoopData,
  CompiledArtifact | null
> = {
  name: "mtoc2-loop",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<Mtoc2LoopData> | null {
    if (lowered.kind !== "loop") return null;
    // Outer-loop only. Inner loops will execute inside the outer
    // loop's compiled artifact once the outer attempt succeeds.
    if (ctx.interp.loopDepth > 0) return null;
    const classification = lowered.classification;
    // mtoc2's user-function body model has no place to put a
    // top-of-function `return`. If the body contains one, decline so
    // the interpreter handles control flow correctly.
    if (classification.hasReturn) return null;

    const mtoc2InputTypes: Mtoc2Type[] = [];
    for (const jt of classification.inputTypes) {
      const mt = jitTypeToMtoc2Type(jt);
      if (mt === null) return null;
      mtoc2InputTypes.push(mt);
    }

    return {
      data: {
        loopStmt: classification.stmt,
        inputs: classification.inputs,
        outputs: classification.outputs,
        inputTypes: classification.inputTypes,
        mtoc2InputTypes,
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
        `mtoc2-loop:${cName}(${typeDesc}) -> outputs=${d.outputs.length}`,
        source
      );
      const factory = new Function(source)() as (
        $h: Mtoc2HostHelpers
      ) => (...args: unknown[]) => unknown;
      const specFn = factory(buildHostHelpers(interp.rt));
      return { specFn };
    } catch (e) {
      if (e instanceof UnsupportedConstruct || e instanceof Mtoc2TypeError) {
        return null;
      }
      throw e;
    }
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "mtoc2-loop: codegen declined" } };
    }
    const interp = ctx.interp as Interpreter;
    try {
      const inputValues: unknown[] = [];
      for (const name of d.inputs) {
        const rv = interp.env.get(name);
        inputValues.push(rv === undefined ? undefined : numblToMtoc2(rv));
      }
      const result = compiled.specFn(...inputValues);
      // Writeback shape mirrors mtoc2's nargout convention.
      if (d.outputs.length === 0) {
        // No live-after-loop assigns — nothing to write back.
      } else if (d.outputs.length === 1) {
        if (result !== undefined) {
          const rv = ensureRuntimeValue(mtoc2ToNumbl(result)) as RuntimeValue;
          interp.env.set(d.outputs[0], rv);
        }
      } else {
        if (!Array.isArray(result)) {
          throw new Error(
            `mtoc2-loop: expected array for outputs.length=${d.outputs.length}`
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
          message: `mtoc2-loop: runtime error: ${e instanceof Error ? e.message : String(e)}`,
        },
      };
    }
  },
};
