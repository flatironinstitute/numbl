/**
 * jit-call — executor that compiles user-function calls via mtoc2's
 * `compileSpec` JIT entry point and runs the emitted JS in-process.
 *
 * Replaces the JS-JIT call executor at `--opt 1`. The shape of the
 * fallback path is the same: on rejection (mtoc2's
 * `UnsupportedConstruct` / `TypeError` during lowering, or a JitType
 * the type adapter can't translate), `propose()` declines and the
 * dispatcher falls through to the next executor / interpreter.
 *
 * Key differences from JS-JIT:
 *
 * - **No mid-run bailouts.** mtoc2 either lowers cleanly or rejects
 *   statically. When `compile()` returns, the artifact will run to
 *   completion. `bailRisk` is `false`.
 * - **Shared `Lowerer.specializations` is the cache.** One Workspace
 *   + Lowerer pair per LoweringContext (i.e. per execution session);
 *   accumulating specs persist across calls so a function called
 *   repeatedly with the same arg signature reuses the prior compile.
 * - **Output-count support.** `nargout >= 1` (single-output and
 *   multi-output `[a, b, ...] = f(x)`). `nargout === 0` (bare-stmt
 *   `f();`) is declined because mtoc2's nargout=0 spec emits a no-
 *   return body, but numbl's interpreter still uses the first
 *   declared output for `ans` binding — let the interpreter handle.
 * - **Loop-depth gate.** Declines when `interp.loopDepth > 0` so a
 *   hot loop iterating function calls doesn't pay per-call JIT
 *   propose / spec-cache overhead. Once an outer call is JIT'd, its
 *   loops run inside mtoc2's compiled code anyway.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import { jitTypeKey, type JitType } from "../../jitTypes.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import type { Stmt, Span } from "../../parser/index.js";
import type { FunctionDef } from "../../interpreter/types.js";
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

const COST = { compileMs: 30, perCallNs: 100, runNs: 100 };

interface JitCallData {
  readonly fn: FunctionDef;
  readonly nargout: number;
  readonly argTypes: readonly JitType[];
  readonly compilerArgTypes: readonly CompilerType[];
  readonly args: readonly unknown[];
}

interface CompiledArtifact {
  readonly specFn: (...args: unknown[]) => unknown;
}

/** Build a parser-shaped `FuncStmt` from numbl's `FunctionDef`.
 *
 *  Numbl drops the span when projecting parsed Function stmts into
 *  `FunctionDef`. mtoc2 keys its spec cache on
 *  `(span.file, argTypes, nargout)`, so we must derive the *real*
 *  defining file — otherwise two functions with the same name from
 *  different files (e.g. when numbl's addpath/rmpath shadows a
 *  workspace function) would collide on a single spec entry and the
 *  old body would be reused after the path change.
 *
 *  The file comes from the first body statement's span. Empty
 *  bodies fall back to `<jit>` (a function with no body has no
 *  shadowing concern). */
function synthesizeFuncStmt(fd: FunctionDef): FuncStmt {
  const fromBody = fd.body[0]?.span?.file;
  const span: Span = {
    file: fromBody ?? "<jit>",
    start: 0,
    end: 0,
  };
  return {
    type: "Function",
    name: fd.name,
    functionId: fd.name,
    params: [...fd.params],
    outputs: [...fd.outputs],
    body: fd.body,
    argumentsBlocks: fd.argumentsBlocks ?? [],
    span,
  };
}

export const jitCallExecutor: Executor<JitCallData, CompiledArtifact | null> = {
  name: "jit-call",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<JitCallData> | null {
    if (lowered.kind !== "call") return null;
    // Disable JIT inside any enclosing for/while loop body. Once an
    // outer call has been JIT'd, its loops execute inside mtoc2's
    // compiled code and never reach the interpreter; so this only
    // gates calls the interpreter is dispatching directly while
    // iterating a hot loop, where per-call propose() / spec-cache
    // overhead is a net loss.
    if (ctx.interp.loopDepth > 0) return null;
    const classification = lowered.classification;
    // `%!numbl:assert_jit c` requires C-JIT at --opt 2. Decline the JS
    // path for such units so they either C-JIT or fall through to the
    // interpreter — which then raises on the directive (see
    // interpreterExec.ts). Plain `assert_jit` only requires JS-JIT at
    // --opt 1, so it does not gate here.
    if (ctx.interp.optimization === "2" && classification.assertsCJit) {
      return null;
    }
    // nargout=0 (bare-statement call like `f();`) is interpreter-only.
    // numbl's convention is that the function STILL returns its first
    // declared output for `ans` binding even with nargout=0, but
    // mtoc2's nargout=0 spec emits a no-return body, dropping the
    // value. Decline and let the interpreter handle the ans semantics.
    if (classification.nargout === 0) return null;

    // Map every JitType to mtoc2; any rejection aborts the proposal.
    const compilerArgTypes: CompilerType[] = [];
    for (const jt of classification.argTypes) {
      const mt = jitTypeToCompilerType(jt);
      if (mt === null) return null;
      compilerArgTypes.push(mt);
    }

    return {
      data: {
        fn: classification.fn,
        nargout: classification.nargout,
        argTypes: classification.argTypes,
        compilerArgTypes,
        args: lowered.args,
      },
      cost: COST,
      // mtoc2 rejects statically — once compile() returns, the
      // artifact runs to completion.
      bailRisk: false,
    };
  },

  cacheKey(d): string {
    // Include nargout so two calls to the same function with different
    // output counts (e.g. `[a,b] = f()` then `[a,b,c] = f()`) get
    // distinct compiled artifacts. mtoc2 specializes on nargout
    // internally; the executor's cache key has to match or the
    // artifact from the first call is wrongly reused.
    return (
      d.fn.name + "|" + d.argTypes.map(jitTypeKey).join(",") + "|n=" + d.nargout
    );
  },

  compile(d, ctx: DispatchContext): CompiledArtifact | null {
    const { workspace, lowerer } = getOrCreateSession(ctx.interp);
    try {
      const { source, cName } = compileSpec({
        workspace,
        lowerer,
        funcDecl: synthesizeFuncStmt(d.fn),
        argTypes: d.compilerArgTypes as CompilerType[],
        nargout: d.nargout,
      });
      // Surface the emitted JS through the same hook the legacy JS-JIT
      // used so it shows up in the IDE's "internals" view and the CLI
      // `--dump-js` flag.
      const interp = ctx.interp as Interpreter;
      const typeDesc = d.argTypes.map(jitTypeKey).join(", ");
      interp.onJitCompile?.(
        `jit-call:${cName}(${typeDesc}) -> nargout=${d.nargout}`,
        source,
        "js"
      );
      const factory = new Function(source)() as (
        $h: JitHostHelpers
      ) => (...args: unknown[]) => unknown;
      const specFn = factory(buildHostHelpers(interp.rt));
      return { specFn };
    } catch (e) {
      if (e instanceof UnsupportedConstruct || e instanceof JitTypeError) {
        return null;
      }
      throw e;
    }
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "jit-call: codegen declined" } };
    }
    try {
      const compilerArgs = d.args.map((v, i) =>
        numblToJit(v as never, d.argTypes[i])
      );
      const result = compiled.specFn(...compilerArgs);
      // mtoc2 return-shape convention:
      //   nargout = 0 → undefined (no value)
      //   nargout = 1 → bare value
      //   nargout >= 2 → array of values
      // numbl's interpreter convention matches at the boundary
      // (`callUserFunction` returns `outputs[0]` for nargout<=1 else
      // `outputs`), so we mirror the shape on the way out.
      if (d.nargout >= 2) {
        if (!Array.isArray(result)) {
          throw new Error(
            `jit-call: expected array for nargout=${d.nargout}, got ${typeof result}`
          );
        }
        return { result: result.map(v => jitToNumbl(v)) };
      }
      return { result: jitToNumbl(result) };
    } catch (e) {
      if (isGrowBail(e)) {
        ctx.interp.onJitBail?.(
          "jit-call: indexed-store array growth; falling back to interpreter"
        );
      }
      return {
        bail: {
          message: `jit-call: runtime error: ${e instanceof Error ? e.message : String(e)}`,
        },
      };
    }
  },
};
