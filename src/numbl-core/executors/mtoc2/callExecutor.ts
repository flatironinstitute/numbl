/**
 * mtoc2-call — executor that compiles user-function calls via mtoc2's
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
 * - **First-cut scope.** `nargout === 1` only; other counts decline.
 *   Coverage grows as the type/value adapters fill in.
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
  Workspace,
  Lowerer,
  UnsupportedConstruct,
  Mtoc2TypeError,
  type Type as Mtoc2Type,
} from "../../mtoc2/index.js";
import { jitTypeToMtoc2Type } from "./typeAdapter.js";
import { numblToMtoc2, mtoc2ToNumbl } from "./valueAdapter.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

const COST = { compileMs: 30, perCallNs: 100, runNs: 100 };

interface Mtoc2CallData {
  readonly fn: FunctionDef;
  readonly nargout: number;
  readonly argTypes: readonly JitType[];
  readonly mtoc2ArgTypes: readonly Mtoc2Type[];
  readonly args: readonly unknown[];
}

interface CompiledArtifact {
  readonly specFn: (...args: unknown[]) => unknown;
}

interface SessionState {
  workspace: Workspace;
  lowerer: Lowerer;
}

/** Per-LoweringContext mtoc2 session state. Survives across all
 *  mtoc2-call dispatches in a single execution session so spec
 *  compilations are reused. WeakMap auto-cleans when the context is
 *  GC'd. */
const sessionStateByCtx = new WeakMap<object, SessionState>();

function getOrCreateSession(ctx: DispatchContext): SessionState {
  const key = ctx.interp.ctx;
  let s = sessionStateByCtx.get(key);
  if (!s) {
    const workspace = Workspace.fromExistingContext(
      ctx.interp.ctx,
      ctx.interp.ctx.mainFileName,
      []
    );
    s = { workspace, lowerer: new Lowerer(workspace) };
    sessionStateByCtx.set(key, s);
  }
  return s;
}

/** Build a parser-shaped `FuncStmt` from numbl's `FunctionDef`.
 *  Numbl drops the span when projecting parsed Function stmts into
 *  `FunctionDef`; we synthesize a placeholder span here. mtoc2 keys
 *  its spec cache on (span.file, argTypes, nargout) — using a
 *  constant "<jit>" file is fine because the source function name
 *  prefix already distinguishes different functions. */
function synthesizeFuncStmt(fd: FunctionDef): FuncStmt {
  const span: Span = { file: "<jit>", start: 0, end: 0 };
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

export const mtoc2CallExecutor: Executor<
  Mtoc2CallData,
  CompiledArtifact | null
> = {
  name: "mtoc2-call",

  propose(lowered: LoweredStmt): Proposal<Mtoc2CallData> | null {
    if (lowered.kind !== "call") return null;
    const classification = lowered.classification;
    // First-cut: single-output only. Multi-output requires a return-
    // array unwrap in run() that's straightforward but not yet wired.
    if (classification.nargout !== 1) return null;

    // Map every JitType to mtoc2; any rejection aborts the proposal.
    const mtoc2ArgTypes: Mtoc2Type[] = [];
    for (const jt of classification.argTypes) {
      const mt = jitTypeToMtoc2Type(jt);
      if (mt === null) return null;
      mtoc2ArgTypes.push(mt);
    }

    return {
      data: {
        fn: classification.fn,
        nargout: classification.nargout,
        argTypes: classification.argTypes,
        mtoc2ArgTypes,
        args: lowered.args,
      },
      cost: COST,
      // mtoc2 rejects statically — once compile() returns, the
      // artifact runs to completion.
      bailRisk: false,
    };
  },

  cacheKey(d): string {
    return d.fn.name + "|" + d.argTypes.map(jitTypeKey).join(",");
  },

  compile(d, ctx: DispatchContext): CompiledArtifact | null {
    const { workspace, lowerer } = getOrCreateSession(ctx);
    try {
      const { source, cName } = compileSpec({
        workspace,
        lowerer,
        funcDecl: synthesizeFuncStmt(d.fn),
        argTypes: d.mtoc2ArgTypes as Mtoc2Type[],
        nargout: d.nargout,
      });
      // Surface the emitted JS through the same hook the legacy JS-JIT
      // used so it shows up in the IDE's "internals" view and the CLI
      // `--dump-js` flag.
      const interp = ctx.interp as Interpreter;
      const typeDesc = d.argTypes.map(jitTypeKey).join(", ");
      interp.onJitCompile?.(
        `mtoc2-call:${cName}(${typeDesc}) -> nargout=${d.nargout}`,
        source
      );
      const factory = new Function(source)() as ($h: {
        write: (s: string) => void;
      }) => (...args: unknown[]) => unknown;
      const rt = interp.rt;
      const specFn = factory({ write: (s: string) => rt.output(s) });
      return { specFn };
    } catch (e) {
      if (e instanceof UnsupportedConstruct || e instanceof Mtoc2TypeError) {
        return null;
      }
      throw e;
    }
  },

  run(compiled, d): RunResult {
    if (compiled === null) {
      return { bail: { message: "mtoc2-call: codegen declined" } };
    }
    try {
      const mtoc2Args = d.args.map(v => numblToMtoc2(v as never));
      const result = compiled.specFn(...mtoc2Args);
      return { result: mtoc2ToNumbl(result) };
    } catch (e) {
      return {
        bail: {
          message: `mtoc2-call: runtime error: ${e instanceof Error ? e.message : String(e)}`,
        },
      };
    }
  },
};
