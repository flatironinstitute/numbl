/**
 * c-jit-loop — C codegen executor for scalar for/while loops.
 *
 *   - `propose()` filters on `lowered.kind === "loop"`, requires a
 *     wired `nativeBridge`, applies the C feasibility whitelist
 *     (scalar-only IR), and rejects loops containing IO + bail-risk.
 *
 *   - `compile()` generates C source from the lowered IR, caches the
 *     compiled `.so` under `~/.cache/numbl/c-jit/<sha>.so`, and loads
 *     via koffi.
 *
 *   - `run()` reads loop inputs from the interpreter env, invokes the
 *     compiled C function with an output Float64Array, writes back
 *     the outputs.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import type { LoopLowered } from "../jsJit/jitLoop.js";
import type { JitType } from "../../jitTypes.js";
import { jitTypeKey } from "../../jitTypes.js";
import { RTV } from "../../runtime/constructors.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { isCJitFeasible } from "./whitelist.js";
import { generateCSource } from "./codegen.js";
import { compileAndLoad, type CompiledC } from "./compile.js";

interface CLoopCompiled {
  readonly compiled: CompiledC;
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  /** Source kept for diagnostics; helpful in --dump-c output. */
  readonly source: string;
}

// Cost is set to win comfortably against the AST interpreter for a
// hot loop. Once a real cost-model exists, this scales with iteration
// count.
const C_JIT_LOOP_COST = { compileMs: 200, perCallNs: 100, runNs: 100 };

export const cJitLoopExecutor: Executor<LoopLowered, CLoopCompiled | null> = {
  name: "c-jit-loop",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<LoopLowered> | null {
    if (lowered.kind !== "loop") return null;
    if (!ctx.interp.nativeBridge) return null;

    const flags = lowered.flags;
    if (flags.hasReturn) return null;
    if (flags.hasIO && flags.hasBailRisk) return null;
    if (flags.hasIO) return null; // C-JIT has no IO emit (no disp/fprintf yet).

    // All inputs must be scalar reals (the only type the codegen
    // handles today). Inputs/outputs are mapped 1:1 to f64 args / out
    // slots in C.
    const cls = lowered.lowered.classification;
    for (const t of cls.inputTypes) {
      if (!isScalarReal(t)) return null;
    }

    if (!isCJitFeasible(lowered.lowered.result.body)) return null;

    return {
      data: lowered.lowered,
      cost: C_JIT_LOOP_COST,
      bailRisk: true,
    };
  },

  cacheKey(d): string {
    return d.classification.cacheKey;
  },

  compile(d, ctx: DispatchContext): CLoopCompiled | null {
    const bridge = ctx.interp.nativeBridge;
    if (!bridge) return null;

    const cls = d.classification;
    const fnName = `c_jit_loop_${cls.kind}`;
    const source = generateCSource(
      fnName,
      cls.inputs,
      cls.outputs,
      d.result.body
    );

    const paramComments = cls.inputs
      .map((p, i) => `${p}: ${jitTypeKey(cls.inputTypes[i])}`)
      .join(", ");
    const line = ctx.interp.rt.$line ?? 0;
    ctx.interp.onCJitCompile?.(
      `loop:${cls.kind}@${line}(${paramComments})`,
      source
    );

    let compiled: CompiledC;
    try {
      compiled = compileAndLoad(
        source,
        { fnName, nInputs: cls.inputs.length, nOutputs: cls.outputs.length },
        bridge
      );
    } catch (e) {
      // Surface the failure once to the user, then disable this
      // proposal via cache-bail.
      console.warn(
        `Warning: c-jit-loop compile failed; falling back to interpreter. ${
          e instanceof Error ? e.message : String(e)
        }`
      );
      return null;
    }

    return {
      compiled,
      inputs: cls.inputs,
      outputs: cls.outputs,
      source,
    };
  },

  run(compiled, _d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "c-jit-loop: codegen rejected" } };
    }

    const interp = ctx.interp;
    const inputs = compiled.inputs;
    const outputs = compiled.outputs;

    // Gather input values — must all be plain JS numbers (whitelist
    // guarantees scalar real types).
    const args: number[] = [];
    for (const name of inputs) {
      const v = interp.env.get(name);
      const n = toScalarNumber(v);
      if (n === null) {
        return {
          bail: {
            message: `c-jit-loop: input '${name}' is not a scalar number at run time`,
          },
          transient: true,
        };
      }
      args.push(n);
    }

    const out = new Float64Array(outputs.length);
    try {
      compiled.compiled.fn(out, ...args);
    } catch (e) {
      return {
        bail: {
          message: `c-jit-loop: native invocation failed: ${
            e instanceof Error ? e.message : String(e)
          }`,
        },
      };
    }

    // Write back outputs.
    for (let i = 0; i < outputs.length; i++) {
      interp.env.set(outputs[i], RTV.num(out[i]) as RuntimeValue);
    }
    return { consumed: 1 };
  },
};

function isScalarReal(t: JitType): boolean {
  if (t.kind === "number") return true;
  if (t.kind === "boolean") return true;
  return false;
}

function toScalarNumber(v: unknown): number | null {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (
    v !== null &&
    typeof v === "object" &&
    "type" in v &&
    (v as { type?: string }).type === "number"
  ) {
    const val = (v as { value?: unknown }).value;
    if (typeof val === "number") return val;
  }
  return null;
}
