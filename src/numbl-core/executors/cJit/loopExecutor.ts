/**
 * c-jit-loop — C codegen executor for scalar for/while loops.
 *
 *   - `propose()` filters on `lowered.kind === "loop"`, requires a
 *     wired `nativeBridge`, applies the C feasibility whitelist, and
 *     rejects loops containing IO + bail-risk.
 *
 *   - `compile()` generates C source from the lowered IR, caches the
 *     compiled `.so` under `~/.cache/numbl/c-jit/<sha>.so`, and loads
 *     via koffi.
 *
 *   - `run()` reads loop inputs from the interpreter env, marshals
 *     complex values to (re, im) pairs, invokes the compiled C
 *     function with an output Float64Array, and writes the outputs
 *     back to env.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import type { LoopLowered } from "../jsJit/jitLoop.js";
import { jitTypeKey } from "../../jitTypes.js";
import { RTV } from "../../runtime/constructors.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { isCJitFeasible, isCScalarType } from "./whitelist.js";
import {
  generateCSource,
  inferVarEncodings,
  totalSlotCount,
  type VarEncoding,
} from "./codegen.js";
import { compileAndLoad, type CompiledC } from "./compile.js";
import { zeroedFloat64 } from "../../runtime/alloc.js";

interface CLoopCompiled {
  readonly compiled: CompiledC;
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly inputEncodings: readonly VarEncoding[];
  readonly outputEncodings: readonly VarEncoding[];
  /** Total `double` slots in `out` — sum of per-output slot counts. */
  readonly outputSlots: number;
  readonly source: string;
}

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
    if (flags.hasIO) return null;

    const cls = lowered.lowered.classification;
    for (const t of cls.inputTypes) {
      if (!isCScalarType(t)) return null;
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
    const varTypes = inferVarEncodings(
      cls.inputs,
      cls.inputTypes,
      d.result.body
    );
    const source = generateCSource(
      fnName,
      cls.inputs,
      cls.outputs,
      d.result.body,
      varTypes
    );

    const paramComments = cls.inputs
      .map((p, i) => `${p}: ${jitTypeKey(cls.inputTypes[i])}`)
      .join(", ");
    const line = ctx.interp.rt.$line ?? 0;
    ctx.interp.onCJitCompile?.(
      `loop:${cls.kind}@${line}(${paramComments})`,
      source
    );

    const inputEncodings = cls.inputs.map(n => varTypes.get(n) ?? "real");
    const outputEncodings = cls.outputs.map(n => varTypes.get(n) ?? "real");
    const nOutputSlots = totalSlotCount(cls.outputs, varTypes);

    const params: string[] = [];
    for (let i = 0; i < cls.inputs.length; i++) {
      if (inputEncodings[i] === "complex") {
        params.push(`double a${i}re`);
        params.push(`double a${i}im`);
      } else {
        params.push(`double a${i}`);
      }
    }
    const declaration = `void ${fnName}(double *out${
      params.length > 0 ? ", " + params.join(", ") : ""
    })`;

    let compiled: CompiledC;
    try {
      compiled = compileAndLoad(source, declaration, bridge, {
        fastMath: ctx.interp.fastMath,
      });
    } catch (e) {
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
      inputEncodings,
      outputEncodings,
      outputSlots: nOutputSlots,
      source,
    };
  },

  run(compiled, _d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "c-jit-loop: codegen rejected" } };
    }

    const interp = ctx.interp;
    const inputs = compiled.inputs;
    const inputEncodings = compiled.inputEncodings;
    const outputs = compiled.outputs;
    const outputEncodings = compiled.outputEncodings;

    // Gather + marshal input values. Complex inputs contribute two
    // f64 args (re, im).
    const args: number[] = [];
    for (let i = 0; i < inputs.length; i++) {
      const v = interp.env.get(inputs[i]);
      const enc = inputEncodings[i];
      if (enc === "complex") {
        const c = toComplexPair(v);
        if (c === null) {
          return {
            bail: {
              message: `c-jit-loop: input '${inputs[i]}' is not a scalar number/complex at run time`,
            },
            transient: true,
          };
        }
        args.push(c.re, c.im);
      } else {
        const n = toScalarNumber(v);
        if (n === null) {
          return {
            bail: {
              message: `c-jit-loop: input '${inputs[i]}' is not a scalar number at run time`,
            },
            transient: true,
          };
        }
        args.push(n);
      }
    }

    const out = zeroedFloat64(compiled.outputSlots);
    try {
      (compiled.compiled.fn as (...a: unknown[]) => unknown)(out, ...args);
    } catch (e) {
      return {
        bail: {
          message: `c-jit-loop: native invocation failed: ${
            e instanceof Error ? e.message : String(e)
          }`,
        },
      };
    }

    // Write back outputs. Each complex output reads two consecutive slots.
    let slot = 0;
    for (let i = 0; i < outputs.length; i++) {
      const enc = outputEncodings[i];
      if (enc === "complex") {
        const re = out[slot];
        const im = out[slot + 1];
        // Decay to a real number when the imaginary part vanishes —
        // this matches MATLAB's "purely real result is a real
        // number" semantics for downstream type tests.
        const value =
          im === 0 ? RTV.num(re) : (RTV.complex(re, im) as RuntimeValue);
        interp.env.set(outputs[i], value as RuntimeValue);
        slot += 2;
      } else {
        interp.env.set(outputs[i], RTV.num(out[slot]) as RuntimeValue);
        slot += 1;
      }
    }
    return { ok: true };
  },
};

function toScalarNumber(v: unknown): number | null {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  return null;
}

function toComplexPair(v: unknown): { re: number; im: number } | null {
  if (typeof v === "number") return { re: v, im: 0 };
  if (typeof v === "boolean") return { re: v ? 1 : 0, im: 0 };
  if (
    v !== null &&
    typeof v === "object" &&
    "kind" in v &&
    (v as { kind?: string }).kind === "complex_number"
  ) {
    const c = v as { re?: unknown; im?: unknown };
    if (typeof c.re === "number" && typeof c.im === "number") {
      return { re: c.re, im: c.im };
    }
  }
  return null;
}
