/**
 * scalar-fn-c-kernel — C codegen executor for the call shape.
 *
 * Consumes the same lowered call IR the JS-JIT call executor sees;
 * codegens to a C kernel via koffi when the args + outputs are all
 * scalar (number/boolean).
 *
 *   - `propose()` filters on `lowered.kind === "call"`, then applies
 *     the e2-specific scalar-args gate (skips on tensors, structs,
 *     cells, etc.). Output scalar-ness is verified during compile.
 *
 *   - `compile()` calls `compileE2ScalarFn` against the lowered IR.
 *     Runs `checkCFeasibility`, `generateC`, koffi compilation.
 *     Cached by the registry. Returns null when the body isn't
 *     pure-scalar — registry marks BAILED.
 *
 *   - `run()` calls `runE2ScalarFn`. The C kernel runs to completion
 *     or raises — no mid-execution bail mechanism, so `bailRisk`
 *     stays false.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import type { CallLowered } from "../jsJit/jitCall.js";
import {
  compileE2ScalarFn,
  runE2ScalarFn,
  type E2ScalarFnCompiled,
} from "./scalarFnDriver.js";

interface ScalarFnData {
  readonly lowered: CallLowered;
  readonly args: readonly unknown[];
}

const SCALAR_FN_C_COST = { compileMs: 80, perCallNs: 300, runNs: 100 };

export const scalarFnCKernelExecutor: Executor<
  ScalarFnData,
  E2ScalarFnCompiled | null
> = {
  name: "scalar-fn-c-kernel",

  propose(lowered: LoweredStmt): Proposal<ScalarFnData> | null {
    if (lowered.kind !== "call") return null;

    const { classification, args } = lowered;
    const fn = classification.fn;

    // Param count must match — generateC's ABI is positional /
    // fixed-arity, no MATLAB-style nargin underflow.
    if (args.length !== fn.params.length) return null;

    // All args must be scalar number/boolean. Tensors, cells,
    // structs, chars, etc. need the tensor-aware lowering path
    // we don't have here.
    for (const a of args) {
      if (typeof a !== "number" && typeof a !== "boolean") return null;
    }

    return {
      data: { lowered: lowered.lowered, args },
      cost: SCALAR_FN_C_COST,
      // C kernel runs to completion or doesn't run at all — no
      // mid-execution bail mechanism.
      bailRisk: false,
    };
  },

  cacheKey(d): string {
    return d.lowered.classification.cacheKey;
  },

  compile(d, ctx: DispatchContext): E2ScalarFnCompiled | null {
    return compileE2ScalarFn(ctx.interp, d.lowered);
  },

  run(compiled, d): RunResult {
    if (compiled === null) {
      return { bail: { message: "scalar-fn-c-kernel: not pure-scalar" } };
    }
    const result = runE2ScalarFn(
      d.args,
      compiled,
      d.lowered.classification.nargout
    );
    return { result };
  },
};
