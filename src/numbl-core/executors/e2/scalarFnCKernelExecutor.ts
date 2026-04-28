/**
 * scalar-fn-c-kernel — port of the e2 whole-function scalar C JIT
 * (`tryE2ScalarFn`).
 *
 * Wraps the existing per-FunctionDef pure-scalar C-kernel path: when
 * all args are scalar number/boolean and all outputs are scalar, the
 * function body compiles to a single C function via the same lowering
 * pipeline `--opt 2` and `--opt e1` use, and dispatches via koffi.
 *
 * Same shim pattern as the other e2 executors: classification and
 * per-FunctionDef caching live in the wrapped layer; the shim
 * declares a transient bail so the registry cache stays out of the
 * way.
 *
 * `bailRisk: false` because the C kernel runs to completion or
 * doesn't run at all — there is no mid-execution bail mechanism for
 * this path.
 */

import type { CallExecutor, CallProposal, CallRunResult } from "../types.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import { tryE2ScalarFn, E2_SKIP } from "./scalarFnDriver.js";

interface ScalarFnData {
  readonly _: 0;
}

const SHARED_DATA: ScalarFnData = { _: 0 };
const SCALAR_FN_C_COST = { compileMs: 80, perCallNs: 300, runNs: 100 };
const SHARED_PROPOSAL: CallProposal<ScalarFnData> = {
  data: SHARED_DATA,
  cost: SCALAR_FN_C_COST,
};

export const scalarFnCKernelExecutor: CallExecutor<ScalarFnData> = {
  name: "scalar-fn-c-kernel",
  bailRisk: false,

  proposeCall(): CallProposal<ScalarFnData> {
    return SHARED_PROPOSAL;
  },

  runCall(_data, fn, args, nargout, interp: Interpreter): CallRunResult {
    const r = tryE2ScalarFn(interp, fn, args, nargout);
    if (r === E2_SKIP) {
      return {
        bail: { message: "scalar-fn-c-kernel: not eligible" },
        transient: true,
      };
    }
    return { result: r };
  },
};
