/**
 * js-jit-call — port of the JS-JIT user-function-call hook
 * (`tryJitCall`).
 *
 * Wraps the existing whole-function JS-JIT path: when a user function
 * is called with eligible arg types, the function body compiles to a
 * type-specialized JS function via `new Function()` and runs in
 * place of the AST interpreter walking the body.
 *
 * Same shim pattern as the stmt-level executors: the wrapped layer
 * (`jit/index.ts`) maintains its own per-FunctionDef cache keyed by
 * argument-type signature, with progressive widening. The shim
 * declares a transient bail so the registry's outer cache stays out
 * of the way.
 *
 * `bailRisk: true` is conservative — `tryJitCall` may throw
 * `JitBailToInterpreter` mid-execution, but the wrapped layer's
 * `runWithCallFrame` already absorbs that into a JIT_SKIP return,
 * which the shim translates to a transient bail. The flag still
 * surfaces the fact that the artifact's correctness depends on type
 * assumptions, in case a future caller wants to filter on it.
 */

import type { CallExecutor, CallProposal, CallRunResult } from "../types.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import { tryJitCall, JIT_SKIP } from "./jitCall.js";

interface JsJitCallData {
  /** No per-call data needed; the wrapped layer reads everything from
   *  fn / args / nargout passed through to runCall. */
  readonly _: 0;
}

// Constant proposal and cost — both shared across all calls, avoiding
// per-call object allocation. Possible because the shim always
// proposes with the same shape; the wrapped layer (`tryJitCall`) does
// the actual eligibility check.
const SHARED_DATA: JsJitCallData = { _: 0 };
const JS_JIT_CALL_COST = { compileMs: 50, perCallNs: 200, runNs: 200 };
const SHARED_PROPOSAL: CallProposal<JsJitCallData> = {
  data: SHARED_DATA,
  cost: JS_JIT_CALL_COST,
  // The artifact's correctness depends on type assumptions that can
  // fail at runtime. (Currently call dispatch has no requireNoBail
  // pathway, but kept for symmetry.)
  bailRisk: true,
};

export const jsJitCallExecutor: CallExecutor<JsJitCallData> = {
  name: "js-jit-call",

  proposeCall(): CallProposal<JsJitCallData> {
    // Propose unconditionally — the wrapped layer does its own
    // eligibility check and returns JIT_SKIP when types are
    // unsuitable. We translate that to a transient bail.
    return SHARED_PROPOSAL;
  },

  runCall(_data, fn, args, nargout, interp: Interpreter): CallRunResult {
    const r = tryJitCall(interp, fn, args, nargout);
    if (r === JIT_SKIP) {
      return {
        bail: { message: "js-jit-call: not jittable" },
        transient: true,
      };
    }
    return { result: r };
  },
};
