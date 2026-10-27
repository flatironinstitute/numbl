/**
 * Build the `$h` helpers object that mtoc2's emitted spec receives
 * when invoked via the numbl JIT bridge.
 *
 *  - `write(s)` — output sink. Routed to `rt.output(...)` so `disp` /
 *    `fprintf` / `warning` etc. land in numbl's output stream.
 *  - `plotDispatch(name, args)` — host hook for plot builtins.
 *    mtoc2's emitted `mtoc2_plot_dispatch(...)` calls this in
 *    preference to the standalone-mode JSON-on-stdout wire format.
 *    We translate args from mtoc2's emit shape into `RuntimeValue`
 *    via `jitToNumbl` and call numbl's `dispatchPlotBuiltin`,
 *    pushing the resulting instruction onto `rt.plotInstructions` —
 *    so `colorbar('off')` / `pcolor(...)` / etc. produce the same
 *    plot-instruction stream as the interpreter would.
 *
 *  - `elemwise` — the interpreter's element-wise tensor kernels (native
 *    addon when loaded). The JS runtime snippets use them for large
 *    tensors, so --opt 0 and --opt 1 compute the same bits and large
 *    element-wise work runs as SIMD C rather than a scalar JS loop.
 *
 * The unknown-name fallthrough (when `dispatchPlotBuiltin` returns
 * false) is a programmer error: every name mtoc2's plot builtin
 * registration covers is in `PLOT_ALL_NAMES`, which is sourced from
 * the same list `dispatchPlotBuiltin` switches on. Throw so the
 * mismatch is loud rather than silently dropping a side effect.
 */

import type { Runtime } from "../../runtime/runtime.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { dispatchPlotBuiltin } from "../../plotting/dispatch.js";
import { rngRandom } from "../../helpers/prng.js";
import { jitToNumbl } from "./valueAdapter.js";
import { tensorOps } from "../../ops/dispatch.js";
import { OpReduce, OpRealBin, OpUnary } from "../../ops/opCodes.js";

/** Element-wise kernels exposed to the JS runtime snippets as
 *  `$elemwise` (see `runtime-globals.d.ts`). `out` has `a`'s length. */
export interface JitElemwiseHost {
  binary(op: string, a: Float64Array, b: Float64Array, out: Float64Array): void;
  scalarBinary(
    op: string,
    s: number,
    a: Float64Array,
    scalarOnLeft: boolean,
    out: Float64Array
  ): void;
  /** False when `op` has no host kernel (the caller runs its own loop). */
  unary(op: string, a: Float64Array, out: Float64Array): boolean;
  /** NaN-skipping `min` / `max` over all of `a` (non-empty). */
  minmax(op: "min" | "max", a: Float64Array): number;
}

const BINARY_OPS: Record<string, number> = {
  plus: OpRealBin.ADD,
  minus: OpRealBin.SUB,
  times: OpRealBin.MUL,
  rdivide: OpRealBin.DIV,
};

const UNARY_OPS: Record<string, number> = {
  exp: OpUnary.EXP,
  log: OpUnary.LOG,
  log2: OpUnary.LOG2,
  log10: OpUnary.LOG10,
  sqrt: OpUnary.SQRT,
  abs: OpUnary.ABS,
  floor: OpUnary.FLOOR,
  ceil: OpUnary.CEIL,
  round: OpUnary.ROUND,
  fix: OpUnary.TRUNC,
  sin: OpUnary.SIN,
  cos: OpUnary.COS,
  tan: OpUnary.TAN,
  atan: OpUnary.ATAN,
  sinh: OpUnary.SINH,
  cosh: OpUnary.COSH,
  tanh: OpUnary.TANH,
  sign: OpUnary.SIGN,
};

const elemwiseHost: JitElemwiseHost = {
  binary(op, a, b, out) {
    tensorOps.realBinaryElemwise(BINARY_OPS[op], out.length, a, b, out);
  },
  scalarBinary(op, s, a, scalarOnLeft, out) {
    tensorOps.realScalarBinaryElemwise(
      BINARY_OPS[op],
      out.length,
      s,
      a,
      scalarOnLeft,
      out
    );
  },
  unary(op, a, out) {
    const code = UNARY_OPS[op];
    if (code === undefined) return false;
    tensorOps.realUnaryElemwise(code, out.length, a, out);
    return true;
  },
  minmax(op, a) {
    const out = new Float64Array(1);
    const code = op === "min" ? OpReduce.MIN : OpReduce.MAX;
    tensorOps.realFlatReduce(code, a.length, a, out);
    return out[0];
  },
};

export interface JitHostHelpers {
  write: (s: string) => void;
  plotDispatch: (name: string, args: unknown[]) => void;
  /** One uniform double in [0, 1). Backed by numbl's process-global PRNG
   *  (the same `rngRandom` the interpreter uses), so JIT'd and interpreted
   *  `rand` draw from one shared, `rng(seed)`-controllable stream. */
  rand: () => number;
  elemwise: JitElemwiseHost;
}

export function buildHostHelpers(rt: Runtime): JitHostHelpers {
  return {
    write: (s: string) => rt.output(s),
    rand: () => rngRandom(),
    elemwise: elemwiseHost,
    plotDispatch: (name: string, args: unknown[]) => {
      const runtimeArgs: RuntimeValue[] = args.map(a => jitToNumbl(a));
      const handled = dispatchPlotBuiltin(
        name,
        runtimeArgs,
        rt.plotInstructions,
        rt
      );
      if (!handled) {
        throw new Error(
          `mtoc2 plot bridge: dispatchPlotBuiltin rejected '${name}' — ` +
            `mtoc2's plot builtin set is out of sync with numbl's dispatch table`
        );
      }
    },
  };
}
