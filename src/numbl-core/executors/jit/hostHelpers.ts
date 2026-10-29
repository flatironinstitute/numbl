/**
 * Build the `$h` helpers object that JS-JIT emitted code receives.
 *
 *  - `write(s)` — output sink. Routed to `rt.output(...)` so `disp` /
 *    `fprintf` etc. land in numbl's output stream.
 *  - `warning(id, msg)` — `jitWarning`: shows a warning unless the
 *    warning state turns it off.
 *  - `plotDispatch(name, args)` — host hook for plot builtins.
 *    The emitted `jit_plot_dispatch(...)` calls this in
 *    preference to the JSON-on-stdout wire format.
 *    We translate args from the emitted value shape into `RuntimeValue`
 *    via `jitToNumbl` and call numbl's `dispatchPlotBuiltin`,
 *    pushing the resulting instruction onto `rt.plotInstructions` —
 *    so `colorbar('off')` / `pcolor(...)` / etc. produce the same
 *    plot-instruction stream as the interpreter would.
 *
 *  - `cancel` — the cancel flag the emitted loops poll (`$cancel`).
 *
 *  - `elemwise` — the interpreter's element-wise tensor kernels (native
 *    addon when loaded). The JS runtime snippets use them for large
 *    tensors, so --opt 0 and --opt 1 compute the same bits and large
 *    element-wise work runs as SIMD C rather than a scalar JS loop.
 *
 *  - `numeric` — matrix multiply through the LAPACK provider stack, the
 *    interpreter's own path (see `JitNumericHost`). The C-JIT reaches the
 *    same functions through callbacks (`bindHostNumeric`).
 *
 * Names `dispatchPlotBuiltin` does not handle are the stubs in
 * `PLOT_STUB_NAMES` (`drawnow`, `pause`, `xlim`, ...): they run as the
 * registered builtin, so a JIT'd `pause` sleeps (cancellably) like the
 * interpreter's. A name that is not a builtin at all is a programmer
 * error: throw so the mismatch is loud rather than silently dropping a
 * side effect.
 */

import type { Runtime } from "../../runtime/runtime.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { dispatchPlotBuiltin } from "../../plotting/dispatch.js";
import { invokeBuiltin, NOT_A_BUILTIN } from "../../runtime/runtimeDispatch.js";
import { RTV } from "../../runtime/constructors.js";
import { rngRandom } from "../../helpers/prng.js";
import { jitToNumbl } from "./valueAdapter.js";
import { tensorOps } from "../../ops/dispatch.js";
import { OpReduce, OpRealBin, OpUnary } from "../../ops/opCodes.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import { matmulIsInline } from "../../native/inline-matmul.js";

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
  /** Reduce all of `a` (non-empty) in sequential order, as the
   *  interpreter's `sum` / `prod` / `mean` and NaN-skipping `min` / `max`
   *  do. */
  reduce(op: "sum" | "prod" | "min" | "max", a: Float64Array): number;
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

const REDUCE_OPS: Record<string, number> = {
  sum: OpReduce.SUM,
  prod: OpReduce.PROD,
  min: OpReduce.MIN,
  max: OpReduce.MAX,
};
const reduceOut = new Float64Array(1);

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
  reduce(op, a) {
    tensorOps.realFlatReduce(REDUCE_OPS[op], a.length, a, reduceOut);
    return reduceOut[0];
  },
};

/** Matrix products for the runtime snippets, exposed as `$numeric` (see
 *  `runtime-globals.d.ts`). Arrays are column-major. Each returns null for
 *  a product small enough to run inline (`matmulIsInline`): the caller then
 *  runs its own loop, the same one the stack's inline provider runs, so
 *  every `--opt` level computes the same bits. */
export interface JitNumericHost {
  matmul(
    a: Float64Array,
    m: number,
    k: number,
    b: Float64Array,
    n: number
  ): Float64Array | null;
  /** An absent imaginary part is zero; the result always has `im`. */
  matmulComplex(
    aRe: Float64Array,
    aIm: Float64Array | undefined,
    m: number,
    k: number,
    bRe: Float64Array,
    bIm: Float64Array | undefined,
    n: number
  ): { re: Float64Array; im: Float64Array } | null;
}

export const numericHost: JitNumericHost = {
  matmul(a, m, k, b, n) {
    if (matmulIsInline(m, k, n)) return null;
    return getLapackBridge().matmul(a, m, k, b, n);
  },
  matmulComplex(aRe, aIm, m, k, bRe, bIm, n) {
    if (matmulIsInline(m, k, n)) return null;
    const r = getLapackBridge().matmulComplex(
      aRe,
      aIm ?? new Float64Array(m * k),
      m,
      k,
      bRe,
      bIm ?? new Float64Array(k * n),
      n
    );
    return { re: r.re, im: r.im ?? new Float64Array(m * n) };
  },
};

export interface JitHostHelpers {
  write: (s: string) => void;
  warning: (identifier: string, msg: string) => void;
  plotDispatch: (name: string, args: unknown[]) => void;
  /** One uniform double in [0, 1). Backed by numbl's process-global PRNG
   *  (the same `rngRandom` the interpreter uses), so JIT'd and interpreted
   *  `rand` draw from one shared, `rng(seed)`-controllable stream. */
  rand: () => number;
  elemwise: JitElemwiseHost;
  numeric: JitNumericHost;
  /** The cancel flag emitted loops poll: the runtime's (a
   *  SharedArrayBuffer view), or a private zero word when it has none. */
  cancel: Int32Array;
  /** Set by the emitted module's factory: reads its current statement
   *  location (`$loc`), so a raise can be attributed to a line. */
  loc?: () => number;
}

/** Run the plot builtin `name` as a statement: through the plot dispatch
 *  table, or else (a stub) as the registered builtin. */
function runPlotBuiltin(rt: Runtime, name: string, args: RuntimeValue[]) {
  if (dispatchPlotBuiltin(name, args, rt.plotInstructions, rt)) return;
  if (invokeBuiltin(rt, name, 0, args) === NOT_A_BUILTIN) {
    throw new Error(
      `JIT plot bridge: '${name}' is neither a plot builtin nor a builtin; ` +
        `the JIT's plot builtin set is out of sync with numbl's dispatch table`
    );
  }
}

export function buildHostHelpers(rt: Runtime): JitHostHelpers {
  return {
    write: (s: string) => rt.output(s),
    warning: (identifier: string, msg: string) =>
      jitWarning(rt, identifier, msg),
    rand: () => rngRandom(),
    elemwise: elemwiseHost,
    numeric: numericHost,
    cancel: rt.cancelFlag ?? new Int32Array(1),
    plotDispatch: (name: string, args: unknown[]) => {
      runPlotBuiltin(
        rt,
        name,
        args.map(a => jitToNumbl(a))
      );
    },
  };
}

/** A warning from compiled code, shown as the interpreter's `warning`
 *  would: unless the warning state turns off `identifier` ("" for a
 *  warning without one) or all warnings. */
function jitWarning(rt: Runtime, identifier: string, msg: string): void {
  const off = rt.warningsOff;
  if (off.has("all") || (identifier !== "" && off.has(identifier))) return;
  rt.output("Warning: " + msg + "\n");
}

/** Prefix of a plot record that compiled C writes to its output (see
 *  `runtime/plot/plot_dispatch.h`). */
const PLOT_RECORD_PREFIX = "\x1ejit:plot\t";

/** Prefix of a warning record (see `runtime/io/warning_fmt.h`): the
 *  identifier and message follow as tab-separated JSON string bodies. */
const WARNING_RECORD_PREFIX = "\x1ejit:warning\t";

/** Decode one argument of a C plot record. */
function decodePlotArg(a: unknown): RuntimeValue {
  if (a === null) return NaN;
  if (typeof a === "number") return a;
  const rec = a as { kind: string; data: unknown; dims?: number[] };
  if (rec.kind === "text") return RTV.char(rec.data as string);
  if (rec.kind === "tensor") {
    const data = Float64Array.from(rec.data as (number | null)[], x =>
      x === null ? NaN : x
    );
    return RTV.tensor(data, rec.dims ?? [1, data.length]);
  }
  throw new Error(`JIT plot bridge: unknown argument kind '${rec.kind}'`);
}

/** The output sink for a compiled C call: passes program output to
 *  `rt.output`, turns plot records into plot instructions and shows
 *  warning records. A record may
 *  arrive in several chunks; it ends at a newline. Call `flush` after the C
 *  call returns. */
export function cOutputSink(rt: Runtime): {
  write: (s: string) => void;
  flush: () => void;
} {
  let pending: string | null = null;
  const write = (s: string): void => {
    while (s.length > 0) {
      if (pending === null) {
        const rs = s.indexOf("\x1e");
        if (rs < 0) {
          rt.output(s);
          return;
        }
        if (rs > 0) rt.output(s.slice(0, rs));
        pending = "";
        s = s.slice(rs);
      }
      const nl = s.indexOf("\n");
      if (nl < 0) {
        pending += s;
        return;
      }
      const line = pending + s.slice(0, nl);
      pending = null;
      s = s.slice(nl + 1);
      if (line.startsWith(WARNING_RECORD_PREFIX)) {
        const [id, msg] = line
          .slice(WARNING_RECORD_PREFIX.length)
          .split("\t")
          .map(part => JSON.parse(`"${part}"`) as string);
        jitWarning(rt, id, msg);
        continue;
      }
      if (!line.startsWith(PLOT_RECORD_PREFIX)) {
        rt.output(line + "\n");
        continue;
      }
      const rec = JSON.parse(line.slice(PLOT_RECORD_PREFIX.length)) as {
        call: string;
        args: unknown[];
      };
      runPlotBuiltin(rt, rec.call, rec.args.map(decodePlotArg));
    }
  };
  const flush = (): void => {
    if (pending !== null) rt.output(pending);
    pending = null;
  };
  return { write, flush };
}
