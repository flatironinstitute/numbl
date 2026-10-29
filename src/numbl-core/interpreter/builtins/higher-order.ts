/**
 * Builtins that call other functions: arrayfun, cellfun, structfun, feval,
 * bsxfun, quadgk, subsref, subsasgn, builtin, onCleanup.
 */

import { RTV, toNumber, toString, RuntimeError } from "../../runtime/index.js";
import {
  type RuntimeValue,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeFunction,
  isRuntimeNumber,
} from "../../runtime/types.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import {
  arrayfunImpl,
  cellfunImpl,
  structfunImpl,
  bsxfunImpl,
  subsrefBuiltin,
  subsasgnBuiltin,
} from "../../runtime/runtimeDispatch.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { quadgkAdaptive } from "../../helpers/quadgk.js";
import { builtin } from "./builtin.js";

builtin({
  name: "arrayfun",
  accepts: { sparse: true },
  nargin: [2, Infinity],
  nargout: Infinity,
  apply: (args, nargout, ctx) =>
    arrayfunImpl(ctx.rt, nargout, args) as RuntimeValue | RuntimeValue[],
});

builtin({
  name: "cellfun",
  accepts: { sparse: true },
  nargin: [2, Infinity],
  nargout: Infinity,
  apply: (args, nargout, ctx) =>
    cellfunImpl(ctx.rt, nargout, args) as RuntimeValue | RuntimeValue[],
});

builtin({
  name: "structfun",
  nargin: [2, Infinity],
  nargout: Infinity,
  apply: (args, nargout, ctx) =>
    structfunImpl(ctx.rt, nargout, args) as RuntimeValue | RuntimeValue[],
});

// feval(name, ...) resolves `name` from the calling scope, without nested
// functions (MATLAB's string-named calls do not see them).
builtin({
  name: "feval",
  accepts: { sparse: true, gpu: true },
  nargin: [1, Infinity],
  nargout: Infinity,
  apply: (args, nargout, ctx) => {
    const fn = args[0];
    const rest = args.slice(1);
    if (isRuntimeFunction(fn))
      return ctx.rt.index(fn, rest, nargout) as RuntimeValue | RuntimeValue[];
    if (isRuntimeChar(fn) || isRuntimeString(fn))
      return ctx.rt.dispatch(toString(fn), nargout, rest) as
        | RuntimeValue
        | RuntimeValue[];
    throw new RuntimeError(
      "feval: first argument must be a function handle or name"
    );
  },
});

builtin({
  name: "bsxfun",
  accepts: { sparse: true },
  nargin: 3,
  apply: (args, nargout, ctx) =>
    bsxfunImpl(ctx.rt, nargout, args) as RuntimeValue | RuntimeValue[],
});

builtin({
  name: "quadgk",
  nargin: [3, Infinity],
  nargout: 2,
  mfile: true,
  apply: (args, nargout, ctx) => {
    const rt = ctx.rt;
    const fnArg = args[0];
    if (!isRuntimeFunction(fnArg))
      throw new RuntimeError("quadgk: first argument must be a function");
    const a = toNumber(args[1]);
    const b = toNumber(args[2]);

    // Parse name-value option pairs.  Unknown names are ignored so that
    // chunkie-style `quadgk(fun, a, b, quadgkparams{:})` spreads that may
    // include options we don't yet support (Waypoints, etc.) don't error.
    let relTol: number | undefined;
    let absTol: number | undefined;
    let maxIntervalCount: number | undefined;
    for (let i = 3; i + 1 < args.length; i += 2) {
      const keyRv = args[i];
      const key = isRuntimeChar(keyRv)
        ? keyRv.value
        : isRuntimeString(keyRv)
          ? (keyRv as string)
          : "";
      const lowerKey = key.toLowerCase();
      const valRv = args[i + 1];
      if (lowerKey === "reltol") relTol = toNumber(valRv);
      else if (lowerKey === "abstol") absTol = toNumber(valRv);
      else if (lowerKey === "maxintervalcount")
        maxIntervalCount = toNumber(valRv);
      // Waypoints and other options are silently ignored for now.
    }

    // Vectorized integrand: pass the 15 nodes as a 1x15 row vector and
    // expect a 1x15 (or 15x1) numeric result.
    const integrand = (pts: number[]): number[] => {
      const vecData = allocFloat64Array(pts);
      const vec = RTV.tensor(vecData, [1, pts.length]);
      const resultRaw = rt.index(fnArg, [vec], 1);
      const rv = ensureRuntimeValue(resultRaw as RuntimeValue);
      if (isRuntimeNumber(rv)) {
        // Scalar result: broadcast (rare — most integrands vectorize).
        return new Array<number>(pts.length).fill(rv as number);
      }
      if (isRuntimeTensor(rv)) {
        if (rv.data.length !== pts.length) {
          throw new RuntimeError(
            `quadgk: integrand returned ${rv.data.length} values for ${pts.length} nodes`
          );
        }
        return Array.from(rv.data);
      }
      throw new RuntimeError("quadgk: integrand must return a numeric vector");
    };

    const result = quadgkAdaptive(integrand, a, b, {
      relTol,
      absTol,
      maxIntervalCount,
    });

    if (nargout >= 2) return [result.value, result.errbnd];
    return result.value;
  },
});

builtin({
  name: "subsref",
  accepts: { gpu: true },
  nargin: 2,
  nargout: Infinity,
  apply: (args, nargout, ctx) =>
    subsrefBuiltin(ctx.rt, nargout, args) as RuntimeValue | RuntimeValue[],
});

builtin({
  name: "subsasgn",
  accepts: { sparse: true, gpu: true },
  nargin: 3,
  apply: (args, nargout, ctx) =>
    subsasgnBuiltin(ctx.rt, nargout, args) as RuntimeValue | RuntimeValue[],
});

builtin({
  name: "builtin",
  accepts: { sparse: true, gpu: true },
  nargin: [1, Infinity],
  nargout: Infinity,
  apply: (args, nargout, ctx) => {
    const fnNameArg = args[0];
    let fnName: string;
    if (isRuntimeFunction(fnNameArg)) {
      fnName = fnNameArg.name;
    } else if (isRuntimeChar(fnNameArg) || isRuntimeString(fnNameArg)) {
      fnName = isRuntimeString(fnNameArg) ? fnNameArg : fnNameArg.value;
    } else {
      throw new RuntimeError(
        "builtin: first argument must be a function name or handle"
      );
    }
    return ctx.rt.callBuiltin(fnName, nargout, args.slice(1)) as
      | RuntimeValue
      | RuntimeValue[];
  },
});

builtin({
  name: "onCleanup",
  nargin: [0, 1],
  mfile: true,
  apply: (args, _nargout, ctx) => {
    if (args.length !== 1 || !isRuntimeFunction(args[0]))
      throw new RuntimeError("onCleanup requires a function handle argument");
    ctx.rt.registerCleanup(args[0]);
    return RTV.classInstance("onCleanup", [], false);
  },
});
