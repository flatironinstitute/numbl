/**
 * `logspace(a, b)` / `logspace(a, b, n)` — n logarithmically-spaced
 * values from 10^a to 10^b as a 1×n row tensor (default n = 50).
 * Matches numbl's interpreter, including the MATLAB special case where
 * an upper limit of exactly `pi` makes the last point `pi` (not 10^pi).
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  DIM_ONE,
  isNumeric,
  isScalar,
  tensorDouble,
  tensorDoubleFromDims,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import { mtoc2_tensor_logspace as jsLogspace } from "../../runtime/snippets.gen.js";

const DEFAULT_N = 50;

function requireScalarReal(t: Type, what: string): void {
  if (!isNumeric(t)) {
    throw new TypeError(
      `${what} must be a scalar real numeric (got ${t.kind})`
    );
  }
  if (t.isComplex) {
    throw new TypeError(`${what} must be real (got complex)`);
  }
  if (t.elem !== "double" && t.elem !== "logical") {
    throw new TypeError(`${what} must be double or logical (got ${t.elem})`);
  }
  if (!isScalar(t)) {
    throw new TypeError(`${what} must be a scalar (got tensor)`);
  }
}

/** A 1×? row tensor type — used when the exact length is unknown, and
 *  for the `n == 1` case (a runtime 1×1, which we deliberately do NOT
 *  type as `[1,1]` because that collapses to a scalar and scalar
 *  indexing — `v(1)` — is not JIT-compiled). */
const ROW_UNKNOWN = (): ReturnType<typeof tensorDoubleFromDims> =>
  tensorDoubleFromDims([DIM_ONE, { kind: "unknown" }]);

export const logspace: Builtin = {
  name: "logspace",
  transfer(argTypes, nargout) {
    if (argTypes.length < 2 || argTypes.length > 3) {
      throw new TypeError(
        `'logspace' expects 2..3 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'logspace' does not support multi-output (nargout=${nargout})`
      );
    }
    requireScalarReal(argTypes[0], "'logspace' arg 1");
    requireScalarReal(argTypes[1], "'logspace' arg 2");
    if (argTypes.length === 3) {
      requireScalarReal(argTypes[2], "'logspace' arg 3");
    }

    const nRaw = argTypes.length === 3 ? exactDouble(argTypes[2]) : DEFAULT_N;
    if (nRaw !== undefined) {
      const n = Math.round(nRaw);
      if (n <= 0) return [tensorDouble([1, 0])];
      if (n >= 2) return [tensorDouble([1, n])];
      // n === 1 falls through to the row-unknown type (see above).
    }
    return [ROW_UNKNOWN()];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_logspace");
    const aC = argsC[0];
    const bC = argsC[1];
    const nC = argTypes.length === 3 ? argsC[2] : `${DEFAULT_N}.0`;
    return `mtoc2_tensor_logspace((double)(${aC}), (double)(${bC}), (long)lround(${nC}))`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_logspace");
    const aJs = argsJs[0];
    const bJs = argsJs[1];
    const nJs = argTypes.length === 3 ? argsJs[2] : String(DEFAULT_N);
    return `mtoc2_tensor_logspace(${aJs}, ${bJs}, Math.round(${nJs}))`;
  },
  call({ args, argTypes }) {
    const av = typeof args[0] === "number" ? args[0] : Number(args[0]);
    const bv = typeof args[1] === "number" ? args[1] : Number(args[1]);
    let n = DEFAULT_N;
    if (argTypes.length === 3) {
      const nv = typeof args[2] === "number" ? args[2] : Number(args[2]);
      n = Math.round(nv);
    }
    return [jsLogspace(av, bv, n) as unknown as RuntimeTensor];
  },
};
