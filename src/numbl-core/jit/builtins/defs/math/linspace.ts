/**
 * `linspace(a, b)` / `linspace(a, b, n)` — n linearly-spaced values
 * from `a` to `b` inclusive as a 1×n row tensor.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  DIM_ONE,
  EXACT_ARRAY_MAX_ELEMENTS,
  isNumeric,
  isScalar,
  scalarDouble,
  signFromNumber,
  tensorDouble,
  tensorDoubleFromDims,
  type DimInfo,
  type NumericType,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import { mtoc2_tensor_linspace as jsLinspace } from "../../runtime/snippets.gen.js";

const DEFAULT_N = 100;

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

function computeLinspaceData(a: number, b: number, n: number): Float64Array {
  const data = new Float64Array(n);
  if (n === 0) return data;
  if (n === 1) {
    data[0] = b;
    return data;
  }
  data[0] = a;
  data[n - 1] = b;
  for (let i = 1; i < n - 1; i++) {
    data[i] = a + ((b - a) * i) / (n - 1);
  }
  if (
    (n & 1) === 1 &&
    !Number.isFinite(a) &&
    !Number.isFinite(b) &&
    Math.sign(a) !== Math.sign(b)
  ) {
    data[(n - 1) / 2] = 0;
  }
  return data;
}

export const linspace: Builtin = {
  name: "linspace",
  transfer(argTypes, nargout) {
    if (argTypes.length < 2 || argTypes.length > 3) {
      throw new TypeError(
        `'linspace' expects 2..3 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'linspace' does not support multi-output (nargout=${nargout})`
      );
    }
    requireScalarReal(argTypes[0], "'linspace' arg 1");
    requireScalarReal(argTypes[1], "'linspace' arg 2");
    if (argTypes.length === 3) {
      requireScalarReal(argTypes[2], "'linspace' arg 3");
    }

    const aV = exactDouble(argTypes[0]);
    const bV = exactDouble(argTypes[1]);
    const nRaw = argTypes.length === 3 ? exactDouble(argTypes[2]) : DEFAULT_N;

    if (nRaw !== undefined) {
      const n = Math.round(nRaw);
      if (n <= 0) {
        return [tensorDouble([1, 0])];
      }
      if (n === 1) {
        if (bV !== undefined && Number.isFinite(bV)) {
          return [scalarDouble(signFromNumber(bV), bV)];
        }
        return [scalarDouble()];
      }
      if (
        aV !== undefined &&
        bV !== undefined &&
        Number.isFinite(aV) &&
        Number.isFinite(bV) &&
        n <= EXACT_ARRAY_MAX_ELEMENTS
      ) {
        const data = computeLinspaceData(aV, bV, n);
        let allFinite = true;
        for (let i = 0; i < data.length; i++) {
          if (!Number.isFinite(data[i])) {
            allFinite = false;
            break;
          }
        }
        if (allFinite) {
          return [tensorDouble([1, n], data)];
        }
      }
      return [tensorDouble([1, n])];
    }

    const dims: DimInfo[] = [DIM_ONE, { kind: "unknown" }];
    return [tensorDoubleFromDims(dims)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_linspace");
    const aC = argsC[0];
    const bC = argsC[1];
    const nC = argTypes.length === 3 ? argsC[2] : `${DEFAULT_N}.0`;

    if (argTypes.length === 3) {
      const nV = exactDouble(argTypes[2] as NumericType);
      if (nV !== undefined && Math.round(nV) === 1) {
        return `((${aC}), (${bC}))`;
      }
    }
    return `mtoc2_tensor_linspace((double)(${aC}), (double)(${bC}), (long)lround(${nC}))`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_linspace");
    const aJs = argsJs[0];
    const bJs = argsJs[1];
    const nJs = argTypes.length === 3 ? argsJs[2] : String(DEFAULT_N);
    if (argTypes.length === 3) {
      const nV = exactDouble(argTypes[2] as NumericType);
      if (nV !== undefined && Math.round(nV) === 1) {
        return `((${aJs}), (${bJs}))`;
      }
    }
    return `mtoc2_tensor_linspace(${aJs}, ${bJs}, Math.round(${nJs}))`;
  },
  call({ args, argTypes }) {
    const av = typeof args[0] === "number" ? args[0] : Number(args[0]);
    const bv = typeof args[1] === "number" ? args[1] : Number(args[1]);
    let n = DEFAULT_N;
    if (argTypes.length === 3) {
      const nv = typeof args[2] === "number" ? args[2] : Number(args[2]);
      n = Math.round(nv);
      if (n === 1) return [bv];
    }
    return [jsLinspace(av, bv, n) as unknown as RuntimeTensor];
  },
};
