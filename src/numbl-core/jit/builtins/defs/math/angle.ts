/**
 * `angle(z)` — phase angle in radians. Returns `atan2(im, re)`. For
 * real input the result is 0 when the value is non-negative, π when
 * negative; complex input yields `carg(z)`.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  type NumericType,
  scalarDouble,
  isMultiElement,
  signFromNumber,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { requireRealOrComplex, exactDouble, exactComplex } from "../_shared.js";

/** Angle of a real value: pi when negative, 0 otherwise (also for -0),
 *  NaN for NaN. */
function realAngle(x: number): number {
  return x < 0 ? Math.PI : x === x ? 0 : x;
}

export const angle: Builtin = {
  name: "angle",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'angle' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'angle' does not support multi-output (nargout=${nargout})`
      );
    }
    requireRealOrComplex(argTypes[0], `'angle' arg`);
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      throw new UnsupportedConstruct(
        `'angle' on a tensor is not yet supported`
      );
    }
    if (a.isComplex) {
      const cx = exactComplex(a);
      if (cx !== undefined) {
        const v = Math.atan2(cx.im, cx.re);
        if (Number.isFinite(v)) return [scalarDouble(signFromNumber(v), v)];
      }
      return [scalarDouble()];
    }
    const v = exactDouble(a);
    if (v !== undefined) {
      const r = realAngle(v);
      if (Number.isFinite(r)) return [scalarDouble(signFromNumber(r), r)];
    }
    return [scalarDouble()];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (a.isComplex) {
      useRuntime("jit_cscalar");
      return `jit_cangle(${argsC[0]})`;
    }
    // Not atan2(0, x), which is pi for x = -0.
    const x = argsC[0];
    return `((${x}) < 0.0 ? 3.141592653589793 : (${x}) == (${x}) ? 0.0 : (${x}))`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (a.isComplex) {
      useRuntime("jit_cscalar");
      return `jit_cangle(${argsJs[0]})`;
    }
    return `((_x) => (_x < 0 ? Math.PI : _x === _x ? 0 : _x))(${argsJs[0]})`;
  },
};
