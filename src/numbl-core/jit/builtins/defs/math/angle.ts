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
      const r = Math.atan2(0, v);
      if (Number.isFinite(r)) return [scalarDouble(signFromNumber(r), r)];
    }
    return [scalarDouble()];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (a.isComplex) {
      useRuntime("mtoc2_cscalar");
      return `mtoc2_cangle(${argsC[0]})`;
    }
    return `atan2(0.0, ${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (a.isComplex) {
      useRuntime("mtoc2_cscalar");
      return `mtoc2_cangle(${argsJs[0]})`;
    }
    return `Math.atan2(0, ${argsJs[0]})`;
  },
  call({ args, argTypes }) {
    const a = argTypes[0] as NumericType;
    if (a.isComplex) {
      const z = args[0] as { re: number; im: number };
      return [Math.atan2(z.im, z.re)];
    }
    const v = typeof args[0] === "number" ? args[0] : Number(args[0]);
    return [Math.atan2(0, v)];
  },
};
