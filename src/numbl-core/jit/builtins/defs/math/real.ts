/**
 * `real(z)` — real part. Accepts scalar real or complex; returns a
 * real double. For real input it's the identity; for complex input
 * the result is `creal(z)`.
 */

import {
  type NumericType,
  scalarDouble,
  tensorDoubleFromDims,
  isMultiElement,
  signFromNumber,
} from "../../../lowering/types.js";
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Builtin } from "../../registry.js";
import { requireRealOrComplex, exactDouble, exactComplex } from "../_shared.js";

export const real: Builtin = {
  name: "real",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'real' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'real' does not support multi-output (nargout=${nargout})`
      );
    }
    requireRealOrComplex(argTypes[0], `'real' arg`);
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      const out = tensorDoubleFromDims(a.dims.slice());
      if (!a.isComplex) out.sign = a.sign;
      return [out];
    }
    if (a.isComplex) {
      const cx = exactComplex(a);
      if (cx !== undefined) {
        return [scalarDouble(signFromNumber(cx.re), cx.re)];
      }
      return [scalarDouble()];
    }
    const v = exactDouble(a);
    if (v !== undefined) return [scalarDouble(signFromNumber(v), v)];
    return [scalarDouble(a.sign)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        useRuntime("jit_tensor_unary_complex_math");
        useRuntime("jit_cscalar");
        return `jit_tensor_real_complex(${argsC[0]})`;
      }
      useRuntime("jit_tensor_copy");
      return `jit_tensor_copy(${argsC[0]})`;
    }
    if (a.isComplex) {
      useRuntime("jit_cscalar");
      return `jit_creal(${argsC[0]})`;
    }
    return `(${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        useRuntime("jit_tensor_unary_complex_math");
        useRuntime("jit_cscalar");
        return `jit_tensor_real_complex(${argsJs[0]})`;
      }
      useRuntime("jit_tensor_copy");
      return `jit_tensor_copy(${argsJs[0]})`;
    }
    if (a.isComplex) {
      useRuntime("jit_cscalar");
      return `jit_creal(${argsJs[0]})`;
    }
    return `(${argsJs[0]})`;
  },
  elementwise: true,
};
