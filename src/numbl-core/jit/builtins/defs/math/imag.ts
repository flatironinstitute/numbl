/**
 * `imag(z)` — imaginary part. Accepts scalar real or complex; returns
 * a real double. For real input the result is always 0; for complex
 * input the result is `cimag(z)`.
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
import { requireRealOrComplex, exactComplex } from "../_shared.js";

export const imag: Builtin = {
  name: "imag",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'imag' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'imag' does not support multi-output (nargout=${nargout})`
      );
    }
    requireRealOrComplex(argTypes[0], `'imag' arg`);
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      const out = tensorDoubleFromDims(a.dims.slice());
      if (!a.isComplex) out.sign = "zero";
      return [out];
    }
    if (a.isComplex) {
      const cx = exactComplex(a);
      if (cx !== undefined) {
        return [scalarDouble(signFromNumber(cx.im), cx.im)];
      }
      return [scalarDouble()];
    }
    return [scalarDouble("zero", 0)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        useRuntime("jit_tensor_unary_complex_math");
        useRuntime("jit_cscalar");
        return `jit_tensor_imag_complex(${argsC[0]})`;
      }
      useRuntime("jit_tensor_elemwise_real");
      return `jit_tensor_times_ts(${argsC[0]}, 0.0)`;
    }
    if (a.isComplex) {
      useRuntime("jit_cscalar");
      return `jit_cimag(${argsC[0]})`;
    }
    return `0.0`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        useRuntime("jit_tensor_unary_complex_math");
        useRuntime("jit_cscalar");
        return `jit_tensor_imag_complex(${argsJs[0]})`;
      }
      useRuntime("jit_tensor_elemwise_real");
      return `jit_tensor_times_ts(${argsJs[0]}, 0)`;
    }
    if (a.isComplex) {
      useRuntime("jit_cscalar");
      return `jit_cimag(${argsJs[0]})`;
    }
    return `0`;
  },
  elementwise: true,
};
