/**
 * `conj(z)` — complex conjugate. Accepts real or complex, scalar or
 * tensor. For real input it's the identity; for complex input it
 * flips the sign of the imaginary part.
 */

import {
  type NumericType,
  scalarDouble,
  scalarComplex,
  tensorComplex,
  tensorComplexFromDims,
  tensorDouble,
  tensorDoubleFromDims,
  isMultiElement,
  shapeNumel,
  signFromNumber,
  EXACT_ARRAY_MAX_ELEMENTS,
} from "../../../lowering/types.js";
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Builtin } from "../../registry.js";
import {
  requireRealOrComplex,
  exactDouble,
  exactComplex,
  exactComplexArray,
  exactRealArray,
} from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import { mtoc2_tensor_conj_complex as jsTensorConjComplex } from "../../runtime/snippets.gen.js";

export const conj: Builtin = {
  name: "conj",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'conj' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'conj' does not support multi-output (nargout=${nargout})`
      );
    }
    requireRealOrComplex(argTypes[0], `'conj' arg`);
    const a = argTypes[0] as NumericType;
    if (!isMultiElement(a)) {
      if (a.isComplex) {
        const cx = exactComplex(a);
        if (cx !== undefined) return [scalarComplex({ re: cx.re, im: -cx.im })];
        return [scalarComplex()];
      }
      const v = exactDouble(a);
      if (v !== undefined) return [scalarDouble(signFromNumber(v), v)];
      return [scalarDouble(a.sign)];
    }
    if (a.isComplex) {
      const cx = exactComplexArray(a);
      if (cx !== undefined && a.shape !== undefined) {
        const total = shapeNumel(a.shape);
        if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
          const re = new Float64Array(cx.re.length);
          const im = new Float64Array(cx.im.length);
          for (let i = 0; i < cx.re.length; i++) {
            re[i] = cx.re[i];
            im[i] = -cx.im[i];
          }
          return [tensorComplex(a.shape, { re, im })];
        }
      }
      return [tensorComplexFromDims(a.dims.slice())];
    }
    const arr = exactRealArray(a);
    if (arr !== undefined && a.shape !== undefined) {
      return [tensorDouble(a.shape, arr)];
    }
    const out = tensorDoubleFromDims(a.dims.slice());
    out.sign = a.sign;
    return [out];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        useRuntime("mtoc2_tensor_unary_complex_math");
        useRuntime("mtoc2_cscalar");
        return `mtoc2_tensor_conj_complex(${argsC[0]})`;
      }
      useRuntime("mtoc2_tensor_copy");
      return `mtoc2_tensor_copy(${argsC[0]})`;
    }
    if (a.isComplex) {
      useRuntime("mtoc2_cscalar");
      return `mtoc2_cconj(${argsC[0]})`;
    }
    return `(${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        useRuntime("mtoc2_tensor_unary_complex_math");
        useRuntime("mtoc2_cscalar");
        return `mtoc2_tensor_conj_complex(${argsJs[0]})`;
      }
      useRuntime("mtoc2_tensor_copy");
      return `mtoc2_tensor_copy(${argsJs[0]})`;
    }
    if (a.isComplex) {
      useRuntime("mtoc2_cscalar");
      return `mtoc2_cconj(${argsJs[0]})`;
    }
    return `(${argsJs[0]})`;
  },
  call({ args, argTypes }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        return [
          jsTensorConjComplex(
            args[0] as RuntimeTensor
          ) as unknown as RuntimeTensor,
        ];
      }
      return [args[0]];
    }
    if (a.isComplex) {
      const z = args[0] as { re: number; im: number };
      return [{ re: z.re, im: -z.im }];
    }
    return [args[0]];
  },
  elementwise: true,
};
