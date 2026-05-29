import {
  type NumericType,
  scalarDouble,
  scalarComplex,
  tensorDouble,
  tensorComplex,
  tensorDoubleFromDims,
  tensorComplexFromDims,
  signFromNumber,
  flipSign,
  isMultiElement,
  isNumeric,
  isScalar,
} from "../../../lowering/types.js";
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Builtin } from "../../registry.js";
import {
  requireRealOrComplex,
  exactDouble,
  exactRealArray,
  exactComplex,
  exactComplexArray,
} from "../_shared.js";
import {
  isComplexValue,
  type RuntimeTensor,
  type RuntimeValue,
} from "../../../runtime/value.js";
import {
  mtoc2_cneg,
  mtoc2_tensor_uminus,
  mtoc2_tensor_uminus_complex,
} from "../../runtime/snippets.gen.js";

export const uminus: Builtin = {
  name: "uminus",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'uminus' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'uminus' does not support multi-output (nargout=${nargout})`
      );
    }
    requireRealOrComplex(argTypes[0], `'uminus' arg`);
    const a = argTypes[0] as NumericType;
    if (a.isComplex) {
      if (!isMultiElement(a)) {
        const cx = exactComplex(a);
        if (cx !== undefined) {
          return [scalarComplex({ re: -cx.re, im: -cx.im })];
        }
        return [scalarComplex()];
      }
      // Complex tensor: fold when both lanes are exact, else runtime.
      const cx = exactComplexArray(a);
      if (cx !== undefined && a.shape !== undefined) {
        const re = new Float64Array(cx.re.length);
        const im = new Float64Array(cx.im.length);
        for (let i = 0; i < cx.re.length; i++) {
          re[i] = -cx.re[i];
          im[i] = -cx.im[i];
        }
        return [tensorComplex(a.shape, { re, im })];
      }
      return [tensorComplexFromDims(a.dims.slice())];
    }
    if (isScalar(a)) {
      const ax = exactDouble(a);
      if (ax !== undefined) {
        const v = -ax;
        return [scalarDouble(signFromNumber(v), v)];
      }
      return [scalarDouble(flipSign(a.sign))];
    }
    // Tensor uminus: fold when exact, else runtime. The C helper
    // iterates `n = prod(dims)` regardless of whether shape is known
    // statically, so unknown-dim tensors are fine — we just can't
    // fold their values.
    const arr = exactRealArray(a);
    if (arr !== undefined && a.shape !== undefined) {
      const out = new Float64Array(arr.length);
      for (let i = 0; i < arr.length; i++) out[i] = -arr[i];
      return [tensorDouble(a.shape, out)];
    }
    const out = tensorDoubleFromDims(a.dims.slice());
    out.sign = flipSign(a.sign);
    return [out];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const ty = argTypes[0] as NumericType;
    if (isMultiElement(ty)) {
      if (ty.isComplex) {
        useRuntime("mtoc2_tensor_elemwise_complex");
        useRuntime("mtoc2_cscalar");
        return `mtoc2_tensor_uminus_complex(${argsC[0]})`;
      }
      useRuntime("mtoc2_tensor_elemwise_real");
      return `mtoc2_tensor_uminus(${argsC[0]})`;
    }
    if (isNumeric(ty) && ty.isComplex) {
      useRuntime("mtoc2_cscalar");
      return `mtoc2_cneg(${argsC[0]})`;
    }
    return `(-${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const ty = argTypes[0] as NumericType;
    if (isMultiElement(ty)) {
      if (ty.isComplex) {
        useRuntime("mtoc2_tensor_elemwise_complex");
        useRuntime("mtoc2_cscalar");
        return `mtoc2_tensor_uminus_complex(${argsJs[0]})`;
      }
      useRuntime("mtoc2_tensor_elemwise_real");
      return `mtoc2_tensor_uminus(${argsJs[0]})`;
    }
    if (isNumeric(ty) && ty.isComplex) {
      useRuntime("mtoc2_cscalar");
      return `mtoc2_cneg(${argsJs[0]})`;
    }
    return `(-${argsJs[0]})`;
  },
  call({ args, argTypes }) {
    const ty = argTypes[0] as NumericType;
    if (isMultiElement(ty)) {
      if (ty.isComplex) {
        return [
          mtoc2_tensor_uminus_complex(
            args[0] as RuntimeTensor
          ) as unknown as RuntimeTensor,
        ];
      }
      return [
        mtoc2_tensor_uminus(
          args[0] as RuntimeTensor
        ) as unknown as RuntimeTensor,
      ];
    }
    if (isNumeric(ty) && ty.isComplex) {
      const v = args[0] as RuntimeValue;
      const cx = isComplexValue(v) ? v : { re: Number(v), im: 0 };
      return [mtoc2_cneg(cx)];
    }
    const v = typeof args[0] === "number" ? args[0] : Number(args[0]);
    return [-v];
  },
  elementwise: true,
};
