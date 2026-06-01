/**
 * `transpose` builtin — backs the `.'` operator. Conjugate `'` on a
 * complex input lowers to `transpose(conj(z))`.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isNumeric,
  isScalar,
  isMultiElement,
  scalarComplex,
  scalarDouble,
  signFromNumber,
  tensorComplex,
  tensorDouble,
  typeToString,
} from "../../../lowering/types.js";
import type { NumericType } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactComplex, exactComplexArray, exactDouble } from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import { isTensor } from "../../../runtime/value.js";
import {
  mtoc2_tensor_transpose as jsTranspose,
  mtoc2_tensor_transpose_complex as jsTransposeComplex,
} from "../../runtime/snippets.gen.js";

function transposeExact(src: Float64Array, m: number, n: number): Float64Array {
  const out = new Float64Array(m * n);
  for (let sc = 0; sc < n; sc++) {
    for (let sr = 0; sr < m; sr++) {
      out[sc + sr * n] = src[sr + sc * m];
    }
  }
  return out;
}

export const transpose: Builtin = {
  name: "transpose",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(
        `'transpose' expects 1 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'transpose' does not support multi-output (nargout=${nargout})`
      );
    }
    const a = argTypes[0];
    if (!isNumeric(a)) {
      throw new TypeError(
        `transpose argument must be numeric (got ${typeToString(a)})`
      );
    }
    if (a.elem !== "double" && a.elem !== "logical") {
      throw new TypeError(
        `transpose argument must be a double or logical (got ${a.elem})`
      );
    }

    if (isScalar(a)) {
      if (a.isComplex) {
        const cx = exactComplex(a);
        if (cx !== undefined) return [scalarComplex(cx)];
        return [scalarComplex()];
      }
      const v = exactDouble(a);
      const r =
        v !== undefined
          ? scalarDouble(signFromNumber(v), v)
          : scalarDouble(a.sign);
      if (a.elem === "logical") r.elem = "logical";
      return [r];
    }

    if (a.dims.length !== 2) {
      throw new UnsupportedConstruct(
        `transpose requires a 2-D operand (got ${a.dims.length}-D); ` +
          `use 'permute' for higher-rank reorderings ` +
          `(numbl flattens trailing dims into cols; mtoc2 follows MATLAB and rejects)`
      );
    }

    if (a.shape === undefined) {
      throw new UnsupportedConstruct(
        `transpose of a tensor with unknown shape is not yet supported`
      );
    }

    const m = a.shape[0];
    const n = a.shape[1];
    const newShape = [n, m];

    if (a.isComplex) {
      const cx = exactComplexArray(a);
      if (cx !== undefined) {
        return [
          tensorComplex(newShape, {
            re: transposeExact(cx.re, m, n),
            im: transposeExact(cx.im, m, n),
          }),
        ];
      }
      return [tensorComplex(newShape)];
    }

    // Preserve logical element-class: class(L') === "logical" in MATLAB
    // / the interpreter (the data stays the 0/1 carrier; only the elem
    // tag is carried through).
    const r =
      a.exact instanceof Float64Array
        ? tensorDouble(newShape, transposeExact(a.exact, m, n))
        : tensorDouble(newShape);
    if (a.elem === "logical") r.elem = "logical";
    return [r];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_transpose");
    useRuntime("mtoc2_tensor_transpose_complex");
    const a = argTypes[0] as NumericType;
    if (!isMultiElement(a)) {
      return argsC[0];
    }
    if (a.isComplex) {
      return `mtoc2_tensor_transpose_complex(${argsC[0]})`;
    }
    return `mtoc2_tensor_transpose(${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (!isMultiElement(a)) return argsJs[0];
    if (a.isComplex) {
      useRuntime("mtoc2_tensor_transpose_complex");
      return `mtoc2_tensor_transpose_complex(${argsJs[0]})`;
    }
    useRuntime("mtoc2_tensor_transpose");
    return `mtoc2_tensor_transpose(${argsJs[0]})`;
  },
  call({ args, argTypes }) {
    const a = argTypes[0] as NumericType;
    if (!isMultiElement(a)) return [args[0]];
    if (!isTensor(args[0])) {
      throw new TypeError(
        `'transpose' runtime arg has type-system shape tensor but runtime value isn't a tensor`
      );
    }
    const fn = a.isComplex ? jsTransposeComplex : jsTranspose;
    return [fn(args[0]) as unknown as RuntimeTensor];
  },
};
