/**
 * `flipud`, `fliplr`, `flip` — axis-flip builtins.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  EXACT_ARRAY_MAX_ELEMENTS,
  isMultiElement,
  isNumeric,
  isScalar,
  scalarComplex,
  scalarDouble,
  shapeNumel,
  signFromNumber,
  tensorComplex,
  tensorDouble,
  typeToString,
} from "../../../lowering/types.js";
import type { NumericType } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  exactComplex,
  exactComplexArray,
  exactDouble,
  exactRealArray,
} from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import {
  mtoc2_tensor_flip as jsFlip,
  mtoc2_tensor_flip_complex as jsFlipComplex,
} from "../../runtime/snippets.gen.js";

function flipExact(
  src: Float64Array,
  shape: number[],
  axisIdx: number
): Float64Array {
  const total = shapeNumel(shape);
  const out = new Float64Array(total);
  const axisSize = axisIdx >= 0 && axisIdx < shape.length ? shape[axisIdx] : 1;
  if (axisSize <= 1) {
    out.set(src);
    return out;
  }
  let strideDim = 1;
  for (let d = 0; d < axisIdx; d++) strideDim *= shape[d];
  const slabSize = strideDim * axisSize;
  const numOuter = total / slabSize;
  for (let outer = 0; outer < numOuter; outer++) {
    const base = outer * slabSize;
    for (let k = 0; k < axisSize; k++) {
      const srcOff = base + k * strideDim;
      const dstOff = base + (axisSize - 1 - k) * strideDim;
      for (let s = 0; s < strideDim; s++) {
        out[dstOff + s] = src[srcOff + s];
      }
    }
  }
  return out;
}

function resolveFlipAxis(
  name: string,
  argTypes: NumericType[],
  fixedAxis: number | null
): number {
  if (fixedAxis !== null) return fixedAxis;
  if (argTypes.length === 1) {
    const a = argTypes[0];
    if (a.shape !== undefined) {
      const idx = a.shape.findIndex(d => d > 1);
      return idx === -1 ? 0 : idx;
    }
    return 0;
  }
  const k = argTypes[1];
  if (!isScalar(k) || k.isComplex) {
    throw new TypeError(
      `'${name}' second arg must be a real scalar (got ${typeToString(k)})`
    );
  }
  const kv = exactDouble(k);
  if (kv === undefined) {
    throw new UnsupportedConstruct(
      `'${name}' with a dynamic dim argument is not yet supported (require statically-known axis)`
    );
  }
  if (!Number.isFinite(kv) || !Number.isInteger(kv) || kv < 1) {
    throw new TypeError(
      `'${name}' dim argument must be a positive integer (got ${kv})`
    );
  }
  return kv - 1;
}

function defineFlip(opts: {
  name: string;
  fixedAxis: number | null;
  minArgs: number;
  maxArgs: number;
}): Builtin {
  return {
    name: opts.name,
    transfer(argTypes, nargout) {
      if (argTypes.length < opts.minArgs || argTypes.length > opts.maxArgs) {
        throw new TypeError(
          `'${opts.name}' expects ${opts.minArgs}..${opts.maxArgs} arg(s), ` +
            `got ${argTypes.length}`
        );
      }
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${opts.name}' does not support multi-output (nargout=${nargout})`
        );
      }
      const a = argTypes[0];
      if (!isNumeric(a)) {
        throw new TypeError(
          `'${opts.name}' arg must be numeric (got ${typeToString(a)})`
        );
      }
      if (a.elem !== "double" && a.elem !== "logical") {
        throw new TypeError(
          `'${opts.name}' arg must be a real double or logical (got ${a.elem})`
        );
      }

      if (isScalar(a)) {
        if (a.isComplex) {
          const cx = exactComplex(a);
          if (cx !== undefined) return [scalarComplex(cx)];
          return [scalarComplex()];
        }
        const v = exactDouble(a);
        if (v !== undefined) return [scalarDouble(signFromNumber(v), v)];
        return [scalarDouble(a.sign)];
      }

      const axisIdx = resolveFlipAxis(
        opts.name,
        argTypes as NumericType[],
        opts.fixedAxis
      );

      if (a.isComplex) {
        const cx = exactComplexArray(a);
        if (a.shape !== undefined && cx !== undefined) {
          const total = shapeNumel(a.shape);
          if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
            const reOut = flipExact(cx.re, a.shape, axisIdx);
            const imOut = flipExact(cx.im, a.shape, axisIdx);
            return [tensorComplex(a.shape, { re: reOut, im: imOut })];
          }
        }
        if (a.shape !== undefined) return [tensorComplex(a.shape)];
        return [{ ...a, exact: undefined }];
      }

      const arr = exactRealArray(a);
      if (a.shape !== undefined && arr !== undefined) {
        const total = shapeNumel(a.shape);
        if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
          const out = flipExact(arr, a.shape, axisIdx);
          return [tensorDouble(a.shape, out)];
        }
      }
      if (a.shape !== undefined) {
        return [tensorDouble(a.shape)];
      }
      return [{ ...a, exact: undefined }];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      const a = argTypes[0] as NumericType;
      if (!isMultiElement(a)) {
        return argsC[0];
      }
      const axisIdx = resolveFlipAxis(
        opts.name,
        argTypes as NumericType[],
        opts.fixedAxis
      );
      if (a.isComplex) {
        useRuntime("mtoc2_tensor_flip");
        return `mtoc2_tensor_flip_complex(${argsC[0]}, ${axisIdx}L)`;
      }
      useRuntime("mtoc2_tensor_flip");
      return `mtoc2_tensor_flip(${argsC[0]}, ${axisIdx}L)`;
    },
    emitJs({ argsJs, argTypes, useRuntime }) {
      const a = argTypes[0] as NumericType;
      if (!isMultiElement(a)) return argsJs[0];
      const axisIdx = resolveFlipAxis(
        opts.name,
        argTypes as NumericType[],
        opts.fixedAxis
      );
      if (a.isComplex) {
        useRuntime("mtoc2_tensor_flip");
        return `mtoc2_tensor_flip_complex(${argsJs[0]}, ${axisIdx})`;
      }
      useRuntime("mtoc2_tensor_flip");
      return `mtoc2_tensor_flip(${argsJs[0]}, ${axisIdx})`;
    },
    call({ args, argTypes }) {
      const a = argTypes[0] as NumericType;
      if (!isMultiElement(a)) return [args[0]];
      const axisIdx = resolveFlipAxis(
        opts.name,
        argTypes as NumericType[],
        opts.fixedAxis
      );
      if (a.isComplex) {
        return [
          jsFlipComplex(
            args[0] as RuntimeTensor,
            axisIdx
          ) as unknown as RuntimeTensor,
        ];
      }
      return [
        jsFlip(args[0] as RuntimeTensor, axisIdx) as unknown as RuntimeTensor,
      ];
    },
  };
}

export const flipud = defineFlip({
  name: "flipud",
  fixedAxis: 0,
  minArgs: 1,
  maxArgs: 1,
});

export const fliplr = defineFlip({
  name: "fliplr",
  fixedAxis: 1,
  minArgs: 1,
  maxArgs: 1,
});

export const flip = defineFlip({
  name: "flip",
  fixedAxis: null,
  minArgs: 1,
  maxArgs: 2,
});
