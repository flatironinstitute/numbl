/**
 * `not` builtin — backs the `~` unary operator.
 *
 * Elementwise logical NOT. Numbl's `not(v)` in
 * `runtime/runtimeOperators.ts` line 100:
 *   - scalar 0          → logical true
 *   - scalar nonzero    → logical false
 *   - tensor            → logical tensor of same shape; element-wise
 *                         `(x == 0) ? 1 : 0`
 *
 * Type result is always logical (mtoc2's `NumericType.elem = "logical"`).
 * Sign is `nonneg` since outputs are in {0, 1}.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  type NumericType,
  type DimInfo,
  EXACT_ARRAY_MAX_ELEMENTS,
  isMultiElement,
  isNumeric,
  isScalar,
  scalarLogical,
  shapeNumel,
  typeToString,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  exactDouble,
  exactRealArray,
  exactComplex,
  exactComplexArray,
} from "../_shared.js";
import {
  mtoc2_cnonzero,
  mtoc2_tensor_not as jsTensorNot,
  mtoc2_tensor_not_complex as jsTensorNotComplex,
} from "../../runtime/snippets.gen.js";
import {
  isComplexValue,
  type RuntimeTensor,
  type RuntimeValue,
} from "../../../runtime/value.js";

function logicalTensor(
  dims: DimInfo[],
  shape: number[] | undefined,
  exact?: Float64Array
): NumericType {
  const t: NumericType = {
    kind: "Numeric",
    elem: "logical",
    isComplex: false,
    dims,
    sign: "nonneg",
  };
  if (shape !== undefined) t.shape = shape.slice();
  if (exact !== undefined) t.exact = exact;
  return t;
}

export const notBuiltin: Builtin = {
  name: "not",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'not' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'not' does not support multi-output (nargout=${nargout})`
      );
    }
    const a = argTypes[0];
    if (!isNumeric(a)) {
      throw new TypeError(
        `'~' / 'not' arg must be numeric (got ${typeToString(a)})`
      );
    }
    if (a.elem !== "double" && a.elem !== "logical") {
      throw new TypeError(
        `'~' / 'not' arg must be a real double or logical (got ${a.elem})`
      );
    }
    // Scalar input → scalar logical, sign nonneg.
    if (isScalar(a)) {
      if (a.isComplex) {
        const cx = exactComplex(a);
        if (cx !== undefined) {
          return [scalarLogical(cx.re === 0 && cx.im === 0)];
        }
        return [scalarLogical()];
      }
      const v = exactDouble(a);
      if (v !== undefined) {
        return [scalarLogical(v === 0)];
      }
      return [scalarLogical()];
    }

    // Tensor input → logical tensor of same shape. Exact-fold within
    // the element-count cap.
    if (a.isComplex) {
      const cx = exactComplexArray(a);
      if (a.shape !== undefined && cx !== undefined) {
        const total = shapeNumel(a.shape);
        if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
          const out = new Float64Array(total);
          for (let i = 0; i < total; i++) {
            out[i] = cx.re[i] === 0 && cx.im[i] === 0 ? 1.0 : 0.0;
          }
          return [logicalTensor(a.dims, a.shape, out)];
        }
      }
      return [logicalTensor(a.dims, a.shape)];
    }
    const arr = exactRealArray(a);
    if (a.shape !== undefined && arr !== undefined) {
      const total = shapeNumel(a.shape);
      if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
        const out = new Float64Array(arr.length);
        for (let i = 0; i < arr.length; i++) {
          out[i] = arr[i] === 0 ? 1.0 : 0.0;
        }
        return [logicalTensor(a.dims, a.shape, out)];
      }
    }
    return [logicalTensor(a.dims, a.shape)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        useRuntime("mtoc2_tensor_logical_real");
        return `mtoc2_tensor_not_complex(${argsC[0]})`;
      }
      useRuntime("mtoc2_tensor_logical_real");
      return `mtoc2_tensor_not(${argsC[0]})`;
    }
    if (a.isComplex) {
      useRuntime("mtoc2_cscalar");
      // Complex scalar is "false" iff both parts are exactly 0.
      return `(!mtoc2_cnonzero(${argsC[0]}))`;
    }
    // Scalar logical: `!x` is a C int (0 or 1).
    return `(!(${argsC[0]}))`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        useRuntime("mtoc2_tensor_logical_real");
        return `mtoc2_tensor_not_complex(${argsJs[0]})`;
      }
      useRuntime("mtoc2_tensor_logical_real");
      return `mtoc2_tensor_not(${argsJs[0]})`;
    }
    if (a.isComplex) {
      useRuntime("mtoc2_cscalar");
      return `(!mtoc2_cnonzero(${argsJs[0]}))`;
    }
    // A logical scalar is a JS boolean (`!b` is correct). A real double
    // must compare `=== 0` rather than use JS `!`, because `!NaN` is
    // `true` in JS but numbl's `~NaN` is 0 (NaN is nonzero → not-false).
    if (a.elem === "logical") return `(!(${argsJs[0]}))`;
    return `((${argsJs[0]}) === 0)`;
  },
  call({ args, argTypes }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      if (a.isComplex) {
        return [
          jsTensorNotComplex(
            args[0] as RuntimeTensor
          ) as unknown as RuntimeTensor,
        ];
      }
      return [
        jsTensorNot(args[0] as RuntimeTensor) as unknown as RuntimeTensor,
      ];
    }
    if (a.isComplex) {
      const v = args[0] as RuntimeValue;
      const cx = isComplexValue(v) ? v : { re: Number(v), im: 0 };
      return [!mtoc2_cnonzero(cx)];
    }
    const v = typeof args[0] === "number" ? args[0] : Number(args[0]);
    // `v === 0`, not `!v`: `!NaN` is `true` in JS but `~NaN` is 0.
    return [v === 0];
  },
};
