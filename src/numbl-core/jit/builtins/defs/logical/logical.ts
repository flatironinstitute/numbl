/**
 * `logical(x)` — convert a real value to logical: nonzero → 1, zero →
 * 0, elementwise. A NaN raises (MATLAB:nologicalnan), as in numbl's
 * `logical`; a logical input cannot hold one, so it skips the check.
 * Complex input is rejected by `requireRealDouble`.
 */
import { UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isMultiElement,
  isNumeric,
  type NumericType,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble, exactRealArray } from "../_shared.js";
import { defineUnaryPred } from "./_unary_pred.js";

const base = defineUnaryPred({
  name: "logical",
  cScalar: arg => `((${arg}) != 0.0)`,
  jsScalar: arg => `((${arg}) !== 0)`,
  jsFn: x => x !== 0,
  tensorHelper: "jit_tensor_predicate",
});

/** A real double input, which may hold a NaN. */
function checksNaN(t: Type | undefined): t is NumericType {
  return t !== undefined && isNumeric(t) && !t.isComplex && t.elem === "double";
}

export const logical: Builtin = {
  ...base,
  transfer(argTypes, nargout) {
    const a = argTypes[0];
    if (checksNaN(a)) {
      const x = exactDouble(a);
      const arr = exactRealArray(a);
      if (Number.isNaN(x) || (arr !== undefined && arr.some(Number.isNaN)))
        throw new UnsupportedConstruct(
          "'logical' of a NaN raises; left to the interpreter"
        );
    }
    return base.transfer(argTypes, nargout);
  },
  emitC(ctx) {
    const a = ctx.argTypes[0];
    if (!checksNaN(a)) return base.emitC!(ctx);
    ctx.useRuntime("jit_tensor_predicate");
    return isMultiElement(a)
      ? `jit_tensor_logical(${ctx.argsC[0]})`
      : `jit_logical_elem(${ctx.argsC[0]})`;
  },
  emitJs(ctx) {
    const a = ctx.argTypes[0];
    if (!checksNaN(a)) return base.emitJs!(ctx);
    ctx.useRuntime("jit_tensor_predicate");
    return isMultiElement(a)
      ? `jit_tensor_logical(${ctx.argsJs[0]})`
      : `jit_logical_elem(${ctx.argsJs[0]})`;
  },
};
