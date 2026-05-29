/**
 * `isnumeric(x)` — folds true if the argument's static type is a
 * Numeric (real or complex, scalar or tensor, double or logical?
 * MATLAB excludes logical). Matches numbl's introspection.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isNumeric, scalarLogical } from "../../../lowering/types.js";
import {
  isTensor,
  isComplexValue,
  type RuntimeValue,
} from "../../../runtime/value.js";
import type { Builtin } from "../../registry.js";

export const isnumeric: Builtin = {
  name: "isnumeric",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(
        `'isnumeric' expects 1 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'isnumeric' does not support multi-output (nargout=${nargout})`
      );
    }
    const t = argTypes[0];
    // MATLAB / numbl: logical is NOT numeric; double / complex IS.
    if (isNumeric(t) && t.elem !== "logical") {
      return [scalarLogical(true)];
    }
    return [scalarLogical(false)];
  },
  emitC({ argTypes }) {
    const t = argTypes[0];
    return isNumeric(t) && t.elem !== "logical" ? `1.0` : `0.0`;
  },
  emitJs({ argTypes }) {
    const t = argTypes[0];
    return isNumeric(t) && t.elem !== "logical" ? `true` : `false`;
  },
  call({ args }) {
    const v = args[0] as RuntimeValue;
    if (typeof v === "number") return [true];
    if (typeof v === "boolean") return [false];
    if (isComplexValue(v)) return [true];
    if (isTensor(v)) {
      return [!(v as { isLogical?: boolean }).isLogical];
    }
    return [false];
  },
};
