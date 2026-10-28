/**
 * `isnumeric(x)` — folds true if the argument's static type is a
 * Numeric (real or complex, scalar or tensor, double or logical?
 * MATLAB excludes logical). Matches numbl's introspection.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isNumeric,
  scalarLogical,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { numericClassName } from "../_shared.js";

function isNumericClass(t: Type): boolean {
  return isNumeric(t) && numericClassName(t) === "double";
}

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
    // Logical is not numeric; double (real or complex) is.
    return [scalarLogical(isNumericClass(t))];
  },
  emitC({ argTypes }) {
    const t = argTypes[0];
    return isNumericClass(t) ? `1.0` : `0.0`;
  },
  emitJs({ argTypes }) {
    const t = argTypes[0];
    return isNumericClass(t) ? `true` : `false`;
  },
};
