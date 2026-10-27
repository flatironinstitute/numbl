/**
 * `isstruct(x)` — folds true if the argument's static type is a
 * struct, false otherwise. Class instances are NOT structs (matches
 * MATLAB / numbl `introspection.ts:182`).
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isStruct, scalarLogical } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";

export const isstruct: Builtin = {
  name: "isstruct",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(
        `'isstruct' expects 1 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'isstruct' does not support multi-output (nargout=${nargout})`
      );
    }
    return [scalarLogical(isStruct(argTypes[0]))];
  },
  emitC({ argTypes }) {
    return isStruct(argTypes[0]) ? `1.0` : `0.0`;
  },
  emitJs({ argTypes }) {
    return isStruct(argTypes[0]) ? `true` : `false`;
  },
};
