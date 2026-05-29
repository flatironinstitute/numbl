/**
 * `iscell(x)` — folds true if the argument's static type is a Cell,
 * false otherwise. Always static because the kind is part of the type.
 *
 * Matches numbl `introspection.ts:173-180` (returns `1` / `0`).
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isCell, scalarLogical } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";

export const iscell: Builtin = {
  name: "iscell",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'iscell' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'iscell' does not support multi-output (nargout=${nargout})`
      );
    }
    return [scalarLogical(isCell(argTypes[0]))];
  },
  emitC({ argTypes }) {
    return isCell(argTypes[0]) ? `1.0` : `0.0`;
  },
  emitJs({ argTypes }) {
    return isCell(argTypes[0]) ? `true` : `false`;
  },
  call({ args }) {
    const v = args[0];
    if (
      typeof v === "object" &&
      v !== null &&
      (v as { mtoc2Tag?: string }).mtoc2Tag === "cell"
    ) {
      return [true];
    }
    return [false];
  },
};
