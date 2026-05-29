/**
 * `mpower` builtin — backs the `^` binary operator.
 *
 * v1 supports only the scalar-on-both-sides case (which delegates to
 * `power`, the `.^` builtin). Matrix power on a square base would
 * require eigendecomposition for non-integer exponents and repeated
 * `mtimes` for integer exponents — a real slope of its own. Numbl's
 * `mPow` (helpers/arithmetic.ts:1006) is the reference; we punt on it
 * entirely for now.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isMultiElement } from "../../../lowering/types.js";
import {
  type Builtin,
  requireEmitC,
  requireEmitJs,
  requireCall,
} from "../../registry.js";
import { power } from "./power.js";

export const mpower: Builtin = {
  name: "mpower",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 2) {
      throw new TypeError(`'mpower' expects 2 arg(s), got ${argTypes.length}`);
    }
    if (isMultiElement(argTypes[0]) || isMultiElement(argTypes[1])) {
      throw new UnsupportedConstruct(
        `'^' on matrices (matrix power) is not yet supported; use '.^' for elementwise power`
      );
    }
    return power.transfer(argTypes, nargout);
  },
  emitC(args) {
    return requireEmitC(power)(args);
  },
  emitJs(args) {
    return requireEmitJs(power)(args);
  },
  call(args) {
    return requireCall(power)(args);
  },
};
