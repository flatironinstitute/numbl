/**
 * `xor(a, b)` — logical exclusive-or of two real scalars. The result is
 * a logical scalar that is `true` iff exactly one operand is nonzero.
 *
 * Scalar-only: tensor or complex operands throw `UnsupportedConstruct`
 * so the call falls back to the interpreter (which broadcasts and
 * handles complex). Booleanization uses `(x != 0)` on the C side and
 * `Number(x) !== 0` on the JS side so both agree element-for-element —
 * including `NaN` (nonzero → true on both).
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isNumeric,
  isScalar,
  scalarLogical,
  type NumericType,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";

function requireScalarReal(t: Type, what: string): NumericType {
  if (!isNumeric(t)) {
    throw new TypeError(`${what} must be numeric (got ${t.kind})`);
  }
  if (t.isComplex) {
    throw new UnsupportedConstruct(
      `'xor' on complex operands is not JIT-compiled`
    );
  }
  if (!isScalar(t)) {
    throw new UnsupportedConstruct(
      `'xor' on tensor operands is not JIT-compiled`
    );
  }
  if (t.elem !== "double" && t.elem !== "logical") {
    throw new TypeError(`${what} must be double or logical (got ${t.elem})`);
  }
  return t;
}

export const xorBuiltin: Builtin = {
  name: "xor",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 2) {
      throw new TypeError(`'xor' expects 2 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'xor' does not support multi-output (nargout=${nargout})`
      );
    }
    const a = requireScalarReal(argTypes[0], "'xor' arg 1");
    const b = requireScalarReal(argTypes[1], "'xor' arg 2");
    const ax = exactDouble(a);
    const bx = exactDouble(b);
    if (ax !== undefined && bx !== undefined) {
      return [scalarLogical((ax !== 0) !== (bx !== 0))];
    }
    return [scalarLogical()];
  },
  emitC({ argsC }) {
    return `(((${argsC[0]}) != 0) != ((${argsC[1]}) != 0))`;
  },
  emitJs({ argsJs }) {
    return `((Number(${argsJs[0]}) !== 0) !== (Number(${argsJs[1]}) !== 0))`;
  },
  call({ args }) {
    const av = Number(args[0]);
    const bv = Number(args[1]);
    return [(av !== 0) !== (bv !== 0)];
  },
};
