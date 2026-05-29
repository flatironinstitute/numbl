/**
 * `true` / `false` — MATLAB's logical constants.
 *
 * Two surface forms, mirroring numbl:
 *   - 0-arg bare name (`if true`, `x = false`, `assert(true)`) →
 *     logical scalar. The bare-ident read resolves here through
 *     `lowerIdent`'s 0-arg builtin probe; the exact value lets the
 *     `if`-cond fold decide branches statically.
 *   - shape-constructor (`true(n)`, `false(m, n)`, `false(1, k)`) →
 *     a logical tensor filled with 1 / 0. Same Float64 data as
 *     `ones` / `zeros`; only the static element type differs
 *     (`logical` vs `double`), so we delegate codegen to the shared
 *     shape constructor and re-tag the result type as logical.
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { scalarLogical, isNumeric } from "../../../lowering/types.js";
import type { NumericType, Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { defineShapeConstructor } from "../shape/_construct.js";

/** Re-tag a double-typed shape-constructor result as `logical`. The
 *  runtime representation is identical (a Float64 buffer of 1s / 0s);
 *  only the static `elem` tag and `sign` change. */
function toLogical(t: Type, value: 0 | 1): Type {
  if (!isNumeric(t)) return t;
  const out: NumericType = { ...t, elem: "logical" };
  out.sign = value === 1 ? "positive" : "zero";
  return out;
}

function boolBuiltin(name: "true" | "false"): Builtin {
  const value: 0 | 1 = name === "true" ? 1 : 0;
  // ones/zeros helpers produce exactly the 1.0 / 0.0 fill we want.
  const shape =
    value === 1
      ? defineShapeConstructor(
          name,
          1,
          "mtoc2_tensor_ones_nd",
          "mtoc2_tensor_ones_square",
          { minArgs: 1 }
        )
      : defineShapeConstructor(
          name,
          0,
          "mtoc2_tensor_zeros_nd",
          "mtoc2_tensor_zeros_square",
          { minArgs: 1 }
        );
  return {
    name,
    transfer(argTypes, nargout) {
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${name}' does not support multi-output (nargout=${nargout})`
        );
      }
      if (argTypes.length === 0) {
        return [scalarLogical(value === 1)];
      }
      return shape.transfer(argTypes, nargout).map(t => toLogical(t, value));
    },
    emitC(args) {
      if (args.argTypes.length === 0) return value === 1 ? "1.0" : "0.0";
      if (!shape.emitC) {
        throw new TypeError(
          `internal: '${name}' shape constructor has no emitC`
        );
      }
      return shape.emitC(args);
    },
    emitJs(args) {
      if (args.argTypes.length === 0) return value === 1 ? "true" : "false";
      if (!shape.emitJs) {
        throw new TypeError(
          `internal: '${name}' shape constructor has no emitJs`
        );
      }
      return shape.emitJs(args);
    },
    call(args) {
      // A logical scalar is a JS boolean in the interpreter (see
      // `inferTypeFromValue`), so the 0-arg form returns true / false
      // rather than 1 / 0 — keeping `islogical` / `class` honest.
      if (args.argTypes.length === 0) return [value === 1];
      if (!shape.call) {
        throw new TypeError(
          `internal: '${name}' shape constructor has no call`
        );
      }
      return shape.call(args);
    },
    // The 0-arg scalar slot is a plain literal, so it can appear inside
    // a fused elementwise expression (e.g. `x & true`).
    elementwise: true,
  };
}

export const trueBuiltin = boolBuiltin("true");
export const falseBuiltin = boolBuiltin("false");
