/**
 * `isscalar(x)` — does `x` hold a single element?
 *
 * Numbl semantics
 * (`numbl-core/interpreter/builtins/introspection.ts`, plus the
 * `jitEmit` shortcut for `string`):
 *   - Real or logical scalar number → `true`.
 *   - Complex scalar → `true`.
 *   - Tensor → `true` iff `numel == 1`.
 *   - String handle (`"..."`) → `true` (single handle = scalar).
 *   - Char array (`'...'`) → `false`, even for `'a'` — char arrays
 *     are 1×N and treated as non-scalar.
 *   - Struct / class instance / function handle → `false`.
 *
 * The result folds at type-check time whenever the shape is known
 * (which is the common case). When a tensor's shape is opaque the
 * emit paths fall back to a `numel == 1` runtime check.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isChar,
  isClass,
  isHandle,
  isNumeric,
  isString,
  isStruct,
  scalarLogical,
  shapeNumel,
  typeToString,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  isTensor,
  isChar as isRuntimeChar,
  isComplexValue,
  isHandleValue,
} from "../../../runtime/value.js";

/** Static fold when the answer is determined by the type. Returns
 *  `undefined` for the one shape that needs a runtime check (Numeric
 *  with no known shape). */
function staticAnswer(t: Type): boolean | undefined {
  if (isNumeric(t)) {
    if (t.shape === undefined) return undefined;
    return shapeNumel(t.shape) === 1;
  }
  if (isString(t)) return true;
  if (isChar(t)) return false;
  if (isStruct(t) || isClass(t) || isHandle(t)) return false;
  // Unknown type: defer to runtime. The lowerer treats Unknown
  // conservatively; this matches.
  return undefined;
}

function requireKnown(t: Type): void {
  // Every concrete type kind has a defined answer (per `staticAnswer`),
  // so the only way we hit `undefined` is Unknown or a Numeric with no
  // shape. Both are fine to accept — the emit paths handle the
  // shape-unknown case via a runtime numel check.
  if (
    t.kind !== "Numeric" &&
    t.kind !== "String" &&
    t.kind !== "Char" &&
    t.kind !== "Struct" &&
    t.kind !== "Class" &&
    t.kind !== "Handle" &&
    t.kind !== "Unknown"
  ) {
    throw new TypeError(
      `'isscalar' got an unsupported argument type (${typeToString(t)})`
    );
  }
}

export const isscalar: Builtin = {
  name: "isscalar",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(
        `'isscalar' expects 1 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'isscalar' does not support multi-output (nargout=${nargout})`
      );
    }
    const t = argTypes[0];
    requireKnown(t);
    const v = staticAnswer(t);
    return [v === undefined ? scalarLogical() : scalarLogical(v)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const t = argTypes[0];
    const v = staticAnswer(t);
    if (v !== undefined) return v ? `1.0` : `0.0`;
    // Numeric with unknown shape — emit a runtime numel check.
    useRuntime("mtoc2_numel");
    return `(mtoc2_numel(${argsC[0]}) == 1)`;
  },
  emitJs({ argsJs, argTypes }) {
    const t = argTypes[0];
    const v = staticAnswer(t);
    if (v !== undefined) return v ? `true` : `false`;
    return `(${argsJs[0]}.shape.reduce((a,b)=>a*b, 1) === 1)`;
  },
  call({ args }) {
    const v = args[0];
    if (typeof v === "number" || typeof v === "boolean") return [true];
    if (isComplexValue(v)) return [true];
    if (typeof v === "string") return [true];
    if (isRuntimeChar(v)) return [false];
    if (isTensor(v)) return [v.data.length === 1];
    if (isHandleValue(v)) return [false];
    // Plain JS object → struct or class instance. Numbl returns false
    // for both. (Class instances carry a non-enumerable `mtoc2Class`
    // tag; we don't need to distinguish them here.)
    return [false];
  },
};
