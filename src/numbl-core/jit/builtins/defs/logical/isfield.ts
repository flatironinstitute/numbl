/**
 * `isfield(s, name)` — does the struct/class `s` have a field/property
 * named `name`?
 *
 * Numbl semantics
 * (`numbl-core/interpreter/builtins/type-constructors.ts`):
 *   - Struct or class instance → `true` iff `name` is in the field list.
 *   - Anything else (numeric tensor, `[]`, char, string, handle, …) →
 *     `false`. There is no "empty matrix as struct" coercion; the
 *     non-struct branch just returns `false` so `isfield([], 'foo')`
 *     naturally yields `false`.
 *
 * mtoc2 restricts the second argument to a Char or String literal
 * (`.exact` set) — sufficient for the common `isfield(s, 'name')` /
 * `isfield(s, "name")` shape. Runtime-only names would need a string
 * comparison against the (compile-time-known) struct field set, which
 * isn't worth wiring until something needs it.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  fieldType,
  isStruct,
  isClass,
  isChar,
  isString,
  scalarLogical,
  typeToString,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";

function staticHasField(s: Type, name: string): boolean {
  if (!isStruct(s) && !isClass(s)) return false;
  return fieldType(s, name) !== undefined;
}

function requireExactName(name: Type): string {
  if (!isChar(name) && !isString(name)) {
    throw new TypeError(
      `'isfield' second arg must be a char or string (got ${typeToString(name)})`
    );
  }
  if (name.exact === undefined) {
    throw new UnsupportedConstruct(
      `'isfield' requires a literal field name; got a non-literal ${name.kind.toLowerCase()}`
    );
  }
  return name.exact;
}

export const isfield: Builtin = {
  name: "isfield",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 2) {
      throw new TypeError(`'isfield' expects 2 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'isfield' does not support multi-output (nargout=${nargout})`
      );
    }
    const name = requireExactName(argTypes[1]);
    return [scalarLogical(staticHasField(argTypes[0], name))];
  },
  emitC({ argTypes }) {
    const name = requireExactName(argTypes[1]);
    return staticHasField(argTypes[0], name) ? `1.0` : `0.0`;
  },
  emitJs({ argTypes }) {
    const name = requireExactName(argTypes[1]);
    return staticHasField(argTypes[0], name) ? `true` : `false`;
  },
  call({ argTypes }) {
    const name = requireExactName(argTypes[1]);
    return [staticHasField(argTypes[0], name)];
  },
};
