/**
 * `isa(value, 'ClassName')` — checks whether `value` is an instance
 * of the named class (or matches one of MATLAB's primitive class
 * names: `double`, `logical`, `char`, `string`, `struct`,
 * `function_handle`, `cell`, `numeric`, `float`).
 *
 * Always static: the JIT knows every value's type at lowering time and
 * folds to a scalar logical literal. Matches numbl's
 * `runtimeDispatch.ts:isa` minus the inheritance-chain walk (the JIT
 * has no inheritance in v1).
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isNumeric,
  isHandle,
  isClass,
  isStruct,
  isChar,
  isString,
  isCell,
  scalarLogical,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { numericClassName } from "../_shared.js";

function staticClassNameOf(t: Type): string | null {
  if (isClass(t)) return t.className;
  if (isStruct(t)) return "struct";
  if (isHandle(t)) return "function_handle";
  if (isChar(t)) return "char";
  if (isString(t)) return "string";
  if (isCell(t)) return "cell";
  if (isNumeric(t)) return numericClassName(t);
  return null;
}

function matches(name: string, valueClass: string): boolean {
  if (valueClass === name) return true;
  // The JIT's only numeric class is double, so 'numeric' and 'float' match
  // the same values and 'integer' none (see runtime/kinds.ts isaValue).
  return (name === "numeric" || name === "float") && valueClass === "double";
}

export const isa: Builtin = {
  name: "isa",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 2) {
      throw new TypeError(`'isa' expects 2 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'isa' does not support multi-output (nargout=${nargout})`
      );
    }
    const nameTy = argTypes[1];
    if (!isChar(nameTy) && !isString(nameTy)) {
      throw new TypeError(
        `'isa' second argument must be a class name (char or string)`
      );
    }
    if (nameTy.exact === undefined) {
      throw new UnsupportedConstruct(
        `'isa' requires a statically-known class name (got a runtime ` +
          `text value)`
      );
    }
    const valueClass = staticClassNameOf(argTypes[0]);
    if (valueClass === null) {
      return [scalarLogical()];
    }
    return [scalarLogical(matches(nameTy.exact, valueClass))];
  },
  emitC({ argTypes }) {
    const nameTy = argTypes[1];
    if ((isChar(nameTy) || isString(nameTy)) && nameTy.exact !== undefined) {
      const valueClass = staticClassNameOf(argTypes[0]);
      if (valueClass !== null) {
        return matches(nameTy.exact, valueClass) ? `1.0` : `0.0`;
      }
    }
    return `0.0`;
  },
  emitJs({ argTypes }) {
    const nameTy = argTypes[1];
    if ((isChar(nameTy) || isString(nameTy)) && nameTy.exact !== undefined) {
      const valueClass = staticClassNameOf(argTypes[0]);
      if (valueClass !== null) {
        return matches(nameTy.exact, valueClass) ? `true` : `false`;
      }
    }
    return `false`;
  },
};
