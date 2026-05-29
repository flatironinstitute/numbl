/**
 * `isa(value, 'ClassName')` — checks whether `value` is an instance
 * of the named class (or matches one of MATLAB's primitive class
 * names: `double`, `logical`, `char`, `string`, `struct`,
 * `function_handle`, `cell`, `numeric`).
 *
 * Always static: mtoc2 knows every value's type at lowering time and
 * folds to a scalar logical literal. Matches numbl's
 * `runtimeDispatch.ts:isa` minus the inheritance-chain walk (mtoc2
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
import {
  isTensor,
  isChar as isCharRV,
  type RuntimeValue,
} from "../../../runtime/value.js";
import type { Builtin } from "../../registry.js";

function staticClassNameOf(t: Type): string | null {
  if (isClass(t)) return t.className;
  if (isStruct(t)) return "struct";
  if (isHandle(t)) return "function_handle";
  if (isChar(t)) return "char";
  if (isString(t)) return "string";
  if (isCell(t)) return "cell";
  if (isNumeric(t)) return t.elem === "logical" ? "logical" : "double";
  return null;
}

function matches(name: string, valueClass: string, t: Type): boolean {
  if (valueClass === name) return true;
  if (name === "numeric") return isNumeric(t) && t.elem !== "logical";
  return false;
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
    return [scalarLogical(matches(nameTy.exact, valueClass, argTypes[0]))];
  },
  emitC({ argTypes }) {
    const nameTy = argTypes[1];
    if ((isChar(nameTy) || isString(nameTy)) && nameTy.exact !== undefined) {
      const valueClass = staticClassNameOf(argTypes[0]);
      if (valueClass !== null) {
        return matches(nameTy.exact, valueClass, argTypes[0]) ? `1.0` : `0.0`;
      }
    }
    return `0.0`;
  },
  emitJs({ argTypes }) {
    const nameTy = argTypes[1];
    if ((isChar(nameTy) || isString(nameTy)) && nameTy.exact !== undefined) {
      const valueClass = staticClassNameOf(argTypes[0]);
      if (valueClass !== null) {
        return matches(nameTy.exact, valueClass, argTypes[0])
          ? `true`
          : `false`;
      }
    }
    return `false`;
  },
  call({ args }) {
    const v = args[0];
    const nameArg = args[1];
    let name: string;
    if (typeof nameArg === "string") name = nameArg;
    else if (
      nameArg &&
      typeof nameArg === "object" &&
      isCharRV(nameArg as RuntimeValue)
    ) {
      name = (nameArg as { value: string }).value;
    } else {
      name = String(nameArg);
    }
    let valueClass = "unknown";
    if (typeof v === "number") valueClass = "double";
    else if (typeof v === "boolean") valueClass = "logical";
    else if (typeof v === "string") valueClass = "string";
    else if (v !== null && typeof v === "object") {
      if (isCharRV(v as RuntimeValue)) valueClass = "char";
      else if (isTensor(v as RuntimeValue)) {
        const tag = (v as { mtoc2Tag?: string }).mtoc2Tag;
        valueClass = tag === "tensor" ? "double" : "double";
      } else if ((v as { mtoc2Tag?: string }).mtoc2Tag === "cell")
        valueClass = "cell";
      else if ((v as { mtoc2Handle?: boolean }).mtoc2Handle === true)
        valueClass = "function_handle";
      else if ((v as { mtoc2Class?: string }).mtoc2Class !== undefined)
        valueClass = (v as { mtoc2Class: string }).mtoc2Class;
      else valueClass = "struct";
    }
    if (valueClass === name) return [true];
    if (name === "numeric") {
      if (valueClass === "double") return [true];
      return [false];
    }
    return [false];
  },
};
