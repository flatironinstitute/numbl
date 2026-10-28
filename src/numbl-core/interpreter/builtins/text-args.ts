/**
 * Argument checks shared by the text, cell/struct, and dictionary builtins.
 */

import { RuntimeError } from "../../runtime/error.js";
import {
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeStringArray,
  type RuntimeChar,
  type RuntimeValue,
} from "../../runtime/types.js";
import { valueClass } from "../../runtime/kinds.js";
import { builtinError } from "./args.js";

/** A char array or a string scalar. */
export function isText(v: RuntimeValue): v is RuntimeChar | string {
  return isRuntimeChar(v) || isRuntimeString(v);
}

/** Text in any of MATLAB's text containers: char, string (scalar or
 *  array), or a cell array (whose elements are checked by the caller). */
export function isTextOrCell(v: RuntimeValue): boolean {
  return isText(v) || isRuntimeStringArray(v) || isRuntimeCell(v);
}

/** MATLAB's error for a non-text first argument to a string function. */
export function mustBeTextError(): RuntimeError {
  return builtinError(
    "MATLAB:string:MustBeCharCellArrayOrString",
    "First argument must be text."
  );
}

/** MATLAB's error when no overload of `name` accepts the argument's class
 *  (e.g. a dictionary method called on a double). */
export function undefinedFunctionError(
  name: string,
  v: RuntimeValue
): RuntimeError {
  return builtinError(
    "MATLAB:UndefinedFunction",
    `Undefined function '${name}' for input arguments of type '${valueClass(v)}'.`
  );
}

/** Error for an input type MATLAB accepts (or handles differently) but
 *  numbl does not implement. */
export function unsupportedClassError(
  name: string,
  v: RuntimeValue
): RuntimeError {
  const what = isRuntimeCell(v)
    ? "cell array"
    : isRuntimeStringArray(v)
      ? "string array"
      : `'${valueClass(v)}'`;
  return new RuntimeError(`${name}: ${what} input is not supported`);
}

/** Error for a call form MATLAB accepts but numbl does not implement. */
export function unsupportedFormError(name: string, what: string): RuntimeError {
  return new RuntimeError(`${name}: ${what} is not supported`);
}
