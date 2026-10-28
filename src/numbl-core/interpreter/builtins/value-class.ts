/**
 * MATLAB class names of runtime values, for builtin error messages.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import { valueClass } from "../../runtime/kinds.js";
import type { RuntimeError } from "../../runtime/error.js";
import { builtinError } from "./args.js";

/** MATLAB's error for calling a function on a type it has no method for:
 *  "Undefined function 'f' for input arguments of type 'cell'." */
export function undefinedForTypeError(
  fn: string,
  v: RuntimeValue
): RuntimeError {
  return builtinError(
    "MATLAB:UndefinedFunction",
    `Undefined function '${fn}' for input arguments of type '${valueClass(v)}'.`
  );
}
