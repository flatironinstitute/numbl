/**
 * MATLAB class names of runtime values, for builtin error messages.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import type { RuntimeError } from "../../runtime/error.js";
import { builtinError } from "./args.js";

/** MATLAB's class name of a value (what `class(v)` returns). */
export function valueClassName(v: RuntimeValue): string {
  if (typeof v === "number") return "double";
  if (typeof v === "boolean") return "logical";
  if (typeof v === "string") return "string";
  switch (v.kind) {
    case "tensor":
      return v._isLogical ? "logical" : "double";
    case "complex_number":
    case "sparse_matrix":
      return "double";
    case "char":
      return "char";
    case "cell":
      return "cell";
    case "struct":
    case "struct_array":
      return "struct";
    case "function":
      return "function_handle";
    case "string_array":
      return "string";
    case "class_instance":
    case "class_instance_array":
      return v.className;
    case "dictionary":
      return "dictionary";
    default:
      return v.kind;
  }
}

/** MATLAB's error for calling a function on a type it has no method for:
 *  "Undefined function 'f' for input arguments of type 'cell'." */
export function undefinedForTypeError(
  fn: string,
  v: RuntimeValue
): RuntimeError {
  return builtinError(
    "MATLAB:UndefinedFunction",
    `Undefined function '${fn}' for input arguments of type '${valueClassName(v)}'.`
  );
}
