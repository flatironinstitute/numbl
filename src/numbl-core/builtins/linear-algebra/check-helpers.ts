/**
 * Helpers for writing builtin check functions.
 * These make check implementations shorter and easier to read.
 */

import {
  type ItemType,
  IType,
  isNum,
  isFullyUnknown,
  isString,
  isChar,
} from "../../lowering/itemTypes.js";
import type { FloatXArrayType } from "../../runtime/types.js";

/** Wraps output types into the check-function return value. */
export function out(...types: ItemType[]): { outputTypes: ItemType[] } {
  return { outputTypes: types };
}

/** A 2-D matrix type. */
export function unknownMatrix(isComplex?: boolean): ItemType {
  return IType.tensor({ isComplex: isComplex || undefined });
}

/** Alias for unknownMatrix — shape tracking was removed. */
export function matrix(
  _shapeOrComplex?:
    | (number | "unknown")[]
    | "unknown"
    | boolean
    | [number | "unknown", number | "unknown"],
  isComplex?: boolean
): ItemType {
  // If first arg is boolean, it's the isComplex flag
  if (typeof _shapeOrComplex === "boolean") {
    return IType.tensor({ isComplex: _shapeOrComplex || undefined });
  }
  return IType.tensor({ isComplex: isComplex || undefined });
}

/**
 * Parses the optional economy-mode argument used in decompositions
 * like qr, svd, etc.
 *
 * Returns:
 *   true      — economy mode confirmed (0 or 'econ')
 *   false     — full mode (no argument provided)
 *   "unknown" — argument present but value not statically known
 *   null      — invalid argument (this branch should be rejected)
 */
export function parseEconArg(
  argType: ItemType | undefined
): true | false | "unknown" | null {
  if (argType === undefined) return false;
  if (isFullyUnknown(argType)) return "unknown";
  if (isNum(argType) === true) return "unknown"; // value tracking removed
  if (isString(argType) === true || isChar(argType) === true) return "unknown"; // value tracking removed
  return null;
}

/** Ensure data is Float64Array (needed by LAPACK bridges). */
export function toF64(data: FloatXArrayType): Float64Array {
  return data instanceof Float64Array ? data : new Float64Array(data);
}

/**
 * Extract and normalize a string argument at runtime.
 * Strips surrounding quotes and lowercases the result.
 * Works with both raw string args and RuntimeChar/RuntimeString values.
 */
export function parseStringArgLower(arg: unknown): string {
  if (typeof arg === "string") {
    return arg.replace(/^['"]|['"]$/g, "").toLowerCase();
  }
  if (arg && typeof arg === "object" && "value" in arg) {
    return String((arg as { value: unknown }).value)
      .replace(/^['"]|['"]$/g, "")
      .toLowerCase();
  }
  return String(arg).toLowerCase();
}
