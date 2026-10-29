/**
 * Helpers shared across multiple builtin files.
 */

import type { Span } from "../../numbl/index.js";
import { TypeError, UnsupportedConstruct } from "../../lowering/errors.js";
import {
  type NumericType,
  type Type,
  isScalarRealDouble,
  isNumeric,
  isScalar,
  isDoubleOrLogical,
} from "../../lowering/types.js";
import type { Builtin } from "../registry.js";

export function requireScalarRealDouble(
  t: Type,
  what: string,
  span?: Span
): void {
  if (!isScalarRealDouble(t)) {
    throw new TypeError(
      `${what} must be a scalar real double (got ${t.kind})`,
      span
    );
  }
}

/** Like `requireScalarRealDouble` but accepts non-scalar real doubles
 *  (the elemwise path). Logical also accepted (stored as double in C). */
export function requireRealDouble(t: Type, what: string, span?: Span): void {
  if (!isNumeric(t) || t.isComplex) {
    throw new TypeError(`${what} must be a real numeric (got ${t.kind})`, span);
  }
  if (!isDoubleOrLogical(t)) {
    throw new TypeError(
      `${what} must be double or logical (got ${t.elem})`,
      span
    );
  }
}

/** Accepts a real or complex numeric (double / logical). Used by the
 *  unary math builtins and the scalar/tensor arithmetic paths that
 *  contaminate to complex when either operand is complex. */
export function requireRealOrComplex(t: Type, what: string, span?: Span): void {
  if (!isNumeric(t)) {
    throw new TypeError(`${what} must be a numeric (got ${t.kind})`, span);
  }
  if (!isDoubleOrLogical(t)) {
    throw new TypeError(
      `${what} must be double or logical (got ${t.elem})`,
      span
    );
  }
}

/** Require a scalar numeric (real or complex). */
export function requireScalarRealOrComplex(
  t: Type,
  what: string,
  span?: Span
): void {
  if (!isNumeric(t) || !isScalar(t)) {
    throw new TypeError(
      `${what} must be a scalar numeric (got ${t.kind})`,
      span
    );
  }
  if (!isDoubleOrLogical(t)) {
    throw new TypeError(
      `${what} must be double or logical (got ${t.elem})`,
      span
    );
  }
}

/** The MATLAB class of a numeric value (complex counts as double). An
 *  element type the JIT does not model declines. */
export function numericClassName(t: NumericType): "double" | "logical" {
  if (t.elem === "double" || t.elem === "logical") return t.elem;
  throw new UnsupportedConstruct(
    `the class of a '${t.elem}' numeric value is not modeled by the JIT`
  );
}

export function exactDouble(t: Type): number | undefined {
  if (!isNumeric(t)) return undefined;
  if (t.exact === undefined) return undefined;
  if (typeof t.exact === "number") return t.exact;
  return undefined;
}

export function exactRealArray(t: Type): Float64Array | undefined {
  if (!isNumeric(t)) return undefined;
  if (t.exact instanceof Float64Array) return t.exact;
  return undefined;
}

/** Read a scalar-complex exact value if present. Returns `{re, im}`
 *  when the type's `exact` is the scalar-complex carrier, or `undefined`
 *  otherwise. */
export function exactComplex(t: Type): { re: number; im: number } | undefined {
  if (!isNumeric(t)) return undefined;
  if (t.exact === undefined) return undefined;
  if (typeof t.exact === "number") return undefined;
  if (t.exact instanceof Float64Array) return undefined;
  if (t.exact.re instanceof Float64Array) return undefined;
  return t.exact as { re: number; im: number };
}

/** Read a complex-tensor split-buffer exact carrier if present.
 *  Returns the `{re, im}` Float64Array pair when the type's `exact` is
 *  the complex-tensor carrier (parallel to `exactRealArray` but for
 *  complex tensors); returns `undefined` for any other shape (no exact,
 *  scalar number, real-array, scalar-complex `{re: number}`). */
export function exactComplexArray(
  t: Type
): { re: Float64Array; im: Float64Array } | undefined {
  if (!isNumeric(t)) return undefined;
  if (t.exact === undefined) return undefined;
  if (typeof t.exact !== "object") return undefined;
  if (t.exact instanceof Float64Array) return undefined;
  if (!(t.exact.re instanceof Float64Array)) return undefined;
  return t.exact as { re: Float64Array; im: Float64Array };
}

/** Convenience: project a scalar exact value (real or complex) into a
 *  `{re, im}` pair, returning `undefined` if neither carrier matches. */
export function exactScalarAsComplex(
  t: Type
): { re: number; im: number } | undefined {
  const r = exactDouble(t);
  if (r !== undefined) return { re: r, im: 0 };
  return exactComplex(t);
}

/** `b` with a logical first output when the arguments `pick` selects are
 *  all real logical: the shape builtins and min/max keep the class of a
 *  logical input (`flip(true)`, `max([true false])`). */
export function keepsLogical(
  b: Builtin,
  pick: (argTypes: Type[]) => Type[]
): Builtin {
  return {
    ...b,
    transfer(argTypes, nargout) {
      const out = b.transfer(argTypes, nargout);
      const picked = pick(argTypes);
      const first = out[0];
      if (
        picked.length > 0 &&
        picked.every(
          t => isNumeric(t) && t.elem === "logical" && !t.isComplex
        ) &&
        first !== undefined &&
        isNumeric(first) &&
        !first.isComplex
      )
        return [{ ...first, elem: "logical" }, ...out.slice(1)];
      return out;
    },
  };
}

/** The first argument, for `keepsLogical`. */
export const firstArg = (argTypes: Type[]): Type[] => argTypes.slice(0, 1);
