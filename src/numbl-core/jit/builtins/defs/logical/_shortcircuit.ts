/**
 * Shared scaffold for the two short-circuit logical builtins
 * (`||` → `oror`, `&&` → `andand`).
 *
 * **Scope decision: scalar operands only.**
 *
 * MATLAB requires scalar operands for `||` and `&&`; numbl is more
 * permissive (routes both operands through `toBool` to accept
 * tensors — see `interpreter/interpreterExec.ts` lines 650–661,
 * `runtime/convert.ts:44`). mtoc2 v1 takes the MATLAB-strict path so
 * the result type stays a scalar logical and the codegen stays
 * trivial (`(a) || (b)` / `(a) && (b)` in C, which short-circuits
 * natively).
 *
 * Compile-time fold:
 *   - both operands exact → fold to scalar logical with concrete value.
 *   - LHS exact + truthy (for `||`) or falsy (for `&&`) → result is
 *     the LHS's logical value regardless of RHS. This honors
 *     short-circuit semantics at the type level so a runtime-only RHS
 *     doesn't pollute the result type. (mtoc2 always evaluates the
 *     RHS at the type-system layer, but the fold value still skips
 *     it.)
 *   - otherwise → scalar logical, no exact.
 *
 * **The runtime side already short-circuits** via C's `||` / `&&`
 * operators, independent of the fold rule.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isNumeric,
  isScalar,
  scalarLogical,
  typeToString,
  type NumericType,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble, exactScalarAsComplex } from "../_shared.js";

export type ShortCircuitKind = "or" | "and";

function asScalarRealOrLogical(t: Type, what: string, surface: string): void {
  if (!isNumeric(t)) {
    throw new TypeError(
      `${what} must be a real or complex scalar (got ${typeToString(t)}); ` +
        `'${surface}' requires scalar operands (numbl accepts tensors via toBool, but mtoc2 follows MATLAB)`
    );
  }
  if (t.elem !== "double" && t.elem !== "logical") {
    throw new TypeError(
      `${what} must be a real double or logical (got ${t.elem})`
    );
  }
  if (!isScalar(t)) {
    throw new TypeError(
      `${what} must be a scalar (got ${typeToString(t)}); ` +
        `'${surface}' requires scalar operands (numbl accepts tensors via toBool, but mtoc2 follows MATLAB — use elementwise '|' / '&' when added)`
    );
  }
}

/** Truthy as numbl/MATLAB define it. Complex is truthy iff either
 *  part is non-zero. Returns `undefined` when the value isn't exact. */
function scalarTruthy(t: Type): boolean | undefined {
  if (!isNumeric(t)) return undefined;
  if (t.isComplex) {
    const cx = exactScalarAsComplex(t);
    if (cx === undefined) return undefined;
    return cx.re !== 0 || cx.im !== 0;
  }
  const v = exactDouble(t);
  if (v === undefined) return undefined;
  return v !== 0;
}

export function defineShortCircuit(
  name: string,
  surface: string,
  kind: ShortCircuitKind
): Builtin {
  const cOp = kind === "or" ? "||" : "&&";
  return {
    name,
    transfer(argTypes, nargout) {
      if (argTypes.length !== 2) {
        throw new TypeError(
          `'${name}' expects 2 arg(s), got ${argTypes.length}`
        );
      }
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${name}' does not support multi-output (nargout=${nargout})`
        );
      }
      asScalarRealOrLogical(argTypes[0], `'${surface}' left operand`, surface);
      asScalarRealOrLogical(argTypes[1], `'${surface}' right operand`, surface);
      const at = scalarTruthy(argTypes[0]);
      const bt = scalarTruthy(argTypes[1]);
      if (kind === "or") {
        if (at === true) return [scalarLogical(true)];
        if (at !== undefined && bt !== undefined) return [scalarLogical(bt)];
        return [scalarLogical()];
      }
      // and: LHS exact + falsy short-circuits to false.
      if (at === false) return [scalarLogical(false)];
      if (at !== undefined && bt !== undefined) return [scalarLogical(bt)];
      return [scalarLogical()];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      const anyComplex =
        (argTypes[0] as NumericType).isComplex ||
        (argTypes[1] as NumericType).isComplex;
      if (anyComplex) useRuntime("mtoc2_cscalar");
      const lhs = (argTypes[0] as NumericType).isComplex
        ? `mtoc2_cnonzero(${argsC[0]})`
        : `(${argsC[0]})`;
      const rhs = (argTypes[1] as NumericType).isComplex
        ? `mtoc2_cnonzero(${argsC[1]})`
        : `(${argsC[1]})`;
      // C's `||` / `&&` short-circuit and yield int 0/1; auto-promotes
      // to double when assigned to a double scalar slot.
      return `(${lhs} ${cOp} ${rhs})`;
    },
    emitJs({ argsJs, argTypes, useRuntime }) {
      const aN = argTypes[0] as NumericType;
      const bN = argTypes[1] as NumericType;
      const anyComplex = aN.isComplex || bN.isComplex;
      if (anyComplex) useRuntime("mtoc2_cscalar");
      // Coerce to a bool with `!!x`. The operand may be a bare double
      // (`0`/`nonzero`) or a JS boolean from an upstream logical op;
      // `!!` works for both. Strict `!== 0` would WRONGLY return
      // `true` for a JS `false` because of strict-equality non-
      // coercion (`false !== 0` is `true` in JS).
      const lhs = aN.isComplex
        ? `mtoc2_cnonzero(${argsJs[0]})`
        : `!!(${argsJs[0]})`;
      const rhs = bN.isComplex
        ? `mtoc2_cnonzero(${argsJs[1]})`
        : `!!(${argsJs[1]})`;
      // Result is a bare JS bool.
      return `(${lhs} ${cOp} ${rhs})`;
    },
    call({ args, argTypes }) {
      const aN = argTypes[0] as NumericType;
      const bN = argTypes[1] as NumericType;
      const toBool = (v: unknown, isComplex: boolean): boolean => {
        if (isComplex) {
          const cx = v as { re: number; im: number };
          return cx.re !== 0 || cx.im !== 0;
        }
        const n = typeof v === "number" ? v : Number(v);
        return n !== 0;
      };
      const a = toBool(args[0], aN.isComplex);
      const b = toBool(args[1], bN.isComplex);
      return [kind === "or" ? a || b : a && b];
    },
  };
}
