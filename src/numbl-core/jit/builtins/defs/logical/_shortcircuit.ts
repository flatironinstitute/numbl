/**
 * Shared scaffold for the two short-circuit logical builtins
 * (`||` → `oror`, `&&` → `andand`).
 *
 * **Scope decision: scalar operands only.**
 *
 * MATLAB requires scalar operands for `||` and `&&`; numbl is more
 * permissive (routes both operands through `toBool` to accept
 * tensors — see `evalBinary` in `interpreter/evalExpressions.ts`,
 * `runtime/convert.ts:44`). The JIT takes the MATLAB-strict path so
 * the result type stays a scalar logical and the codegen stays
 * trivial (`(a) || (b)` / `(a) && (b)` in C, which short-circuits
 * natively).
 *
 * Compile-time fold:
 *   - both operands exact → fold to scalar logical with concrete value.
 *   - LHS exact + truthy (for `||`) or falsy (for `&&`) → result is
 *     the LHS's logical value regardless of RHS. This honors
 *     short-circuit semantics at the type level so a runtime-only RHS
 *     doesn't pollute the result type. (the JIT always evaluates the
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
  isDoubleOrLogical,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble, exactScalarAsComplex } from "../_shared.js";

export type ShortCircuitKind = "or" | "and";

function asScalarRealOrLogical(t: Type, what: string, surface: string): void {
  if (!isNumeric(t)) {
    throw new TypeError(
      `${what} must be a real or complex scalar (got ${typeToString(t)}); ` +
        `'${surface}' requires scalar operands (numbl accepts tensors via toBool, but the JIT follows MATLAB)`
    );
  }
  if (!isDoubleOrLogical(t)) {
    throw new TypeError(
      `${what} must be a real double or logical (got ${t.elem})`
    );
  }
  if (!isScalar(t)) {
    throw new TypeError(
      `${what} must be a scalar (got ${typeToString(t)}); ` +
        `'${surface}' requires scalar operands (numbl accepts tensors via toBool, but the JIT follows MATLAB — use elementwise '|' / '&' when added)`
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
      if (argTypes.some(t => isNumeric(t) && t.isComplex)) {
        throw new UnsupportedConstruct(
          `'${surface}' of a complex operand raises; left to the interpreter`
        );
      }
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
      if (anyComplex) useRuntime("jit_cscalar");
      const lhs = (argTypes[0] as NumericType).isComplex
        ? `jit_cnonzero(${argsC[0]})`
        : `(${argsC[0]})`;
      const rhs = (argTypes[1] as NumericType).isComplex
        ? `jit_cnonzero(${argsC[1]})`
        : `(${argsC[1]})`;
      // C's `||` / `&&` short-circuit and yield int 0/1; auto-promotes
      // to double when assigned to a double scalar slot.
      return `(${lhs} ${cOp} ${rhs})`;
    },
    emitJs({ argsJs, argTypes, useRuntime }) {
      const aN = argTypes[0] as NumericType;
      const bN = argTypes[1] as NumericType;
      const anyComplex = aN.isComplex || bN.isComplex;
      if (anyComplex) useRuntime("jit_cscalar");
      // Coerce each operand to a numbl-truthiness bool. The
      // representation depends on the operand's element type:
      //   - complex → jit_cnonzero (object would be JS-truthy)
      //   - logical → already a JS boolean, use directly
      //   - real double → `!== 0`, NOT `!!x`: `!!NaN` is `false` in JS
      //     but numbl treats a nonzero NaN as true. (`!== 0` would be
      //     wrong for a JS `false`, hence the per-type branch.)
      const coerce = (n: NumericType, x: string): string =>
        n.isComplex
          ? `jit_cnonzero(${x})`
          : n.elem === "logical"
            ? `(${x})`
            : `((${x}) !== 0)`;
      const lhs = coerce(aN, argsJs[0]);
      const rhs = coerce(bN, argsJs[1]);
      // Result is a bare JS bool.
      return `(${lhs} ${cOp} ${rhs})`;
    },
  };
}
