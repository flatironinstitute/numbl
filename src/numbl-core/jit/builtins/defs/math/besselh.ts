/**
 * `besselh(nu, kind, x)` — Hankel function of the first kind.
 */
import {
  type NumericType,
  scalarComplex,
  tensorComplexFromDims,
  isMultiElement,
  isNumeric,
  typeToString,
} from "../../../lowering/types.js";
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";

function requireExactInt(t: NumericType, what: string): number {
  const v = exactDouble(t);
  if (v === undefined || !Number.isInteger(v)) {
    throw new UnsupportedConstruct(
      `'besselh' ${what} must be a compile-time integer (got non-exact or non-integer)`
    );
  }
  return v;
}

export const besselh: Builtin = {
  name: "besselh",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 3) {
      throw new TypeError(`'besselh' expects 3 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'besselh' does not support multi-output (nargout=${nargout})`
      );
    }
    for (let i = 0; i < 3; i++) {
      if (!isNumeric(argTypes[i])) {
        throw new TypeError(
          `'besselh' arg ${i + 1} must be a real numeric (got ${typeToString(argTypes[i])})`
        );
      }
      if ((argTypes[i] as NumericType).isComplex) {
        throw new UnsupportedConstruct(
          `'besselh' with complex args is not yet supported`
        );
      }
    }
    const nuT = argTypes[0] as NumericType;
    const kindT = argTypes[1] as NumericType;
    const xT = argTypes[2] as NumericType;
    const nu = requireExactInt(nuT, "arg 1 (nu)");
    const kind = requireExactInt(kindT, "arg 2 (kind)");
    if (nu !== 0 && nu !== 1) {
      throw new UnsupportedConstruct(
        `'besselh' supports only nu in {0, 1} today (got ${nu})`
      );
    }
    if (kind !== 1) {
      throw new UnsupportedConstruct(
        `'besselh' supports only kind == 1 today (got ${kind})`
      );
    }
    if (xT.elem !== "double") {
      throw new TypeError(
        `'besselh' arg 3 (x) must be double (got ${xT.elem})`
      );
    }
    if (!isMultiElement(xT)) {
      return [scalarComplex()];
    }
    return [tensorComplexFromDims(xT.dims.slice())];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_besselh");
    useRuntime("mtoc2_cscalar");
    const nu = (argTypes[0] as NumericType).exact as number;
    const xT = argTypes[2] as NumericType;
    if (isMultiElement(xT)) {
      return nu === 0
        ? `mtoc2_tensor_besselh0(${argsC[2]})`
        : `mtoc2_tensor_besselh1(${argsC[2]})`;
    }
    return nu === 0
      ? `mtoc2_besselh0_scalar(${argsC[2]})`
      : `mtoc2_besselh1_scalar(${argsC[2]})`;
  },
  // js-aot and interpreter parity is deliberately deferred — the
  // c-aot path leans on POSIX `j0` / `j1` / `y0` / `y1` from
  // `<math.h>`; JS has no equivalent in the standard library. A
  // proper port would vendor numbl's `besselj` / `bessely` helpers
  // (see `../numbl/src/numbl-core/helpers/bessel.ts`). Throwing a
  // clean UnsupportedConstruct here is better than the framework's
  // "internal: ... no emitJs hook" surface — every builtin in the
  // registry now structurally implements all three hooks.
  emitJs() {
    throw new UnsupportedConstruct(
      `'besselh' is not yet implemented for the js-aot backend ` +
        `(needs a JS port of POSIX j0/j1/y0/y1; see numbl/helpers/bessel.ts)`
    );
  },
  call() {
    throw new UnsupportedConstruct(
      `'besselh' is not yet implemented for the interpreter ` +
        `(needs a JS port of POSIX j0/j1/y0/y1; see numbl/helpers/bessel.ts)`
    );
  },
};
