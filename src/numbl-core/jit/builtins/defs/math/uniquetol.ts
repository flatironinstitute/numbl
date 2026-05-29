/**
 * `uniquetol(x)` / `uniquetol(x, tol)` — first-occurrence dedup of
 * a real-double tensor with absolute tolerance.
 *
 * Numbl semantics
 * (`numbl-core/interpreter/builtins/set-operations.ts`):
 *   - Default `tol = 1e-6` when no second arg.
 *   - Algorithm: naive pairwise scan against the running unique
 *     list — first-occurrence order preserved, NOT sorted.
 *     Transitive chaining is a real consequence (e.g.
 *     `uniquetol([0 0.6 1.2 1.8], 0.7) == [0 1.2]`).
 *   - NaN is never within tol of anything; each NaN survives.
 *   - Output shape: row input (1×N) → row output `[1, ?]`;
 *     anything else (column / matrix / empty) → column output
 *     `[?, 1]`. Empty input gives `[0, 1]`.
 *
 * The multi-output `[c, ia, ic] = uniquetol(...)` form, the
 * `'ByRows'` name-value pair, and complex inputs are not yet
 * supported and raise `UnsupportedConstruct`.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  DIM_ONE,
  type DimInfo,
  type NumericType,
  isNumeric,
  isScalar,
  scalarDouble,
  signFromNumber,
  tensorDoubleFromDims,
  type Type,
  typeToString,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import { mtoc2_uniquetol_real as jsUniquetolReal } from "../../runtime/snippets.gen.js";

const DEFAULT_TOL = 1e-6;

function requireRealInput(t: Type): NumericType {
  if (!isNumeric(t)) {
    throw new TypeError(
      `'uniquetol' first arg must be numeric (got ${typeToString(t)})`
    );
  }
  if (t.isComplex) {
    throw new UnsupportedConstruct(
      `'uniquetol' on complex inputs is not yet supported`
    );
  }
  if (t.elem !== "double" && t.elem !== "logical") {
    throw new TypeError(
      `'uniquetol' first arg must be double or logical (got ${t.elem})`
    );
  }
  return t;
}

function requireScalarRealTol(t: Type): void {
  if (!isNumeric(t) || t.isComplex || !isScalar(t)) {
    throw new TypeError(
      `'uniquetol' tol arg must be a real scalar (got ${typeToString(t)})`
    );
  }
  if (t.elem !== "double" && t.elem !== "logical") {
    throw new TypeError(
      `'uniquetol' tol arg must be double or logical (got ${t.elem})`
    );
  }
}

/** True iff the input shape is `1×N` (row-like, including scalar). */
function isRowInput(a: NumericType): boolean {
  if (a.dims.length < 2) return false;
  const a0 = a.dims[0];
  return a0.kind === "exact" && a0.value === 1;
}

export const uniquetol: Builtin = {
  name: "uniquetol",
  transfer(argTypes, nargout) {
    if (argTypes.length < 1 || argTypes.length > 2) {
      throw new TypeError(
        `'uniquetol' expects 1..2 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'uniquetol' multi-output form ([c, ia, ic] = uniquetol(...)) is not yet supported`
      );
    }
    const a = requireRealInput(argTypes[0]);
    if (argTypes.length === 2) requireScalarRealTol(argTypes[1]);
    // Scalar input → scalar output: dedup of a single value is itself.
    if (isScalar(a)) {
      const x = exactDouble(a);
      if (x !== undefined && Number.isFinite(x)) {
        return [scalarDouble(signFromNumber(x), x)];
      }
      return [scalarDouble()];
    }
    const row = isRowInput(a);
    // Output length is runtime-only; one axis is `unknown`.
    const dims: DimInfo[] = row
      ? [DIM_ONE, { kind: "unknown" }]
      : [{ kind: "unknown" }, DIM_ONE];
    return [tensorDoubleFromDims(dims)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isScalar(a)) return `(${argsC[0]})`;
    useRuntime("mtoc2_uniquetol_real");
    const tolC =
      argTypes.length === 2 ? `(double)(${argsC[1]})` : `${DEFAULT_TOL}`;
    const rowOut = isRowInput(a) ? 1 : 0;
    return `mtoc2_uniquetol_real(${argsC[0]}, ${tolC}, ${rowOut})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isScalar(a)) return `(${argsJs[0]})`;
    useRuntime("mtoc2_uniquetol_real");
    const tolJs = argTypes.length === 2 ? argsJs[1] : String(DEFAULT_TOL);
    const rowOut = isRowInput(a) ? 1 : 0;
    return `mtoc2_uniquetol_real(${argsJs[0]}, ${tolJs}, ${rowOut})`;
  },
  call({ args, argTypes }) {
    const a = argTypes[0] as NumericType;
    if (isScalar(a)) {
      return [typeof args[0] === "number" ? args[0] : Number(args[0])];
    }
    let tol = DEFAULT_TOL;
    if (argTypes.length === 2) {
      const t = argTypes[1] as NumericType;
      const x = exactDouble(t);
      tol = x !== undefined ? x : Number(args[1]);
    }
    const rowOut = isRowInput(a) ? 1 : 0;
    return [
      jsUniquetolReal(
        args[0] as RuntimeTensor,
        tol,
        rowOut
      ) as unknown as RuntimeTensor,
    ];
  },
};
