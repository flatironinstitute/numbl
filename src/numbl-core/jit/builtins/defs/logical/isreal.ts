/**
 * `isreal(x)` — true iff `x` carries no imaginary part.
 *
 * Mirrors numbl's `isreal` (interpreter/builtins/predicates.ts):
 *   - real numeric / char / string / struct → true;
 *   - a complex tensor is always "complex" (it owns an imag lane), so
 *     → false;
 *   - a complex scalar is real iff its imaginary part is exactly 0
 *     (numbl checks `v.im === 0` at runtime).
 *
 * Result is a logical scalar. Most cases fold statically from the
 * type; only a non-exact complex scalar needs a runtime `cimag`
 * check (the C compiler folds it away when the value is constant).
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { scalarLogical, isNumeric, isScalar } from "../../../lowering/types.js";
import type { Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { isTensor, isComplexValue } from "../../../runtime/value.js";

function checkArity(argTypes: Type[], nargout: number): void {
  if (argTypes.length !== 1) {
    throw new TypeError(`'isreal' expects 1 arg(s), got ${argTypes.length}`);
  }
  if (nargout > 1) {
    throw new UnsupportedConstruct(
      `'isreal' does not support multi-output (nargout=${nargout})`
    );
  }
}

/** Static verdict: `true` / `false` when known at type time,
 *  `"runtime"` when it depends on a complex scalar's imag value. */
function staticVerdict(t: Type): boolean | "runtime" {
  if (!isNumeric(t)) return true; // char / string / struct / cell
  if (!t.isComplex) return true;
  if (!isScalar(t)) return false; // complex tensor
  // Complex scalar: fold if we know the exact value.
  const ex = t.exact;
  if (ex !== undefined && typeof ex === "object" && "im" in ex) {
    const im = ex.im;
    if (typeof im === "number") return im === 0;
  }
  return "runtime";
}

export const isreal: Builtin = {
  name: "isreal",
  transfer(argTypes, nargout) {
    checkArity(argTypes, nargout);
    const v = staticVerdict(argTypes[0]);
    if (v === "runtime") return [scalarLogical()];
    return [scalarLogical(v)];
  },
  emitC({ argTypes, argsC, useRuntime }) {
    const v = staticVerdict(argTypes[0]);
    if (v === true) return "1.0";
    if (v === false) return "0.0";
    useRuntime("mtoc2_cscalar");
    return `(cimag(${argsC[0]}) == 0.0)`;
  },
  emitJs({ argTypes, argsJs }) {
    const v = staticVerdict(argTypes[0]);
    if (v === true) return "true";
    if (v === false) return "false";
    return `(${argsJs[0]}.im === 0)`;
  },
  call({ args }) {
    const v = args[0];
    if (typeof v === "number" || typeof v === "boolean") return [true];
    if (isComplexValue(v)) return [v.im === 0];
    if (isTensor(v)) return [!v.imag];
    return [true];
  },
};
