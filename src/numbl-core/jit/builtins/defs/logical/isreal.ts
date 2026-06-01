/**
 * `isreal(x)` — true iff `x` carries no imaginary content.
 *
 * Mirrors numbl's `isreal` (interpreter/builtins/predicates.ts):
 *   - real numeric / char / string / struct → true;
 *   - a complex scalar is real iff its imaginary part is exactly 0
 *     (runtime `cimag` check);
 *   - a complex tensor is real iff *every* imaginary element is 0
 *     (runtime scan). The JIT routinely produces complex-typed tensors
 *     whose imaginary lane is entirely zero when it cannot prove
 *     realness at compile time (e.g. `sqrt(1 - x.^2/2)`), so a
 *     value-based test is required to agree with the interpreter.
 *
 * Result is a logical scalar. Real cases fold statically from the type;
 * complex scalars/tensors need a runtime check (the C compiler folds it
 * away when the value is constant).
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { scalarLogical, isNumeric, isScalar } from "../../../lowering/types.js";
import type { Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { isTensor, isComplexValue } from "../../../runtime/value.js";
import { mtoc2_tensor_imag_all_zero } from "../../runtime/snippets.gen.js";

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

/** Static verdict: `true` when known real at type time, `"runtime"`
 *  when it depends on the value's imaginary content (a complex scalar's
 *  imag, or a complex tensor's imag lane). */
function staticVerdict(t: Type): boolean | "runtime" {
  if (!isNumeric(t)) return true; // char / string / struct / cell
  if (!t.isComplex) return true;
  if (!isScalar(t)) return "runtime"; // complex tensor → scan imag lane
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
    const t = argTypes[0];
    const v = staticVerdict(t);
    if (v === true) return "1.0";
    if (v === false) return "0.0";
    if (isNumeric(t) && t.isComplex && !isScalar(t)) {
      useRuntime("mtoc2_tensor_imag_all_zero");
      return `mtoc2_tensor_imag_all_zero(${argsC[0]})`;
    }
    useRuntime("mtoc2_cscalar");
    return `(cimag(${argsC[0]}) == 0.0)`;
  },
  emitJs({ argTypes, argsJs, useRuntime }) {
    const t = argTypes[0];
    const v = staticVerdict(t);
    if (v === true) return "true";
    if (v === false) return "false";
    if (isNumeric(t) && t.isComplex && !isScalar(t)) {
      useRuntime("mtoc2_tensor_imag_all_zero");
      return `mtoc2_tensor_imag_all_zero(${argsJs[0]})`;
    }
    return `(${argsJs[0]}.im === 0)`;
  },
  call({ args }) {
    const v = args[0];
    if (typeof v === "number" || typeof v === "boolean") return [true];
    if (isComplexValue(v)) return [v.im === 0];
    if (isTensor(v)) return [mtoc2_tensor_imag_all_zero(v)];
    return [true];
  },
};
