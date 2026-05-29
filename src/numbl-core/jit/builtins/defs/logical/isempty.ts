/**
 * `isempty(x)` — true iff `x` has zero elements.
 *
 * Mirrors numbl's `isempty` (interpreter/builtins/introspection.ts):
 *   - scalars (real / complex) → false;
 *   - tensors / cells → numel === 0;
 *   - char → zero-length text; string scalar → false;
 *   - structs → false (a scalar struct is never empty).
 *
 * Result is a logical scalar. When the numeric shape is statically
 * known (the common case) it folds; otherwise it emits a runtime
 * `mtoc2_numel(x) == 0` check.
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  scalarLogical,
  shapeNumel,
  isNumeric,
  isScalar,
  isChar,
} from "../../../lowering/types.js";
import type { Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { isTensor, isChar as isRtChar } from "../../../runtime/value.js";

/** The only runtime-shaped AOT path is a numeric tensor of unknown
 *  extent. Char of unknown length isn't representable as an
 *  `mtoc2_tensor_t`, so the AOT backends defer it to the interpreter. */
function requireNumericRuntime(t: Type): void {
  if (!isNumeric(t)) {
    throw new UnsupportedConstruct(
      `'isempty' on a non-numeric value of unknown size is not supported ` +
        `by the AOT backends; the interpreter handles it`
    );
  }
}

function checkArity(argTypes: Type[], nargout: number): void {
  if (argTypes.length !== 1) {
    throw new TypeError(`'isempty' expects 1 arg(s), got ${argTypes.length}`);
  }
  if (nargout > 1) {
    throw new UnsupportedConstruct(
      `'isempty' does not support multi-output (nargout=${nargout})`
    );
  }
}

/** Static verdict, or `"runtime"` when the numeric shape is unknown. */
function staticVerdict(t: Type): boolean | "runtime" {
  if (isNumeric(t)) {
    if (isScalar(t)) return false;
    if (t.shape !== undefined) return shapeNumel(t.shape) === 0;
    return "runtime";
  }
  if (isChar(t)) {
    // Char carries its text only when statically exact; an empty char
    // literal ('') has length 0.
    if (t.exact !== undefined) return t.exact.length === 0;
    return "runtime";
  }
  // string scalar / struct / class / handle → never empty in v1.
  return false;
}

export const isempty: Builtin = {
  name: "isempty",
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
    requireNumericRuntime(argTypes[0]);
    useRuntime("mtoc2_numel");
    return `(mtoc2_numel(${argsC[0]}) == 0)`;
  },
  emitJs({ argTypes, argsJs }) {
    const v = staticVerdict(argTypes[0]);
    if (v === true) return "true";
    if (v === false) return "false";
    requireNumericRuntime(argTypes[0]);
    return `(${argsJs[0]}.shape.reduce((a,b)=>a*b,1) === 0)`;
  },
  call({ args }) {
    const v = args[0];
    if (typeof v === "number" || typeof v === "boolean") return [false];
    if (isTensor(v)) return [v.data.length === 0];
    if (isRtChar(v)) return [v.value.length === 0];
    if (typeof v === "string") return [false];
    // struct / complex / handle
    return [false];
  },
};
