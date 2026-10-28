/**
 * `sort` — stable sort with optional 'ascend' / 'descend' mode arg.
 *
 * Supported forms in v1:
 *   - `b = sort(a)`                       — ascending
 *   - `b = sort(a, 'ascend' | 'descend')` — explicit mode
 *   - `[v, i] = sort(...)`                — values + 1-based positions
 *
 * Real and complex inputs both supported. Complex sort orders by
 * magnitude (then phase as tiebreak), matching numbl / MATLAB.
 *
 * `dim` arg and `sort(a, dim, mode)` form are out of scope for v1 —
 * the runtime helper walks the flat column-major buffer and the
 * type system restricts inputs to 1×N / N×1 vectors. The descending
 * comparator preserves the stable tie-break (ascending original
 * index) in both directions, matching numbl.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isDimOne,
  isMultiElement,
  isNumeric,
  isText,
  tensorComplexFromDims,
  tensorDoubleFromDims,
  type NumericType,
  type Type,
  typeToString,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { requireRealOrComplex } from "../_shared.js";

function requireVectorInput(a: Type): NumericType {
  requireRealOrComplex(a, "'sort' arg 1");
  const aN = a as NumericType;
  if (aN.elem === "logical") {
    // The runtime helpers return a double tensor; MATLAB keeps logical.
    throw new UnsupportedConstruct(
      `'sort' on a logical input is not supported`
    );
  }
  if (!isMultiElement(aN)) {
    throw new UnsupportedConstruct(
      `'sort' on a scalar is a no-op; pass a 1-D vector (1×N or N×1)`
    );
  }
  if (aN.dims.length !== 2) {
    throw new UnsupportedConstruct(
      `'sort' currently only supports 1-D vectors (got ${typeToString(aN)}); ` +
        `matrix and N-D forms (with explicit 'dim' arg) are not yet supported`
    );
  }
  const isRow = isDimOne(aN.dims[0]);
  const isCol = isDimOne(aN.dims[1]);
  if (!isRow && !isCol) {
    throw new UnsupportedConstruct(
      `'sort' currently only supports 1-D vectors (1×N or N×1); got ` +
        `${typeToString(aN)} — pass an explicit 'dim' (not yet supported) or ` +
        `reshape first`
    );
  }
  return aN;
}

/** Read the mode arg (if present) and return `descending: boolean`. */
function parseMode(argTypes: Type[]): boolean {
  if (argTypes.length < 2) return false;
  const m = argTypes[1];
  if (!isText(m) || (m.kind !== "Char" && m.kind !== "String")) {
    throw new TypeError(
      `'sort' mode arg must be a literal 'ascend' or 'descend' (got ${typeToString(m)})`
    );
  }
  if (m.exact === undefined) {
    throw new UnsupportedConstruct(
      `'sort' mode arg must be a literal; non-literal text is not supported`
    );
  }
  const mode = m.exact.toLowerCase();
  if (mode === "ascend") return false;
  if (mode === "descend") return true;
  throw new TypeError(
    `'sort' mode arg must be 'ascend' or 'descend' (got '${m.exact}')`
  );
}

export const sort: Builtin = {
  name: "sort",
  transfer(argTypes, nargout) {
    if (argTypes.length < 1 || argTypes.length > 2) {
      throw new TypeError(`'sort' expects 1..2 arg(s), got ${argTypes.length}`);
    }
    if (nargout < 1 || nargout > 2) {
      throw new UnsupportedConstruct(
        `'sort' supports 1..2 output(s); got nargout=${nargout}`
      );
    }
    const aN = requireVectorInput(argTypes[0]);
    parseMode(argTypes);
    const v = aN.isComplex
      ? tensorComplexFromDims(aN.dims.slice())
      : tensorDoubleFromDims(aN.dims.slice());
    if (nargout === 1) return [v];
    // The index output is always real (positive ints), regardless of
    // input complexity.
    const idx = tensorDoubleFromDims(aN.dims.slice());
    idx.sign = "positive";
    return [v, idx];
  },
  emitC({ argsC, argTypes, nargout, outArgsC, useRuntime }) {
    useRuntime("jit_sort_real");
    const aN = argTypes[0] as NumericType;
    const isComplex = isNumeric(aN) && aN.isComplex;
    const desc = parseMode(argTypes) ? 1 : 0;
    const base = isComplex ? "jit_sort_complex" : "jit_sort_real";
    if (nargout === 1) {
      return `${base}(${argsC[0]}, ${desc})`;
    }
    const outs = outArgsC ?? [];
    return `${base}_2(${argsC[0]}, ${desc}, ${outs.join(", ")})`;
  },
  emitJs({ argsJs, argTypes, nargout, useRuntime }) {
    useRuntime("jit_sort_real");
    const aN = argTypes[0] as NumericType;
    const isComplex = isNumeric(aN) && aN.isComplex;
    const desc = parseMode(argTypes) ? "true" : "false";
    const base = isComplex ? "jit_sort_complex" : "jit_sort_real";
    if (nargout === 1) return `${base}(${argsJs[0]}, ${desc})`;
    return `(o => [o.v, o.ix])(${base}_2(${argsJs[0]}, ${desc}))`;
  },
};
