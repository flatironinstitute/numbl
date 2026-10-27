/**
 * `isequal(A, B, ...)` — true iff every argument is equal to the
 * first. Always returns a logical scalar.
 *
 * Equality follows numbl's `valuesEqualSimple` (interpreter/utility.ts):
 *   - real / logical scalars compare by `==` (NaN ≠ NaN);
 *   - a scalar equals a 1×1 tensor with the matching value;
 *   - tensors are equal iff same rank, same extent, same data — and a
 *     real tensor is never equal to a complex tensor (numbl's rule);
 *   - complex scalars compare real and imaginary parts;
 *   - char / string compare as text; structs compare field-by-field.
 *
 * The JIT covers the dominant real-numeric scalar/tensor combinations
 * inline / via the `mtoc2_isequal_*` helpers and rejects the rarer
 * combinations (complex tensors, char/string, struct/cell) with an
 * `UnsupportedConstruct` so the limitation is explicit.
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { scalarLogical, isNumeric, isScalar } from "../../../lowering/types.js";
import type { Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";

type NumKind = "rs" | "cs" | "rt" | "ct" | "other";

function classify(t: Type): NumKind {
  if (isNumeric(t)) {
    const scalar = isScalar(t);
    if (scalar) return t.isComplex ? "cs" : "rs";
    return t.isComplex ? "ct" : "rt";
  }
  return "other";
}

function checkArity(name: string, argTypes: Type[], nargout: number): void {
  if (argTypes.length < 2) {
    throw new TypeError(
      `'${name}' expects at least 2 arg(s), got ${argTypes.length}`
    );
  }
  if (nargout > 1) {
    throw new UnsupportedConstruct(
      `'${name}' does not support multi-output (nargout=${nargout})`
    );
  }
}

/** C expression (0.0/1.0) comparing two args by static kind. */
function pairC(
  tA: Type,
  aC: string,
  tB: Type,
  bC: string,
  useRuntime: (s: string) => void
): string {
  const ka = classify(tA);
  const kb = classify(tB);
  const useTensorHelper = () => useRuntime("mtoc2_isequal");
  const useComplex = () => useRuntime("mtoc2_cscalar");
  // Real scalar vs real scalar.
  if (ka === "rs" && kb === "rs") return `(${aC} == ${bC} ? 1.0 : 0.0)`;
  // Complex scalar combinations.
  if (ka === "rs" && kb === "cs") {
    useComplex();
    return `(cimag(${bC}) == 0.0 && creal(${bC}) == ${aC} ? 1.0 : 0.0)`;
  }
  if (ka === "cs" && kb === "rs") {
    useComplex();
    return `(cimag(${aC}) == 0.0 && creal(${aC}) == ${bC} ? 1.0 : 0.0)`;
  }
  if (ka === "cs" && kb === "cs") {
    useComplex();
    return `(creal(${aC}) == creal(${bC}) && cimag(${aC}) == cimag(${bC}) ? 1.0 : 0.0)`;
  }
  // Real scalar vs tensor.
  if (ka === "rs" && kb === "rt") {
    useTensorHelper();
    return `mtoc2_isequal_st(${aC}, ${bC})`;
  }
  if (ka === "rt" && kb === "rs") {
    useTensorHelper();
    return `mtoc2_isequal_st(${bC}, ${aC})`;
  }
  // Real scalar vs complex tensor → numbl: a complex tensor always
  // carries an imag lane, so a real scalar can never equal it.
  if ((ka === "rs" && kb === "ct") || (ka === "ct" && kb === "rs"))
    return `0.0`;
  // Complex scalar vs real tensor: equal iff imag is zero and the
  // tensor is a 1×1 with the matching real value.
  if (ka === "cs" && kb === "rt") {
    useTensorHelper();
    useComplex();
    return `(cimag(${aC}) == 0.0 ? mtoc2_isequal_st(creal(${aC}), ${bC}) : 0.0)`;
  }
  if (ka === "rt" && kb === "cs") {
    useTensorHelper();
    useComplex();
    return `(cimag(${bC}) == 0.0 ? mtoc2_isequal_st(creal(${bC}), ${aC}) : 0.0)`;
  }
  // Real tensor vs real tensor.
  if (ka === "rt" && kb === "rt") {
    useTensorHelper();
    return `mtoc2_isequal_tt(${aC}, ${bC})`;
  }
  // Real tensor vs complex tensor → never equal (numbl tensorsEqual).
  if ((ka === "rt" && kb === "ct") || (ka === "ct" && kb === "rt"))
    return `0.0`;
  throw new UnsupportedConstruct(
    `'isequal' with these argument types is not yet supported by the AOT ` +
      `backends (complex-tensor, char/string, struct or cell comparison); ` +
      `the interpreter handles it`
  );
}

/** JS expression (0/1) comparing two args by static kind. */
function pairJs(
  tA: Type,
  aJs: string,
  tB: Type,
  bJs: string,
  useRuntime: (s: string) => void
): string {
  const ka = classify(tA);
  const kb = classify(tB);
  const useTensorHelper = () => useRuntime("mtoc2_isequal");
  // A logical *scalar* is a JS boolean; coerce it to a number so a
  // strict `===` against a numeric operand works (`true === 1` is
  // false in JS). Logical *tensors* are Float64-backed, so only scalars
  // need this — guard with isScalar to avoid `+tensorObject` → NaN.
  const numJs = (t: Type, x: string): string =>
    isNumeric(t) && isScalar(t) && t.elem === "logical" ? `(+(${x}))` : x;
  const aN = numJs(tA, aJs);
  const bN = numJs(tB, bJs);
  if (ka === "rs" && kb === "rs") return `(${aN} === ${bN} ? 1 : 0)`;
  if (ka === "rs" && kb === "cs")
    return `(${bJs}.im === 0 && ${bJs}.re === ${aN} ? 1 : 0)`;
  if (ka === "cs" && kb === "rs")
    return `(${aJs}.im === 0 && ${aJs}.re === ${bN} ? 1 : 0)`;
  if (ka === "cs" && kb === "cs")
    return `(${aJs}.re === ${bJs}.re && ${aJs}.im === ${bJs}.im ? 1 : 0)`;
  if (ka === "rs" && kb === "rt") {
    useTensorHelper();
    return `mtoc2_isequal_st(${aN}, ${bJs})`;
  }
  if (ka === "rt" && kb === "rs") {
    useTensorHelper();
    return `mtoc2_isequal_st(${bN}, ${aJs})`;
  }
  if ((ka === "rs" && kb === "ct") || (ka === "ct" && kb === "rs")) return `0`;
  if (ka === "cs" && kb === "rt") {
    useTensorHelper();
    return `(${aJs}.im === 0 ? mtoc2_isequal_st(${aJs}.re, ${bJs}) : 0)`;
  }
  if (ka === "rt" && kb === "cs") {
    useTensorHelper();
    return `(${bJs}.im === 0 ? mtoc2_isequal_st(${bJs}.re, ${aJs}) : 0)`;
  }
  if (ka === "rt" && kb === "rt") {
    useTensorHelper();
    return `mtoc2_isequal_tt(${aJs}, ${bJs})`;
  }
  if ((ka === "rt" && kb === "ct") || (ka === "ct" && kb === "rt")) return `0`;
  throw new UnsupportedConstruct(
    `'isequal' with these argument types is not yet supported by the AOT ` +
      `backends (complex-tensor, char/string, struct or cell comparison); ` +
      `the interpreter handles it`
  );
}

// ── interpreter equality (full coverage) ─────────────────────────────

// ── builtin ──────────────────────────────────────────────────────────

export const isequal: Builtin = {
  name: "isequal",
  transfer(argTypes, nargout) {
    checkArity("isequal", argTypes, nargout);
    // Transfer stays permissive (any value kind type-checks); specific
    // JIT gaps surface at emit time.
    return [scalarLogical()];
  },
  emitC({ argTypes, argsC, useRuntime }) {
    const parts: string[] = [];
    for (let i = 1; i < argTypes.length; i++) {
      parts.push(
        `(${pairC(argTypes[0], argsC[0], argTypes[i], argsC[i], useRuntime)}) != 0.0`
      );
    }
    return `(${parts.join(" && ")})`;
  },
  emitJs({ argTypes, argsJs, useRuntime }) {
    const parts: string[] = [];
    for (let i = 1; i < argTypes.length; i++) {
      // Each pairJs returns 1/0 (boolean-equivalent in JS); wrap with
      // `!== 0` to coerce to a real JS bool so the `&&` chain stays
      // boolean rather than returning the last truthy operand.
      parts.push(
        `(${pairJs(argTypes[0], argsJs[0], argTypes[i], argsJs[i], useRuntime)}) !== 0`
      );
    }
    return `(${parts.join(" && ")})`;
  },
};
