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
 * The interpreter `call` hook handles every value kind. The AOT
 * backends cover the dominant real-numeric scalar/tensor combinations
 * inline / via the `mtoc2_isequal_*` helpers and reject the rarer
 * combinations (complex tensors, char/string, struct/cell) with an
 * `UnsupportedConstruct` so the limitation is explicit.
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { scalarLogical, isNumeric, isScalar } from "../../../lowering/types.js";
import type { Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  isTensor,
  isChar as isRtChar,
  isComplexValue,
  type RuntimeValue,
  type RuntimeTensor,
} from "../../../runtime/value.js";

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
  if (ka === "rs" && kb === "rs") return `(${aJs} === ${bJs} ? 1 : 0)`;
  if (ka === "rs" && kb === "cs")
    return `(${bJs}.im === 0 && ${bJs}.re === ${aJs} ? 1 : 0)`;
  if (ka === "cs" && kb === "rs")
    return `(${aJs}.im === 0 && ${aJs}.re === ${bJs} ? 1 : 0)`;
  if (ka === "cs" && kb === "cs")
    return `(${aJs}.re === ${bJs}.re && ${aJs}.im === ${bJs}.im ? 1 : 0)`;
  if (ka === "rs" && kb === "rt") {
    useTensorHelper();
    return `mtoc2_isequal_st(${aJs}, ${bJs})`;
  }
  if (ka === "rt" && kb === "rs") {
    useTensorHelper();
    return `mtoc2_isequal_st(${bJs}, ${aJs})`;
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

function textValue(v: RuntimeValue): string | null {
  if (isRtChar(v)) return v.value;
  if (typeof v === "string") return v;
  return null;
}

function numOf(v: RuntimeValue): number | null {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  return null;
}

function tensorsEqual(a: RuntimeTensor, b: RuntimeTensor): boolean {
  if (a.data.length !== b.data.length) return false;
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  for (let i = 0; i < a.data.length; i++) {
    if (a.data[i] !== b.data[i]) return false;
  }
  if (a.imag || b.imag) {
    if (!a.imag || !b.imag) return false;
    for (let i = 0; i < a.imag.length; i++) {
      if (a.imag[i] !== b.imag[i]) return false;
    }
  }
  return true;
}

function valuesEqual(a: RuntimeValue, b: RuntimeValue): boolean {
  const aText = textValue(a);
  const bText = textValue(b);
  if (aText !== null && bText !== null) return aText === bText;
  if (aText !== null || bText !== null) return false;

  const an = numOf(a);
  const bn = numOf(b);
  if (an !== null && bn !== null) return an === bn;

  if (isComplexValue(a) && isComplexValue(b))
    return a.re === b.re && a.im === b.im;
  if (isComplexValue(a) && bn !== null) return a.re === bn && a.im === 0;
  if (an !== null && isComplexValue(b)) return b.re === an && b.im === 0;

  if (isTensor(a) && isTensor(b)) return tensorsEqual(a, b);
  if (isTensor(a) && bn !== null)
    return a.data.length === 1 && !a.imag && a.data[0] === bn;
  if (an !== null && isTensor(b))
    return b.data.length === 1 && !b.imag && b.data[0] === an;
  if (isTensor(a) && isComplexValue(b))
    return (
      a.data.length === 1 &&
      a.data[0] === b.re &&
      (a.imag ? a.imag[0] === b.im : b.im === 0)
    );
  if (isComplexValue(a) && isTensor(b))
    return (
      b.data.length === 1 &&
      b.data[0] === a.re &&
      (b.imag ? b.imag[0] === a.im : a.im === 0)
    );

  // Structs: plain objects without an mtoc2 tag. Compare field sets.
  if (isPlainStruct(a) && isPlainStruct(b)) {
    const ao = a as Record<string, RuntimeValue>;
    const bo = b as Record<string, RuntimeValue>;
    const ak = Object.keys(ao);
    const bk = Object.keys(bo);
    if (ak.length !== bk.length) return false;
    for (const k of ak) {
      if (!(k in bo)) return false;
      if (!valuesEqual(ao[k], bo[k])) return false;
    }
    return true;
  }
  return false;
}

function isPlainStruct(v: RuntimeValue): v is Record<string, RuntimeValue> {
  return (
    typeof v === "object" &&
    v !== null &&
    !isTensor(v) &&
    !isRtChar(v) &&
    !isComplexValue(v) &&
    !("mtoc2Handle" in (v as object))
  );
}

// ── builtin ──────────────────────────────────────────────────────────

export const isequal: Builtin = {
  name: "isequal",
  transfer(argTypes, nargout) {
    checkArity("isequal", argTypes, nargout);
    // Transfer stays permissive (any value kind type-checks); specific
    // AOT gaps surface at emit time so the interpreter still runs the
    // char/string/struct/complex-tensor cases.
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
  call({ args }) {
    for (let i = 1; i < args.length; i++) {
      if (!valuesEqual(args[0], args[i])) return [false];
    }
    return [true];
  },
};
