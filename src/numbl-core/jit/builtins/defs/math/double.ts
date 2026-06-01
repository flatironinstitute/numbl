/**
 * `double(x)` — numeric conversion to double.
 *
 * Matches numbl `type-constructors.ts`:
 *   - Numeric scalar (double / logical / complex_or_number):
 *     scalar real number out. For complex scalars, returns the real
 *     part (numbl's `apply` returns `v.re`).
 *   - Numeric tensor: same shape; logical lane → double (drop the
 *     logical flag); complex tensors pass through as complex
 *     (numbl preserves `isComplex` in the match shape).
 *   - Char (single-quoted): char-code conversion. Empty → 0×0
 *     tensor; 1 char → scalar double; N chars → 1×N row.
 *
 * Scope notes:
 *   - The mtoc2 C representation already stores logical and double
 *     scalars as `double`, and logical tensors as `mtoc2_tensor_t`
 *     with a `double *` buffer. So a numeric → double conversion is
 *     either identity (scalar) or a fresh struct copy (tensor); no
 *     per-element rewrite is needed.
 *   - Char input is supported only when the char carries `exact`
 *     (the common case — char literals always carry exact text).
 *     Non-exact char input (e.g. `double(sprintf(...))`) raises
 *     `UnsupportedConstruct`. Folding at transfer time avoids
 *     introducing a new runtime helper for the char-tensor → numeric
 *     tensor path.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isMultiElement,
  isNumeric,
  isScalar,
  scalarDouble,
  signFromNumber,
  tensorDouble,
  type NumericType,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  isChar,
  isComplexValue,
  isTensor,
  makeComplexTensor,
  makeTensor,
  type RuntimeValue,
} from "../../../runtime/value.js";

function charExactToTensorType(s: string): Type {
  if (s.length === 0) {
    // numbl: RTV.tensor(allocFloat64Array(0), [0, 0])
    return tensorDouble([0, 0], new Float64Array(0));
  }
  if (s.length === 1) {
    const v = s.charCodeAt(0);
    return scalarDouble(signFromNumber(v), v);
  }
  const data = new Float64Array(s.length);
  for (let i = 0; i < s.length; i++) data[i] = s.charCodeAt(i);
  return tensorDouble([1, s.length], data);
}

function formatCNumber(v: number): string {
  if (Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}.0`;
  return String(v);
}

/** Inline ASCII-only C-string literal builder for char-code values.
 *  Only called for printable ASCII bytes (0x20..0x7e) in practice
 *  (most char literals); falls back to a `(double[]){...}` array
 *  emission elsewhere, so this is only used inside `(double[]){}`
 *  initializer lists. */
function compoundLiteralFromChar(s: string): string {
  const parts: string[] = [];
  for (let i = 0; i < s.length; i++) {
    parts.push(formatCNumber(s.charCodeAt(i)));
  }
  return parts.join(", ");
}

function jsArrayFromChar(s: string): string {
  const parts: string[] = [];
  for (let i = 0; i < s.length; i++) {
    parts.push(String(s.charCodeAt(i)));
  }
  return parts.join(", ");
}

export const doubleBuiltin: Builtin = {
  name: "double",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'double' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'double' does not support multi-output (nargout=${nargout})`
      );
    }
    const t = argTypes[0];
    if (isNumeric(t)) {
      if (t.isComplex && isScalar(t)) {
        // numbl scalar complex → real part
        if (
          t.exact !== undefined &&
          typeof t.exact === "object" &&
          !(t.exact instanceof Float64Array) &&
          typeof (t.exact as { re?: unknown }).re === "number"
        ) {
          const re = (t.exact as { re: number; im: number }).re;
          return [scalarDouble(signFromNumber(re), re)];
        }
        return [scalarDouble()];
      }
      // Real or complex tensor / scalar real — preserve shape/complex,
      // drop logical → double in the result type. Exact data carries
      // through unchanged (logical exact values 0/1 are valid doubles;
      // numeric exact data is the same double values).
      const out: NumericType = { ...t, elem: "double" };
      return [out];
    }
    if (t.kind === "Char") {
      if (t.exact === undefined) {
        throw new UnsupportedConstruct(
          `'double' on a non-exact char value is not supported (only char literals fold to numeric)`
        );
      }
      return [charExactToTensorType(t.exact)];
    }
    throw new TypeError(`'double' arg must be numeric or char (got ${t.kind})`);
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const t = argTypes[0];
    if (isNumeric(t)) {
      if (t.isComplex && isScalar(t)) {
        useRuntime("mtoc2_cscalar");
        return `mtoc2_creal(${argsC[0]})`;
      }
      if (isMultiElement(t)) {
        // Fresh tensor copy. Stripping the logical flag isn't
        // represented at the C level — `mtoc2_tensor_t` doesn't carry
        // one — so a structural copy is sufficient.
        if (t.isComplex) {
          useRuntime("mtoc2_tensor_copy_complex");
          return `mtoc2_tensor_copy_complex(${argsC[0]})`;
        }
        useRuntime("mtoc2_tensor_copy");
        return `mtoc2_tensor_copy(${argsC[0]})`;
      }
      // Scalar real (double / logical) — identity. mtoc2 stores both
      // as `double` in C, so no coercion is needed.
      return argsC[0];
    }
    if (t.kind === "Char") {
      // transfer rejected non-exact char input, so `t.exact` is set.
      const s = t.exact!;
      if (s.length === 0) {
        useRuntime("mtoc2_tensor_alloc");
        return `mtoc2_tensor_alloc(0, 0)`;
      }
      if (s.length === 1) {
        return formatCNumber(s.charCodeAt(0));
      }
      useRuntime("mtoc2_tensor_from_row");
      return `mtoc2_tensor_from_row((double[]){${compoundLiteralFromChar(s)}}, ${s.length})`;
    }
    throw new TypeError(`internal: 'double' emitC reached on type ${t.kind}`);
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const t = argTypes[0];
    if (isNumeric(t)) {
      if (t.isComplex && isScalar(t)) {
        return `(${argsJs[0]}).re`;
      }
      if (isMultiElement(t)) {
        // Fresh tensor object, dropping any `isLogical` flag (numbl
        // strips `_isLogical` on the double(...) result). Use an
        // inline object literal — argsJs[0] is a Var read so the
        // dup-evaluation is harmless.
        if (t.isComplex) {
          return `({mtoc2Tag:"tensor", shape:${argsJs[0]}.shape.slice(), data:new Float64Array(${argsJs[0]}.data), imag:new Float64Array(${argsJs[0]}.imag)})`;
        }
        return `({mtoc2Tag:"tensor", shape:${argsJs[0]}.shape.slice(), data:new Float64Array(${argsJs[0]}.data)})`;
      }
      // Scalar real. A scalar JS boolean must coerce to 0/1 — unary
      // `+` does this; for numbers it's a no-op.
      if (t.elem === "logical") {
        return `(+${argsJs[0]})`;
      }
      return argsJs[0];
    }
    if (t.kind === "Char") {
      const s = t.exact!;
      if (s.length === 0) {
        useRuntime("mtoc2_tensor_alloc");
        return `mtoc2_tensor_alloc(0, 0)`;
      }
      if (s.length === 1) {
        return String(s.charCodeAt(0));
      }
      useRuntime("mtoc2_tensor_from_row");
      return `mtoc2_tensor_from_row([${jsArrayFromChar(s)}], ${s.length})`;
    }
    throw new TypeError(`internal: 'double' emitJs reached on type ${t.kind}`);
  },
  call({ args }) {
    const v: RuntimeValue = args[0];
    if (typeof v === "number") return [v];
    if (typeof v === "boolean") return [v ? 1 : 0];
    if (isComplexValue(v)) return [v.re];
    if (isTensor(v)) {
      // Fresh data buffer; the resulting tensor is plain "double"
      // (no `isLogical` flag).
      if (v.imag !== undefined) {
        return [
          makeComplexTensor(
            v.shape.slice(),
            new Float64Array(v.data),
            new Float64Array(v.imag)
          ),
        ];
      }
      return [makeTensor(v.shape.slice(), new Float64Array(v.data))];
    }
    if (isChar(v)) {
      const s = v.value;
      if (s.length === 0) {
        return [makeTensor([0, 0], new Float64Array(0))];
      }
      if (s.length === 1) {
        return [s.charCodeAt(0)];
      }
      const data = new Float64Array(s.length);
      for (let i = 0; i < s.length; i++) data[i] = s.charCodeAt(i);
      return [makeTensor([1, s.length], data)];
    }
    throw new TypeError(
      `'double' got an unsupported runtime value (typeof = ${typeof v})`
    );
  },
};
