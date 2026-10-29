/**
 * The `single` element class.
 *
 * A single array is a tensor with `cls: "single"`. Its elements live in a
 * Float64Array like every other class, holding values that are exactly
 * representable in single precision: every operation that produces a
 * single result computes in double and rounds (`Math.fround`). For + - * /
 * and sqrt this gives the same bits as computing in single; elsewhere the
 * result can differ from MATLAB's in the last place. A single scalar is a
 * 1x1 tensor (a JS number is always double).
 */

import { allocFloat64Array } from "./alloc.js";
import {
  type RuntimeValue,
  RuntimeTensor,
  RuntimeComplexNumber,
  isRuntimeTensor,
  withElementClass,
} from "./types.js";

/** Whether `v` is a single array. */
export function isSingle(v: unknown): v is RuntimeTensor {
  return v instanceof RuntimeTensor && v.cls === "single";
}

/** Whether any of `args` is a single array. */
export function hasSingleArg(args: readonly unknown[]): boolean {
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a instanceof RuntimeTensor && a.cls === "single") return true;
  }
  return false;
}

/** Round every element to single precision, in place. A no-op on values
 *  that are already single, so it is safe on storage shared with an
 *  input. */
export function froundInPlace(a: Float64Array): void {
  for (let i = 0; i < a.length; i++) a[i] = Math.fround(a[i]);
}

/** A copy of `a` rounded to single precision. */
export function froundCopy(a: Float64Array): Float64Array {
  const out = allocFloat64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.fround(a[i]);
  return out;
}

/** A single 1x1 array holding `re + im*i` (rounded). */
export function singleScalar(re: number, im = 0): RuntimeTensor {
  return new RuntimeTensor(
    Float64Array.of(Math.fround(re)),
    [1, 1],
    im !== 0 ? Float64Array.of(Math.fround(im)) : undefined,
    "single"
  );
}

/** A single copy of real or complex data (rounded; inputs untouched). */
export function singleTensor(
  data: Float64Array,
  shape: number[],
  imag?: Float64Array
): RuntimeTensor {
  return new RuntimeTensor(
    froundCopy(data),
    shape.slice(),
    imag && froundCopy(imag),
    "single"
  );
}

/**
 * The result of an operation that has a single operand, as single: a
 * double number, complex number or tensor is rounded to single; a single
 * tensor is re-rounded in place (a builtin may have built it from `v.cls`
 * around computed values); anything else (logical, char, cell, ...) is
 * returned as it is. A double tensor is rounded in place unless it is (or
 * wraps the storage of) one of the operation's `operands`; any other
 * result of an operation is fresh.
 */
export function toSingleResult(
  v: RuntimeValue,
  operands: readonly unknown[]
): RuntimeValue {
  if (typeof v === "number") return singleScalar(v);
  if (typeof v !== "object") return v;
  if (v instanceof RuntimeComplexNumber) return singleScalar(v.re, v.im);
  if (!isRuntimeTensor(v)) return v;
  if (v.cls === "single") {
    froundInPlace(v.data);
    if (v.imag) froundInPlace(v.imag);
    return v;
  }
  if (v.cls !== "double") return v;
  if (
    operands.some(
      o => o === v || (o instanceof RuntimeTensor && o.data === v.data)
    )
  )
    return singleTensor(v.data, v.shape, v.imag);
  froundInPlace(v.data);
  if (v.imag) froundInPlace(v.imag);
  return withElementClass(v, "single");
}

/** A single array as double: a 1x1 array becomes a number (or complex
 *  number); otherwise a double copy. */
export function singleToDouble(t: RuntimeTensor): RuntimeValue {
  if (t.data.length === 1 && t.shape.length === 2) {
    const im = t.imag ? t.imag[0] : 0;
    return im !== 0 ? new RuntimeComplexNumber(t.data[0], im) : t.data[0];
  }
  return new RuntimeTensor(
    allocFloat64Array(t.data),
    t.shape.slice(),
    t.imag && allocFloat64Array(t.imag),
    "double"
  );
}

/** Element `k` of a tensor as a scalar value of its class: a number, a
 *  logical, a complex number, or a single 1x1 array. */
export function tensorElement(t: RuntimeTensor, k: number): RuntimeValue {
  const im = t.imag !== undefined ? t.imag[k] : 0;
  if (t.cls === "single") return singleScalar(t.data[k], im);
  if (im !== 0) return new RuntimeComplexNumber(t.data[k], im);
  if (t.cls === "logical") return t.data[k] !== 0;
  return t.data[k];
}
