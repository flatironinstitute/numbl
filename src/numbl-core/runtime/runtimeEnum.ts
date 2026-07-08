/**
 * Enumeration-class value semantics.
 *
 * An enumeration member is represented as a `RuntimeClassInstance` whose
 * `_enumMember` field holds the member name and whose `_builtinData` holds the
 * underlying (superclass) numeric value — e.g. for `classdef patchtype < uint32`
 * the member `tri (1)` is stored with `_enumMember = "tri"`, `_builtinData = 1`.
 *
 * MATLAB comparison rules (verified against R2025b):
 *   - enum == enum      → compare underlying values (inherited from the numeric
 *                         superclass); falls back to name equality for plain
 *                         enumerations that carry no superclass value.
 *   - enum == numeric   → compare underlying value to the number.
 *   - enum == char/str  → compare the member NAME to the text (a non-matching
 *                         name yields false, never an error), NOT an
 *                         element-wise char-code comparison.
 * Comparisons are element-wise with scalar broadcasting; arrays of members
 * produce a logical array the size of the member array.
 */

import { BinaryOperation } from "../lowering/index.js";
import { toNumber } from "./convert.js";
import { RTV } from "./constructors.js";
import { RuntimeError } from "./error.js";
import {
  type RuntimeValue,
  type RuntimeClassInstance,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeStringArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
} from "./types.js";

/** True if `v` is a single enumeration-member instance. */
export function isEnumInstance(v: RuntimeValue): v is RuntimeClassInstance {
  return isRuntimeClassInstance(v) && v._enumMember !== undefined;
}

/** True if `v` is an enumeration member or an array of enumeration members. */
export function isEnumValue(v: RuntimeValue): boolean {
  if (isEnumInstance(v)) return true;
  if (isRuntimeClassInstanceArray(v))
    return v.elements.some(e => e._enumMember !== undefined);
  return false;
}

/** Text value of a char / scalar string, or null if `v` is neither. */
function asText(v: RuntimeValue): string | null {
  if (isRuntimeChar(v)) return v.value;
  if (isRuntimeString(v)) return v;
  return null;
}

/** Compare one enumeration member `e` against a scalar operand `other`
 *  (another member, a number, or char/string text). */
export function enumScalarEquals(
  e: RuntimeClassInstance,
  other: RuntimeValue
): boolean {
  const text = asText(other);
  if (text !== null) return e._enumMember === text;
  if (isEnumInstance(other)) {
    if (e._builtinData !== undefined && other._builtinData !== undefined) {
      return toNumber(e._builtinData) === toNumber(other._builtinData);
    }
    return e._enumMember === other._enumMember;
  }
  if (e._builtinData !== undefined) {
    if (isRuntimeNumber(other)) return toNumber(e._builtinData) === other;
    if (isRuntimeLogical(other))
      return toNumber(e._builtinData) === (other ? 1 : 0);
  }
  return false;
}

/** An operand decomposed into scalar elements plus a 2-D shape. */
interface Operand {
  elems: RuntimeValue[];
  shape: [number, number];
}

/** Decompose a comparison operand into per-element RuntimeValues. Char/string
 *  scalars stay whole (they are matched against a member name, not per char).
 *  Returns null for kinds enum comparison does not handle. */
function decompose(v: RuntimeValue): Operand | null {
  if (isEnumInstance(v)) return { elems: [v], shape: [1, 1] };
  if (isRuntimeClassInstanceArray(v))
    return { elems: [...v.elements], shape: v.shape };
  if (isRuntimeChar(v)) return { elems: [v], shape: [1, 1] };
  if (isRuntimeString(v)) return { elems: [v], shape: [1, 1] };
  if (isRuntimeStringArray(v))
    return { elems: v.data.map(s => s as RuntimeValue), shape: v.shape };
  if (isRuntimeNumber(v) || isRuntimeLogical(v))
    return { elems: [v], shape: [1, 1] };
  if (isRuntimeTensor(v)) {
    const shape: [number, number] = [v.shape[0] ?? 0, v.shape[1] ?? 1];
    return { elems: Array.from(v.data, x => x as RuntimeValue), shape };
  }
  return null;
}

/** Equality of two scalar elements where at least one is an enum member. */
function elementEquals(x: RuntimeValue, y: RuntimeValue): boolean {
  if (isEnumInstance(x)) return enumScalarEquals(x, y);
  if (isEnumInstance(y)) return enumScalarEquals(y, x);
  return false;
}

/**
 * Element-wise `==` / `~=` for enumeration operands. Returns a logical result
 * (a boolean for scalars, a logical tensor for arrays), or null if the op is
 * not equality/inequality or neither operand is an enumeration value.
 */
export function enumEqualityOp(
  op: string,
  a: RuntimeValue,
  b: RuntimeValue
): RuntimeValue | null {
  if (op !== BinaryOperation.Equal && op !== BinaryOperation.NotEqual)
    return null;
  if (!isEnumValue(a) && !isEnumValue(b)) return null;

  const oa = decompose(a);
  const ob = decompose(b);
  if (!oa || !ob) return null;

  const na = oa.elems.length;
  const nb = ob.elems.length;
  let n: number;
  let shape: [number, number];
  if (na === nb) {
    n = na;
    shape = oa.shape;
  } else if (na === 1) {
    n = nb;
    shape = ob.shape;
  } else if (nb === 1) {
    n = na;
    shape = oa.shape;
  } else {
    throw new RuntimeError("Array dimensions must match for comparison");
  }

  const negate = op === BinaryOperation.NotEqual;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const eq = elementEquals(
      oa.elems[na === 1 ? 0 : i],
      ob.elems[nb === 1 ? 0 : i]
    );
    out[i] = (negate ? !eq : eq) ? 1 : 0;
  }
  if (n === 1) return RTV.logical(out[0] === 1);
  const t = RTV.tensor(out, [shape[0], shape[1]]);
  t._isLogical = true;
  return t;
}
