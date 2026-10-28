/**
 * Input types of an executor unit, in the JIT compiler's `Type` lattice.
 *
 * Every dispatch types its inputs (`typeOfValue`), widens them against the
 * previous dispatch (`widenAgainst`) and keys the compile cache on them
 * (`typeKey`), so this runs on every interpreted call of a user function.
 * To keep that cheap, the types are interned: a value maps through a short
 * signature string to a shared, frozen `Type`, the canonical key of an
 * interned type is computed once, and widening is memoized on the pair of
 * interned types. In the steady state a dispatch builds no types at all.
 */

import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeStruct,
  isRuntimeTensor,
  type RuntimeValue,
} from "../runtime/types.js";
import {
  canonicalizeType,
  scalarComplex,
  scalarDouble,
  scalarLogical,
  signFromNumber,
  structType,
  tensorComplex,
  tensorDouble,
  widenArgType,
  UNKNOWN,
  type Sign,
  type Type,
} from "../jit/index.js";

/** Bound on each intern table; a table is cleared when it fills (tensor
 *  shapes, say, can vary without limit). */
const MAX_INTERNED = 4096;

/** Interned types by value signature (see `signature`). */
const bySig = new Map<string, Type>();
/** Interned types by canonical key (widening results). */
const byKey = new Map<string, Type>();
/** The canonical key of each interned type. */
const keys = new WeakMap<Type, string>();
/** Memoized `widenArgType(a, prev)` on interned types (null: no widening). */
const widened = new WeakMap<Type, WeakMap<Type, Type | null>>();

function deepFreeze<T>(t: T): T {
  if (t !== null && typeof t === "object" && !Object.isFrozen(t)) {
    Object.freeze(t);
    for (const v of Object.values(t)) deepFreeze(v);
  }
  return t;
}

function remember<K>(table: Map<K, Type>, k: K, t: Type): Type {
  if (table.size >= MAX_INTERNED) table.clear();
  deepFreeze(t);
  table.set(k, t);
  return t;
}

/** The canonical key (`canonicalizeType`) of a type, computed once per
 *  interned type. */
export function typeKey(t: Type): string {
  let k = keys.get(t);
  if (k === undefined) {
    k = canonicalizeType(t);
    keys.set(t, k);
  }
  return k;
}

/** A scalar argument's sign. Zero is `nonneg` rather than `zero`: a counter
 *  or time that starts at 0 then goes positive would otherwise compile a
 *  spec for zero, one for nonneg, and only then settle. */
function scalarSign(v: number): Sign {
  const s = signFromNumber(v);
  return s === "zero" ? "nonneg" : s;
}

/**
 * A string that determines the type of a runtime value, or null for a value
 * no backend takes (a cell, object, sparse matrix, function handle, ...).
 */
function signature(v: RuntimeValue): string | null {
  if (isRuntimeNumber(v)) return scalarSign(v);
  if (isRuntimeLogical(v)) return "b";
  if (isRuntimeString(v)) return "S";
  if (isRuntimeChar(v)) return "C";
  if (isRuntimeComplexNumber(v)) return "z";
  if (isRuntimeTensor(v)) {
    const tag = v.imag ? "tz" : v._isLogical ? "tb" : "t";
    return v.shape.length >= 2
      ? `${tag}${v.shape.join("x")}`
      : `${tag}1x${v.shape.join("x")}`;
  }
  if (isRuntimeStruct(v)) {
    let s = "s{";
    for (const [name, fv] of v.fields) {
      const f = signature(fv);
      if (f === null) return null;
      s += `${name}:${f};`;
    }
    return s + "}";
  }
  return null;
}

function buildType(v: RuntimeValue): Type {
  if (isRuntimeNumber(v)) return scalarDouble(scalarSign(v));
  if (isRuntimeLogical(v)) return scalarLogical();
  if (isRuntimeString(v)) return { kind: "String" };
  if (isRuntimeChar(v)) return { kind: "Char" };
  if (isRuntimeComplexNumber(v)) return scalarComplex();
  if (isRuntimeTensor(v)) {
    const shape = v.shape.length >= 2 ? v.shape.slice() : [1, ...v.shape];
    if (v.imag) return tensorComplex(shape);
    const t = tensorDouble(shape);
    // A logical tensor used as an index is a mask, not a gather.
    if (v._isLogical) t.elem = "logical";
    return t;
  }
  // A struct (the `p.tol` parameter idiom). Field order is the struct's
  // own, as in the compiler's StructType; `signature` vetted the fields.
  const fields: { name: string; ty: Type }[] = [];
  for (const [name, fv] of (v as Extract<RuntimeValue, { kind: "struct" }>)
    .fields)
    fields.push({ name, ty: typeOfValue(fv) });
  return structType(fields);
}

/**
 * The compiler `Type` the JIT specializes on for a runtime value (shared
 * and frozen), or `UNKNOWN` for a value no backend takes as an input;
 * `propose` then declines.
 *
 * A real scalar carries its sign but never its value: specializing on
 * values would compile one spec per call. Tensors carry their shape but
 * not their sign, which would cost a pass over the data on every dispatch.
 */
export function typeOfValue(value: unknown): Type {
  const v = value as RuntimeValue;
  const sig = signature(v);
  if (sig === null) return UNKNOWN;
  return bySig.get(sig) ?? remember(bySig, sig, buildType(v));
}

/** `widenArgType` on interned types, memoized; returns an interned type,
 *  or null when the two should not share a specialization. */
function widenInterned(a: Type, prev: Type): Type | null {
  let memo = widened.get(a);
  if (memo === undefined) {
    memo = new WeakMap();
    widened.set(a, memo);
  }
  let w = memo.get(prev);
  if (w === undefined) {
    const u = widenArgType(a, prev);
    if (u === undefined) w = null;
    else {
      const k = typeKey(u);
      w = byKey.get(k) ?? remember(byKey, k, u);
      keys.set(w, k);
    }
    memo.set(prev, w);
  }
  return w;
}

/**
 * Progressive type widening: in place, widen each entry of `types`
 * against the corresponding entry of `prev` (`widenArgType`). No-op when
 * the arities differ (a different specialization, no widening). An entry
 * that should not widen keeps this call's concrete type, so a fresh,
 * specific spec gets built rather than one poisoned by a mismatched
 * earlier call.
 */
export function widenAgainst(
  types: Type[],
  prev: readonly Type[] | undefined
): void {
  if (!prev || prev.length !== types.length) return;
  for (let i = 0; i < types.length; i++) {
    const a = types[i];
    const p = prev[i];
    if (a === p || a.kind === "Unknown" || p.kind === "Unknown") continue;
    const w = widenInterned(a, p);
    if (w !== null) types[i] = w;
  }
}
