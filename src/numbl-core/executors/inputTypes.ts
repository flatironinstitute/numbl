/**
 * Input types of an executor unit, in the JIT compiler's `Type` lattice.
 *
 * Every dispatch types its inputs (`typeOfValue`), widens them against the
 * earlier dispatches (`widenAgainst`) and keys the compile cache on them
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
    const tag = v.imag ? "tz" : v.cls === "logical" ? "tb" : "t";
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
    if (v.cls === "logical") t.elem = "logical";
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
 * The input types a unit has been dispatched with, per input name (a
 * variable name, or a call's parameter position): a few type _families_,
 * most recent first, no two of which widen against each other (a scalar
 * and a tensor, say, or a 2-D and a 3-D tensor).
 *
 * Keeping one entry per family, rather than only the previous dispatch's
 * type, lets an input whose type alternates between families converge: a
 * `find` result that is sometimes 1x1 and sometimes kx1 widens each kx1
 * against the earlier kx1 types, instead of compiling a new spec for every
 * k that follows a 1x1. Keying by name keeps the history when a loop's
 * input set changes between dispatches (a variable that exists only after
 * some branch has run).
 */
export type WideningHistory = ReadonlyMap<string, readonly Type[]>;

/** Families kept per input. */
const MAX_FAMILIES = 4;

/**
 * Progressive type widening: in place, widen each entry of `types` (the
 * input named `names[i]`) against the first of that input's families in
 * `prev` it widens with (`widenArgType`). An entry that widens with no
 * family keeps this call's concrete type, so a fresh, specific spec gets
 * built rather than one poisoned by a mismatched earlier call.
 */
export function widenAgainst(
  types: Type[],
  names: readonly string[],
  prev: WideningHistory | undefined
): void {
  if (!prev) return;
  for (let i = 0; i < types.length; i++) {
    const a = types[i];
    if (a.kind === "Unknown") continue;
    const fams = prev.get(names[i]);
    if (!fams) continue;
    for (const p of fams) {
      if (a === p) break;
      if (p.kind === "Unknown") continue;
      const w = widenInterned(a, p);
      if (w !== null) {
        types[i] = w;
        break;
      }
    }
  }
}

/**
 * The history after a dispatch with (widened) `types`: each input's type
 * moves to the front of its families, replacing the families it widens
 * with; inputs absent from this dispatch keep theirs. Returns `prev`
 * itself when nothing changed (the steady state).
 */
export function recordWidening(
  prev: WideningHistory | undefined,
  names: readonly string[],
  types: readonly Type[]
): WideningHistory {
  let next: Map<string, readonly Type[]> | null = prev ? null : new Map();
  for (let i = 0; i < types.length; i++) {
    const t = types[i];
    const fams = prev?.get(names[i]);
    if (fams?.[0] === t) continue;
    const kept: Type[] = [t];
    for (const p of fams ?? []) {
      if (kept.length >= MAX_FAMILIES) break;
      if (p === t) continue;
      if (t.kind !== "Unknown" && p.kind !== "Unknown") {
        if (widenInterned(t, p) !== null) continue;
      }
      kept.push(p);
    }
    next ??= new Map(prev);
    next.set(names[i], kept);
  }
  return next ?? prev!;
}

const positionNames: (readonly string[])[] = [];

/** The names a call's parameters are widened under: their positions. */
export function paramPositions(n: number): readonly string[] {
  return (positionNames[n] ??= Array.from({ length: n }, (_, i) => `#${i}`));
}
