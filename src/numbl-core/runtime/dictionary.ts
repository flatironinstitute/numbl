/**
 * Dictionary storage operations (key hashing, type coercion, insert,
 * lookup, remove), shared by the dictionary builtins and indexing.
 */

import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeCell,
  type RuntimeValue,
  type RuntimeDictionary,
} from "./types.js";
import { kstr, valueClass } from "./kinds.js";
import { RTV } from "./constructors.js";
import { RuntimeError } from "./error.js";

// ── Key hashing ──────────────────────────────────────────────────────────

/** Produce a unique string key from a RuntimeValue for Map lookup. */
export function hashKey(v: RuntimeValue): string {
  if (isRuntimeNumber(v)) return `n:${v}`;
  if (isRuntimeLogical(v)) return `b:${v ? 1 : 0}`;
  if (isRuntimeString(v)) return `s:${v}`;
  if (isRuntimeChar(v)) return `s:${v.value}`;
  if (isRuntimeTensor(v) && v.data.length === 1) return `n:${v.data[0]}`;
  throw new RuntimeError(
    `Dictionary keys of type '${kstr(v)}' are not supported`
  );
}

// ── Type label ───────────────────────────────────────────────────────────

/** The MATLAB class of a key or value; char auto-promotes to string. */
function typeLabel(v: RuntimeValue): string {
  return isRuntimeChar(v) ? "string" : valueClass(v);
}

// ── Type enforcement / auto-conversion ───────────────────────────────────

/** Convert a key to the configured key type, or throw. */
export function coerceKey(v: RuntimeValue, keyType: string): RuntimeValue {
  // char → string promotion
  if (isRuntimeChar(v) && keyType === "string") return v.value;
  const actual = typeLabel(v);
  if (actual === keyType) return v;
  // numeric coercions
  if (keyType === "double") {
    if (isRuntimeLogical(v)) return v ? 1 : 0;
    if (isRuntimeString(v)) {
      const n = Number(v);
      if (!isNaN(n)) return n;
    }
  }
  throw new RuntimeError(
    `Unable to use a '${actual}' as a dictionary key of type '${keyType}'`
  );
}

/** Convert a value to the configured value type, or throw. */
function coerceValue(v: RuntimeValue, valueType: string): RuntimeValue {
  if (isRuntimeChar(v) && valueType === "string") return v.value;
  const actual = typeLabel(v);
  if (actual === valueType) return v;
  if (valueType === "double") {
    if (isRuntimeLogical(v)) return v ? 1 : 0;
    if (isRuntimeString(v)) {
      const n = Number(v);
      if (!isNaN(n)) return n;
    }
  }
  if (valueType === "cell") {
    return RTV.cell([v], [1, 1]);
  }
  throw new RuntimeError(
    `Unable to use a '${actual}' as a dictionary value of type '${valueType}'`
  );
}

// ── Helpers for expanding keys/values arrays ─────────────────────────────

/** Expand a key argument into an array of individual keys. */
export function expandKeys(k: RuntimeValue): RuntimeValue[] {
  if (isRuntimeTensor(k)) {
    const out: RuntimeValue[] = [];
    for (let i = 0; i < k.data.length; i++) out.push(k.data[i]);
    return out;
  }
  if (isRuntimeCell(k)) return k.data;
  return [k];
}

/** Expand a value argument into an array of individual values. */
function expandValues(v: RuntimeValue, count: number): RuntimeValue[] {
  if (isRuntimeTensor(v) && v.data.length === count) {
    const out: RuntimeValue[] = [];
    for (let i = 0; i < v.data.length; i++) out.push(v.data[i]);
    return out;
  }
  if (isRuntimeCell(v) && v.data.length === count) return v.data;
  // scalar expansion
  if (count > 1 && !isRuntimeTensor(v) && !isRuntimeCell(v)) {
    return Array(count).fill(v);
  }
  if (count === 1) return [v];
  if (isRuntimeTensor(v) && v.data.length !== count) {
    throw new RuntimeError(
      `Number of values (${v.data.length}) does not match number of keys (${count})`
    );
  }
  return [v];
}

// ── Core insertion logic ─────────────────────────────────────────────────

/** Insert a single key-value pair without expanding arrays. */
export function dictInsertSingle(
  d: RuntimeDictionary,
  key: RuntimeValue,
  value: RuntimeValue
): RuntimeDictionary {
  const entries = new Map(d.entries);
  let keyType = d.keyType;
  let valueType = d.valueType;
  if (!keyType) keyType = typeLabel(key);
  if (!valueType) valueType = typeLabel(value);
  const k = coerceKey(key, keyType);
  const v = coerceValue(value, valueType);
  entries.set(hashKey(k), { key: k, value: v });
  return RTV.dictionary(entries, keyType, valueType);
}

/** Insert key-value pairs into a dictionary, returning a new dictionary. */
export function dictInsert(
  d: RuntimeDictionary,
  keysArg: RuntimeValue,
  valuesArg: RuntimeValue
): RuntimeDictionary {
  const keys = expandKeys(keysArg);
  const values = expandValues(valuesArg, keys.length);
  const entries = new Map(d.entries);
  let keyType = d.keyType;
  let valueType = d.valueType;

  for (let i = 0; i < keys.length; i++) {
    let k = keys[i];
    let v = values[i];
    // Auto-configure on first insert
    if (!keyType) keyType = typeLabel(k);
    if (!valueType) valueType = typeLabel(v);
    k = coerceKey(k, keyType);
    v = coerceValue(v, valueType);
    entries.set(hashKey(k), { key: k, value: v });
  }
  return RTV.dictionary(entries, keyType, valueType);
}

/** Remove keys from a dictionary, returning a new dictionary. */
export function dictRemove(
  d: RuntimeDictionary,
  keysArg: RuntimeValue
): RuntimeDictionary {
  const keys = expandKeys(keysArg);
  const entries = new Map(d.entries);
  for (const k of keys) {
    const coerced = d.keyType ? coerceKey(k, d.keyType) : k;
    const h = hashKey(coerced);
    if (!entries.has(h)) throw new RuntimeError("Key not found in dictionary");
    entries.delete(h);
  }
  return RTV.dictionary(entries, d.keyType, d.valueType);
}

/** Lookup values by keys. Returns single value or tensor/cell array. */
export function dictLookup(
  d: RuntimeDictionary,
  keysArg: RuntimeValue,
  fallbackValue?: RuntimeValue
): RuntimeValue {
  const keys = expandKeys(keysArg);
  const results: RuntimeValue[] = [];
  for (const k of keys) {
    const coerced = d.keyType ? coerceKey(k, d.keyType) : k;
    const h = hashKey(coerced);
    const entry = d.entries.get(h);
    if (!entry) {
      if (fallbackValue !== undefined) {
        results.push(fallbackValue);
        continue;
      }
      throw new RuntimeError("Key not found in dictionary");
    }
    results.push(entry.value);
  }
  if (results.length === 1) return results[0];
  // Return array of results
  if (d.valueType === "double" || d.valueType === "logical") {
    const data = results.map(r =>
      isRuntimeNumber(r) ? r : isRuntimeLogical(r) ? (r ? 1 : 0) : 0
    );
    return RTV.tensorFromArray(data, [results.length, 1]);
  }
  if (d.valueType === "string") {
    return RTV.cell(results, [results.length, 1]);
  }
  return RTV.cell(results, [results.length, 1]);
}
