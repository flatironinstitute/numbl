/**
 * Dictionary builtins: dictionary, configureDictionary, keys, values,
 * numEntries, isConfigured, isKey, entries, insert, lookup, remove, types.
 */

import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeDictionary,
  kstr,
  type RuntimeChar,
  type RuntimeValue,
  type RuntimeDictionary,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { builtin, type Arity } from "./builtin.js";
import { builtinError } from "./args.js";
import { undefinedFunctionError, unsupportedFormError } from "./text-args.js";

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

/** Return the MATLAB type name string for a runtime value. */
function typeLabel(v: RuntimeValue): string {
  if (isRuntimeNumber(v)) return "double";
  if (isRuntimeLogical(v)) return "logical";
  if (isRuntimeString(v)) return "string";
  if (isRuntimeChar(v)) return "string"; // char keys auto-promote to string
  if (isRuntimeTensor(v)) return v._isLogical ? "logical" : "double";
  if (isRuntimeCell(v)) return "cell";
  if (isRuntimeDictionary(v)) return "dictionary";
  return kstr(v);
}

// ── Type enforcement / auto-conversion ───────────────────────────────────

/** Convert a key to the configured key type, or throw. */
function coerceKey(v: RuntimeValue, keyType: string): RuntimeValue {
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
function expandKeys(k: RuntimeValue): RuntimeValue[] {
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
    return RTV.tensor(data, [results.length, 1]);
  }
  if (d.valueType === "string") {
    return RTV.cell(results, [results.length, 1]);
  }
  return RTV.cell(results, [results.length, 1]);
}

// ── Builtins ─────────────────────────────────────────────────────────────

function oddNameValueError(): RuntimeError {
  return builtinError(
    "MATLAB:dictionary:OddNumberNVPairs",
    "Incorrect number of arguments. Each parameter name must be followed by a corresponding value."
  );
}

/** Define a dictionary method: the first argument must be a dictionary. */
function dictMethod(
  name: string,
  nargin: Arity,
  apply: (
    d: RuntimeDictionary,
    args: RuntimeValue[],
    nargout: number
  ) => RuntimeValue | RuntimeValue[],
  nargout?: Arity
): void {
  builtin({
    name,
    nargin,
    nargout,
    apply: (args, n) => {
      const d = args[0];
      if (!isRuntimeDictionary(d)) throw undefinedFunctionError(name, d);
      return apply(d, args, n);
    },
  });
}

// dictionary(), dictionary(keys, values), dictionary(k1,v1,...,kN,vN)
builtin({
  name: "dictionary",
  apply: args => {
    if (args.length % 2 !== 0)
      throw builtinError(
        "MATLAB:dictionary:IncorrectNumberOfInputs",
        "Incorrect number of input arguments. Each key must be followed by a value."
      );
    let d = RTV.dictionary();
    for (let i = 0; i < args.length; i += 2) {
      d = dictInsert(d, args[i], args[i + 1]);
    }
    return d;
  },
});

// configureDictionary(keyType, valueType)
builtin({
  name: "configureDictionary",
  nargin: 2,
  apply: args => {
    const typeName = (v: RuntimeValue): string => {
      if (isRuntimeString(v)) return v;
      if (isRuntimeChar(v)) return v.value;
      throw builtinError(
        "MATLAB:string:MustBeStringScalarOrCharacterVector",
        "Argument must be a text scalar."
      );
    };
    return RTV.dictionary(new Map(), typeName(args[0]), typeName(args[1]));
  },
});

// keys(d)
dictMethod("keys", [1, 2], (d, args) => {
  if (args.length > 1) throw unsupportedFormError("keys", "keys(d, format)");
  const k: RuntimeValue[] = [];
  for (const entry of d.entries.values()) k.push(entry.key);
  if (k.length === 0) {
    if (d.keyType === "string") return RTV.cell([], [0, 1]);
    return RTV.tensor([], [0, 1]);
  }
  if (d.keyType === "double" || d.keyType === "logical") {
    const data = k.map(v =>
      isRuntimeNumber(v) ? v : isRuntimeLogical(v) ? (v ? 1 : 0) : 0
    );
    return RTV.tensor(data, [k.length, 1]);
  }
  // string keys or other
  return RTV.cell(k, [k.length, 1]);
});

// values(d)
dictMethod("values", [1, 2], (d, args) => {
  if (args.length > 1)
    throw unsupportedFormError("values", "the second argument of values");
  const v: RuntimeValue[] = [];
  for (const entry of d.entries.values()) v.push(entry.value);
  if (v.length === 0) {
    if (d.valueType === "double") return RTV.tensor([], [0, 1]);
    return RTV.cell([], [0, 1]);
  }
  if (d.valueType === "double" || d.valueType === "logical") {
    const data = v.map(x =>
      isRuntimeNumber(x) ? x : isRuntimeLogical(x) ? (x ? 1 : 0) : 0
    );
    return RTV.tensor(data, [v.length, 1]);
  }
  return RTV.cell(v, [v.length, 1]);
});

// numEntries(d)
dictMethod("numEntries", 1, d => d.entries.size);

// isConfigured(d)
dictMethod(
  "isConfigured",
  1,
  d => d.keyType !== undefined && d.valueType !== undefined
);

// isKey(d, key)
dictMethod("isKey", 2, (d, args) => {
  const keys = expandKeys(args[1]);
  if (keys.length === 1) {
    try {
      const coerced = d.keyType ? coerceKey(keys[0], d.keyType) : keys[0];
      return d.entries.has(hashKey(coerced));
    } catch {
      return false;
    }
  }
  // Return logical tensor for multiple keys
  const data = keys.map(k => {
    try {
      const coerced = d.keyType ? coerceKey(k, d.keyType) : k;
      return d.entries.has(hashKey(coerced)) ? 1 : 0;
    } catch {
      return 0;
    }
  });
  const t = RTV.tensor(data, [1, data.length]);
  t._isLogical = true;
  return t;
});

// entries(d). MATLAB returns a table; numbl returns a struct with keys and
// values fields, or [keys, values] with two outputs.
dictMethod(
  "entries",
  [1, 2],
  (d, args, nargout) => {
    if (args.length > 1)
      throw unsupportedFormError("entries", "entries(d, format)");
    const k: RuntimeValue[] = [];
    const v: RuntimeValue[] = [];
    for (const entry of d.entries.values()) {
      k.push(entry.key);
      v.push(entry.value);
    }
    if (nargout >= 2) {
      // [keys, values] = entries(d)
      const keysOut =
        d.keyType === "double" || d.keyType === "logical"
          ? RTV.tensor(
              k.map(x => (isRuntimeNumber(x) ? x : 0)),
              [k.length, 1]
            )
          : RTV.cell(k, [k.length, 1]);
      const valsOut =
        d.valueType === "double" || d.valueType === "logical"
          ? RTV.tensor(
              v.map(x => (isRuntimeNumber(x) ? x : 0)),
              [v.length, 1]
            )
          : RTV.cell(v, [v.length, 1]);
      return [keysOut, valsOut];
    }
    // Single output: struct with keys and values fields
    return RTV.struct({
      keys: RTV.cell(k, [k.length, 1]),
      values: RTV.cell(v, [v.length, 1]),
    });
  },
  2
);

// types(d): a 1x2 cell {keyType, valueType}, or [keyType, valueType] with
// two outputs.
dictMethod(
  "types",
  1,
  (d, _args, nargout) => {
    const kt = d.keyType ?? "unset";
    const vt = d.valueType ?? "unset";
    if (nargout >= 2) return [RTV.string(kt), RTV.string(vt)];
    return RTV.cell([kt, vt], [1, 2]);
  },
  2
);

// insert(d, keys, values)
dictMethod("insert", [3, Infinity], (d, args) => {
  if (args.length > 3) {
    if ((args.length - 3) % 2 !== 0) throw oddNameValueError();
    throw unsupportedFormError("insert", "name-value options");
  }
  return dictInsert(d, args[1], args[2]);
});

// lookup(d, keys) or lookup(d, keys, 'FallbackValue', fb)
dictMethod("lookup", [2, Infinity], (d, args) => {
  if ((args.length - 2) % 2 !== 0) throw oddNameValueError();
  let fallback: RuntimeValue | undefined;
  for (let i = 2; i < args.length; i += 2) {
    const optName = isRuntimeChar(args[i])
      ? (args[i] as RuntimeChar).value
      : isRuntimeString(args[i])
        ? (args[i] as string)
        : "";
    if (optName.toLowerCase() !== "fallbackvalue") {
      throw new RuntimeError(`Unrecognized option '${optName}' for lookup`);
    }
    fallback = args[i + 1];
  }
  return dictLookup(d, args[1], fallback);
});

// remove(d, keys)
dictMethod("remove", 2, (d, args) => dictRemove(d, args[1]));
