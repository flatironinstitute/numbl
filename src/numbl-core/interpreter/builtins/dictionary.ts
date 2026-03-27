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
  type RuntimeValue,
  type RuntimeDictionary,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { registerIBuiltin, type IBuiltinResolution } from "./types.js";

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
  keysArg: RuntimeValue
): RuntimeValue {
  const keys = expandKeys(keysArg);
  const results: RuntimeValue[] = [];
  for (const k of keys) {
    const coerced = d.keyType ? coerceKey(k, d.keyType) : k;
    const h = hashKey(coerced);
    const entry = d.entries.get(h);
    if (!entry) throw new RuntimeError("Key not found in dictionary");
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

// dictionary()
registerIBuiltin({
  name: "dictionary",
  resolve: (argTypes): IBuiltinResolution | null => {
    // dictionary() — no args
    if (argTypes.length === 0) {
      return {
        outputTypes: [{ kind: "dictionary" }],
        apply: () => RTV.dictionary(),
      };
    }
    // dictionary(keys, values) or dictionary(k1,v1,...,kN,vN)
    if (argTypes.length >= 2) {
      return {
        outputTypes: [{ kind: "dictionary" }],
        apply: args => {
          let d = RTV.dictionary();
          if (argTypes.length === 2) {
            // dictionary(keys, values)
            d = dictInsert(d, args[0], args[1]);
          } else {
            // dictionary(k1,v1,...,kN,vN)
            if (args.length % 2 !== 0)
              throw new RuntimeError("dictionary requires key-value pairs");
            for (let i = 0; i < args.length; i += 2) {
              d = dictInsert(d, args[i], args[i + 1]);
            }
          }
          return d;
        },
      };
    }
    return null;
  },
});

// configureDictionary(keyType, valueType)
registerIBuiltin({
  name: "configureDictionary",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [{ kind: "dictionary" }],
      apply: args => {
        const kt = isRuntimeString(args[0])
          ? args[0]
          : isRuntimeChar(args[0])
            ? args[0].value
            : null;
        const vt = isRuntimeString(args[1])
          ? args[1]
          : isRuntimeChar(args[1])
            ? args[1].value
            : null;
        if (!kt || !vt)
          throw new RuntimeError(
            "configureDictionary requires string type names"
          );
        return RTV.dictionary(new Map(), kt, vt);
      },
    };
  },
});

// keys(d)
registerIBuiltin({
  name: "keys",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 1 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
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
      },
    };
  },
});

// values(d)
registerIBuiltin({
  name: "values",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 1 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
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
      },
    };
  },
});

// numEntries(d)
registerIBuiltin({
  name: "numEntries",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 1 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
        return d.entries.size;
      },
    };
  },
});

// isConfigured(d)
registerIBuiltin({
  name: "isConfigured",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 1 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
        return d.keyType !== undefined && d.valueType !== undefined;
      },
    };
  },
});

// isKey(d, key)
registerIBuiltin({
  name: "isKey",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 2 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
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
      },
    };
  },
});

// entries(d)
registerIBuiltin({
  name: "entries",
  resolve: (argTypes, nargout): IBuiltinResolution | null => {
    if (argTypes.length !== 1 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes:
        nargout >= 2
          ? [{ kind: "unknown" }, { kind: "unknown" }]
          : [{ kind: "unknown" }],
      apply: (args, nargout) => {
        const d = args[0] as RuntimeDictionary;
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
    };
  },
});

// types(d)
registerIBuiltin({
  name: "types",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 1 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
        const kt = d.keyType ?? "unset";
        const vt = d.valueType ?? "unset";
        return RTV.cell([kt, vt], [1, 2]);
      },
    };
  },
});

// insert(d, keys, values)
registerIBuiltin({
  name: "insert",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 3 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "dictionary" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
        return dictInsert(d, args[1], args[2]);
      },
    };
  },
});

// lookup(d, keys)
registerIBuiltin({
  name: "lookup",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 2 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
        return dictLookup(d, args[1]);
      },
    };
  },
});

// remove(d, keys)
registerIBuiltin({
  name: "remove",
  resolve: (argTypes): IBuiltinResolution | null => {
    if (argTypes.length !== 2 || argTypes[0].kind !== "dictionary") return null;
    return {
      outputTypes: [{ kind: "dictionary" }],
      apply: args => {
        const d = args[0] as RuntimeDictionary;
        return dictRemove(d, args[1]);
      },
    };
  },
});
