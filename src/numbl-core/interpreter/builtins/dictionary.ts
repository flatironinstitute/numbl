/**
 * Dictionary builtins: dictionary, configureDictionary, keys, values,
 * numEntries, isConfigured, isKey, entries, insert, lookup, remove, types.
 */

import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeDictionary,
  type RuntimeChar,
  type RuntimeValue,
  type RuntimeDictionary,
} from "../../runtime/types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { builtin, type Arity } from "./builtin.js";
import { builtinError } from "./args.js";
import { undefinedFunctionError, unsupportedFormError } from "./text-args.js";
import {
  coerceKey,
  dictInsert,
  dictLookup,
  dictRemove,
  expandKeys,
  hashKey,
} from "../../runtime/dictionary.js";

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
    // Values are stored as given (keys reject sparse themselves).
    accepts: { sparse: true },
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
  accepts: { sparse: true },
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
    return RTV.tensorFromArray([], [0, 1]);
  }
  if (d.keyType === "double" || d.keyType === "logical") {
    const data = k.map(v =>
      isRuntimeNumber(v) ? v : isRuntimeLogical(v) ? (v ? 1 : 0) : 0
    );
    return RTV.tensorFromArray(data, [k.length, 1]);
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
    if (d.valueType === "double") return RTV.tensorFromArray([], [0, 1]);
    return RTV.cell([], [0, 1]);
  }
  if (d.valueType === "double" || d.valueType === "logical") {
    const data = v.map(x =>
      isRuntimeNumber(x) ? x : isRuntimeLogical(x) ? (x ? 1 : 0) : 0
    );
    return RTV.tensorFromArray(data, [v.length, 1]);
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
  return RTV.logicalTensor(allocFloat64Array(data), [1, data.length]);
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
          ? RTV.tensorFromArray(
              k.map(x => (isRuntimeNumber(x) ? x : 0)),
              [k.length, 1]
            )
          : RTV.cell(k, [k.length, 1]);
      const valsOut =
        d.valueType === "double" || d.valueType === "logical"
          ? RTV.tensorFromArray(
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
