/**
 * Conversion of parsed JSON to MATLAB values, as jsondecode does it
 * (shared by jsondecode, webread and the uihtml data bridge).
 */

import { type RuntimeValue } from "./types.js";
import { RTV } from "./constructors.js";
import { allocFloat64Array } from "./alloc.js";

function makeValidFieldName(name: string): string {
  // Replace invalid characters with underscores
  let valid = name.replace(/[^a-zA-Z0-9_]/g, "_");
  // Ensure it starts with a letter
  if (!/^[a-zA-Z]/.test(valid)) valid = "x" + valid;
  // Truncate to reasonable length
  if (valid.length > 63) valid = valid.slice(0, 63);
  return valid;
}

export function convertJsonValue(val: unknown): RuntimeValue {
  if (val === null) return NaN;
  if (typeof val === "boolean") return val;
  if (typeof val === "number") return val;
  if (typeof val === "string") return RTV.char(val);
  if (Array.isArray(val)) return convertJsonArray(val);
  if (typeof val === "object")
    return convertJsonObject(val as Record<string, unknown>);
  return NaN;
}

function convertJsonObject(obj: Record<string, unknown>): RuntimeValue {
  const fields: Record<string, RuntimeValue> = {};
  for (const [key, value] of Object.entries(obj)) {
    fields[makeValidFieldName(key)] = convertJsonValue(value);
  }
  return RTV.struct(fields);
}

function convertJsonArray(arr: unknown[]): RuntimeValue {
  if (arr.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);

  // Classify elements
  let allBooleans = true;
  let allNumbers = true;
  let allStrings = true;
  let allObjects = true;

  for (const el of arr) {
    if (typeof el !== "boolean" && el !== null) allBooleans = false;
    if (typeof el !== "number" && el !== null) allNumbers = false;
    if (typeof el !== "string") allStrings = false;
    if (typeof el !== "object" || el === null || Array.isArray(el))
      allObjects = false;
  }

  // Booleans also count as numbers in JSON, so prioritize booleans check
  // but only if ALL are booleans (not mixed with numbers)
  if (allBooleans && !allNumbers) {
    // Array of booleans → logical column vector
    const data = allocFloat64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      data[i] = arr[i] === null ? 0 : arr[i] ? 1 : 0;
    }
    const t = RTV.tensor(data, [arr.length, 1]);
    t._isLogical = true;
    return t;
  }

  if (allNumbers) {
    // Array of numbers → double column vector (null → NaN)
    const data = allocFloat64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      data[i] = arr[i] === null ? NaN : (arr[i] as number);
    }
    return RTV.tensor(data, [arr.length, 1]);
  }

  // All booleans (with no nulls that would make allNumbers true too)
  if (allBooleans) {
    const data = allocFloat64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      data[i] = arr[i] ? 1 : 0;
    }
    const t = RTV.tensor(data, [arr.length, 1]);
    t._isLogical = true;
    return t;
  }

  if (allStrings) {
    // Array of strings → cell array of char vectors
    const cells = arr.map(el => RTV.char(el as string));
    return RTV.cell(cells, [arr.length, 1]);
  }

  if (allObjects) {
    // Check if all objects have the same field names
    const objs = arr as Record<string, unknown>[];
    const fieldSets = objs.map(o =>
      Object.keys(o).map(makeValidFieldName).sort().join("\0")
    );
    const allSameFields = fieldSets.every(s => s === fieldSets[0]);

    if (allSameFields && fieldSets[0].length > 0) {
      // Same fields → struct array
      const fieldNames = Object.keys(objs[0]).map(makeValidFieldName);
      const elements = objs.map(o => {
        const fields = new Map<string, RuntimeValue>();
        for (const [key, value] of Object.entries(o)) {
          fields.set(makeValidFieldName(key), convertJsonValue(value));
        }
        return RTV.struct(fields);
      });
      return RTV.structArray(fieldNames, elements, [elements.length, 1]);
    }

    // Different fields → cell array of structs
    const cells = objs.map(o => convertJsonObject(o));
    return RTV.cell(cells, [arr.length, 1]);
  }

  // Mixed types → cell array
  const cells = arr.map(el => convertJsonValue(el));
  return RTV.cell(cells, [arr.length, 1]);
}
