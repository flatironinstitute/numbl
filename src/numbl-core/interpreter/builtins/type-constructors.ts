/**
 * Type constructor builtins: double, logical, complex, cell, struct, full,
 * isfield, deal, func2str.
 */

import {
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeSparseMatrix,
  isRuntimeFunction,
  isRuntimeStringArray,
  isRuntimeClassInstanceArray,
  RuntimeTensor,
} from "../../runtime/types.js";
import type {
  RuntimeValue,
  RuntimeCell,
  RuntimeStruct,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber, toString } from "../../runtime/convert.js";
import { makeTensor, mkc } from "./types.js";
import { builtin } from "./builtin.js";
import { undefinedForTypeError } from "./value-class.js";
import { valueClass } from "../../runtime/kinds.js";
import { builtinError } from "./args.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { imagAllZero } from "../../helpers/effectively-real.js";
import { charToNumericTensor } from "../../runtime/indexing.js";
import { parseShapeArgs } from "../../helpers/shape-utils.js";
import { unparseExpr, type Expr } from "../../parser/index.js";

// ── Shared helpers ──────────────────────────────────────────────────────

/** MATLAB's "Conversion to <to> from <class> is not possible." */
function invalidConversion(to: string, v: RuntimeValue): RuntimeError {
  return builtinError(
    "MATLAB:invalidConversion",
    `Conversion to ${to} from ${valueClass(v)} is not possible.`
  );
}

/** The value underlying a built-in-derived class instance (enumeration
 *  member) or an array of them; other values unchanged. */
function underlyingValue(v: RuntimeValue): RuntimeValue {
  if (isRuntimeClassInstance(v) && v._builtinData !== undefined)
    return v._builtinData;
  return classInstanceArrayNumeric(v) ?? v;
}

// ── double ──────────────────────────────────────────────────────────────

/** Underlying numeric tensor of an array of built-in-derived class instances
 *  (e.g. an enumeration-member array), or null if `v` is not such an array.
 *  Each element's `_builtinData` supplies one value; the array shape carries
 *  over. */
function classInstanceArrayNumeric(v: RuntimeValue): RuntimeTensor | null {
  if (!isRuntimeClassInstanceArray(v)) return null;
  const data = allocFloat64Array(v.elements.length);
  for (let i = 0; i < v.elements.length; i++) {
    const bd = v.elements[i]._builtinData;
    data[i] = bd !== undefined ? toNumber(bd) : NaN;
  }
  return RTV.tensor(data, [v.shape[0], v.shape[1]]);
}

/** Parse string text to a double (MATLAB double(string) semantics):
 *  numeric text parses, anything else is NaN. */
function parseDoubleText(s: string): number {
  const t = s.trim();
  if (t === "") return NaN;
  if (/^[+-]?inf(inity)?$/i.test(t)) return t[0] === "-" ? -Infinity : Infinity;
  return Number(t);
}

builtin({
  name: "double",
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (isRuntimeChar(v)) {
      if (v.value.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
      if (v.value.length === 1) return RTV.num(v.value.charCodeAt(0));
      const codes = Array.from(v.value).map(c => c.charCodeAt(0));
      return RTV.row(codes);
    }
    // Strings convert by PARSING the text (unlike char code points);
    // non-numeric text becomes NaN.
    if (isRuntimeString(v)) return RTV.num(parseDoubleText(v));
    if (isRuntimeStringArray(v)) {
      return RTV.tensor(allocFloat64Array(v.data.map(parseDoubleText)), [
        v.shape[0],
        v.shape[1],
      ]);
    }
    if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
    if (isRuntimeNumber(v)) return v;
    if (isRuntimeComplexNumber(v)) return v;
    if (isRuntimeSparseMatrix(v)) return v;
    if (isRuntimeTensor(v)) {
      if (v.cls === "logical") {
        return RTV.tensor(allocFloat64Array(v.data), v.shape);
      }
      return v;
    }
    if (isRuntimeClassInstance(v) && v._builtinData !== undefined) {
      return v._builtinData;
    }
    const arr = classInstanceArrayNumeric(v);
    if (arr) return arr;
    throw invalidConversion("double", v);
  },
});

// ── single ──────────────────────────────────────────────────────────────
//
// numbl stores all numerics as double, so `single` cannot produce a distinct
// runtime class. It DOES, however, round values to single precision (via
// Math.fround), which is observable: code relies on `single(x)` to drop the
// low-order bits (e.g. ultraSEM classifies boundary points with `single(x) ==
// real(single(x))`). The result reports class 'double' in numbl.

function froundArray(src: Float64Array): Float64Array {
  const out = allocFloat64Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = Math.fround(src[i]);
  return out;
}

function singleApply(v: RuntimeValue): RuntimeValue {
  if (isRuntimeChar(v)) {
    if (v.value.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);
    if (v.value.length === 1)
      return RTV.num(Math.fround(v.value.charCodeAt(0)));
    return RTV.row(Array.from(v.value).map(c => Math.fround(c.charCodeAt(0))));
  }
  if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
  if (isRuntimeNumber(v)) return RTV.num(Math.fround(v));
  if (isRuntimeComplexNumber(v))
    return RTV.complex(Math.fround(v.re), Math.fround(v.im));
  if (isRuntimeTensor(v)) {
    const imag = v.imag ? froundArray(v.imag) : undefined;
    return RTV.tensor(froundArray(v.data), v.shape.slice(), imag);
  }
  if (isRuntimeClassInstance(v) && v._builtinData !== undefined)
    return singleApply(v._builtinData);
  const arr = classInstanceArrayNumeric(v);
  if (arr) return singleApply(arr);
  if (isRuntimeSparseMatrix(v))
    throw new RuntimeError("single: sparse input is not supported");
  throw invalidConversion("single", v);
}

builtin({
  name: "single",
  nargin: 1,
  apply: args => singleApply(args[0]),
});

// ── Integer types ───────────────────────────────────────────────────────
//
// numbl represents all numeric data as double-precision floats, so the
// int8/int16/...uint64 constructors don't really produce a distinct
// runtime class — they round toward zero and saturate at the integer
// range's limits, then return a double-backed tensor.  This matches
// MATLAB's numeric behavior closely enough for code that uses the
// `idivide(int64(a), int64(b))` idiom for integer division.

interface IntRange {
  name: string;
  min: number;
  max: number;
}

const INT_RANGES: IntRange[] = [
  { name: "int8", min: -128, max: 127 },
  { name: "int16", min: -32768, max: 32767 },
  { name: "int32", min: -2147483648, max: 2147483647 },
  // int64/uint64 can't represent their full native range as doubles;
  // clamp at Number.MAX_SAFE_INTEGER to avoid silent precision loss.
  {
    name: "int64",
    min: -Number.MAX_SAFE_INTEGER,
    max: Number.MAX_SAFE_INTEGER,
  },
  { name: "uint8", min: 0, max: 255 },
  { name: "uint16", min: 0, max: 65535 },
  { name: "uint32", min: 0, max: 4294967295 },
  { name: "uint64", min: 0, max: Number.MAX_SAFE_INTEGER },
];

function saturateRoundToward(x: number, min: number, max: number): number {
  if (isNaN(x)) return 0;
  // MATLAB int* conversion rounds to nearest, ties away from zero.
  const r = x >= 0 ? Math.floor(x + 0.5) : -Math.floor(-x + 0.5);
  if (r < min) return min;
  if (r > max) return max;
  return r;
}

for (const { name, min, max } of INT_RANGES) {
  const sat = (x: number) => saturateRoundToward(x, min, max);
  builtin({
    name,
    nargin: 1,
    apply: args => {
      // Enumeration member / built-in-derived instance → its underlying
      // value (e.g. `uint32(patchtype.tri)` → the stored uint32).
      const v = underlyingValue(args[0]);
      if (isRuntimeNumber(v)) return RTV.num(sat(v));
      if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
      // Complex values keep their (rounded, saturated) imaginary part.
      if (isRuntimeComplexNumber(v)) return mkc(sat(v.re), sat(v.im));
      if (isRuntimeChar(v)) {
        if (v.value.length === 0)
          return RTV.tensor(allocFloat64Array(0), [0, 0]);
        if (v.value.length === 1) return RTV.num(sat(v.value.charCodeAt(0)));
        const out = allocFloat64Array(v.value.length);
        for (let i = 0; i < v.value.length; i++) {
          out[i] = sat(v.value.charCodeAt(i));
        }
        return RTV.row(Array.from(out));
      }
      if (isRuntimeTensor(v)) {
        const data = allocFloat64Array(v.data.length);
        for (let i = 0; i < v.data.length; i++) data[i] = sat(v.data[i]);
        let imag: Float64Array | undefined;
        if (v.imag && !imagAllZero(v.imag)) {
          imag = allocFloat64Array(v.data.length);
          for (let i = 0; i < v.data.length; i++) imag[i] = sat(v.imag[i]);
        }
        return RTV.tensor(data, [...v.shape], imag);
      }
      if (isRuntimeSparseMatrix(v))
        throw builtinError(
          "MATLAB:unimplementedSparseType",
          "Attempt to convert to unimplemented sparse type"
        );
      throw invalidConversion(name, v);
    },
  });
}

// ── typecast ────────────────────────────────────────────────────────────
//
// Reinterpret the raw bytes of a numeric array as another numeric class.
// numbl stores all numerics as double, so the INPUT is treated as
// double-precision (8 bytes/element) and its IEEE-754 little-endian bytes are
// reinterpreted as the requested class. This is the "serialize to bytes"
// direction (e.g. `typecast(double(x), 'uint8')` to write binary data); the
// reverse (bytes -> double) is not supported because numbl cannot represent a
// genuine integer-typed array.

const TYPECAST_VIEWS: Record<
  string,
  (b: ArrayBufferLike) => ArrayLike<number>
> = {
  uint8: b => new Uint8Array(b),
  int8: b => new Int8Array(b),
  uint16: b => new Uint16Array(b),
  int16: b => new Int16Array(b),
  uint32: b => new Uint32Array(b),
  int32: b => new Int32Array(b),
  single: b => new Float32Array(b),
  double: b => new Float64Array(b),
};

builtin({
  name: "typecast",
  help: {
    signatures: ["B = typecast(X, CLASS)"],
    description:
      "Reinterpret the raw bytes of numeric array X as CLASS (e.g. 'uint8', 'single', 'int32'). numbl stores all numerics as double, so X is treated as double-precision; this is mainly for serializing numeric data to bytes, e.g. typecast(double(x), 'uint8').",
  },
  nargin: 2,
  apply: args => {
    const X = args[0];
    const clsArg = args[1];
    if (!isRuntimeChar(clsArg) && !isRuntimeString(clsArg))
      throw builtinError(
        "MATLAB:typecast:invalidSyntax",
        "Second argument must be either a character row vector or string scalar."
      );
    const cls = toString(clsArg).toLowerCase();

    let src: Float64Array;
    if (isRuntimeNumber(X)) src = Float64Array.of(X);
    else if (isRuntimeLogical(X)) src = Float64Array.of(X ? 1 : 0);
    else if (isRuntimeTensor(X)) {
      if (X.imag && !imagAllZero(X.imag))
        throw new RuntimeError("typecast: complex input not supported");
      src = X.data;
    } else if (isRuntimeComplexNumber(X)) {
      if (X.im !== 0)
        throw new RuntimeError("typecast: complex input not supported");
      src = Float64Array.of(X.re);
    } else if (isRuntimeChar(X)) {
      throw new RuntimeError("typecast: char input is not supported");
    } else {
      throw builtinError(
        "MATLAB:typecast:mustBeNumericCharOrLogical",
        "First argument must be numeric, char, or logical."
      );
    }

    const makeView = TYPECAST_VIEWS[cls];
    if (!makeView) {
      if (cls === "int64" || cls === "uint64")
        throw new RuntimeError(
          `typecast: conversion to '${cls}' is not supported`
        );
      throw builtinError(
        "MATLAB:typecast:unsupportedClass",
        `Data conversion from 'double' to '${toString(clsArg)}' is not supported.`
      );
    }

    // Copy to a tightly-packed buffer so the typed-array view is aligned.
    const buffer = src.buffer.slice(
      src.byteOffset,
      src.byteOffset + src.byteLength
    );
    const view = makeView(buffer);
    const out = allocFloat64Array(view.length);
    for (let i = 0; i < view.length; i++) out[i] = view[i];
    return RTV.tensor(out, [1, view.length]);
  },
});

// ── jsonencode ──────────────────────────────────────────────────────────
//
// Encode a numbl value as a JSON string (returned as a char row vector).
// Mirrors MATLAB's jsonencode for the common cases: struct -> object,
// cell -> array, char/string -> string, logical -> true/false, numeric
// scalar -> number, numeric vector/matrix -> (nested) array, [] -> [].
// Non-finite numbers (NaN/Inf) are emitted as null to keep the output valid
// JSON.

/** jsonencode's ConvertInfAndNaN option: when false, non-finite numbers
 *  are written as NaN / Infinity / -Infinity instead of null. */
let jsonConvertInfAndNaN = true;

function jsonEncodeNumber(x: number): string {
  if (Number.isFinite(x)) return String(x);
  if (jsonConvertInfAndNaN) return "null";
  return Number.isNaN(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity";
}

function jsonEncodeTensor(v: RuntimeTensor): string {
  // A tensor with an all-zero imaginary lane is real in value (the JIT often
  // produces such tensors when it can't prove realness at compile time). Treat
  // it as real — matching `isreal` — instead of rejecting on the lane's mere
  // presence. Genuinely complex data is unsupported (as in MATLAB).
  if (v.imag && !imagAllZero(v.imag))
    throw new RuntimeError("jsonencode: complex values are not supported");
  const data = v.data;
  const n = data.length;
  const enc =
    v.cls === "logical"
      ? (x: number) => (x ? "true" : "false")
      : jsonEncodeNumber;
  const shape = v.shape ?? [n];
  // scalar
  if (n === 1 && shape.every(s => s === 1)) return enc(data[0]);
  if (n === 0) return "[]";
  // vector (at most one non-singleton dimension)
  const nonSingleton = shape.filter(s => s > 1).length;
  if (shape.length <= 1 || nonSingleton <= 1) {
    const parts: string[] = [];
    for (let i = 0; i < n; i++) parts.push(enc(data[i]));
    return "[" + parts.join(",") + "]";
  }
  // 2-D matrix: nest by rows (column-major storage -> data[j*m + i])
  if (shape.length === 2) {
    const [m, cols] = shape;
    const rows: string[] = [];
    for (let i = 0; i < m; i++) {
      const rowParts: string[] = [];
      for (let j = 0; j < cols; j++) rowParts.push(enc(data[j * m + i]));
      rows.push("[" + rowParts.join(",") + "]");
    }
    return "[" + rows.join(",") + "]";
  }
  // N-D fallback: flat array
  const parts: string[] = [];
  for (let i = 0; i < n; i++) parts.push(enc(data[i]));
  return "[" + parts.join(",") + "]";
}

function jsonEncodeCell(v: RuntimeCell): string {
  if (v.data.length === 0) return "[]";
  return "[" + v.data.map(jsonEncodeValue).join(",") + "]";
}

function jsonEncodeStruct(v: RuntimeStruct): string {
  const parts: string[] = [];
  for (const [key, val] of v.fields) {
    parts.push(JSON.stringify(key) + ":" + jsonEncodeValue(val));
  }
  return "{" + parts.join(",") + "}";
}

function jsonEncodeValue(v: RuntimeValue): string {
  if (v === undefined || v === null) return "null";
  if (isRuntimeNumber(v)) return jsonEncodeNumber(v);
  if (isRuntimeLogical(v)) return v ? "true" : "false";
  if (isRuntimeString(v)) return JSON.stringify(v);
  if (isRuntimeChar(v)) {
    // Multi-row char array -> array of row strings; otherwise a single string.
    if (v.shape && v.shape.length === 2 && v.shape[0] > 1) {
      const rows = v.shape[0];
      const cols = v.shape[1];
      const out: string[] = [];
      for (let i = 0; i < rows; i++) {
        let row = "";
        for (let j = 0; j < cols; j++) row += v.value[j * rows + i] ?? "";
        out.push(JSON.stringify(row));
      }
      return "[" + out.join(",") + "]";
    }
    return JSON.stringify(v.value);
  }
  if (isRuntimeComplexNumber(v)) {
    // A complex scalar with zero imaginary part is real in value (consistent
    // with the all-zero-lane tensor case and with `isreal`).
    if (v.im !== 0)
      throw new RuntimeError("jsonencode: complex values are not supported");
    return jsonEncodeNumber(v.re);
  }
  if (isRuntimeTensor(v)) return jsonEncodeTensor(v);
  if (isRuntimeCell(v)) return jsonEncodeCell(v);
  if (isRuntimeStruct(v)) return jsonEncodeStruct(v);
  if (isRuntimeStructArray(v))
    return "[" + v.elements.map(jsonEncodeStruct).join(",") + "]";
  throw new RuntimeError("jsonencode: unsupported value type");
}

/** jsonencode's name-value options. PrettyPrint is accepted only when
 *  false. Returns the ConvertInfAndNaN setting. */
function jsonencodeOptions(opts: RuntimeValue[]): boolean {
  let convert = true;
  for (let i = 0; i < opts.length; i += 2) {
    const nameArg = opts[i];
    const name =
      isRuntimeChar(nameArg) || isRuntimeString(nameArg)
        ? toString(nameArg)
        : "";
    if (name !== "ConvertInfAndNaN" && name !== "PrettyPrint")
      throw builtinError(
        "MATLAB:json:UnmatchedParameter",
        "Name argument must be 'ConvertInfAndNaN' or 'PrettyPrint'."
      );
    if (i + 1 >= opts.length)
      throw builtinError(
        "MATLAB:json:ParamMissingValue",
        `Name '${name}' requires a value.`
      );
    const v = opts[i + 1];
    let flag: boolean;
    if (typeof v === "boolean") flag = v;
    else if (typeof v === "number" && (v === 0 || v === 1)) flag = v === 1;
    else
      throw builtinError(
        "MATLAB:json:InvalidValueType",
        `Value type must be logical scalar for name '${name}'.`
      );
    if (name === "ConvertInfAndNaN") convert = flag;
    else if (flag)
      throw new RuntimeError("jsonencode: 'PrettyPrint' is not supported");
  }
  return convert;
}

builtin({
  name: "jsonencode",
  help: {
    signatures: [
      "txt = jsonencode(V)",
      "txt = jsonencode(V, 'ConvertInfAndNaN', TF)",
    ],
    description:
      "Encode value V (struct, cell, char/string, logical, or numeric array) as a JSON-formatted char row vector. NaN and Inf are written as null unless ConvertInfAndNaN is false.",
  },
  nargin: [1, 5],
  apply: args => {
    const convert = jsonencodeOptions(args.slice(1));
    const saved = jsonConvertInfAndNaN;
    jsonConvertInfAndNaN = convert;
    try {
      return RTV.char(jsonEncodeValue(args[0]));
    } finally {
      jsonConvertInfAndNaN = saved;
    }
  },
});

// ── idivide ─────────────────────────────────────────────────────────────
//
// Integer division with a selectable rounding mode.  The optional third
// argument is a string: 'fix' (default, truncate toward zero), 'floor'
// (toward -Inf), 'ceil' (toward +Inf), or 'round' (to nearest, ties
// away from zero).  MATLAB semantics: fix/floor/ceil/round act on the
// true quotient before the result lands in an integer class.

type IdivMode = "fix" | "floor" | "ceil" | "round";

function idivideMode(args: RuntimeValue[]): IdivMode {
  if (args.length < 3) return "fix";
  const m = args[2];
  const s = isRuntimeChar(m) || isRuntimeString(m) ? toString(m) : "";
  if (s === "fix" || s === "floor" || s === "ceil" || s === "round") return s;
  throw builtinError(
    "MATLAB:idivide:InvalidRoundingOption",
    "Unrecognized rounding option."
  );
}

/** An idivide operand as a real number or real tensor. */
function idivideOperand(v: RuntimeValue): number | RuntimeTensor {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  const complexErr = () =>
    builtinError(
      "MATLAB:idivide:complexInts",
      "Complex integer arithmetic is not supported."
    );
  if (isRuntimeComplexNumber(v)) {
    if (v.im !== 0) throw complexErr();
    return v.re;
  }
  if (isRuntimeTensor(v)) {
    if (v.imag) {
      if (!imagAllZero(v.imag)) throw complexErr();
      return RTV.tensor(v.data, v.shape) as RuntimeTensor;
    }
    return v;
  }
  throw new RuntimeError("idivide: arguments must be numeric");
}

builtin({
  name: "idivide",
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const mode = idivideMode(args);
    const divFn = ((): ((a: number, b: number) => number) => {
      switch (mode) {
        case "fix":
          return (a, b) => {
            if (b === 0) return 0;
            const q = a / b;
            return q >= 0 ? Math.floor(q) : -Math.floor(-q);
          };
        case "floor":
          return (a, b) => (b === 0 ? 0 : Math.floor(a / b));
        case "ceil":
          return (a, b) => (b === 0 ? 0 : Math.ceil(a / b));
        case "round":
          return (a, b) => {
            if (b === 0) return 0;
            // MATLAB round: ties away from zero.
            const q = a / b;
            return q >= 0 ? Math.floor(q + 0.5) : -Math.floor(-q + 0.5);
          };
      }
    })();
    const a = idivideOperand(args[0]);
    const b = idivideOperand(args[1]);
    if (isRuntimeNumber(a) && isRuntimeNumber(b)) {
      return RTV.num(divFn(a, b));
    }
    if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
      const bv = b;
      const data = allocFloat64Array(a.data.length);
      for (let i = 0; i < a.data.length; i++) data[i] = divFn(a.data[i], bv);
      return RTV.tensor(data, [...a.shape]);
    }
    if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
      const av = a;
      const data = allocFloat64Array(b.data.length);
      for (let i = 0; i < b.data.length; i++) data[i] = divFn(av, b.data[i]);
      return RTV.tensor(data, [...b.shape]);
    }
    if (isRuntimeTensor(a) && isRuntimeTensor(b)) {
      if (a.data.length !== b.data.length)
        throw new RuntimeError("idivide: arrays must be the same size");
      const data = allocFloat64Array(a.data.length);
      for (let i = 0; i < a.data.length; i++)
        data[i] = divFn(a.data[i], b.data[i]);
      return RTV.tensor(data, [...a.shape]);
    }
    throw new RuntimeError("idivide: arguments must be numeric");
  },
});

// ── logical ─────────────────────────────────────────────────────────────

/** One element converted to logical, rejecting NaN as MATLAB does. */
function toLogicalElem(x: number): number {
  if (Number.isNaN(x))
    throw builtinError(
      "MATLAB:nologicalnan",
      "NaN values cannot be converted to logicals."
    );
  return x !== 0 ? 1 : 0;
}

builtin({
  name: "logical",
  nargin: 1,
  apply: args => {
    // Enumeration member with a `logical` superclass → underlying value.
    let v = underlyingValue(args[0]);
    const complexErr = () =>
      builtinError(
        "MATLAB:nologicalcomplex",
        "Complex values cannot be converted to logicals."
      );
    if (typeof v === "boolean") return v;
    if (typeof v === "number") return toLogicalElem(v) === 1;
    if (isRuntimeComplexNumber(v)) throw complexErr();
    if (isRuntimeChar(v)) {
      if (v.value.length === 1) return v.value.charCodeAt(0) !== 0;
      v = charToNumericTensor(v);
    }
    if (isRuntimeTensor(v)) {
      if (v.imag && !imagAllZero(v.imag)) throw complexErr();
      const result = allocFloat64Array(v.data.length);
      for (let i = 0; i < v.data.length; i++) {
        result[i] = toLogicalElem(v.data[i]);
      }
      const t = RTV.logicalTensor(result, v.shape);
      return t;
    }
    if (isRuntimeSparseMatrix(v))
      throw new RuntimeError("logical: sparse input is not supported");
    throw invalidConversion("logical", v);
  },
});

// ── complex ─────────────────────────────────────────────────────────────

/** A real part / imaginary part operand of complex(a, b): a real number or
 *  a real tensor (chars become their codes, logicals 0/1). */
function complexPart(
  v: RuntimeValue,
  which: "realPart" | "imagPart"
): number | RuntimeTensor {
  const err = () =>
    builtinError(
      `MATLAB:complex:${which}InputNonNumericOrNonReal`,
      `Input for ${which === "realPart" ? "real" : "imaginary"} part must be a real numeric value.`
    );
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (isRuntimeChar(v))
    return v.value.length === 1
      ? v.value.charCodeAt(0)
      : charToNumericTensor(v);
  if (isRuntimeComplexNumber(v)) {
    if (v.im !== 0) throw err();
    return v.re;
  }
  if (isRuntimeTensor(v)) {
    if (v.imag && !imagAllZero(v.imag)) throw err();
    return v;
  }
  if (isRuntimeSparseMatrix(v))
    throw new RuntimeError("complex: sparse input is not supported");
  throw err();
}

builtin({
  name: "complex",
  nargin: [1, 2],
  apply: args => {
    if (args.length === 1) {
      // complex(a) — ensure complex
      let a = args[0];
      if (isRuntimeComplexNumber(a)) return a;
      if (isRuntimeChar(a))
        a =
          a.value.length === 1 ? a.value.charCodeAt(0) : charToNumericTensor(a);
      if (isRuntimeNumber(a)) return RTV.complex(a, 0);
      if (isRuntimeLogical(a)) return RTV.complex(a ? 1 : 0, 0);
      if (isRuntimeTensor(a)) {
        const im = a.imag || allocFloat64Array(a.data.length);
        return makeTensor(
          allocFloat64Array(a.data),
          allocFloat64Array(im),
          a.shape.slice()
        );
      }
      if (isRuntimeSparseMatrix(a))
        throw new RuntimeError("complex: sparse input is not supported");
      throw builtinError(
        "MATLAB:complex:expectsNumeric",
        "First input must be numeric."
      );
    }
    // complex(a, b) — a + bi. A one-element array counts as a scalar.
    const asScalar = (x: number | RuntimeTensor) =>
      typeof x === "number" ? x : x.data.length === 1 ? x.data[0] : x;
    const a = asScalar(complexPart(args[0], "realPart"));
    const b = asScalar(complexPart(args[1], "imagPart"));
    if (typeof a === "number" && typeof b === "number")
      return RTV.complex(a, b);
    if (
      typeof a !== "number" &&
      typeof b !== "number" &&
      (a.data.length !== b.data.length ||
        a.shape.length !== b.shape.length ||
        a.shape.some((d, i) => d !== b.shape[i]))
    )
      throw builtinError(
        "MATLAB:complex:inputSizeMismatch",
        "Input arrays must have the same size."
      );
    const ref = (typeof a !== "number" ? a : b) as RuntimeTensor;
    const len = ref.data.length;
    const reArr = allocFloat64Array(len);
    const imArr = allocFloat64Array(len);
    for (let i = 0; i < len; i++) {
      reArr[i] = typeof a === "number" ? a : a.data[i];
      imArr[i] = typeof b === "number" ? b : b.data[i];
    }
    return makeTensor(reArr, imArr, ref.shape.slice());
  },
});

// ── cell ────────────────────────────────────────────────────────────────

/** One cell() size argument, checked as MATLAB does. */
function cellSizeArg(v: RuntimeValue): RuntimeValue {
  if (typeof v === "number") return v;
  if (isRuntimeTensor(v) && v.cls !== "logical") {
    if (v.imag && !imagAllZero(v.imag))
      throw builtinError(
        "MATLAB:obsoleteComplexInput",
        "Size inputs must be real."
      );
    return v;
  }
  if (isRuntimeComplexNumber(v))
    throw builtinError(
      "MATLAB:obsoleteComplexInput",
      "Size inputs must be real."
    );
  throw invalidConversion("cell", v);
}

builtin({
  name: "cell",
  nargin: [1, Infinity],
  apply: args => {
    if (args.length === 1 && isRuntimeCell(args[0])) return args[0];
    const sizeArgs = args.map(cellSizeArg);
    let dims = parseShapeArgs(sizeArgs);
    if (dims.length === 0) dims = [0, 0];
    else if (dims.length === 1) dims = [dims[0], dims[0]];
    while (dims.length > 2 && dims[dims.length - 1] === 1) dims.pop();
    const total = dims.reduce((a, b) => a * b, 1);
    const data: RuntimeValue[] = new Array(total);
    for (let i = 0; i < total; i++)
      data[i] = RTV.tensor(allocFloat64Array(0), [0, 0]);
    return RTV.cell(data, dims);
  },
});

// ── struct ──────────────────────────────────────────────────────────────

builtin({
  name: "struct",
  apply: args => {
    if (args.length === 0) return RTV.struct(new Map());
    if (args.length === 1) {
      const a = args[0];
      // struct(s) returns s.
      if (isRuntimeStruct(a) || isRuntimeStructArray(a)) return a;
      // struct([]) → 0x0 empty struct array with no fields
      if (isRuntimeTensor(a) && a.data.length === 0)
        return RTV.structArray([], [], [0, 0]);
      if (isRuntimeClassInstance(a) || isRuntimeClassInstanceArray(a))
        throw new RuntimeError("struct: object input is not supported");
      throw invalidConversion("struct", a);
    }
    if (args.length % 2 !== 0)
      throw builtinError(
        "MATLAB:StructConversion:NonPairedArgs",
        "Field and value input arguments must come in pairs."
      );
    const fieldNames: string[] = [];
    const fieldValues: RuntimeValue[] = [];
    let hasCell = false;
    // A one-element cell value is copied to every element; the other
    // cell values must all have the same number of elements.
    let arrayLen = -1;
    let arrayArg = 0;
    let arrayShape: number[] = [1, 1];
    for (let i = 0; i < args.length; i += 2) {
      const nameArg = args[i];
      if (
        !(isRuntimeChar(nameArg) || isRuntimeString(nameArg)) ||
        toString(nameArg) === ""
      )
        throw builtinError(
          "MATLAB:Cell2Struct:NonStringFieldNames",
          "Field names must be non-empty character vectors or string scalars."
        );
      const name = toString(nameArg);
      const val = args[i + 1];
      fieldNames.push(name);
      fieldValues.push(val);
      if (isRuntimeCell(val)) {
        hasCell = true;
        const len = val.data.length;
        if (len === 1) continue;
        if (arrayLen === -1) {
          arrayLen = len;
          arrayArg = i + 2;
          arrayShape = val.shape.slice();
        } else if (len !== arrayLen)
          throw builtinError(
            "MATLAB:StructConversion:DimensionMismatch",
            `Array dimensions of input '${i + 2}' must match those of input '${arrayArg}', or be scalar.`
          );
      }
    }
    if (hasCell && arrayLen === -1) arrayLen = 1;
    if (!hasCell) {
      const fields = new Map<string, RuntimeValue>();
      for (let i = 0; i < fieldNames.length; i++) {
        fields.set(fieldNames[i], fieldValues[i]);
      }
      return RTV.struct(fields);
    }
    // Struct array: each element gets one value from each cell
    const elements = [];
    for (let k = 0; k < arrayLen; k++) {
      const fields = new Map<string, RuntimeValue>();
      for (let i = 0; i < fieldNames.length; i++) {
        const val = fieldValues[i];
        if (isRuntimeCell(val)) {
          fields.set(fieldNames[i], val.data[val.data.length === 1 ? 0 : k]);
        } else {
          fields.set(fieldNames[i], val);
        }
      }
      elements.push(RTV.struct(fields));
    }
    // The result has the size of the cell values; 1x1 is a scalar struct.
    if (elements.length === 1) return elements[0];
    return RTV.structArray(fieldNames, elements, arrayShape);
  },
});

// ── full (sparse → dense) ───────────────────────────────────────────────

builtin({
  name: "full",
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (!isRuntimeSparseMatrix(v)) return v; // passthrough for non-sparse
    const { m, n, ir, jc, pr, pi } = v;
    const data = allocFloat64Array(m * n);
    const imag = pi ? allocFloat64Array(m * n) : undefined;
    for (let col = 0; col < n; col++) {
      for (let k = jc[col]; k < jc[col + 1]; k++) {
        data[col * m + ir[k]] = pr[k]; // column-major
        if (imag && pi) imag[col * m + ir[k]] = pi[k];
      }
    }
    return RTV.tensor(data, [m, n], imag);
  },
});

// ── isfield ─────────────────────────────────────────────────────────────

builtin({
  name: "isfield",
  nargin: 2,
  apply: args => {
    const v = args[0];
    const hasField = (name: string): boolean => {
      if (isRuntimeStructArray(v)) return v.fieldNames.includes(name);
      // As in MATLAB, an object has properties, not fields.
      if (!isRuntimeStruct(v)) return false;
      return v.fields.has(name);
    };
    // Cell array of field names → logical array of the same size
    if (isRuntimeCell(args[1])) {
      const cell = args[1] as RuntimeCell;
      const result = allocFloat64Array(cell.data.length);
      for (let i = 0; i < cell.data.length; i++) {
        result[i] = hasField(toString(cell.data[i])) ? 1 : 0;
      }
      return new RuntimeTensor(
        result,
        cell.shape.slice(),
        undefined,
        "logical"
      );
    }
    return RTV.logical(hasField(toString(args[1])));
  },
});

// ── deal ────────────────────────────────────────────────────────────────

builtin({
  name: "deal",
  nargin: [0, Infinity],
  nargout: [0, Infinity],
  mfile: true,
  apply: (args, nargout) => {
    // deal(X) copies X to every output; otherwise inputs map 1:1 to
    // outputs.
    if (args.length === 1) {
      if (nargout <= 1) return args[0];
      return new Array(nargout).fill(args[0]);
    }
    if (args.length !== Math.max(nargout, 1))
      throw builtinError(
        "MATLAB:deal:narginNargoutMismatch",
        "The number of outputs should match the number of inputs."
      );
    if (nargout <= 1) return args[0];
    return args;
  },
});

// ── func2str ────────────────────────────────────────────────────────────

builtin({
  name: "func2str",
  help: {
    signatures: ["str = func2str(fh)"],
    description:
      "Text of a function handle: the function name for a named handle " +
      "('sin'), or the definition of an anonymous function ('@(x)x+1').",
  },
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (!isRuntimeFunction(v)) throw undefinedForTypeError("func2str", v);
    const ast = v.handleAst as Expr | undefined;
    if (ast?.type === "AnonFunc") return RTV.char(unparseExpr(ast));
    return RTV.char(v.name);
  },
});
