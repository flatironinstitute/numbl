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
} from "../../runtime/types.js";
import type {
  RuntimeValue,
  RuntimeTensor,
  RuntimeCell,
  RuntimeStruct,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber, toBool, toString } from "../../runtime/convert.js";
import type { JitType } from "../../jitTypes.js";
import { defineBuiltin, registerIBuiltin, makeTensor } from "./types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";

// ── double ──────────────────────────────────────────────────────────────

defineBuiltin({
  name: "double",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (
          a.kind === "number" ||
          a.kind === "boolean" ||
          a.kind === "char" ||
          a.kind === "complex_or_number" ||
          a.kind === "class_instance"
        )
          return [{ kind: "number" }];
        if (a.kind === "tensor")
          return [
            {
              kind: "tensor",
              isComplex: a.isComplex,
              shape: a.shape,
              ndim: a.ndim,
            },
          ];
        return null;
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeChar(v)) {
          if (v.value.length === 0)
            return RTV.tensor(allocFloat64Array(0), [0, 0]);
          if (v.value.length === 1) return RTV.num(v.value.charCodeAt(0));
          const codes = Array.from(v.value).map(c => c.charCodeAt(0));
          return RTV.row(codes);
        }
        if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
        if (isRuntimeNumber(v)) return v;
        if (isRuntimeComplexNumber(v)) return v.re;
        if (isRuntimeTensor(v)) {
          if (v._isLogical) {
            return RTV.tensor(allocFloat64Array(v.data), v.shape);
          }
          return v;
        }
        if (isRuntimeClassInstance(v) && v._builtinData !== undefined) {
          return v._builtinData;
        }
        return RTV.num(toNumber(v));
      },
    },
  ],
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
  defineBuiltin({
    name,
    cases: [
      {
        match: argTypes => {
          if (argTypes.length !== 1) return null;
          const a = argTypes[0];
          if (
            a.kind === "number" ||
            a.kind === "boolean" ||
            a.kind === "char" ||
            a.kind === "complex_or_number"
          )
            return [{ kind: "number" }];
          if (a.kind === "tensor")
            return [
              {
                kind: "tensor",
                isComplex: false,
                shape: a.shape,
                ndim: a.ndim,
              },
            ];
          return null;
        },
        apply: args => {
          const v = args[0];
          if (isRuntimeNumber(v))
            return RTV.num(saturateRoundToward(v as number, min, max));
          if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
          if (isRuntimeComplexNumber(v))
            return RTV.num(saturateRoundToward(v.re, min, max));
          if (isRuntimeChar(v)) {
            if (v.value.length === 0)
              return RTV.tensor(allocFloat64Array(0), [0, 0]);
            if (v.value.length === 1)
              return RTV.num(
                saturateRoundToward(v.value.charCodeAt(0), min, max)
              );
            const out = allocFloat64Array(v.value.length);
            for (let i = 0; i < v.value.length; i++) {
              out[i] = saturateRoundToward(v.value.charCodeAt(i), min, max);
            }
            return RTV.row(Array.from(out));
          }
          if (isRuntimeTensor(v)) {
            const data = allocFloat64Array(v.data.length);
            for (let i = 0; i < v.data.length; i++) {
              data[i] = saturateRoundToward(v.data[i], min, max);
            }
            return RTV.tensor(data, [...v.shape]);
          }
          return RTV.num(saturateRoundToward(toNumber(v), min, max));
        },
      },
    ],
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

defineBuiltin({
  name: "typecast",
  help: {
    signatures: ["B = typecast(X, CLASS)"],
    description:
      "Reinterpret the raw bytes of numeric array X as CLASS (e.g. 'uint8', 'single', 'int32'). numbl stores all numerics as double, so X is treated as double-precision; this is mainly for serializing numeric data to bytes, e.g. typecast(double(x), 'uint8').",
  },
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const a = argTypes[0];
        if (
          a.kind === "number" ||
          a.kind === "boolean" ||
          a.kind === "tensor" ||
          a.kind === "complex_or_number"
        )
          return [{ kind: "tensor", isComplex: false }];
        return null;
      },
      apply: args => {
        const X = args[0];
        const cls = toString(args[1]).toLowerCase();
        const makeView = TYPECAST_VIEWS[cls];
        if (!makeView)
          throw new RuntimeError(`typecast: unsupported class '${cls}'`);

        let src: Float64Array;
        if (isRuntimeNumber(X)) src = Float64Array.of(X);
        else if (isRuntimeLogical(X)) src = Float64Array.of(X ? 1 : 0);
        else if (isRuntimeTensor(X)) {
          if (X.imag)
            throw new RuntimeError("typecast: complex input not supported");
          src = X.data;
        } else {
          throw new RuntimeError("typecast: X must be a numeric array");
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
    },
  ],
});

// ── jsonencode ──────────────────────────────────────────────────────────
//
// Encode a numbl value as a JSON string (returned as a char row vector).
// Mirrors MATLAB's jsonencode for the common cases: struct -> object,
// cell -> array, char/string -> string, logical -> true/false, numeric
// scalar -> number, numeric vector/matrix -> (nested) array, [] -> [].
// Non-finite numbers (NaN/Inf) are emitted as null to keep the output valid
// JSON.

function jsonEncodeNumber(x: number): string {
  return Number.isFinite(x) ? String(x) : "null";
}

function jsonEncodeTensor(v: RuntimeTensor): string {
  if (v.imag)
    throw new RuntimeError("jsonencode: complex values are not supported");
  const data = v.data;
  const n = data.length;
  const enc = v._isLogical
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
  if (isRuntimeTensor(v)) return jsonEncodeTensor(v);
  if (isRuntimeCell(v)) return jsonEncodeCell(v);
  if (isRuntimeStruct(v)) return jsonEncodeStruct(v);
  if (isRuntimeStructArray(v))
    return "[" + v.elements.map(jsonEncodeStruct).join(",") + "]";
  throw new RuntimeError("jsonencode: unsupported value type");
}

defineBuiltin({
  name: "jsonencode",
  help: {
    signatures: ["txt = jsonencode(V)"],
    description:
      "Encode value V (struct, cell, char/string, logical, or numeric array) as a JSON-formatted char row vector.",
  },
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1) return null;
        return [{ kind: "char" }];
      },
      apply: args => RTV.char(jsonEncodeValue(args[0])),
    },
  ],
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
  if (!isRuntimeChar(m))
    throw new RuntimeError("idivide: OPT argument must be a string");
  const s = m.value;
  if (s === "fix" || s === "floor" || s === "ceil" || s === "round") return s;
  throw new RuntimeError(
    "idivide: OPT must be 'fix', 'floor', 'ceil', or 'round'"
  );
}

defineBuiltin({
  name: "idivide",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        return [{ kind: "unknown" }];
      },
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
        const a = args[0];
        const b = args[1];
        if (isRuntimeNumber(a) && isRuntimeNumber(b)) {
          return RTV.num(divFn(a as number, b as number));
        }
        if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
          const bv = b as number;
          const data = allocFloat64Array(a.data.length);
          for (let i = 0; i < a.data.length; i++)
            data[i] = divFn(a.data[i], bv);
          return RTV.tensor(data, [...a.shape]);
        }
        if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
          const av = a as number;
          const data = allocFloat64Array(b.data.length);
          for (let i = 0; i < b.data.length; i++)
            data[i] = divFn(av, b.data[i]);
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
    },
  ],
});

// ── logical ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "logical",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (a.kind === "number" || a.kind === "boolean")
          return [{ kind: "boolean" }];
        if (a.kind === "tensor")
          return [
            {
              kind: "tensor",
              isComplex: false,
              isLogical: true,
              shape: a.shape,
              ndim: a.ndim,
            },
          ];
        return null;
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeTensor(v)) {
          const result = allocFloat64Array(v.data.length);
          for (let i = 0; i < v.data.length; i++) {
            result[i] = v.data[i] !== 0 ? 1 : 0;
          }
          const t = RTV.tensor(result, v.shape);
          t._isLogical = true;
          return t;
        }
        return RTV.logical(toBool(v));
      },
    },
  ],
});

// ── complex ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "complex",
  cases: [
    // complex(a) — ensure complex
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (
          a.kind === "number" ||
          a.kind === "boolean" ||
          a.kind === "complex_or_number"
        )
          return [{ kind: "complex_or_number" }];
        if (a.kind === "tensor")
          return [
            { kind: "tensor", isComplex: true, shape: a.shape, ndim: a.ndim },
          ];
        return null;
      },
      apply: args => {
        const a = args[0];
        if (isRuntimeComplexNumber(a)) return a;
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
        throw new RuntimeError("complex requires numeric arguments");
      },
    },
    // complex(a, b) — a + bi
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const [a, b] = argTypes;
        const scalarA =
          a.kind === "number" ||
          a.kind === "boolean" ||
          a.kind === "complex_or_number";
        const scalarB =
          b.kind === "number" ||
          b.kind === "boolean" ||
          b.kind === "complex_or_number";
        if (scalarA && scalarB) return [{ kind: "complex_or_number" }];
        // At least one tensor
        if (a.kind === "tensor" || b.kind === "tensor") {
          const shape =
            a.kind === "tensor"
              ? a.shape
              : (b as Extract<JitType, { kind: "tensor" }>).shape;
          return [{ kind: "tensor", isComplex: true, shape }];
        }
        return null;
      },
      apply: args => {
        const [a, b] = args;
        if (
          (isRuntimeNumber(a) || isRuntimeLogical(a)) &&
          (isRuntimeNumber(b) || isRuntimeLogical(b))
        ) {
          const re = isRuntimeLogical(a) ? (a ? 1 : 0) : (a as number);
          const im = isRuntimeLogical(b) ? (b ? 1 : 0) : (b as number);
          return RTV.complex(re, im);
        }
        // Tensor cases
        if (isRuntimeTensor(a) || isRuntimeTensor(b)) {
          const aIsT = isRuntimeTensor(a);
          const bIsT = isRuntimeTensor(b);
          const aData = aIsT ? (a as RuntimeTensor).data : null;
          const bData = bIsT ? (b as RuntimeTensor).data : null;
          const shape = aIsT
            ? (a as RuntimeTensor).shape
            : (b as RuntimeTensor).shape;
          const len = aIsT
            ? (a as RuntimeTensor).data.length
            : (b as RuntimeTensor).data.length;
          const reArr = allocFloat64Array(len);
          const imArr = allocFloat64Array(len);
          const aScalar = !aIsT ? toNumber(a) : 0;
          const bScalar = !bIsT ? toNumber(b) : 0;
          for (let i = 0; i < len; i++) {
            reArr[i] = aData ? aData[i] : aScalar;
            imArr[i] = bData ? bData[i] : bScalar;
          }
          return makeTensor(reArr, imArr, shape.slice());
        }
        throw new RuntimeError("complex requires numeric arguments");
      },
    },
  ],
});

// ── cell ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cell",
  cases: [
    {
      match: argTypes => {
        // cell(), cell(n), cell(m,n), cell([m,n])
        if (argTypes.length === 0) return [{ kind: "unknown" }];
        if (argTypes.length <= 2) {
          for (const a of argTypes) {
            if (
              a.kind !== "number" &&
              a.kind !== "boolean" &&
              a.kind !== "tensor"
            )
              return null;
          }
          return [{ kind: "unknown" }];
        }
        return null;
      },
      apply: args => {
        const empty = () => RTV.tensor(allocFloat64Array(0), [0, 0]);
        if (args.length === 0) return RTV.cell([], [0, 0]);
        if (args.length === 1) {
          const arg = args[0];
          // Vector arg → dimensions
          if (isRuntimeTensor(arg) && arg.data.length > 1) {
            const dims = Array.from(arg.data).map(d => Math.round(d));
            const total = dims.reduce((a, b) => a * b, 1);
            const data: RuntimeValue[] = new Array(total);
            for (let i = 0; i < total; i++) data[i] = empty();
            return RTV.cell(data, dims);
          }
          const n = Math.round(toNumber(arg));
          const data: RuntimeValue[] = new Array(n * n);
          for (let i = 0; i < n * n; i++) data[i] = empty();
          return RTV.cell(data, [n, n]);
        }
        // cell(m, n)
        const r = Math.round(toNumber(args[0]));
        const c = Math.round(toNumber(args[1]));
        const data: RuntimeValue[] = new Array(r * c);
        for (let i = 0; i < r * c; i++) data[i] = empty();
        return RTV.cell(data, [r, c]);
      },
    },
  ],
});

// ── struct ──────────────────────────────────────────────────────────────

defineBuiltin({
  name: "struct",
  cases: [
    {
      match: argTypes => {
        // struct() or struct(field, value, ...)
        if (argTypes.length === 0) return [{ kind: "struct", fields: {} }];
        if (argTypes.length % 2 !== 0) return null;
        // Verify field names are strings/chars
        for (let i = 0; i < argTypes.length; i += 2) {
          const k = argTypes[i].kind;
          if (k !== "string" && k !== "char") return null;
        }
        return [{ kind: "struct", fields: {} }];
      },
      apply: args => {
        if (args.length === 0) return RTV.struct(new Map());
        if (args.length % 2 !== 0)
          throw new RuntimeError("struct: requires field-value pairs");
        const fieldNames: string[] = [];
        const fieldValues: RuntimeValue[] = [];
        let hasCell = false;
        let arrayLen = -1;
        for (let i = 0; i < args.length; i += 2) {
          const name = toString(args[i]);
          const val = args[i + 1];
          fieldNames.push(name);
          fieldValues.push(val);
          if (isRuntimeCell(val)) {
            hasCell = true;
            const len = val.data.length;
            if (arrayLen === -1) arrayLen = len;
            else if (len !== arrayLen)
              throw new RuntimeError(
                "struct: cell array values must have the same length"
              );
          }
        }
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
              fields.set(fieldNames[i], val.data[k]);
            } else {
              fields.set(fieldNames[i], val);
            }
          }
          elements.push(RTV.struct(fields));
        }
        return RTV.structArray(fieldNames, elements);
      },
    },
  ],
});

// ── full (sparse → dense) ───────────────────────────────────────────────

defineBuiltin({
  name: "full",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        // Passthrough for non-sparse types
        if (a.kind === "number" || a.kind === "boolean")
          return [{ kind: "number" }];
        if (a.kind === "tensor")
          return [
            {
              kind: "tensor",
              isComplex: a.isComplex,
              shape: a.shape,
              ndim: a.ndim,
            },
          ];
        // unknown covers sparse and others
        return [{ kind: "unknown" }];
      },
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
    },
  ],
});

// ── isfield ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "isfield",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        return [{ kind: "boolean" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeStructArray(v))
          return RTV.logical(v.fieldNames.includes(toString(args[1])));
        if (!isRuntimeStruct(v) && !isRuntimeClassInstance(v))
          return RTV.logical(false);
        return RTV.logical(v.fields.has(toString(args[1])));
      },
    },
  ],
});

// ── deal ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "deal",
  resolve: (argTypes, nargout) => {
    if (argTypes.length === 0) return null;
    if (argTypes.length === 1) {
      // Replicate single input to nargout outputs
      const outTypes = new Array(Math.max(nargout, 1)).fill(argTypes[0]);
      return {
        outputTypes: outTypes,
        apply: (args, nargout) => {
          if (nargout <= 1) return args[0];
          return new Array(nargout).fill(args[0]);
        },
      };
    }
    // N inputs → N outputs
    return {
      outputTypes: argTypes.slice(),
      apply: (args, nargout) => {
        if (args.length !== nargout)
          throw new RuntimeError(
            `deal: number of inputs (${args.length}) must match number of outputs (${nargout})`
          );
        if (nargout <= 1) return args[0];
        return args;
      },
    };
  },
});

// ── func2str ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "func2str",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "string" }];
      },
      apply: args => {
        const v = args[0];
        if (!isRuntimeFunction(v))
          throw new RuntimeError(
            "func2str: argument must be a function handle"
          );
        return RTV.string(v.name);
      },
    },
  ],
});
