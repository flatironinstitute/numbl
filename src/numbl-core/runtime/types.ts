// Determine float precision from environment variable
import { Refcounted, type RefcountRuntime, decref } from "./refcount.js";

const useFloat32 = import.meta.env?.NUMBL_USE_FLOAT32 === "true" ? true : false;

export const USE_FLOAT32 = useFloat32;

export type RuntimeValue =
  | number
  | boolean
  | string
  | RuntimeTensor
  | RuntimeChar
  | RuntimeCell
  | RuntimeStruct
  | RuntimeFunction
  | RuntimeClassInstance
  | RuntimeComplexNumber
  | RuntimeDummyHandle
  | RuntimeGraphicsHandle
  | RuntimeStructArray
  | RuntimeClassInstanceArray
  | RuntimeSparseMatrix
  | RuntimeDictionary;

export type RuntimeNumber = number;
export type RuntimeLogical = boolean;
export type RuntimeString = string;

export const isRuntimeNumber = (value: RuntimeValue): value is RuntimeNumber =>
  typeof value === "number";
export const isRuntimeLogical = (
  value: RuntimeValue
): value is RuntimeLogical => typeof value === "boolean";
export const isRuntimeString = (value: RuntimeValue): value is RuntimeString =>
  typeof value === "string";
export const isRuntimeChar = (value: RuntimeValue): value is RuntimeChar =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeChar).kind === "char";
export const isRuntimeTensor = (value: RuntimeValue): value is RuntimeTensor =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeTensor).kind === "tensor";
export const isRuntimeCell = (value: RuntimeValue): value is RuntimeCell =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeCell).kind === "cell";
export const isRuntimeStruct = (value: RuntimeValue): value is RuntimeStruct =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeStruct).kind === "struct";
export const isRuntimeFunction = (
  value: RuntimeValue
): value is RuntimeFunction =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeFunction).kind === "function";
export const isRuntimeClassInstance = (
  value: RuntimeValue
): value is RuntimeClassInstance =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeClassInstance).kind === "class_instance";
export const isRuntimeComplexNumber = (
  value: RuntimeValue
): value is RuntimeComplexNumber =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeComplexNumber).kind === "complex_number";
export const isRuntimeDummyHandle = (
  value: RuntimeValue
): value is RuntimeDummyHandle =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeDummyHandle).kind === "dummy_handle";
export const isRuntimeGraphicsHandle = (
  value: RuntimeValue
): value is RuntimeGraphicsHandle =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeGraphicsHandle).kind === "graphics_handle";
export const isRuntimeStructArray = (
  value: RuntimeValue
): value is RuntimeStructArray =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeStructArray).kind === "struct_array";
export const isRuntimeClassInstanceArray = (
  value: RuntimeValue
): value is RuntimeClassInstanceArray =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeClassInstanceArray).kind === "class_instance_array";
export const isRuntimeSparseMatrix = (
  value: RuntimeValue
): value is RuntimeSparseMatrix =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeSparseMatrix).kind === "sparse_matrix";
export const isRuntimeDictionary = (
  value: RuntimeValue
): value is RuntimeDictionary =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeDictionary).kind === "dictionary";

export const kstr = (value: RuntimeValue): string => {
  if (isRuntimeNumber(value)) return "number";
  if (isRuntimeLogical(value)) return "logical";
  if (isRuntimeString(value)) return "string";
  if (isRuntimeChar(value)) return "char";
  if (isRuntimeTensor(value)) return "tensor";
  if (isRuntimeCell(value)) return "cell";
  if (isRuntimeStruct(value)) return "struct";
  if (isRuntimeFunction(value)) return "function";
  if (isRuntimeClassInstance(value)) return `instance of ${value.className}`;
  if (isRuntimeClassInstanceArray(value))
    return `array of ${value.className} instances`;
  if (isRuntimeComplexNumber(value)) return "complex number";
  if (isRuntimeDummyHandle(value)) return "dummy handle";
  if (isRuntimeGraphicsHandle(value)) return "graphics handle";
  if (isRuntimeStructArray(value)) return "struct array";
  if (isRuntimeSparseMatrix(value)) return "sparse matrix";
  if (isRuntimeDictionary(value)) return "dictionary";
  return "unknown";
};

// ── Container classes ────────────────────────────────────────────────────
//
// Each class extends Refcounted. The `kind` field is preserved as an
// instance property so the existing `value.kind === "..."` checks still
// narrow correctly. Public mutable fields are kept for now — strict-API
// enforcement comes in phase 3 when mutation methods replace direct
// field writes.

export class RuntimeTensor extends Refcounted {
  readonly kind = "tensor" as const;
  data: Float64Array;
  imag: Float64Array | undefined;
  shape: number[];
  /** When true, this tensor represents a logical (boolean) array from
   *  comparisons/logical ops. */
  _isLogical: boolean | undefined;

  constructor(
    data: Float64Array,
    shape: number[],
    imag?: Float64Array,
    _isLogical?: boolean
  ) {
    super();
    this.data = data;
    this.imag = imag;
    this.shape = shape;
    this._isLogical = _isLogical;
  }

  protected _destroy(rt: RefcountRuntime): void {
    if (rt.poolReclaim) {
      rt.pool.release(this.data);
      if (this.imag !== undefined) rt.pool.release(this.imag);
    }
  }
}

export class RuntimeChar extends Refcounted {
  readonly kind = "char" as const;
  value: string;
  /** Optional shape for multi-row char arrays. If absent, shape is
   *  [1, value.length]. For multi-row arrays, value contains all rows
   *  concatenated (each row is shape[1] chars). */
  shape: number[] | undefined;

  constructor(value: string, shape?: number[]) {
    super();
    this.value = value;
    this.shape = shape;
  }
}

export class RuntimeCell extends Refcounted {
  readonly kind = "cell" as const;
  data: RuntimeValue[];
  shape: number[];

  constructor(data: RuntimeValue[], shape: number[]) {
    super();
    this.data = data;
    this.shape = shape;
  }

  protected _destroy(rt: RefcountRuntime): void {
    for (const v of this.data) decref(rt, v);
  }
}

export class RuntimeStruct extends Refcounted {
  readonly kind = "struct" as const;
  fields: Map<string, RuntimeValue>;

  constructor(fields: Map<string, RuntimeValue>) {
    super();
    this.fields = fields;
  }

  protected _destroy(rt: RefcountRuntime): void {
    for (const v of this.fields.values()) decref(rt, v);
  }
}

export class RuntimeFunction extends Refcounted {
  readonly kind = "function" as const;
  name: string;
  /** For closures: captured variables */
  captures: RuntimeValue[];
  /** The underlying callable — either a builtin or user-defined function name */
  impl: "builtin" | "user";
  /** For anonymous functions and user function handles: the underlying JS closure */
  jsFn: ((...args: unknown[]) => unknown) | undefined;
  /** When true, jsFn expects nargout as its first argument */
  jsFnExpectsNargout: boolean | undefined;
  /** Number of input parameters (for nargin(handle)) */
  nargin: number | undefined;

  constructor(
    name: string,
    impl: "builtin" | "user",
    captures: RuntimeValue[],
    jsFn?: (...args: unknown[]) => unknown,
    jsFnExpectsNargout?: boolean,
    nargin?: number
  ) {
    super();
    this.name = name;
    this.impl = impl;
    this.captures = captures;
    this.jsFn = jsFn;
    this.jsFnExpectsNargout = jsFnExpectsNargout;
    this.nargin = nargin;
  }

  protected _destroy(rt: RefcountRuntime): void {
    for (const v of this.captures) decref(rt, v);
  }
}

export class RuntimeClassInstance extends Refcounted {
  readonly kind = "class_instance" as const;
  className: string;
  fields: Map<string, RuntimeValue>;
  /** True if this class inherits from handle (reference semantics). */
  isHandleClass: boolean;
  /** For classes that inherit from built-in types (e.g. classdef Foo < double),
   *  stores the underlying built-in data. */
  _builtinData: RuntimeValue | undefined;

  constructor(
    className: string,
    fields: Map<string, RuntimeValue>,
    isHandleClass: boolean,
    _builtinData?: RuntimeValue
  ) {
    super();
    this.className = className;
    this.fields = fields;
    this.isHandleClass = isHandleClass;
    this._builtinData = _builtinData;
  }

  protected _destroy(rt: RefcountRuntime): void {
    for (const v of this.fields.values()) decref(rt, v);
    if (this._builtinData !== undefined) decref(rt, this._builtinData);
  }
}

/** A 1-D array of class instances that all share the same class.
 *  Created by default horzcat/vertcat when the class doesn't overload them. */
export class RuntimeClassInstanceArray extends Refcounted {
  readonly kind = "class_instance_array" as const;
  className: string;
  elements: RuntimeClassInstance[];

  constructor(className: string, elements: RuntimeClassInstance[]) {
    super();
    this.className = className;
    this.elements = elements;
  }

  protected _destroy(rt: RefcountRuntime): void {
    for (const el of this.elements) decref(rt, el);
  }
}

export class RuntimeComplexNumber extends Refcounted {
  readonly kind = "complex_number" as const;
  re: number;
  im: number;

  constructor(re: number, im: number) {
    super();
    this.re = re;
    this.im = im;
  }
}

export class RuntimeDummyHandle extends Refcounted {
  readonly kind = "dummy_handle" as const;

  constructor() {
    super();
  }
}

/** Handle to a graphics object (e.g. surface returned by pcolor) with a mutable trace reference. */
export class RuntimeGraphicsHandle extends Refcounted {
  readonly kind = "graphics_handle" as const;
  _trace: Record<string, unknown>;
  _traceType: string;

  constructor(_trace: Record<string, unknown>, _traceType: string) {
    super();
    this._trace = _trace;
    this._traceType = _traceType;
  }
}

/** A 1-D array of structs that all share the same field names. */
export class RuntimeStructArray extends Refcounted {
  readonly kind = "struct_array" as const;
  fieldNames: string[];
  elements: RuntimeStruct[];

  constructor(fieldNames: string[], elements: RuntimeStruct[]) {
    super();
    this.fieldNames = fieldNames;
    this.elements = elements;
  }

  protected _destroy(rt: RefcountRuntime): void {
    for (const el of this.elements) decref(rt, el);
  }
}

/** Sparse matrix in CSC (Compressed Sparse Column) format, matching MATLAB's
 *  internal representation. */
export class RuntimeSparseMatrix extends Refcounted {
  readonly kind = "sparse_matrix" as const;
  m: number;
  n: number;
  ir: Int32Array;
  jc: Int32Array;
  pr: Float64Array;
  pi: Float64Array | undefined;

  constructor(
    m: number,
    n: number,
    ir: Int32Array,
    jc: Int32Array,
    pr: Float64Array,
    pi?: Float64Array
  ) {
    super();
    this.m = m;
    this.n = n;
    this.ir = ir;
    this.jc = jc;
    this.pr = pr;
    this.pi = pi;
  }

  protected _destroy(rt: RefcountRuntime): void {
    if (rt.poolReclaim) {
      rt.pool.release(this.pr);
      if (this.pi !== undefined) rt.pool.release(this.pi);
    }
    // ir/jc are Int32Array — pool is Float64-only, so they're left to JS GC.
  }
}

/** Dictionary mapping unique keys to values (MATLAB R2022b+). */
export class RuntimeDictionary extends Refcounted {
  readonly kind = "dictionary" as const;
  /** Entries keyed by a hash string of the RuntimeValue key, preserving insertion order. */
  entries: Map<string, { key: RuntimeValue; value: RuntimeValue }>;
  keyType: string | undefined;
  valueType: string | undefined;

  constructor(
    entries?: Map<string, { key: RuntimeValue; value: RuntimeValue }>,
    keyType?: string,
    valueType?: string
  ) {
    super();
    this.entries = entries ?? new Map();
    this.keyType = keyType;
    this.valueType = valueType;
  }

  protected _destroy(rt: RefcountRuntime): void {
    for (const { key, value } of this.entries.values()) {
      decref(rt, key);
      decref(rt, value);
    }
  }
}
