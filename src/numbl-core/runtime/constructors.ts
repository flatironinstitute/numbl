/**
 * RuntimeValue constructor helpers (the RTV namespace).
 */

import { allocFloat64Array } from "./alloc.js";
import {
  type RuntimeNumber,
  RuntimeTensor,
  type RuntimeString,
  RuntimeChar,
  type RuntimeLogical,
  RuntimeCell,
  RuntimeStruct,
  RuntimeFunction,
  RuntimeClassInstance,
  RuntimeClassInstanceArray,
  RuntimeComplexNumber,
  RuntimeDummyHandle,
  RuntimeGraphicsHandle,
  RuntimeStructArray,
  RuntimeSparseMatrix,
  RuntimeDictionary,
  RuntimeStringArray,
  type RuntimeValue,
  type ElementClass,
} from "./types.js";

export const RTV = {
  num(value: number): RuntimeNumber {
    return value;
  },

  /** Tensor over `data` (taken, not copied), with trailing singleton
   *  dimensions stripped. `data` and `imag` must be Float64Arrays: other
   *  storage is rejected rather than silently widened to double. */
  tensor(
    data: Float64Array,
    shape: number[],
    imag?: Float64Array,
    cls: ElementClass = "double"
  ): RuntimeTensor {
    if (!(data instanceof Float64Array))
      throw new Error("internal error: RTV.tensor data must be a Float64Array");
    if (imag !== undefined && !(imag instanceof Float64Array))
      throw new Error("internal error: RTV.tensor imag must be a Float64Array");
    // Strip trailing singleton dimensions (always keeps minimum 2D)
    const s = [...shape];
    while (s.length > 2 && s[s.length - 1] === 1) s.pop();
    return new RuntimeTensor(data, s, imag, cls);
  },

  /** Logical tensor over `data` (0/1 values, taken, not copied). */
  logicalTensor(data: Float64Array, shape: number[]): RuntimeTensor {
    return RTV.tensor(data, shape, undefined, "logical");
  },

  /** Double tensor copied from plain number arrays. */
  tensorFromArray(
    data: number[],
    shape: number[],
    imag?: number[]
  ): RuntimeTensor {
    return RTV.tensor(
      allocFloat64Array(data),
      shape,
      imag && allocFloat64Array(imag)
    );
  },

  /** Fast tensor constructor — data must be Float64Array, shape already normalized (no trailing singletons). */
  tensorRaw(
    data: Float64Array,
    shape: number[],
    cls: ElementClass = "double"
  ): RuntimeTensor {
    return new RuntimeTensor(data, shape, undefined, cls);
  },

  /** Create a scalar tensor (1x1) */
  scalar(value: number): RuntimeNumber {
    return value;
  },

  /** Create a row vector [1 x n] */
  row(data: number[], imag?: number[]): RuntimeTensor {
    const im = imag ? allocFloat64Array(imag) : undefined;
    return new RuntimeTensor(allocFloat64Array(data), [1, data.length], im);
  },

  /** Create a column vector [n x 1] */
  col(data: number[], imag?: number[]): RuntimeTensor {
    const im = imag ? allocFloat64Array(imag) : undefined;
    return new RuntimeTensor(allocFloat64Array(data), [data.length, 1], im);
  },

  string(value: string): RuntimeString {
    return value;
  },

  /** String array (does NOT collapse 1x1 — use stringArrayValue for that). */
  stringArray(data: string[], shape: [number, number]): RuntimeStringArray {
    return new RuntimeStringArray(data, shape);
  },

  char(value: string): RuntimeChar {
    return new RuntimeChar(value);
  },

  logical(value: boolean): RuntimeLogical {
    return value;
  },

  cell(data: RuntimeValue[], shape: number[]): RuntimeCell {
    return new RuntimeCell(data, [...shape]);
  },

  struct(
    fields: Map<string, RuntimeValue> | Record<string, RuntimeValue>
  ): RuntimeStruct {
    const map =
      fields instanceof Map ? fields : new Map(Object.entries(fields));
    return new RuntimeStruct(map);
  },

  func(
    name: string,
    impl: "builtin" | "user",
    captures: RuntimeValue[] = []
  ): RuntimeFunction {
    return new RuntimeFunction(name, impl, captures);
  },

  classInstance(
    className: string,
    propertyNames: string[],
    isHandleClass = false,
    defaults?: Map<string, RuntimeValue>
  ): RuntimeClassInstance {
    const fields = new Map<string, RuntimeValue>();
    for (const name of propertyNames) {
      // Default for unspecified properties is [] (empty double)
      fields.set(
        name,
        defaults?.get(name) ?? RTV.tensor(allocFloat64Array(0), [0, 0])
      );
    }
    return new RuntimeClassInstance(className, fields, isHandleClass);
  },

  classInstanceArray(
    className: string,
    elements: RuntimeClassInstance[],
    shape?: [number, number]
  ): RuntimeClassInstanceArray {
    return new RuntimeClassInstanceArray(className, elements, shape);
  },

  complex(re: number, im: number): RuntimeComplexNumber {
    return new RuntimeComplexNumber(re, im);
  },

  dummyHandle(): RuntimeDummyHandle {
    return new RuntimeDummyHandle();
  },

  graphicsHandle(
    trace: Record<string, unknown>,
    traceType: string
  ): RuntimeGraphicsHandle {
    return new RuntimeGraphicsHandle(trace, traceType);
  },

  structArray(
    fieldNames: string[],
    elements: RuntimeStruct[],
    shape?: number[]
  ): RuntimeStructArray {
    return new RuntimeStructArray(fieldNames, elements, shape);
  },

  sparseMatrix(
    m: number,
    n: number,
    ir: Int32Array,
    jc: Int32Array,
    pr: Float64Array,
    pi?: Float64Array
  ): RuntimeSparseMatrix {
    return new RuntimeSparseMatrix(m, n, ir, jc, pr, pi);
  },

  dictionary(
    entries?: Map<string, { key: RuntimeValue; value: RuntimeValue }>,
    keyType?: string,
    valueType?: string
  ): RuntimeDictionary {
    return new RuntimeDictionary(entries, keyType, valueType);
  },
};
