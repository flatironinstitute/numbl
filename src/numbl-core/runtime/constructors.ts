/**
 * RuntimeValue constructor helpers (the RTV namespace).
 */

import { allocFloat64Array } from "./alloc.js";
import { ItemType } from "../lowering/itemTypes.js";
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
  RuntimeComplexNumber,
  RuntimeDummyHandle,
  RuntimeGraphicsHandle,
  RuntimeStructArray,
  RuntimeSparseMatrix,
  RuntimeDictionary,
  type RuntimeValue,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
} from "./types.js";

export const RTV = {
  num(value: number): RuntimeNumber {
    return value;
  },

  tensor(
    data: Float64Array | number[],
    shape: number[],
    imag?: Float64Array | number[]
  ): RuntimeTensor {
    const d = data instanceof Float64Array ? data : allocFloat64Array(data);
    const im = imag
      ? imag instanceof Float64Array
        ? imag
        : allocFloat64Array(imag)
      : undefined;
    // Strip trailing singleton dimensions (always keeps minimum 2D)
    const s = [...shape];
    while (s.length > 2 && s[s.length - 1] === 1) s.pop();
    return new RuntimeTensor(d, s, im);
  },

  /** Fast tensor constructor — data must be Float64Array, shape already normalized (no trailing singletons). */
  tensorRaw(data: Float64Array, shape: number[]): RuntimeTensor {
    return new RuntimeTensor(data, shape);
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

  /** Create a matrix from row-major data */
  matrix(
    rows: number,
    cols: number,
    data: number[] | Float64Array,
    imag?: number[] | Float64Array
  ): RuntimeTensor {
    const d = data instanceof Float64Array ? data : allocFloat64Array(data);
    const im = imag
      ? imag instanceof Float64Array
        ? imag
        : allocFloat64Array(imag)
      : undefined;
    return new RuntimeTensor(d, [rows, cols], im);
  },

  string(value: string): RuntimeString {
    return value;
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
    elements: RuntimeStruct[]
  ): RuntimeStructArray {
    return new RuntimeStructArray(fieldNames, elements);
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

export const getItemTypeFromRuntimeValue = (value: RuntimeValue): ItemType => {
  if (isRuntimeNumber(value)) {
    return { kind: "Number" }; // important to not include the value here or we end up jitting every numeric literal as a separate type!
  }
  if (isRuntimeLogical(value)) {
    return { kind: "Boolean" };
  }
  if (isRuntimeString(value)) {
    return { kind: "String" }; // same note as for RuntimeNumber
  }
  switch (value.kind) {
    case "complex_number":
      return { kind: "ComplexNumber" };
    case "char":
      return { kind: "Char" };
    case "tensor":
      return {
        kind: "Tensor",
        isComplex: value.imag !== undefined,
        isLogical: value._isLogical || undefined,
      };
    case "cell":
      return {
        kind: "Cell",
        elementType:
          value.data.length > 0
            ? getItemTypeFromRuntimeValue(value.data[0])
            : { kind: "Unknown" },
        // length: value.shape.reduce((a, b) => a * b, 1),
        length: "unknown", // do not include length or we get too many jits
      };
    case "struct": {
      const knownFields: Record<
        string,
        import("../lowering/itemTypes.js").ItemType
      > = {};
      for (const [k, v] of value.fields) {
        knownFields[k] = getItemTypeFromRuntimeValue(v);
      }
      return { kind: "Struct", knownFields };
    }
    case "function":
      return {
        kind: "Function",
        params: [],
        returns: { kind: "Unknown" },
      };
    case "class_instance":
      return {
        kind: "ClassInstance",
        className: value.className,
      };
    case "class_instance_array":
      return {
        kind: "ClassInstance",
        className: value.className,
      };
    case "dummy_handle":
      return { kind: "DummyHandle" };
    case "graphics_handle":
      return { kind: "DummyHandle" };
    case "struct_array":
      return { kind: "Unknown" };
    case "sparse_matrix":
      return { kind: "SparseMatrix", isComplex: value.pi !== undefined };
    case "dictionary":
      return { kind: "Dictionary" };
    default:
      return { kind: "Unknown" };
  }
};
