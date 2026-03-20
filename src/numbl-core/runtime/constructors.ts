/**
 * RuntimeValue constructor helpers (the MV namespace).
 */

import { ItemType } from "../lowering/itemTypes.js";
import {
  type RuntimeNumber,
  type RuntimeTensor,
  type RuntimeString,
  type RuntimeChar,
  type RuntimeLogical,
  type RuntimeCell,
  type RuntimeStruct,
  type RuntimeFunction,
  type RuntimeClassInstance,
  type RuntimeComplexNumber,
  type RuntimeDummyHandle,
  type RuntimeStructArray,
  type RuntimeSparseMatrix,
  type RuntimeValue,
  type FloatXArrayType,
  FloatXArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
} from "./types.js";

export const RTV = {
  num(value: number): RuntimeNumber {
    return value;
  },

  tensor(
    data: FloatXArrayType | number[],
    shape: number[],
    imag?: FloatXArrayType | number[]
  ): RuntimeTensor {
    const d = data instanceof FloatXArray ? data : new FloatXArray(data);
    const im = imag
      ? imag instanceof FloatXArray
        ? imag
        : new FloatXArray(imag)
      : undefined;
    // Strip trailing singleton dimensions (always keeps minimum 2D)
    const s = [...shape];
    while (s.length > 2 && s[s.length - 1] === 1) s.pop();
    return { kind: "tensor", data: d, imag: im, shape: s, _rc: 1 };
  },

  /** Fast tensor constructor — data must be FloatXArray, shape already normalized (no trailing singletons). */
  tensorRaw(data: FloatXArrayType, shape: number[]): RuntimeTensor {
    return { kind: "tensor", data, imag: undefined, shape, _rc: 1 };
  },

  /** Create a scalar tensor (1x1) */
  scalar(value: number): RuntimeNumber {
    return value;
  },

  /** Create a row vector [1 x n] */
  row(data: number[], imag?: number[]): RuntimeTensor {
    const im = imag ? new FloatXArray(imag) : undefined;
    return {
      kind: "tensor",
      data: new FloatXArray(data),
      imag: im,
      shape: [1, data.length],
      _rc: 1,
    };
  },

  /** Create a column vector [n x 1] */
  col(data: number[], imag?: number[]): RuntimeTensor {
    const im = imag ? new FloatXArray(imag) : undefined;
    return {
      kind: "tensor",
      data: new FloatXArray(data),
      imag: im,
      shape: [data.length, 1],
      _rc: 1,
    };
  },

  /** Create a matrix from row-major data */
  matrix(
    rows: number,
    cols: number,
    data: number[] | FloatXArrayType,
    imag?: number[] | FloatXArrayType
  ): RuntimeTensor {
    const d = data instanceof FloatXArray ? data : new FloatXArray(data);
    const im = imag
      ? imag instanceof FloatXArray
        ? imag
        : new FloatXArray(imag)
      : undefined;
    return { kind: "tensor", data: d, imag: im, shape: [rows, cols], _rc: 1 };
  },

  string(value: string): RuntimeString {
    return value;
  },

  char(value: string): RuntimeChar {
    return { kind: "char", value };
  },

  logical(value: boolean): RuntimeLogical {
    return value;
  },

  cell(data: RuntimeValue[], shape: number[]): RuntimeCell {
    return { kind: "cell", data, shape: [...shape], _rc: 1 };
  },

  struct(
    fields: Map<string, RuntimeValue> | Record<string, RuntimeValue>
  ): RuntimeStruct {
    const map =
      fields instanceof Map ? fields : new Map(Object.entries(fields));
    return { kind: "struct", fields: map };
  },

  func(
    name: string,
    impl: "builtin" | "user",
    captures: RuntimeValue[] = []
  ): RuntimeFunction {
    return { kind: "function", name, impl, captures };
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
        defaults?.get(name) ?? RTV.tensor(new FloatXArray(0), [0, 0])
      );
    }
    return {
      kind: "class_instance",
      className,
      fields,
      isHandleClass,
    };
  },

  complex(re: number, im: number): RuntimeComplexNumber {
    return { kind: "complex_number", re, im };
  },

  dummyHandle(): RuntimeDummyHandle {
    return { kind: "dummy_handle" };
  },

  structArray(
    fieldNames: string[],
    elements: RuntimeStruct[]
  ): RuntimeStructArray {
    return { kind: "struct_array", fieldNames, elements };
  },

  sparseMatrix(
    m: number,
    n: number,
    ir: Int32Array,
    jc: Int32Array,
    pr: Float64Array,
    pi?: Float64Array
  ): RuntimeSparseMatrix {
    return { kind: "sparse_matrix", m, n, ir, jc, pr, pi, _rc: 1 };
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
    case "dummy_handle":
      return { kind: "DummyHandle" };
    case "struct_array":
      return { kind: "Unknown" };
    case "sparse_matrix":
      return { kind: "SparseMatrix", isComplex: value.pi !== undefined };
    default:
      return { kind: "Unknown" };
  }
};
