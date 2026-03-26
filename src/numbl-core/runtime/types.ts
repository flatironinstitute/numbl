// Determine float precision from environment variable
const useFloat32 = import.meta.env?.NUMBL_USE_FLOAT32 === "true" ? true : false;

export const USE_FLOAT32 = useFloat32;
export const FloatXArray = useFloat32 ? Float32Array : Float64Array;
export type FloatXArrayType = Float32Array | Float64Array;

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
  | RuntimeStructArray
  | RuntimeSparseMatrix;

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
export const isRuntimeStructArray = (
  value: RuntimeValue
): value is RuntimeStructArray =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeStructArray).kind === "struct_array";
export const isRuntimeSparseMatrix = (
  value: RuntimeValue
): value is RuntimeSparseMatrix =>
  typeof value === "object" &&
  value !== null &&
  (value as RuntimeSparseMatrix).kind === "sparse_matrix";

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
  if (isRuntimeComplexNumber(value)) return "complex number";
  if (isRuntimeDummyHandle(value)) return "dummy handle";
  if (isRuntimeStructArray(value)) return "struct array";
  if (isRuntimeSparseMatrix(value)) return "sparse matrix";
  return "unknown";
};

export type RuntimeTensor = {
  kind: "tensor";
  data: FloatXArrayType; // real part
  imag?: FloatXArrayType; // imaginary part (optional, undefined means all zeros)
  shape: number[]; // e.g. [3,4] for 3x4 matrix
  /** When true, this tensor represents a logical (boolean) array from comparisons/logical ops. */
  _isLogical?: boolean;
  /** Reference count for copy-on-write. When > 1, data is shared and must be copied before mutation. */
  _rc: number;
};

export type RuntimeChar = {
  kind: "char";
  value: string;
  /** Optional shape for multi-row char arrays. If absent, shape is [1, value.length].
   *  For multi-row arrays, value contains all rows concatenated (each row is shape[1] chars). */
  shape?: number[];
};

export type RuntimeCell = {
  kind: "cell";
  data: RuntimeValue[];
  shape: number[]; // e.g. [1,3] for {a, b, c}
  /** Reference count for copy-on-write. When > 1, data is shared and must be copied before mutation. */
  _rc: number;
};

export type RuntimeStruct = {
  kind: "struct";
  fields: Map<string, RuntimeValue>;
};

export type RuntimeFunction = {
  kind: "function";
  name: string;
  /** For closures: captured variables */
  captures: RuntimeValue[];
  /** The underlying callable — either a builtin or user-defined function name */
  impl: "builtin" | "user";
  /** For anonymous functions and user function handles: the underlying JS closure */
  jsFn?: (...args: unknown[]) => unknown;
  /** When true, jsFn expects nargout as its first argument */
  jsFnExpectsNargout?: boolean;
  /** Number of input parameters (for nargin(handle)) */
  nargin?: number;
};

export type RuntimeClassInstance = {
  kind: "class_instance";
  className: string;
  fields: Map<string, RuntimeValue>;
  /** True if this class inherits from handle (reference semantics). */
  isHandleClass: boolean;
  /** For classes that inherit from built-in types (e.g. classdef Foo < double),
   *  stores the underlying built-in data. */
  _builtinData?: RuntimeValue;
};

export type RuntimeComplexNumber = {
  kind: "complex_number";
  re: number;
  im: number;
};

export type RuntimeDummyHandle = {
  kind: "dummy_handle";
};

/** A 1-D array of structs that all share the same field names. **/
export type RuntimeStructArray = {
  kind: "struct_array";
  fieldNames: string[];
  elements: RuntimeStruct[];
};

/** Sparse matrix in CSC (Compressed Sparse Column) format, matching MATLAB's internal representation. */
export type RuntimeSparseMatrix = {
  kind: "sparse_matrix";
  m: number; // number of rows
  n: number; // number of columns
  ir: Int32Array; // row indices for each nonzero (length = nnz)
  jc: Int32Array; // column pointers (length = n + 1)
  pr: Float64Array; // nonzero values (length = nnz)
  pi?: Float64Array; // imaginary values (length = nnz), undefined means real
  _rc: number; // reference count for copy-on-write
};
