/**
 * One descriptor per kind of runtime value: its MATLAB class, size, and
 * the facts behind `class`, `isa`, `isobject`, `isreal`, `size`, `numel`
 * and `isempty`. `KINDS` is a mapped type over every kind, so adding a
 * member to `RuntimeValue` without an entry here fails to compile.
 */

import type { RuntimeValue } from "./types.js";
import { hasClassMethods } from "./builtinRegistry.js";

/** Every kind of runtime value: the three primitives, then each
 *  container class's `kind`. */
export type ValueKind =
  | "number"
  | "boolean"
  | "string"
  | Exclude<RuntimeValue, number | boolean | string>["kind"];

type ValueOf<K extends ValueKind> = K extends "number"
  ? number
  : K extends "boolean"
    ? boolean
    : K extends "string"
      ? string
      : Extract<RuntimeValue, { kind: K }>;

export interface KindInfo<V> {
  /** What `class(v)` returns. */
  className(v: V): string;
  /** `size(v)`, with at least two dimensions. May be the value's own
   *  shape array: callers must not mutate it. */
  shape(v: V): number[];
  /** `numel(v)` (built-in, ignoring class overloads). */
  numel(v: V): number;
  /** `isobject(v)`: classdef objects and the built-in classes MATLAB
   *  implements as objects (string, dictionary, graphics). */
  isObject: boolean;
  /** `isreal(v)`. */
  isReal(v: V): boolean;
  /** Short internal description, for error messages. */
  label(v: V): string;
}

/** True when `imag` is absent or all zero. A complex tensor whose
 *  imaginary part is all zero (a JIT artifact) counts as real. */
function noImag(imag: ArrayLike<number> | undefined): boolean {
  if (!imag) return true;
  for (let i = 0; i < imag.length; i++) if (imag[i] !== 0) return false;
  return true;
}

const scalar = () => [1, 1];

/** Classes of graphics objects by trace type (default: Surface). */
const GRAPHICS_CLASSES: Record<string, string> = {
  contour: "matlab.graphics.chart.primitive.Contour",
  patch: "matlab.graphics.primitive.Patch",
  quiver3: "matlab.graphics.chart.primitive.Quiver",
};

export const KINDS: { [K in ValueKind]: KindInfo<ValueOf<K>> } = {
  number: {
    className: () => "double",
    shape: scalar,
    numel: () => 1,
    isObject: false,
    isReal: () => true,
    label: () => "number",
  },
  boolean: {
    className: () => "logical",
    shape: scalar,
    numel: () => 1,
    isObject: false,
    isReal: () => true,
    label: () => "logical",
  },
  string: {
    className: () => "string",
    shape: scalar,
    numel: () => 1,
    isObject: true,
    isReal: () => false,
    label: () => "string",
  },
  tensor: {
    className: v => (v._isLogical ? "logical" : "double"),
    shape: v =>
      v.shape.length >= 2
        ? v.shape
        : v.shape.length === 1
          ? [1, v.shape[0]]
          : [1, 1],
    numel: v => v.data.length,
    isObject: false,
    isReal: v => noImag(v.imag),
    label: () => "tensor",
  },
  complex_number: {
    className: () => "double",
    shape: scalar,
    numel: () => 1,
    isObject: false,
    isReal: v => v.im === 0,
    label: () => "complex number",
  },
  sparse_matrix: {
    className: () => "double",
    shape: v => [v.m, v.n],
    numel: v => v.m * v.n,
    isObject: false,
    isReal: v => noImag(v.pi),
    label: () => "sparse matrix",
  },
  char: {
    className: () => "char",
    shape: v =>
      v.shape ?? (v.value.length === 0 ? [0, 0] : [1, v.value.length]),
    numel: v => v.value.length,
    isObject: false,
    isReal: () => true,
    label: () => "char",
  },
  string_array: {
    className: () => "string",
    shape: v => [...v.shape],
    numel: v => v.data.length,
    isObject: true,
    isReal: () => false,
    label: () => "string array",
  },
  cell: {
    className: () => "cell",
    shape: v => v.shape,
    numel: v => v.data.length,
    isObject: false,
    isReal: () => false,
    label: () => "cell",
  },
  struct: {
    className: () => "struct",
    shape: scalar,
    numel: () => 1,
    isObject: false,
    isReal: () => false,
    label: () => "struct",
  },
  struct_array: {
    className: () => "struct",
    shape: v => v.shape,
    numel: v => v.elements.length,
    isObject: false,
    isReal: () => false,
    label: () => "struct array",
  },
  function: {
    className: () => "function_handle",
    shape: scalar,
    numel: () => 1,
    isObject: false,
    isReal: () => false,
    label: () => "function",
  },
  class_instance: {
    className: v => v.className,
    shape: scalar,
    numel: () => 1,
    isObject: true,
    // An object is real only through its built-in superclass data
    // (`classdef Foo < double`, a numeric enumeration).
    isReal: v => v._builtinData !== undefined && isRealValue(v._builtinData),
    label: v => `instance of ${v.className}`,
  },
  class_instance_array: {
    className: v => v.className,
    shape: v => [...v.shape],
    numel: v => v.elements.length,
    isObject: true,
    isReal: v =>
      v.elements.length > 0 &&
      v.elements.every(e => KINDS.class_instance.isReal(e)),
    label: v => `array of ${v.className} instances`,
  },
  dictionary: {
    className: () => "dictionary",
    shape: scalar,
    numel: () => 1,
    isObject: true,
    isReal: () => false,
    label: () => "dictionary",
  },
  graphics_handle: {
    className: v =>
      GRAPHICS_CLASSES[v._traceType] ?? "matlab.graphics.primitive.Surface",
    shape: scalar,
    numel: () => 1,
    isObject: true,
    isReal: () => false,
    label: () => "graphics handle",
  },
  dummy_handle: {
    className: () => "dummy_handle",
    shape: scalar,
    numel: () => 1,
    isObject: true,
    isReal: () => false,
    label: () => "dummy handle",
  },
};

export function kindOf(v: RuntimeValue): ValueKind {
  if (typeof v === "number") return "number";
  if (typeof v === "boolean") return "boolean";
  if (typeof v === "string") return "string";
  return v.kind;
}

/** The descriptor of a value's kind. */
export function kindInfo(v: RuntimeValue): KindInfo<RuntimeValue> {
  return KINDS[kindOf(v)] as unknown as KindInfo<RuntimeValue>;
}

/** MATLAB's class name of a value (what `class(v)` returns). */
export function valueClass(v: RuntimeValue): string {
  return kindInfo(v).className(v);
}

/** The MATLAB size of a value, with at least two dimensions. Do not
 *  mutate the result. */
export function valueShape(v: RuntimeValue): number[] {
  return kindInfo(v).shape(v);
}

export function valueNumel(v: RuntimeValue): number {
  return kindInfo(v).numel(v);
}

export function isEmptyValue(v: RuntimeValue): boolean {
  return kindInfo(v).numel(v) === 0;
}

export function isObjectValue(v: RuntimeValue): boolean {
  return kindInfo(v).isObject;
}

export function isRealValue(v: RuntimeValue): boolean {
  return kindInfo(v).isReal(v);
}

/** The class whose methods calls and operators on `v` dispatch to, or null
 *  for a value of a built-in type (double, char, cell, struct, ...). A
 *  classdef object (scalar or array) dispatches to its class, as does any
 *  object (`isObject`) whose class has methods registered from TypeScript
 *  (`registerClassMethod`). Resolution (a method of an object argument)
 *  and the operator hooks in `Runtime` use this one rule. */
export function dispatchClassOf(v: unknown): string | null {
  if (typeof v === "string") return hasClassMethods("string") ? "string" : null;
  if (typeof v !== "object" || v === null) return null;
  const rv = v as Exclude<RuntimeValue, number | boolean | string>;
  const info = KINDS[rv.kind] as KindInfo<RuntimeValue> | undefined;
  if (!info?.isObject) return null;
  if (rv.kind === "class_instance" || rv.kind === "class_instance_array")
    return rv.className;
  const cls = info.className(rv);
  return hasClassMethods(cls) ? cls : null;
}

/** For the default of an exhaustive switch: a new kind that reaches it
 *  is a compile error. */
export function assertNever(x: never): never {
  throw new Error(
    `unhandled value kind: ${String((x as { kind?: string })?.kind ?? x)}`
  );
}

/** Short internal description of a value's kind, for error messages. */
export function kstr(v: RuntimeValue): string {
  return kindInfo(v).label(v);
}

// ── isa ──────────────────────────────────────────────────────────────────

export const INTEGER_CLASSES: ReadonlySet<string> = new Set([
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
]);

const FLOAT_CLASSES: ReadonlySet<string> = new Set(["double", "single"]);

/** isa's category names and the built-in classes each one contains. */
const CATEGORIES = new Map<string, ReadonlySet<string>>([
  ["numeric", new Set([...FLOAT_CLASSES, ...INTEGER_CLASSES])],
  ["float", FLOAT_CLASSES],
  ["integer", INTEGER_CLASSES],
]);

function classIs(cls: string, name: string): boolean {
  return cls === name || (CATEGORIES.get(name)?.has(cls) ?? false);
}

/** `isa(v, name)`: `name` is the value's class, one of its superclasses,
 *  or a category ('numeric', 'float', 'integer') containing either.
 *  `ancestors` gives a class's superclasses (`Runtime.classAncestors`);
 *  without it, superclasses are not considered. */
export function isaValue(
  v: RuntimeValue,
  name: string,
  ancestors?: (className: string) => readonly string[]
): boolean {
  const cls = valueClass(v);
  if (classIs(cls, name)) return true;
  const kind = kindOf(v);
  if (
    !ancestors ||
    (kind !== "class_instance" && kind !== "class_instance_array")
  )
    return false;
  return ancestors(cls).some(a => classIs(a, name));
}
