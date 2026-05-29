/**
 * mtoc2 type system. Built from scratch — not derived from numbl's
 * JitType. Designed to be the long-term home for static type info,
 * with "exact known value" tracking as a first-class feature on
 * every scalar (and eventually small arrays).
 *
 * MVP scope: scalar real double only. The full Type union and the
 * NumericType field layout already accommodate growth (complex,
 * arrays, strings, logicals, structs, classes) but the lowerer and
 * codegen reject anything outside scope with UnsupportedConstruct.
 */

import type { Span, Stmt } from "../parser/index.js";

type FunctionStmt = Extract<Stmt, { type: "Function" }>;

// ── Sign lattice ────────────────────────────────────────────────────────

export type Sign =
  | "positive" // > 0
  | "nonneg" // >= 0
  | "negative" // < 0
  | "nonpositive" // <= 0
  | "zero" // == 0
  | "nonzero" // != 0
  | "unknown";

export function signFromNumber(v: number): Sign {
  if (Number.isNaN(v)) return "unknown";
  if (v > 0) return "positive";
  if (v < 0) return "negative";
  return "zero";
}

/** Derive the sign of a tensor from its exact data. Returns the
 *  tightest lattice state that holds across every element:
 *  - all `>0` → `positive`
 *  - all `<0` → `negative`
 *  - all `===0` → `zero`
 *  - mix of `>0` and `==0` → `nonneg`
 *  - mix of `<0` and `==0` → `nonpositive`
 *  - mix of `>0` and `<0` (no zeros) → `nonzero`
 *  - any NaN, or mix of all three (positives, negatives, zeros) → `unknown`.
 *
 *  Empty data returns `unknown` (no elements to constrain). */
export function signFromExactArray(data: Float64Array): Sign {
  if (data.length === 0) return "unknown";
  let anyPos = false;
  let anyNeg = false;
  let anyZero = false;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isNaN(v)) return "unknown";
    if (v > 0) anyPos = true;
    else if (v < 0) anyNeg = true;
    else anyZero = true;
  }
  if (anyPos && anyNeg && anyZero) return "unknown";
  if (anyPos && anyNeg) return "nonzero";
  if (anyPos && anyZero) return "nonneg";
  if (anyNeg && anyZero) return "nonpositive";
  if (anyPos) return "positive";
  if (anyNeg) return "negative";
  return "zero";
}

export function unifySign(a: Sign, b: Sign): Sign {
  if (a === b) return a;
  const s = new Set([a, b]);
  if (s.has("positive") && s.has("zero")) return "nonneg";
  if (s.has("positive") && s.has("nonneg")) return "nonneg";
  if (s.has("negative") && s.has("zero")) return "nonpositive";
  if (s.has("negative") && s.has("nonpositive")) return "nonpositive";
  if (s.has("nonzero") && s.has("positive")) return "nonzero";
  if (s.has("nonzero") && s.has("negative")) return "nonzero";
  return "unknown";
}

export function flipSign(s: Sign): Sign {
  switch (s) {
    case "positive":
      return "negative";
    case "negative":
      return "positive";
    case "nonneg":
      return "nonpositive";
    case "nonpositive":
      return "nonneg";
    default:
      return s;
  }
}

// ── Dimensions ──────────────────────────────────────────────────────────

/** Per-axis dimension knowledge. Either the exact non-negative integer
 *  size is known, or we have no info. There's no separate ">1" lattice
 *  state — call sites that previously cared about "definitely > 1"
 *  now check `d.kind === "exact" && d.value !== 1` (or `value > 1`,
 *  depending on whether a zero-sized axis should count).
 *  When all entries in a `dims` array are `exact`, the surrounding
 *  `NumericType.shape` is also set; the two views never disagree. */
export type DimInfo = { kind: "exact"; value: number } | { kind: "unknown" };

export const DIM_ONE: DimInfo = { kind: "exact", value: 1 };

/** Convenience: does this `DimInfo` statically pin the axis to 1? */
export function isDimOne(d: DimInfo): boolean {
  return d.kind === "exact" && d.value === 1;
}

/** Product of every entry in `shape` — i.e. `numel` for a tensor with
 *  this concrete shape. Returns 1 for an empty shape array (a 0-D
 *  scalar in MATLAB terms; mtoc2 always pads to 2-D so this is rare
 *  but well-defined). */
export function shapeNumel(shape: ReadonlyArray<number>): number {
  let n = 1;
  for (const s of shape) n *= s;
  return n;
}

// ── Exact value (for scalars; arrays later, capped) ─────────────────────

/** Cap for how big an "exact array" we'll propagate through the type
 *  system. Anything larger drops to non-exact. */
export const EXACT_ARRAY_MAX_ELEMENTS = 256;

/** Max rank for `mtoc2_tensor_t`. Mirror of `MTOC2_MAX_NDIM` in
 *  `src/codegen/runtime/tensor.h` — kept in lockstep with the runtime
 *  helpers that allocate per-axis stride/index buffers on the stack. */
export const MTOC2_MAX_NDIM = 8;

/** Exact-value variants.
 *  - `number`: scalar real.
 *  - `{re,im}`: scalar complex.
 *  - `Float64Array`: dense real array, column-major (matches numbl's
 *    `RuntimeTensor.data`). Shape is carried by `dims` on `NumericType`.
 *  - `{re: Float64Array; im: Float64Array}`: dense complex array,
 *    column-major split-buffer (mirrors the runtime layout). */
export type NumericExact =
  | number
  | { re: number; im: number }
  | Float64Array
  | { re: Float64Array; im: Float64Array };

// ── Numeric scalar/tensor type ──────────────────────────────────────────

export type NumericElem = "double" | "logical" | "char";

export interface NumericType {
  kind: "Numeric";
  elem: NumericElem;
  isComplex: boolean;
  /** Per-axis dim info: each axis is either `{exact, value}` or
   *  `unknown`. Always present. Length 2 for scalars
   *  (`[{exact,1},{exact,1}]`); same length as `shape` when `shape`
   *  is set. */
  dims: DimInfo[];
  /** Statically-known integer shape, when available (always set when
   *  every entry in `dims` is `exact`; mirrors those values).
   *  `shape[i]` equals `dims[i].value` for every axis. */
  shape?: number[];
  sign: Sign;
  exact?: NumericExact;
}

/** Scalar string handle — the double-quoted (`"foo"`) text type.
 *  `length("foo") == 1` (numbl treats it as a single value, not a
 *  byte array). Backed by `mtoc2_string_t` in emitted C; participates
 *  in the standard owned-value lifecycle (empty/assign/copy/free).
 *  `exact` carries the literal value when the source is a `"..."`
 *  expression. */
export interface StringType {
  kind: "String";
  exact?: string;
}

/** 1×N row-vector of bytes — the single-quoted (`'foo'`) char-array
 *  type. `length('foo') == 3` (numbl exposes the byte count). Backed
 *  by `mtoc2_char_tensor_t` in emitted C; same owned-value lifecycle
 *  as `String`. Multi-row chars aren't constructable in v1, so the
 *  type doesn't carry a shape — `exact.length` is the column count.
 *  `exact` carries the literal value when the source is a `'...'`
 *  expression. */
export interface CharType {
  kind: "Char";
  exact?: string;
}

export interface UnknownType {
  kind: "Unknown";
}

// ── Function-handle type ────────────────────────────────────────────────

/** One captured variable in a `HandleType`. The `name` is both the
 *  enclosing-scope identifier the @-site snapshot reads from AND the
 *  synthesized function's tail-param name; the `ty` is the captured
 *  value's type at the @-site. Captures of owned types (tensors,
 *  structs, classes, other handles) are deep-copied into the handle's
 *  C struct at the `@(...)` site, matching MATLAB by-value capture
 *  semantics; the handle's per-shape `_free` helper releases each
 *  owned field at scope exit. */
export interface HandleCapture {
  name: string;
  ty: Type;
}

/** Function handle. mtoc2 supports only user-function targets
 *  (named `@user_func` and anonymous `@(p1,...,pN) <body>`); `@builtin`
 *  is rejected. Captures may be any non-Void / non-Unknown / non-String
 *  value type — scalar real numeric, tensor, struct, class instance, or
 *  another handle. The handle's C representation is a per-capture-shape
 *  typedef with `_empty / _copy / _assign / _free` helpers, matching
 *  the struct/class owned-value lifecycle. */
export interface HandleType {
  kind: "Handle";
  /** Source-level identifier for the target function. For named
   *  handles, this is the user's name (e.g. `sq`); for anonymous
   *  handles, the synthesized name (e.g. `anon_0`). Used as the
   *  source-name half of the specialization mangling. */
  targetName: string;
  /** Synthesized or pre-scanned `Function` AST handed to
   *  `specializeUserFunction` at every call site. The params list
   *  contains `[...userParams, ...captureNames]` (in that order); the
   *  body's references to a captured variable resolve naturally to the
   *  matching synthesized tail param. */
  ast: FunctionStmt;
  /** Variables captured from the enclosing scope at the `@(...)` site.
   *  Empty for named handles and for capture-free anonymous functions.
   *  Field order matches the synth function's tail params. */
  captures: ReadonlyArray<HandleCapture>;
}

/** "No value" — the type of a call to a user function with zero
 *  outputs. Valid only as the expression type of an `ExprStmt`. Every
 *  other lowering site (Assign RHS, sub-expression of a Binary / Unary
 *  / Call, tensor-literal element, if/while cond, for bounds) rejects
 *  Void with `UnsupportedConstruct`. */
export interface VoidType {
  kind: "Void";
}

/** A struct value. Field order is canonical (sorted by name) so two
 *  StructType values with the same shape are structurally identical
 *  regardless of source-level field-write order. Construct via
 *  `structType()` rather than the raw interface so the sort is
 *  applied. */
export interface StructType {
  kind: "Struct";
  fields: ReadonlyArray<{ name: string; ty: Type }>;
}

/** A class instance value. The `className` is the source-level class
 *  name (the `classdef Foo` identifier); `properties` is the full
 *  flattened-and-sorted property list with each property's type
 *  derived from its `properties` block default expression. v1
 *  forbids inheritance, so the property list is always exactly the
 *  one declared in the class body. */
export interface ClassType {
  kind: "Class";
  className: string;
  properties: ReadonlyArray<{ name: string; ty: Type }>;
}

/** Cell array. Two modes:
 *  - **tuple**: per-slot types, used when the cell's shape is exact
 *    AND the total slot count fits `EXACT_ARRAY_MAX_ELEMENTS`. Cell
 *    literals `{a, b, c}` and small `cell(n, m)` constructors land
 *    here. `elements.length === shape product`; slot k is the
 *    column-major-flat index. Each `elements[k]` is the *static*
 *    type of slot k.
 *  - **uniform**: a single `elem` type covers every slot. Used for
 *    `cell(n, m)` with a non-exact dim or a slot count above the
 *    cap, and for tuple cells that "demote" on a non-static-index
 *    write whose existing slot types all unify with the rhs type.
 *
 *  Cells have no LUB / heterogeneous mode. When neither tuple nor
 *  uniform applies (a non-static-index write into a tuple whose
 *  slot types don't all unify with the rhs), the lowerer raises
 *  `UnsupportedConstruct`. */
export interface CellType {
  kind: "Cell";
  mode: "tuple" | "uniform";
  dims: DimInfo[];
  shape?: number[];
  /** Tuple-mode only: per-slot types in column-major order. */
  elements?: Type[];
  /** Uniform-mode only: every slot has this type. */
  elem?: Type;
}

export type Type =
  | NumericType
  | StringType
  | CharType
  | UnknownType
  | VoidType
  | HandleType
  | StructType
  | ClassType
  | CellType;

export const VOID: VoidType = { kind: "Void" };

// ── Factories ───────────────────────────────────────────────────────────

export function scalarDouble(
  sign: Sign = "unknown",
  exact?: number
): NumericType {
  const t: NumericType = {
    kind: "Numeric",
    elem: "double",
    isComplex: false,
    dims: [DIM_ONE, DIM_ONE],
    shape: [1, 1],
    sign,
  };
  if (exact !== undefined) t.exact = exact;
  return t;
}

/** Scalar complex double. Sign is always `unknown` for complex values
 *  — the sign lattice describes ordering on the real line, which has
 *  no analogue for complex numbers. When `exact` is provided, both
 *  `re` and `im` must be finite. */
export function scalarComplex(exact?: { re: number; im: number }): NumericType {
  const t: NumericType = {
    kind: "Numeric",
    elem: "double",
    isComplex: true,
    dims: [DIM_ONE, DIM_ONE],
    shape: [1, 1],
    sign: "unknown",
  };
  if (exact !== undefined) t.exact = { re: exact.re, im: exact.im };
  return t;
}

export function scalarLogical(exact?: boolean): NumericType {
  const t: NumericType = {
    kind: "Numeric",
    elem: "logical",
    isComplex: false,
    dims: [DIM_ONE, DIM_ONE],
    shape: [1, 1],
    sign: exact === undefined ? "nonneg" : exact ? "positive" : "zero",
  };
  if (exact !== undefined) t.exact = exact ? 1 : 0;
  return t;
}

/** Real-double tensor built from a per-axis `dims` lattice. Used by
 *  slice-read result typing when the slot pattern leaves at least one
 *  axis with a runtime-only length (an `unknown` `DimInfo`). When every
 *  dim turns out to be `exact`, this function also populates `shape`
 *  so downstream code can take the concrete-shape paths uniformly.
 *
 *  Trailing singletons in `dims` are stripped subject to a 2-axis
 *  minimum, matching numbl's tensor shape-normalization rule. */
export function tensorDoubleFromDims(dims: DimInfo[]): NumericType {
  const trimmed = dims.slice();
  while (trimmed.length > 2 && isDimOne(trimmed[trimmed.length - 1])) {
    trimmed.pop();
  }
  const t: NumericType = {
    kind: "Numeric",
    elem: "double",
    isComplex: false,
    dims: trimmed,
    sign: "unknown",
  };
  if (trimmed.every(d => d.kind === "exact")) {
    t.shape = trimmed.map(d => (d as { kind: "exact"; value: number }).value);
  }
  return t;
}

/** Real-double tensor with statically-known shape. `dims` is derived
 *  from `shape` (`{kind:"exact", value: s}` per axis). When `exact` is
 *  provided, its length must equal the shape's product; the layout is
 *  column-major (matching numbl's `RuntimeTensor.data`).
 *
 *  When `exact` is provided, `sign` is derived from the actual values
 *  via `signFromExactArray`. This is what lets `sqrt([0 1 4 9])` pass
 *  the requireDomain check (without it the tensor would carry
 *  `sign:"unknown"`). For tensors without exact data the caller can
 *  set `sign` post-construction (e.g. `zeros`/`ones` know their fill
 *  value even when the result is too large to carry exact data). */
export function tensorDouble(
  shape: number[],
  exact?: Float64Array
): NumericType {
  const dims: DimInfo[] = shape.map(s =>
    s === 1 ? DIM_ONE : { kind: "exact", value: s }
  );
  const t: NumericType = {
    kind: "Numeric",
    elem: "double",
    isComplex: false,
    dims,
    shape: shape.slice(),
    sign: "unknown",
  };
  if (exact !== undefined) {
    const total = shapeNumel(shape);
    if (exact.length !== total) {
      throw new Error(
        `tensorDouble: shape [${shape.join(",")}] requires ${total} elements, got ${exact.length}`
      );
    }
    t.exact = exact;
    t.sign = signFromExactArray(exact);
  }
  return t;
}

/** Complex-double tensor with statically-known shape. Mirrors
 *  `tensorDouble` but the result type carries `isComplex: true` and
 *  the optional `exact` is the split-buffer carrier whose `re` /
 *  `im` Float64Arrays each match the shape product. Sign is always
 *  `"unknown"` for complex (the lattice describes ordering on the
 *  real line; complex numbers have no analogue). */
export function tensorComplex(
  shape: number[],
  exact?: { re: Float64Array; im: Float64Array }
): NumericType {
  const dims: DimInfo[] = shape.map(s =>
    s === 1 ? DIM_ONE : { kind: "exact", value: s }
  );
  const t: NumericType = {
    kind: "Numeric",
    elem: "double",
    isComplex: true,
    dims,
    shape: shape.slice(),
    sign: "unknown",
  };
  if (exact !== undefined) {
    const total = shapeNumel(shape);
    if (exact.re.length !== total || exact.im.length !== total) {
      throw new Error(
        `tensorComplex: shape [${shape.join(",")}] requires ${total} elements, got re=${exact.re.length} im=${exact.im.length}`
      );
    }
    t.exact = { re: exact.re, im: exact.im };
  }
  return t;
}

/** Complex-double tensor built from a per-axis `dims` lattice. Sibling
 *  of `tensorDoubleFromDims` for slice-read result typing when at
 *  least one axis is runtime-only. */
export function tensorComplexFromDims(dims: DimInfo[]): NumericType {
  const trimmed = dims.slice();
  while (trimmed.length > 2 && isDimOne(trimmed[trimmed.length - 1])) {
    trimmed.pop();
  }
  const t: NumericType = {
    kind: "Numeric",
    elem: "double",
    isComplex: true,
    dims: trimmed,
    sign: "unknown",
  };
  if (trimmed.every(d => d.kind === "exact")) {
    t.shape = trimmed.map(d => (d as { kind: "exact"; value: number }).value);
  }
  return t;
}

export const UNKNOWN: UnknownType = { kind: "Unknown" };

/** Constructor for a `@<user_func>` handle (named or anonymous). */
export function handleType(
  targetName: string,
  ast: FunctionStmt,
  captures: ReadonlyArray<HandleCapture> = []
): HandleType {
  return { kind: "Handle", targetName, ast, captures };
}

/** Construct a `StructType` with canonical (sorted-by-name) field
 *  order. Two structs built from the same fields end up structurally
 *  equal regardless of the order their fields were specified. */
export function structType(
  fields: ReadonlyArray<{ name: string; ty: Type }>
): StructType {
  const sorted = fields.slice().sort((a, b) => (a.name < b.name ? -1 : 1));
  return { kind: "Struct", fields: sorted };
}

/** Construct a `ClassType` with canonical (sorted-by-name) property
 *  order. */
export function classType(
  className: string,
  properties: ReadonlyArray<{ name: string; ty: Type }>
): ClassType {
  const sorted = properties.slice().sort((a, b) => (a.name < b.name ? -1 : 1));
  return { kind: "Class", className, properties: sorted };
}

/** Construct a tuple-mode `CellType`. The caller must provide a fully
 *  exact shape (every entry in `dims` is `exact`) and an `elements`
 *  array whose length equals the shape product. */
export function cellTuple(shape: number[], elements: Type[]): CellType {
  const total = shapeNumel(shape);
  if (elements.length !== total) {
    throw new Error(
      `cellTuple: shape [${shape.join(",")}] requires ${total} slots, got ${elements.length}`
    );
  }
  const dims: DimInfo[] = shape.map(s =>
    s === 1 ? DIM_ONE : { kind: "exact", value: s }
  );
  return {
    kind: "Cell",
    mode: "tuple",
    dims,
    shape: shape.slice(),
    elements: elements.slice(),
  };
}

/** Construct a uniform-mode `CellType`. Used when the cell's slot
 *  count exceeds `EXACT_ARRAY_MAX_ELEMENTS`, when a dim is non-exact,
 *  or when a tuple cell demotes after a non-static-index write. */
export function cellUniform(dims: DimInfo[], elem: Type): CellType {
  const t: CellType = {
    kind: "Cell",
    mode: "uniform",
    dims: dims.slice(),
    elem,
  };
  if (dims.every(d => d.kind === "exact")) {
    t.shape = dims.map(d => (d as { kind: "exact"; value: number }).value);
  }
  return t;
}

/** The canonical "empty `[]` double tensor" slot type — what numbl
 *  initialises every fresh slot in `cell(n, m)` to
 *  (see `numbl/.../type-constructors.ts:442-469`). Shape is `[0, 0]`,
 *  no exact data, sign unknown. Mtoc2 reuses this exact representation
 *  so cross-runner displays of an unwritten slot match byte-for-byte. */
export function emptyDoubleTensorType(): NumericType {
  return {
    kind: "Numeric",
    elem: "double",
    isComplex: false,
    dims: [
      { kind: "exact", value: 0 },
      { kind: "exact", value: 0 },
    ],
    shape: [0, 0],
    sign: "unknown",
  };
}

// ── Predicates ──────────────────────────────────────────────────────────

export function isNumeric(t: Type): t is NumericType {
  return t.kind === "Numeric";
}

export function isVoid(t: Type): t is VoidType {
  return t.kind === "Void";
}

export function isHandle(t: Type): t is HandleType {
  return t.kind === "Handle";
}

export function isStruct(t: Type): t is StructType {
  return t.kind === "Struct";
}

export function isClass(t: Type): t is ClassType {
  return t.kind === "Class";
}

export function isString(t: Type): t is StringType {
  return t.kind === "String";
}

export function isChar(t: Type): t is CharType {
  return t.kind === "Char";
}

export function isCell(t: Type): t is CellType {
  return t.kind === "Cell";
}

/** True when `t` can be consumed by a "text-accepting" runtime helper
 *  via `mtoc2_text_view_t` — today: a `String` handle or a `Char` array.
 *  The shared predicate lets builtins like `disp` / future
 *  `error` / `fprintf` accept either source kind. */
export function isText(t: Type): boolean {
  return isString(t) || isChar(t);
}

/** Find a field on a struct/class by name. Returns the field's type
 *  or undefined if the type isn't a struct/class or no such field
 *  exists. */
export function fieldType(t: Type, name: string): Type | undefined {
  if (t.kind === "Struct") {
    const f = t.fields.find(f => f.name === name);
    return f?.ty;
  }
  if (t.kind === "Class") {
    const p = t.properties.find(p => p.name === name);
    return p?.ty;
  }
  return undefined;
}

export function isScalar(t: Type): boolean {
  if (!isNumeric(t)) return false;
  return t.dims.every(isDimOne);
}

export function isScalarRealDouble(t: Type): boolean {
  return isNumeric(t) && isScalar(t) && t.elem === "double" && !t.isComplex;
}

/** Scalar real numeric: double or logical. Both are stored as `double`
 *  in emitted C, so anything operating on real values accepts either. */
export function isScalarRealNumeric(t: Type): boolean {
  return (
    isNumeric(t) &&
    isScalar(t) &&
    (t.elem === "double" || t.elem === "logical") &&
    !t.isComplex
  );
}

/** True when any axis is statically known to be ≠ 1 (or unknown).
 *  Drives the scalar/tensor split in codegen: scalars compile to bare
 *  `double`; multi-element values compile to `mtoc2_tensor_t`. A
 *  zero-sized axis (`{exact, 0}`) also returns true — an empty tensor
 *  still needs tensor storage. */
export function isMultiElement(t: Type): boolean {
  return isNumeric(t) && t.dims.some(d => !isDimOne(d));
}

/** Row-vector shape: 2-D with a singleton first axis and an exact
 *  non-singleton second axis. Matches MATLAB's "row" classification:
 *  1×0 (empty row) qualifies, 1×1 (scalar) does not. */
export function isRowVecTy(t: NumericType): boolean {
  return (
    t.dims.length === 2 &&
    isDimOne(t.dims[0]) &&
    t.dims[1].kind === "exact" &&
    t.dims[1].value !== 1
  );
}

/** Column-vector shape: 2-D with an exact non-singleton first axis and
 *  a singleton second axis. Matches MATLAB's "column" classification:
 *  0×1 (empty column) qualifies, 1×1 (scalar) does not. */
export function isColVecTy(t: NumericType): boolean {
  return (
    t.dims.length === 2 &&
    t.dims[0].kind === "exact" &&
    t.dims[0].value !== 1 &&
    isDimOne(t.dims[1])
  );
}

/** Statically provable to contain at least one element. True iff every
 *  dim is `exact` with a positive value (equivalently, when `shape` is
 *  set, every entry is > 0). Used by reductions to decide whether the
 *  empty-input edge case (sum→0, prod→1, min/max→NaN, mean→NaN) is
 *  reachable; tighter sign rules apply only on the provably-non-empty
 *  branch. */
export function provablyNonEmpty(t: NumericType): boolean {
  if (t.shape !== undefined) {
    return t.shape.every(s => s > 0);
  }
  return t.dims.every(d => d.kind === "exact" && d.value > 0);
}

/** Owned-heap-value types — i.e. types whose C representation holds a
 *  heap pointer the codegen must `free` at scope exit. Multi-element
 *  tensors are the original owned kind. Structs, class instances, and
 *  function handles all count as owned because their per-shape generated
 *  typedef carries the same `_empty()`/`_assign()`/`_copy()`/`_free()`
 *  lifecycle — a struct or handle with all-scalar fields would
 *  technically be POD, but tracking ownership uniformly keeps the
 *  codegen pipeline simple and lets struct fields and handle captures
 *  hold tensors transparently. */
export function isOwned(t: Type): boolean {
  if (isMultiElement(t)) return true;
  if (t.kind === "Struct") return true;
  if (t.kind === "Class") return true;
  if (t.kind === "Handle") return true;
  if (t.kind === "String") return true;
  if (t.kind === "Char") return true;
  if (t.kind === "Cell") return true;
  return false;
}

/** Types that may appear in an N≥2-output multi-assign slot. Scalar
 *  real numeric outputs are stored by `*_mtoc2_o<i> = <local>` struct-
 *  copy; owned outputs (tensor, struct, class instance, function
 *  handle, Char, String) use the kind's `_assign` helper to transfer
 *  ownership of the buffer pointer. Only Void / Unknown stay rejected. */
export function isMultiOutputSlotType(t: Type): boolean {
  if (isScalarRealNumeric(t)) return true;
  if (isOwned(t)) return true;
  return false;
}

export function signIsNonneg(s: Sign): boolean {
  return s === "positive" || s === "nonneg" || s === "zero";
}

export function signIsPositive(s: Sign): boolean {
  return s === "positive";
}

// ── Exact helpers ───────────────────────────────────────────────────────

export function numericExactsEqual(
  a: NumericExact | undefined,
  b: NumericExact | undefined
): boolean {
  if (a === undefined || b === undefined) return false;
  if (typeof a === "number" && typeof b === "number") {
    return Object.is(a, b);
  }
  const aIsArr = a instanceof Float64Array;
  const bIsArr = b instanceof Float64Array;
  if (aIsArr && bIsArr) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!Object.is(a[i], b[i])) return false;
    }
    return true;
  }
  if (aIsArr || bIsArr) return false;
  // Object case: scalar {re, im} or complex-tensor {re, im} (Float64Arrays).
  if (typeof a === "object" && typeof b === "object") {
    const aReArr = (a as { re: unknown }).re instanceof Float64Array;
    const bReArr = (b as { re: unknown }).re instanceof Float64Array;
    if (aReArr !== bReArr) return false;
    if (aReArr) {
      const ax = a as { re: Float64Array; im: Float64Array };
      const bx = b as { re: Float64Array; im: Float64Array };
      if (ax.re.length !== bx.re.length) return false;
      for (let i = 0; i < ax.re.length; i++) {
        if (!Object.is(ax.re[i], bx.re[i])) return false;
        if (!Object.is(ax.im[i], bx.im[i])) return false;
      }
      return true;
    }
    const ax = a as { re: number; im: number };
    const bx = b as { re: number; im: number };
    return Object.is(ax.re, bx.re) && Object.is(ax.im, bx.im);
  }
  return false;
}

/** Widen every entry in `env` whose name is in `names` before lowering
 *  a loop body. Used by `lowerFor` / `lowerWhile` — see lower.ts for
 *  rationale. Strips `exact` (the one-pass lowerer can't carry exact
 *  values across iterations) AND widens narrowing fields like `sign`
 *  to `"unknown"`, since a `sign: "nonneg"` set before the loop is no
 *  longer valid if the body reassigns the variable to a negative value.
 *  Without the sign widen, `sqrt(x)` / `log(x)` domain checks would
 *  spuriously pass at lowering time and produce NaN at runtime. For
 *  struct / class env entries we recurse via `withoutExact` so any
 *  precise field/property exact values introduced by a `struct(...)`
 *  literal or a constructor preSeed don't leak from iteration 1 into
 *  the rest of the loop's body. */
export function stripExactFromEnv(
  env: Map<string, { cName: string; ty: Type }>,
  names: Iterable<string>
): void {
  for (const n of names) {
    const e = env.get(n);
    if (e === undefined) continue;
    if (e.ty.kind === "Numeric") {
      const needsExactStrip = e.ty.exact !== undefined;
      const needsSignWiden = e.ty.sign !== "unknown";
      if (needsExactStrip || needsSignWiden) {
        env.set(n, {
          cName: e.cName,
          ty: { ...e.ty, exact: undefined, sign: "unknown" },
        });
      }
    } else if (e.ty.kind === "String" && e.ty.exact !== undefined) {
      env.set(n, { cName: e.cName, ty: { kind: "String" } });
    } else if (e.ty.kind === "Char" && e.ty.exact !== undefined) {
      env.set(n, { cName: e.cName, ty: { kind: "Char" } });
    } else if (e.ty.kind === "Struct" || e.ty.kind === "Class") {
      env.set(n, { cName: e.cName, ty: withoutExact(e.ty) });
    }
  }
}

/** Drop `exact` and widen `sign` toward the rhs sign on the named
 *  entry. Called after an indexed write (`x(i) = rhs`, `x(:) = rhs`, …)
 *  where the rhs is general enough that we can't carry the pre-write
 *  sign forward unchanged.
 *
 *  Without this, e.g. `x = zeros(1, 5); x(3) = -10; sqrt(x)` would slip
 *  past `sqrt`'s `requireDomain` check because the env still claims
 *  `sign = nonneg`. We use `unifySign(current, rhsSign)` so a same-sign
 *  rhs (e.g. `x(3) = 4` after `zeros(...)`) doesn't unnecessarily
 *  collapse the lattice — the new sign covers both the surviving
 *  pre-write elements and the just-written one.
 *
 *  `rhsSign` defaults to `"unknown"` for callers that don't have a
 *  rhs to inspect (slice writes from an opaque source, etc.). */
export function widenAfterIndexedWrite(
  env: Map<string, { cName: string; ty: Type }>,
  name: string,
  rhsSign: Sign = "unknown"
): void {
  const e = env.get(name);
  if (e === undefined) return;
  if (e.ty.kind === "Numeric") {
    const t = e.ty;
    const newSign = unifySign(t.sign, rhsSign);
    if (t.exact !== undefined || t.sign !== newSign) {
      env.set(name, {
        cName: e.cName,
        ty: { ...t, exact: undefined, sign: newSign },
      });
    }
  } else if (e.ty.kind === "String" && e.ty.exact !== undefined) {
    env.set(name, { cName: e.cName, ty: { kind: "String" } });
  } else if (e.ty.kind === "Struct" || e.ty.kind === "Class") {
    env.set(name, { cName: e.cName, ty: withoutExact(e.ty) });
  }
}

/** Return a copy of `t` with the type at `fieldPath` replaced by
 *  `newLeafTy`. Walks struct/class types; if the path can't be
 *  resolved (shouldn't happen — the caller already validated), the
 *  original type is returned unchanged. Other type kinds at a path
 *  step are returned as-is.
 *
 *  Used by the env refresh after a MemberStore / member-rooted
 *  IndexStore so subsequent reads of the touched field report the
 *  post-write rhs type rather than the construction-site default. */
export function withPathTypeUpdated(
  t: Type,
  fieldPath: ReadonlyArray<string>,
  newLeafTy: Type
): Type {
  if (fieldPath.length === 0) return newLeafTy;
  const [head, ...rest] = fieldPath;
  if (t.kind === "Struct") {
    return structType(
      t.fields.map(f =>
        f.name === head
          ? { name: f.name, ty: withPathTypeUpdated(f.ty, rest, newLeafTy) }
          : f
      )
    );
  }
  if (t.kind === "Class") {
    return {
      kind: "Class",
      className: t.className,
      properties: t.properties.map(p =>
        p.name === head
          ? { name: p.name, ty: withPathTypeUpdated(p.ty, rest, newLeafTy) }
          : p
      ),
    };
  }
  return t;
}

/** Companion to `widenAfterIndexedWrite` for member-rooted indexed
 *  writes (`obj.field(i) = rhs`). Strips `exact` and widens the
 *  leaf field's `sign` to `"unknown"`, then rebuilds the parent
 *  struct/class type via `withPathTypeUpdated`. The optimisation of
 *  refreshing the leaf's exact carrier in place (the way bare-Ident
 *  writes do) is deferred — see `tryRefreshExactAfterIndexedWrite`. */
export function widenMemberLeafAfterIndexedWrite(
  env: Map<string, { cName: string; ty: Type }>,
  rootName: string,
  fieldPath: ReadonlyArray<string>,
  oldLeafTy: NumericType
): void {
  const e = env.get(rootName);
  if (e === undefined) return;
  if (oldLeafTy.exact === undefined && oldLeafTy.sign === "unknown") return;
  const widened: NumericType = {
    ...oldLeafTy,
    exact: undefined,
    sign: "unknown",
  };
  const newTy = withPathTypeUpdated(e.ty, fieldPath, widened);
  env.set(rootName, { cName: e.cName, ty: newTy });
}

export function withoutExact(t: Type): Type {
  if (t.kind === "Numeric" && t.exact !== undefined) {
    const { exact: _e, ...rest } = t;
    void _e;
    return rest;
  }
  if (t.kind === "String" && t.exact !== undefined) {
    return { kind: "String" };
  }
  if (t.kind === "Char" && t.exact !== undefined) {
    return { kind: "Char" };
  }
  if (t.kind === "Struct") {
    return {
      kind: "Struct",
      fields: t.fields.map(f => ({ name: f.name, ty: withoutExact(f.ty) })),
    };
  }
  if (t.kind === "Class") {
    return {
      kind: "Class",
      className: t.className,
      properties: t.properties.map(p => ({
        name: p.name,
        ty: withoutExact(p.ty),
      })),
    };
  }
  if (t.kind === "Cell") {
    if (t.mode === "tuple") {
      return {
        ...t,
        elements: t.elements!.map(withoutExact),
      };
    }
    return { ...t, elem: withoutExact(t.elem!) };
  }
  return t;
}

// ── Unify (control-flow merge) ──────────────────────────────────────────

/** Merge two types at a join point. Drops exact unless both sides agree.
 *  Sign widens via `unifySign`. For MVP, both sides should be scalar real
 *  double — mismatches return `UNKNOWN`. */
export function unify(a: Type, b: Type): Type {
  if (a.kind === "Unknown" || b.kind === "Unknown") return UNKNOWN;
  if (a.kind === "Void" || b.kind === "Void") return UNKNOWN;
  if (a.kind === "Numeric" && b.kind === "Numeric") {
    if (a.elem !== b.elem) return UNKNOWN;
    if (a.isComplex !== b.isComplex) return UNKNOWN;
    // Normalize trailing-singleton dims to a 2-axis minimum on both
    // sides before comparing. Otherwise an IF arm producing `zeros(2,3)`
    // (dims=[2,3]) and an ELSE arm producing `zeros(2,3,1)`
    // (dims=[2,3,1]) would unify to UNKNOWN even though their shapes
    // are equivalent under MATLAB's trailing-singleton rule.
    const aDims = trimTrailingSingletons(a.dims);
    const bDims = trimTrailingSingletons(b.dims);
    if (aDims.length !== bDims.length) return UNKNOWN;
    const dims = aDims.map((d, i) => unifyDim(d, bDims[i]));
    const sign = unifySign(a.sign, b.sign);
    // Shape survives only when both sides agree exactly after the same
    // trailing-singleton trim.
    const aShape =
      a.shape !== undefined ? trimTrailingShapeOnes(a.shape) : undefined;
    const bShape =
      b.shape !== undefined ? trimTrailingShapeOnes(b.shape) : undefined;
    let shape: number[] | undefined;
    if (
      aShape !== undefined &&
      bShape !== undefined &&
      aShape.length === bShape.length &&
      aShape.every((s, i) => s === bShape[i])
    ) {
      shape = aShape;
    }
    const out: NumericType = {
      kind: "Numeric",
      elem: a.elem,
      isComplex: a.isComplex,
      dims,
      sign,
    };
    if (shape !== undefined) out.shape = shape;
    // Exact survives when bit-identical AND shape matched (so tensor
    // shapes can't unify if the data layout differs).
    if (numericExactsEqual(a.exact, b.exact) && shape !== undefined) {
      out.exact = a.exact;
    }
    return out;
  }
  if (a.kind === "String" && b.kind === "String") {
    if (a.exact !== undefined && b.exact !== undefined && a.exact === b.exact) {
      return { kind: "String", exact: a.exact };
    }
    return { kind: "String" };
  }
  if (a.kind === "Char" && b.kind === "Char") {
    if (a.exact !== undefined && b.exact !== undefined && a.exact === b.exact) {
      return { kind: "Char", exact: a.exact };
    }
    return { kind: "Char" };
  }
  if (a.kind === "Handle" && b.kind === "Handle") {
    if (canonicalizeType(a) === canonicalizeType(b)) return a;
    return UNKNOWN;
  }
  if (a.kind === "Struct" && b.kind === "Struct") {
    // Union of field names; per shared field, unify the types. Fields
    // present only on one side carry through unchanged — useful when a
    // conditional branch only writes a subset of the eventual fields.
    const names = new Set<string>();
    for (const f of a.fields) names.add(f.name);
    for (const f of b.fields) names.add(f.name);
    const out: { name: string; ty: Type }[] = [];
    for (const name of names) {
      const fa = a.fields.find(f => f.name === name);
      const fb = b.fields.find(f => f.name === name);
      if (fa && fb) out.push({ name, ty: unify(fa.ty, fb.ty) });
      else if (fa) out.push({ name, ty: fa.ty });
      else if (fb) out.push({ name, ty: fb.ty });
    }
    return structType(out);
  }
  if (a.kind === "Class" && b.kind === "Class") {
    if (a.className !== b.className) return UNKNOWN;
    // Same class: per-property unify. Property sets must already
    // match (defined by the classdef body, which is global).
    if (a.properties.length !== b.properties.length) return UNKNOWN;
    const props: { name: string; ty: Type }[] = [];
    for (const pa of a.properties) {
      const pb = b.properties.find(p => p.name === pa.name);
      if (!pb) return UNKNOWN;
      props.push({ name: pa.name, ty: unify(pa.ty, pb.ty) });
    }
    return classType(a.className, props);
  }
  if (a.kind === "Cell" && b.kind === "Cell") {
    // Two tuple cells with the same shape and pairwise-unifiable
    // elements merge into a tuple with the unified slot types.
    // Otherwise both sides demote to uniform (via the existing tuple-
    // to-uniform widening rule) before unifying. If either uniform's
    // elem fails to unify, the result is UNKNOWN.
    if (
      a.mode === "tuple" &&
      b.mode === "tuple" &&
      a.shape !== undefined &&
      b.shape !== undefined &&
      a.shape.length === b.shape.length &&
      a.shape.every((s, i) => s === b.shape![i])
    ) {
      const aElems = a.elements!;
      const bElems = b.elements!;
      const out = aElems.map((ea, i) => unify(ea, bElems[i]));
      return cellTuple(a.shape, out);
    }
    const aUni = cellWidenToUniform(a);
    const bUni = cellWidenToUniform(b);
    if (aUni === undefined || bUni === undefined) return UNKNOWN;
    if (aUni.dims.length !== bUni.dims.length) return UNKNOWN;
    const dims = aUni.dims.map((d, i) => unifyDim(d, bUni.dims[i]));
    const elem = unify(aUni.elem!, bUni.elem!);
    if (elem.kind === "Unknown") return UNKNOWN;
    return cellUniform(dims, elem);
  }
  return UNKNOWN;
}

/** Demote a tuple-mode cell to uniform mode if every slot type
 *  unifies. Returns undefined if the cell is already uniform (no
 *  work) or if the demotion isn't possible (heterogeneous slot
 *  types). Used inside `unify` and at non-static-index writes. */
export function cellWidenToUniform(t: CellType): CellType | undefined {
  if (t.mode === "uniform") return t;
  const els = t.elements!;
  if (els.length === 0) {
    // Empty tuple cell — represent as uniform of empty-double sentinel.
    return cellUniform(t.dims, emptyDoubleTensorType());
  }
  let elem = els[0];
  for (let i = 1; i < els.length; i++) {
    elem = unify(elem, els[i]);
    if (elem.kind === "Unknown") return undefined;
  }
  return cellUniform(t.dims, elem);
}

function unifyDim(a: DimInfo, b: DimInfo): DimInfo {
  if (a.kind === "exact" && b.kind === "exact" && a.value === b.value) return a;
  if (a.kind === "unknown" && b.kind === "unknown") return a;
  return { kind: "unknown" };
}

/** Strip trailing axes that are statically known to be 1, down to a
 *  2-axis minimum (matching `tensorDoubleFromDims` and MATLAB's
 *  trailing-singleton rule). Used by `unify` to align ranks before
 *  the dimwise compare so equivalent shapes don't trip the
 *  `dims.length !== dims.length` early-out. */
function trimTrailingSingletons(dims: ReadonlyArray<DimInfo>): DimInfo[] {
  const trimmed = dims.slice();
  while (trimmed.length > 2 && isDimOne(trimmed[trimmed.length - 1])) {
    trimmed.pop();
  }
  return trimmed;
}

function trimTrailingShapeOnes(shape: ReadonlyArray<number>): number[] {
  const trimmed = shape.slice();
  while (trimmed.length > 2 && trimmed[trimmed.length - 1] === 1) {
    trimmed.pop();
  }
  return trimmed;
}

// ── Pretty-print (for diagnostics) ──────────────────────────────────────

export function typeToString(t: Type): string {
  switch (t.kind) {
    case "Unknown":
      return "unknown";
    case "Void":
      return "void";
    case "String":
      return t.exact !== undefined ? `string="${t.exact}"` : "string";
    case "Char":
      return t.exact !== undefined ? `char='${t.exact}'` : "char";
    case "Numeric": {
      let s: string = t.elem;
      if (t.isComplex) s = `complex(${s})`;
      const dimsStr =
        t.shape !== undefined
          ? t.shape.join("×")
          : t.dims
              .map(d => (d.kind === "exact" ? String(d.value) : "?"))
              .join("×");
      s += `[${dimsStr}]`;
      if (t.sign !== "unknown") s += `:${t.sign}`;
      if (t.exact !== undefined) s += `=${formatExactForType(t.exact)}`;
      return s;
    }
    case "Handle": {
      const caps =
        t.captures.length === 0
          ? ""
          : `{${t.captures.map(c => `${c.name}:${typeToString(c.ty)}`).join(",")}}`;
      return `@${t.targetName}${caps}`;
    }
    case "Struct": {
      const inner = t.fields
        .map(f => `${f.name}:${typeToString(f.ty)}`)
        .join(", ");
      return `struct{${inner}}`;
    }
    case "Class": {
      const inner = t.properties
        .map(p => `${p.name}:${typeToString(p.ty)}`)
        .join(", ");
      return `class ${t.className}{${inner}}`;
    }
    case "Cell": {
      const dimsStr =
        t.shape !== undefined
          ? t.shape.join("×")
          : t.dims
              .map(d => (d.kind === "exact" ? String(d.value) : "?"))
              .join("×");
      if (t.mode === "tuple") {
        const inner = t.elements!.map(typeToString).join(", ");
        return `cell[${dimsStr}]{${inner}}`;
      }
      return `cell[${dimsStr}]<${typeToString(t.elem!)}>`;
    }
  }
}

function formatExactForType(e: NumericExact): string {
  if (typeof e === "number") return JSON.stringify(e);
  if (e instanceof Float64Array) {
    const cap = 8;
    const preview = Array.from(e.slice(0, cap)).map(v => v.toString());
    if (e.length > cap) preview.push("…");
    return `[${preview.join(", ")}]`;
  }
  if (e.re instanceof Float64Array) {
    const cap = 4;
    const cx = e as { re: Float64Array; im: Float64Array };
    const preview: string[] = [];
    const n = Math.min(cap, cx.re.length);
    for (let i = 0; i < n; i++) preview.push(`${cx.re[i]}+${cx.im[i]}i`);
    if (cx.re.length > cap) preview.push("…");
    return `[${preview.join(", ")}]`;
  }
  const sx = e as { re: number; im: number };
  return `(${sx.re}+${sx.im}i)`;
}

// ── Canonicalize + hash (for function specialization keys) ──────────────

/** Stable string key used for the specialization-key hash. Includes
 *  exact when set, so each unique exact-value gets its own spec. */
export function canonicalizeType(t: Type): string {
  return JSON.stringify(canon(t));
}

/** True iff two types have the same canonical form. Convenience
 *  wrapper around `canonicalizeType` for callers that don't need the
 *  string key. */
export function canonicalEq(a: Type, b: Type): boolean {
  return canonicalizeType(a) === canonicalizeType(b);
}

/** Storage compatibility: do two types occupy the same C-level slot?
 *  This is COARSER than canonical equality — every multi-element
 *  tensor shares the same `mtoc2_tensor_t` storage regardless of
 *  shape, every scalar real numeric maps to `double`, and every
 *  struct/class needs its canonical typedef to match.
 *
 *  Used by `MemberStore` to validate that a write doesn't try to
 *  cram a tensor into a scalar slot (or vice versa) — the typedef
 *  hash is built from storage shapes, so changing the underlying
 *  representation would break the C side. Shape, sign, and exact
 *  differences are fine: the slot still holds an `mtoc2_tensor_t`. */
export function storageEquivalent(a: Type, b: Type): boolean {
  // Two types occupy the same C slot iff they reduce to the same
  // C-type string. `cFieldTypeStr` already collapses everything that
  // doesn't matter at the C level (sign, exact, tensor shape, the
  // distinction between same-class instances), so this is one line.
  try {
    return cFieldTypeStr(a) === cFieldTypeStr(b);
  } catch {
    // One side is a String/Void/Unknown (or similar non-field-typed
    // value). Those never share a slot with anything legitimate.
    return false;
  }
}

/** Serialize an `exact` numeric value for the canonical spec key.
 *  `JSON.stringify` renders NaN, ±Infinity, and -0 ambiguously (NaN and
 *  both infinities all collapse to `"null"`; -0 prints as `0`), which
 *  lets distinct exact values collide on the spec key. Round-trip each
 *  through a tagged string when the bit pattern wouldn't survive a
 *  JSON round-trip; finite non-zero numbers (including ordinary
 *  negative values) pass through as-is so the common path stays
 *  identical to the previous encoding. */
function encodeExactNumber(v: number): unknown {
  if (Number.isNaN(v)) return "nan";
  if (v === Infinity) return "+inf";
  if (v === -Infinity) return "-inf";
  if (v === 0 && Object.is(v, -0)) return "-0";
  return v;
}

function canon(t: Type): unknown {
  switch (t.kind) {
    case "Unknown":
      return { k: "U" };
    case "Void":
      return { k: "V" };
    case "String":
      return t.exact !== undefined ? { k: "S", x: t.exact } : { k: "S" };
    case "Char":
      return t.exact !== undefined ? { k: "Ch", x: t.exact } : { k: "Ch" };
    case "Numeric": {
      const out: Record<string, unknown> = {
        k: "N",
        e: t.elem,
        c: t.isComplex,
        d: t.dims.map(d => (d.kind === "exact" ? d.value : "?")),
        s: t.sign,
      };
      if (t.shape !== undefined) out.sh = t.shape;
      if (t.exact !== undefined) {
        if (t.exact instanceof Float64Array) {
          out.x = Array.from(t.exact, encodeExactNumber);
        } else if (typeof t.exact === "number") {
          out.x = encodeExactNumber(t.exact);
        } else if (t.exact.re instanceof Float64Array) {
          // Complex-tensor split-buffer exact.
          const cx = t.exact as { re: Float64Array; im: Float64Array };
          out.x = {
            re: Array.from(cx.re, encodeExactNumber),
            im: Array.from(cx.im, encodeExactNumber),
          };
        } else {
          // Scalar complex { re, im } — sanitize each component.
          const sx = t.exact as { re: number; im: number };
          out.x = {
            re: encodeExactNumber(sx.re),
            im: encodeExactNumber(sx.im),
          };
        }
      }
      return out;
    }
    case "Handle":
      // The AST is intentionally NOT serialized — it would bloat the
      // canonical string and isn't part of the type's observable
      // shape. Two handles with the same targetName + captures (by
      // name and canonical type) share a specialization key.
      return {
        k: "H",
        n: t.targetName,
        c: t.captures.map(c => [c.name, canon(c.ty)]),
      };
    case "Struct":
      return {
        k: "St",
        f: t.fields.map(f => [f.name, canon(f.ty)]),
      };
    case "Class":
      return {
        k: "C",
        n: t.className,
        p: t.properties.map(p => [p.name, canon(p.ty)]),
      };
    case "Cell":
      if (t.mode === "tuple") {
        return {
          k: "Cl",
          m: "t",
          sh: t.shape,
          e: t.elements!.map(canon),
        };
      }
      return {
        k: "Cl",
        m: "u",
        d: t.dims.map(d => (d.kind === "exact" ? d.value : "?")),
        sh: t.shape,
        e: canon(t.elem!),
      };
  }
}

/** FNV-1a 32-bit; matches the mangling scheme used in current mtoc. */
export function hashType(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

export function specializationKey(argTypes: Type[]): string {
  return hashType(argTypes.map(canonicalizeType).join("|"));
}

/** Mangled C typedef name for a handle's capture shape. All no-capture
 *  handles share `mtoc2_handle_empty_t` regardless of target identity
 *  (the function dispatch is static — the struct is just a carrier).
 *  Handles with captures get a per-shape `mtoc2_handle__<8hex>` typedef
 *  whose hash covers the `(name, cFieldTypeStr)` tuple of each capture
 *  in order. Matches the struct/class precedent: the typedef hash sees
 *  only the C-level type of each capture, so two handle types that
 *  differ only in lattice precision (sign, exact, tensor shape) share
 *  one typedef and one set of owned-helpers. */
export function handleTypedefName(t: HandleType): string {
  if (t.captures.length === 0) return "mtoc2_handle_empty_t";
  const canonical = JSON.stringify(
    t.captures.map(c => [c.name, cFieldTypeStr(c.ty)])
  );
  return `mtoc2_handle__${hashType(canonical)}`;
}

/** C-level type string for a struct/class field or function-handle
 *  capture. This is the load-bearing identity for typedef hashing:
 *  the typedef hash depends only on `cFieldTypeStr` per field, so
 *  two structs whose internal field types differ only in lattice
 *  precision (sign, exact, tensor shape) collapse to the SAME C
 *  typedef. The internal `StructType.fields[*].ty` is free to keep
 *  full precision — that precision drives function specialization
 *  keys and builtin transfer functions but does NOT shard the C
 *  typedef.
 *
 *  - Scalar real numeric (any sign/exact)   → "double"
 *  - Multi-element tensor (any shape/exact) → "mtoc2_tensor_t"
 *  - Handle                                  → handleTypedefName(t)
 *  - Struct (recurse on fields)              → structTypedefName(t)
 *  - Class (recurse on properties)           → classTypedefName(t)
 *
 *  String / Void / Unknown are not valid struct/class field types
 *  in v1 — the lowerer rejects them at construction sites. */
export function cFieldTypeStr(t: Type): string {
  if (t.kind === "Numeric") {
    if (isMultiElement(t)) return "mtoc2_tensor_t";
    if (t.isComplex) return "double _Complex";
    return "double";
  }
  if (t.kind === "String") return "mtoc2_string_t";
  if (t.kind === "Char") return "mtoc2_char_tensor_t";
  if (t.kind === "Handle") return handleTypedefName(t);
  if (t.kind === "Struct") return structTypedefName(t);
  if (t.kind === "Class") return classTypedefName(t);
  if (t.kind === "Cell") return cellTypedefName(t);
  throw new Error(
    `cFieldTypeStr: type '${t.kind}' is not a valid struct/class field type`
  );
}

/** Mangled C typedef name for a struct's shape. Keyed only on the
 *  C-level type of each field (`cFieldTypeStr`), so two struct
 *  values whose fields' internal types differ only in lattice
 *  precision share one typedef. The internal `StructType.fields[*].ty`
 *  is still carried at full precision for spec keying + transfer-fn
 *  use — but it's separate from the typedef identity. */
export function structTypedefName(t: StructType): string {
  const canonical = JSON.stringify(
    t.fields.map(f => [f.name, cFieldTypeStr(f.ty)])
  );
  return `mtoc2_struct__${hashType(canonical)}`;
}

/** Replace every non-C-identifier character (`[^A-Za-z0-9_]`) with `_`.
 *  Used to make `<name>__<hex>` mangled names always valid C identifiers
 *  when the source carries class/path-like characters (e.g. `pkg.foo`,
 *  `MyClass.method`). v1 class names are typically already C-safe, so
 *  this is a passthrough for those. */
export function sanitizeCIdent(s: string): string {
  return s.replace(/[^A-Za-z0-9_]/g, "_");
}

/** Mangled C typedef name for a class instance. Same C-level
 *  contract as `structTypedefName` — the hash sees only the C-level
 *  type of each property — but salted with `className` so two
 *  distinct classes with the same C-level property shape still pick
 *  distinct typedefs. */
export function classTypedefName(t: ClassType): string {
  const canonical = JSON.stringify({
    n: t.className,
    p: t.properties.map(p => [p.name, cFieldTypeStr(p.ty)]),
  });
  return `mtoc2_class_${sanitizeCIdent(t.className)}__${hashType(canonical)}`;
}

/** Mangled C typedef name for a cell shape. The hash sees the cell's
 *  mode, dim shape (with `?` for non-exact axes), and either the
 *  list of per-slot `cFieldTypeStr` values (tuple) or the elem
 *  `cFieldTypeStr` (uniform). Two cell types whose internal types
 *  differ only in lattice precision (sign, exact, tensor shape)
 *  share one typedef. */
export function cellTypedefName(t: CellType): string {
  const dimsTag = t.dims.map(d => (d.kind === "exact" ? d.value : "?"));
  let canonical: string;
  if (t.mode === "tuple") {
    canonical = JSON.stringify({
      m: "t",
      d: dimsTag,
      e: t.elements!.map(cFieldTypeStr),
    });
  } else {
    canonical = JSON.stringify({
      m: "u",
      d: dimsTag,
      e: cFieldTypeStr(t.elem!),
    });
  }
  return `mtoc2_cell__${hashType(canonical)}`;
}

/** Class-method specialization-name source. Becomes the input to
 *  `mangleClassMethodName` along with arg-type canonicalization. */
export function classMethodSpecSource(
  className: string,
  methodName: string
): string {
  return `${className}__${methodName}`;
}

// ── Span re-export for convenience ──────────────────────────────────────

export type { Span };
