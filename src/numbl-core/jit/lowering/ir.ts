/**
 * mtoc2 intermediate representation. Built from scratch; only what the
 * MVP scope needs (scalar real double, arith + comparisons, disp,
 * if/while/for, user functions with specialization).
 *
 * Every node carries a `Span` for error attribution. Every IRExpr
 * carries a `Type`.
 */

import type { Span } from "../parser/index.js";
import type { NumericType, Type } from "./types.js";
import { BinaryOperation, UnaryOperation } from "../parser/index.js";

// ── Expressions ─────────────────────────────────────────────────────────

export interface NumLit {
  kind: "NumLit";
  value: number;
  ty: Type;
  span: Span;
}

/** Imaginary-unit literal — produced for `1i`, `2.5i`, etc. `value` is
 *  the imaginary coefficient (`2.5` for `2.5i`); the corresponding
 *  real part is implicitly `0`. The lowerer collapses the parser's
 *  `Number * ImagUnit` shape and bare-`ImagUnit` reads into this node;
 *  codegen emits `(<value> * I)` (using `<complex.h>`'s `I` macro). */
export interface ImagLit {
  kind: "ImagLit";
  value: number;
  ty: Type;
  span: Span;
}

/** Source-level string or char literal. Today only consumed by
 *  reducer builtins (`sum(A, 'all')`, `min(A, [], 'all')`, etc.) at
 *  transfer time — they read `ty.exact` to dispatch on the literal.
 *  Codegen renders the IR node as a C string literal so the bare
 *  expression compiles, but the reducer builtins' `codegenC` ignores
 *  the slot entirely (the dim choice is encoded in the helper name).
 *  No other context accepts a `String`-typed value; the lowerer
 *  rejects it via `requireValueType`-adjacent checks per call site. */
export interface StringLit {
  kind: "StringLit";
  value: string;
  ty: Type;
  span: Span;
}

/** Runtime tensor construction for every tensor source-literal. Codegen
 *  emits `mtoc2_tensor_from_row` (1×N) or `mtoc2_tensor_from_matrix`
 *  (rows×cols) with a C99 compound literal of the per-element
 *  expressions. The shape is statically known; the element values may
 *  be any IR expressions (NumLit for literal cells, Var/Binary/... for
 *  computed cells).
 *
 *  `elements` is column-major and length-matches `shape`. */
export interface TensorBuild {
  kind: "TensorBuild";
  elements: IRExpr[];
  shape: number[];
  ty: Type;
  span: Span;
}

export interface Var {
  kind: "Var";
  /** Source name (for diagnostics). */
  name: string;
  /** Allocated C identifier. Each SSA-like assignment gets a fresh one. */
  cName: string;
  ty: Type;
  span: Span;
}

export interface Binary {
  kind: "Binary";
  /** Builtin name (e.g. "plus", "minus"). */
  builtin: string;
  op: BinaryOperation;
  left: IRExpr;
  right: IRExpr;
  ty: Type;
  span: Span;
}

export interface Unary {
  kind: "Unary";
  builtin: string;
  op: UnaryOperation;
  operand: IRExpr;
  ty: Type;
  span: Span;
}

/** Single-output function call. `Call` covers three distinct emit
 *  paths discriminated by how codegen interprets `cName`:
 *
 *  1. **User-function specialization** — the lowerer's `buildUserFunctionCall`
 *     path. `cName` is the mangled spec name returned by
 *     `specializeUserFunction` (e.g. `apply__a1b2c3d4`); codegen emits
 *     a direct call to that symbol. The `IRFunc` for the spec lives in
 *     `IRProgram.functions[cName]`.
 *  2. **Bundled builtin** — `cName` equals the source-level builtin
 *     name (e.g. `sqrt`, `zeros`, `disp`). Codegen routes through
 *     `getBuiltin(cName).codegenC(argsC, argTypes)` to synthesize the
 *     emitted C expression — there's no Standalone IRFunc.
 *  3. **Synthesized runtime call** — emitted by the lowerer for a
 *     handful of fixed runtime helpers (`mtoc2_toc_print`,
 *     `mtoc2_toc_handle_print`, the complex `'` rewrite to
 *     `transpose(conj(z))`, etc.). The `name` field carries the
 *     source-facing identifier the lowerer wants in diagnostics; the
 *     pattern is the same as the builtin route — codegen treats them
 *     as builtin-name lookups.
 *
 *  The `name` field is purely for diagnostics (user-facing call name
 *  e.g. `pkg.foo` or `Class.method` or `f`). The semantic key is
 *  `cName`. */
export interface Call {
  kind: "Call";
  /** Resolved name. For user functions: mangled spec name. For
   *  builtins / synthesized runtime calls: the source-level builtin
   *  name routed through `getBuiltin`. */
  cName: string;
  /** Source-level name (for diagnostics). */
  name: string;
  args: IRExpr[];
  ty: Type;
  span: Span;
}

/** Function-handle literal — produced by `@user_func` (no captures) or
 *  `@(p1, ..., pN) <body>` (zero+ captures). `ty` is a `HandleType`
 *  whose `targetName`/`ast` identify the underlying user function and
 *  whose `captures` list shapes the C struct. The literal renders as a
 *  C compound literal: `(mtoc2_handle_empty_t){0}` for no captures, or
 *  `(mtoc2_handle__<hex>){.cap_<name> = <value>, ...}` per shape.
 *
 *  Dispatch is static — every `h(args)` call site reads the handle
 *  variable's type, specializes against the target AST with
 *  `[...argTypes, ...captureTypes]`, and emits a direct call. The
 *  handle struct only carries captures. */
export interface HandleLit {
  kind: "HandleLit";
  /** One entry per `ty.captures` field, in the same order. Each value
   *  is the lowered IR expression for the captured local at the
   *  `@(...)` site (typically a `Var` read of the enclosing scope's
   *  binding). v1 restricts capture types to scalar real numeric, so
   *  these values are plain `double` C expressions. */
  captures: ReadonlyArray<{ name: string; value: IRExpr }>;
  ty: Type;
  span: Span;
}

/** Field read of a captured value inside a handle struct. Used at
 *  `h(args)` dispatch sites: each capture of `h`'s `HandleType` becomes
 *  a `HandleCaptureLoad` IR node that reads `h.cap_<name>` and is
 *  passed as an extra positional arg to the underlying specialization.
 *  Codegen emits `<base.cName>.cap_<captureName>`. */
export interface HandleCaptureLoad {
  kind: "HandleCaptureLoad";
  base: Var;
  captureName: string;
  ty: Type;
  span: Span;
}

/** Struct/class construction. Produced by `struct('f1', v1, ...)`
 *  literals and by the synthesized initial-receiver of a class
 *  constructor call (a StructLit whose `ty` is the constructor's
 *  initial `ClassType`). The fields list is in canonical (sorted-by-
 *  name) order to match the typedef-shape hash.
 *
 *  Codegen emits a C99 designated initializer like
 *  `(<typedef>){.f1 = <v1>, .f2 = <v2>}`. Owned-typed field values
 *  must be fresh producers (ANF in the lowerer guarantees this). */
export interface StructLit {
  kind: "StructLit";
  fields: ReadonlyArray<{ name: string; value: IRExpr }>;
  ty: Type;
  span: Span;
}

/** Field/property read: `s.f` (struct) or `obj.prop` (class instance),
 *  one level. Chained reads like `s.inner.f` are nested
 *  MemberLoads. `ty` is the field/property's static type. In an
 *  owned-consuming context (Assign RHS, Call arg of owned param) the
 *  codegen wraps the read in the field helper's `_copy` so the
 *  consumer receives a freshly-owned value. */
export interface MemberLoad {
  kind: "MemberLoad";
  base: IRExpr;
  field: string;
  ty: Type;
  span: Span;
}

/** Scalar element read of a multi-element tensor `base`. `indices` has
 *  length 1 (linear addressing into the column-major buffer) or
 *  `base.ty.dims.length` (full per-axis addressing). Each index lowers
 *  to a scalar real IR expression; the result `ty` is the base's
 *  element scalar type. Codegen emits `<base.cName>.real[<offset>]`.
 *
 *  The base is typed `IRExpr` rather than `Var` so the lowerer can
 *  install a `MemberLoad` (e.g. `obj.field(i)`) before ANF runs; the
 *  ANF pass hoists any non-`Var` owned producer to a fresh temp so
 *  emit-time code always sees a `Var` here. */
export interface IndexLoad {
  kind: "IndexLoad";
  base: IRExpr;
  indices: IRExpr[];
  ty: Type;
  span: Span;
}

/** Slice arg shape for `IndexSlice` / `IndexSliceStore`. `Range`'s
 *  `step` is always populated (defaults to scalar `1` when omitted in
 *  the source); v1 requires `step` to be a `NumLit` so codegen can
 *  derive the loop count and source-index arithmetic at compile time
 *  for index-position ranges. `IndexVec` carries a tensor expression
 *  whose values are 1-based indices into the corresponding axis —
 *  gather-style fancy indexing (read-only on v1's IndexSlice; not yet
 *  plumbed through IndexSliceStore). `LogicalMask` carries a logical-
 *  typed tensor whose truthy positions select indices into the
 *  corresponding axis (or the linear column-major flat buffer for
 *  the single-slot form). The result-axis size equals `sum(mask)`,
 *  which is only known at runtime. The carried expression is ANF'd
 *  to a `Var` so codegen can iterate it without re-evaluation. */
export type IndexSliceArg =
  | { kind: "Range"; start: IRExpr; step: IRExpr; end: IRExpr; span: Span }
  | { kind: "Colon"; span: Span }
  | { kind: "Scalar"; expr: IRExpr; span: Span }
  | { kind: "IndexVec"; expr: IRExpr; span: Span }
  | { kind: "LogicalMask"; expr: IRExpr; span: Span };

/** Range / colon / scalar-mix slice read. `index.length` is 1
 *  (linear) or `base.ty.dims.length` (per-axis). The result is a freshly-
 *  allocated tensor; this IR node is an owned producer and ANFs like
 *  every other tensor-producing expression.
 *
 *  The base is typed `IRExpr` for the same reason as `IndexLoad.base`:
 *  the lowerer can install a `MemberLoad` directly for `obj.f(args)`,
 *  and ANF hoists it to a fresh `Var` before emit. */
export interface IndexSlice {
  kind: "IndexSlice";
  base: IRExpr;
  index: ReadonlyArray<IndexSliceArg>;
  ty: Type;
  span: Span;
}

/** Reference to the `end` keyword inside an index slot. Renders as an
 *  axis size of the enclosing index's base — `numel(base)` for a
 *  single-slot context (`axis === "linear"`) or `base.dims[axis]`
 *  otherwise. */
export interface EndRef {
  kind: "EndRef";
  baseCName: string;
  baseTy: Type;
  axis: number | "linear";
  ty: Type;
  span: Span;
}

/** Bracket concatenation `[a, b; c, d]` where one or more cells is a
 *  multi-element tensor (not just a scalar). The all-scalar case
 *  stays on `TensorBuild` for the existing fast-path emission.
 *
 *  Layout discipline (column-major destination throughout):
 *  - `cells` is a list of rows; each row is a list of cells in source
 *    order. After lowering, every cell is either a scalar real
 *    numeric or a multi-element tensor `Var` (ANF hoists owned-
 *    producing non-Var cells to temps first).
 *  - `rowHeights[i]` is the row count of every cell in row `i` (rows
 *    are vertically aligned). `null` means at least one cell on the
 *    row has a runtime-only row count — codegen queries the witness
 *    cell's `dims[0]` and trusts that all cells on the row match
 *    (`mtoc2_check_concat_*` validates at runtime when both sides
 *    are uncertain).
 *  - `cellCols[i][j]` is the column count of cell `j` in row `i`
 *    (cells horizontally concatenate along the row). `null` mirrors
 *    `rowHeights`: runtime-only, queried from the cell's `dims[1]`.
 *  - The result's shape `[totalRows, totalCols]` may contain `null`
 *    for runtime-only axes; codegen computes the value from per-row
 *    / per-cell dim queries and feeds it to `mtoc2_tensor_alloc_nd`.
 *
 *  Empty cells (any cell whose shape contains a statically-known 0)
 *  are filtered out during lowering — they evaporate per numbl's
 *  `catAlongDim` rule. The cell grid here only contains cells whose
 *  product is statically positive OR runtime-only.
 *
 *  Owned producer. Same ANF discipline as `TensorBuild`. */
export interface TensorConcat {
  kind: "TensorConcat";
  cells: IRExpr[][];
  rowHeights: (number | null)[];
  cellCols: (number | null)[][];
  shape: (number | null)[];
  ty: Type;
  span: Span;
}

/** Cell-array source literal `{a, b, c; d, e, f}`. `elements` lists
 *  the slot expressions in column-major order; `shape` is the
 *  declared `[rows, cols]` of the literal. `ty` is a `CellType`
 *  whose `mode` is always `"tuple"` for literals — one slot type per
 *  slot, even when several happen to coincide. Owned producer; ANFs
 *  like other owned-producing expressions.
 *
 *  An empty literal `{}` produces a `CellLit` with `elements: []` and
 *  `shape: [0, 0]`, matching numbl. */
export interface CellLit {
  kind: "CellLit";
  elements: IRExpr[];
  shape: number[];
  ty: Type;
  span: Span;
}

/** Runtime cell construction from `cell(n, m, ...)` (or `cell(n)`
 *  → `n×n`). `dims` carries the IR expressions for each axis size
 *  (parallel to `ty.dims`). When every axis is statically exact and
 *  the slot count fits the exact cap, `ty.mode === "tuple"` and the
 *  emitted code lays out per-slot empty-double slots inline;
 *  otherwise `ty.mode === "uniform"` with `elem` set to the empty-
 *  double sentinel and the helper allocates the slot buffer at
 *  runtime. Owned producer; ANFs. */
export interface CellEmpty {
  kind: "CellEmpty";
  dims: IRExpr[];
  ty: Type;
  span: Span;
}

/** Brace-indexed cell read: `c{i}`, `c{i, j}`, `c{linear}`. Yields
 *  the slot's value (no cell-wrapping). `base` is always a `Var`
 *  after ANF; `indices` is one expr per source-level subscript
 *  (1-D linear access or per-axis). `ty` is the static slot type:
 *  - Tuple mode + every index exact → the specific slot's `Type`.
 *  - Tuple mode + at least one non-exact → the unified slot type
 *    (the lowerer rejects with `UnsupportedConstruct` if not all
 *    slots unify).
 *  - Uniform mode → the cell's `elem` type. */
export interface CellIndexLoad {
  kind: "CellIndexLoad";
  base: IRExpr;
  indices: IRExpr[];
  ty: Type;
  span: Span;
}

/** Range used as a value (outside index slots and for-loop bounds).
 *  Emits a freshly-allocated `1×N` row tensor at runtime via
 *  `mtoc2_tensor_make_range`. Owned producer; ANFs. The `step` may
 *  be any scalar real IR expression — unlike index-slot ranges,
 *  codegen routes through the runtime helper which takes the step
 *  at runtime. */
export interface MakeRange {
  kind: "MakeRange";
  start: IRExpr;
  step: IRExpr;
  end: IRExpr;
  ty: Type;
  span: Span;
}

export type IRExpr =
  | NumLit
  | ImagLit
  | StringLit
  | TensorBuild
  | TensorConcat
  | CellLit
  | CellEmpty
  | CellIndexLoad
  | Var
  | Binary
  | Unary
  | Call
  | HandleLit
  | HandleCaptureLoad
  | StructLit
  | MemberLoad
  | IndexLoad
  | IndexSlice
  | EndRef
  | MakeRange;

// ── Statements ──────────────────────────────────────────────────────────

export interface ExprStmt {
  kind: "ExprStmt";
  expr: IRExpr;
  span: Span;
}

export interface Assign {
  kind: "Assign";
  /** Source name. */
  name: string;
  /** Fresh C name allocated for this assignment. */
  cName: string;
  ty: Type;
  expr: IRExpr;
  span: Span;
}

export interface If {
  kind: "If";
  cond: IRExpr;
  thenBody: IRStmt[];
  elseBody: IRStmt[];
  span: Span;
}

export interface While {
  kind: "While";
  cond: IRExpr;
  body: IRStmt[];
  span: Span;
}

export interface For {
  kind: "For";
  /** Source name of loop variable. */
  varName: string;
  /** Allocated C name for the loop variable. */
  cVar: string;
  start: IRExpr;
  /** Literal numeric step. */
  step: number;
  end: IRExpr;
  body: IRStmt[];
  span: Span;
}

export interface ReturnFromFunction {
  kind: "ReturnFromFunction";
  span: Span;
}

export interface Break {
  kind: "Break";
  span: Span;
}

export interface Continue {
  kind: "Continue";
  span: Span;
}

/** Field/property write: `s.f = rhs` or chained `s.inner.f = rhs`.
 *  The `base` is always the root variable (a `Var`); `fieldPath`
 *  walks from the outermost field inward (e.g. `["inner", "f"]`).
 *  `leafTy` is the type of the leaf slot (the field actually being
 *  written). For scalar leaves codegen emits a plain
 *  `<base.cName>.<f1>...<fn> = <rhs>;`; for owned leaves it emits
 *  `<typedef>_assign(&<base.cName>.<f1>...<fn>, <rhs>);` where the
 *  typedef matches the leaf's owned-kind. */
export interface MemberStore {
  kind: "MemberStore";
  base: Var;
  fieldPath: ReadonlyArray<string>;
  leafTy: Type;
  rhs: IRExpr;
  span: Span;
}

/** Pure annotation node produced by the `%!numbl:showtype` directive.
 *  Carries a snapshot of `{name, cName, ty}` for each named variable
 *  at the directive's source position. Walk / liveness / dataflow
 *  treat it as a no-op; codegen renders one `/_ type ... _/` line
 *  per entry (using real C comment delimiters). No runtime effect. */
export interface TypeComment {
  kind: "TypeComment";
  entries: ReadonlyArray<{ name: string; cName: string; ty: Type }>;
  span: Span;
}

/** Multi-output / drop-all user-function call statement. Drives:
 *    `[a, b] = foo(x);`        (N≥2 outputs, mix of named lvalues
 *                               and ignored `~` slots; trailing
 *                               outputs may be omitted)
 *    `foo(x);`                 (N≥2-output bare statement; every
 *                               output dropped via a discard temp)
 *
 *  1-output and 0-output user-function calls do NOT use this node —
 *  they keep the simpler return-by-value (`Assign` / `ExprStmt`) and
 *  `void`-returning (`ExprStmt(Call)`) shapes respectively, because
 *  their existing C ABI doesn't need out-pointers.
 *
 *  Each entry of `outputs` is either a real binding (the slot's typed
 *  destination — driven through `recordAssignment` like any other
 *  Assign) or `null` for an ignored slot. Codegen wraps the call in a
 *  `{ … }` block and declares one inline `_mtoc2_discard_<callIdx>_<i>`
 *  per `null` slot so those temporaries stay scoped to the call. */
export interface MultiAssignCall {
  kind: "MultiAssignCall";
  /** Resolved C identifier of the callee. Two interpretations:
   *  - **User-function specialization** (`isBuiltin === false`) — the
   *    mangled spec name from `specializeUserFunction` (e.g.
   *    `apply__a1b2c3d4`); codegen emits a direct call.
   *  - **Multi-output builtin** (`isBuiltin === true`) — purely
   *    informational at this point; codegen routes through the
   *    builtin's `emit` hook for the full call string. */
  cName: string;
  /** Source-level name (for diagnostics and for builtin lookup at
   *  codegen when `isBuiltin === true`). */
  name: string;
  /** When true, `name` resolves to a registered builtin and codegen
   *  invokes its `emit` hook with `nargout = outputs.length` and the
   *  built `outArgsC` list. When false (or omitted), `cName` is a
   *  user-function spec and codegen emits a direct call. */
  isBuiltin?: boolean;
  args: IRExpr[];
  /** One entry per output slot of the callee. `ty` is the slot's
   *  static type (always populated so codegen can declare a typed
   *  discard temp for ignored slots). `binding === null` → ignored
   *  output; codegen synthesizes a discard temp. A non-null binding
   *  means "store the call's i-th output into <binding.cName>"; the
   *  binding's declaration is hoisted to function top by the emitter
   *  (owned slots via `collectOwnedLocals`, scalar slots via
   *  `collectHoistedScalarLocals`). */
  outputs: ReadonlyArray<{
    ty: Type;
    binding: { name: string; cName: string } | null;
  }>;
  span: Span;
}

/** Scalar element write into a multi-element tensor `base`. Mutates
 *  the existing buffer in place — NOT an owned re-assignment (the
 *  codegen emits `<base.cName>.real[<offset>] = <rhs>;`). The base
 *  is recorded as both a use and a def by liveness so its buffer
 *  stays live across the store.
 *
 *  `fieldPath` carries a member-rooted LHS like `chnkr.r(i, j)`: the
 *  `base` Var still names the OWNING root (struct / class instance)
 *  for liveness purposes, and the codegen targets the slot
 *  `chnkr.r` by joining `base.cName` with the field path. `leafTy`
 *  is the field's NumericType; for the bare-Ident case (`v(i) =
 *  rhs`) both are absent and `base.ty` is the NumericType. */
export interface IndexStore {
  kind: "IndexStore";
  base: Var;
  fieldPath?: ReadonlyArray<string>;
  leafTy?: NumericType;
  indices: IRExpr[];
  rhs: IRExpr;
  span: Span;
}

/** Brace-indexed cell write: `c{i} = rhs`, `c{i, j} = rhs`. The
 *  cell takes ownership of `rhs`; codegen wraps owned aliases in
 *  `_copy` per the standard owned-consume rule (ANF guarantees
 *  fresh producers / Var reads for the rhs).
 *
 *  Tuple-mode + every index exact: codegen targets the specific
 *  `slot_<i>` field directly. Tuple-mode + non-exact index falls
 *  back to a runtime switch over the slot index. Uniform-mode
 *  writes through `base.slots[<offset>]` with a runtime bounds
 *  check (no auto-grow; mtoc2 rejects out-of-bounds writes —
 *  see docs/cells_plan.md). */
export interface CellIndexStore {
  kind: "CellIndexStore";
  base: Var;
  indices: IRExpr[];
  rhs: IRExpr;
  span: Span;
}

/** Range / colon / scalar-mix slice write. Same arity rules as
 *  `IndexSlice`. RHS is either a scalar (broadcast into every slot)
 *  or a `Var` reading a named tensor (per-slot copy). Mutates the
 *  base buffer in place — not an owned re-assignment.
 *
 *  See `IndexStore` for the meaning of `fieldPath` / `leafTy` —
 *  same convention. */
export interface IndexSliceStore {
  kind: "IndexSliceStore";
  base: Var;
  fieldPath?: ReadonlyArray<string>;
  leafTy?: NumericType;
  index: ReadonlyArray<IndexSliceArg>;
  rhs: IRExpr;
  span: Span;
}

export type IRStmt =
  | ExprStmt
  | Assign
  | If
  | While
  | For
  | ReturnFromFunction
  | Break
  | Continue
  | TypeComment
  | MemberStore
  | MultiAssignCall
  | IndexStore
  | IndexSliceStore
  | CellIndexStore;

// ── Functions ───────────────────────────────────────────────────────────

export interface IRFunc {
  /** Source-level function name. */
  name: string;
  /** Mangled C identifier (name__<8-hex>). */
  cName: string;
  /** Parameter source names. */
  params: string[];
  /** Parameter C names (parallel to `params`). */
  cParams: string[];
  /** Per-parameter type at this specialization. */
  paramTypes: Type[];
  /** Source output names. 0 outputs → C `void` return type; 1 output
   *  → classic return-by-value; N≥2 outputs → C `void` return + one
   *  trailing `T_i *_mtoc2_o<i>` parameter per output. */
  outputs: string[];
  /** C names for outputs (parallel to `outputs`). */
  cOutputs: string[];
  /** Output types after lowering. */
  outputTypes: Type[];
  body: IRStmt[];
  span: Span;
}

// ── Program ─────────────────────────────────────────────────────────────

export interface IRProgram {
  /** Script-level top-level statements. */
  topLevelStmts: IRStmt[];
  /** All specialized function definitions, keyed by their mangled cName. */
  functions: Map<string, IRFunc>;
}
