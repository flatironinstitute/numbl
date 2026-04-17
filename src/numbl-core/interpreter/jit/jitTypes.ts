/**
 * JIT type system and IR node definitions.
 */

import type { BinaryOperation, UnaryOperation } from "../../parser/types.js";

// ── JIT Type System ─────────────────────────────────────────────────────

export type SignCategory = "positive" | "nonneg" | "nonpositive" | "negative";

export type JitType =
  | { kind: "number"; exact?: number; sign?: SignCategory; isInteger?: boolean }
  | { kind: "boolean"; value?: boolean }
  | { kind: "complex_or_number"; pureImaginary?: boolean }
  | {
      kind: "tensor";
      isComplex: boolean;
      shape?: number[];
      ndim?: number;
      isLogical?: boolean;
      nonneg?: boolean;
    }
  | { kind: "string"; value?: string }
  | { kind: "char"; value?: string }
  | { kind: "struct"; fields?: Record<string, JitType> }
  | {
      /**
       * A 1-D homogeneous array of structs (matlab struct array form
       * `T.nodes(i)`). `elemFields` records the per-element field types
       * unified across all elements. `length` is optional and carries
       * the statically-known element count when available.
       */
      kind: "struct_array";
      elemFields?: Record<string, JitType>;
      length?: number;
    }
  | {
      kind: "class_instance";
      className: string;
      isHandleClass?: boolean;
      fields?: Record<string, JitType>;
    }
  | { kind: "sparse_matrix"; isComplex: boolean; m?: number; n?: number }
  | { kind: "cell"; shape?: number[] }
  | { kind: "dictionary" }
  | { kind: "function_handle" }
  | { kind: "unknown" };

// ── Sign helpers ─────────────────────────────────────────────────────────

export function signFromNumber(v: number): SignCategory | undefined {
  if (v > 0) return "positive";
  if (v === 0) return "nonneg";
  if (v < 0) return "negative";
  return undefined; // NaN
}

export function isNonneg(t: JitType): boolean {
  if (t.kind === "number") return t.sign === "nonneg" || t.sign === "positive";
  if (t.kind === "boolean") return true;
  if (t.kind === "tensor") return !!t.nonneg;
  return false;
}

export function flipSign(s?: SignCategory): SignCategory | undefined {
  if (s === "positive") return "negative";
  if (s === "negative") return "positive";
  if (s === "nonneg") return "nonpositive";
  if (s === "nonpositive") return "nonneg";
  return undefined;
}

export function unifySign(
  a?: SignCategory,
  b?: SignCategory
): SignCategory | undefined {
  if (a === b) return a;
  if (!a || !b) return undefined;
  const set = new Set([a, b]);
  if (set.has("positive") && set.has("nonneg")) return "nonneg";
  if (set.has("negative") && set.has("nonpositive")) return "nonpositive";
  return undefined;
}

export function jitTypeKey(t: JitType): string {
  switch (t.kind) {
    case "number": {
      let k = "number";
      if (t.exact !== undefined) k += `=${t.exact}`;
      if (t.sign) k += `:${t.sign}`;
      if (t.isInteger) k += ":int";
      return k;
    }
    case "boolean": {
      let k = "boolean";
      if (t.value !== undefined) k += `=${t.value}`;
      return k;
    }
    case "complex_or_number":
      return t.pureImaginary ? "complex:imag" : "complex_or_number";
    case "tensor": {
      const s = t.shape
        ? t.shape.map(d => (d === -1 ? "?" : d)).join("x")
        : t.ndim !== undefined
          ? Array(t.ndim).fill("?").join("x")
          : "?";
      let k = `tensor[${s}]`;
      if (t.isComplex === true) k += "C";
      else if (t.isComplex === false) k += "R";
      if (t.nonneg) k += "+";
      if (t.isLogical) k += "L";
      return k;
    }
    case "string":
      return t.value != null ? `string:${t.value}` : "string";
    case "char":
      return t.value != null ? `char:${t.value}` : "char";
    case "struct": {
      if (!t.fields) return "struct";
      const keys = Object.keys(t.fields).sort();
      const parts = keys.map(k => `${k}:${jitTypeKey(t.fields![k])}`);
      return `struct{${parts.join(",")}}`;
    }
    case "struct_array": {
      if (!t.elemFields) return "struct_array";
      const keys = Object.keys(t.elemFields).sort();
      const parts = keys.map(k => `${k}:${jitTypeKey(t.elemFields![k])}`);
      return `struct_array{${parts.join(",")}}`;
    }
    case "class_instance": {
      let k = `class:${t.className}`;
      if (t.isHandleClass) k += ":handle";
      if (t.fields) {
        const keys = Object.keys(t.fields).sort();
        const parts = keys.map(f => `${f}:${jitTypeKey(t.fields![f])}`);
        k += `{${parts.join(",")}}`;
      }
      return k;
    }
    case "function_handle":
      return "function_handle";
    case "sparse_matrix": {
      let k = "sparse";
      if (t.m !== undefined && t.n !== undefined) k += `[${t.m}x${t.n}]`;
      if (t.isComplex) k += "C";
      return k;
    }
    case "cell": {
      const s = t.shape ? t.shape.join("x") : "?";
      return `cell[${s}]`;
    }
    case "dictionary":
      return "dictionary";
    case "unknown":
      return "unknown";
  }
}

export function computeJitCacheKey(
  nargout: number,
  argTypes: JitType[]
): string {
  return JSON.stringify({ nargout, argTypes });
}

/** Compute a unique JS function name for a JIT'd specialization. */
export function computeJitFnName(identity: string, funcName: string): string {
  // FNV-1a hash (same as lowering/specKey.ts hashForJsId)
  let hash = 0x811c9dc5;
  for (let i = 0; i < identity.length; i++) {
    hash ^= identity.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  const hex = (hash >>> 0).toString(16).padStart(8, "0");
  return `$jit_${funcName}_${hex}`;
}

/** Widen/unify two types at control-flow join points. */
export function unifyJitTypes(a: JitType, b: JitType): JitType {
  if (a.kind === b.kind) {
    if (a.kind === "number" && b.kind === "number") {
      const exact =
        a.exact !== undefined && a.exact === b.exact ? a.exact : undefined;
      const sign =
        exact !== undefined ? signFromNumber(exact) : unifySign(a.sign, b.sign);
      const isInteger = a.isInteger && b.isInteger;
      return {
        kind: "number",
        ...(exact !== undefined ? { exact } : {}),
        ...(sign ? { sign } : {}),
        ...(isInteger ? { isInteger: true } : {}),
      };
    }
    if (a.kind === "complex_or_number" && b.kind === "complex_or_number") {
      return {
        kind: "complex_or_number",
        ...(a.pureImaginary && b.pureImaginary ? { pureImaginary: true } : {}),
      };
    }
    if (a.kind === "tensor" && b.kind === "tensor") {
      // Unify isComplex: same→keep, different→widen to true
      const isComplex = a.isComplex || b.isComplex;
      // Unify shape
      let shape: number[] | undefined;
      let ndim: number | undefined;
      if (a.shape && b.shape) {
        if (a.shape.length !== b.shape.length) {
          // Different ndim → drop shape, drop ndim
          shape = undefined;
          ndim = undefined;
        } else {
          shape = a.shape.map((d, i) => (d === b.shape![i] ? d : -1));
          ndim = shape.length;
        }
      } else {
        shape = undefined;
        // Unify ndim from shape.length or ndim field
        const aNdim = a.shape ? a.shape.length : a.ndim;
        const bNdim = b.shape ? b.shape.length : b.ndim;
        ndim = aNdim !== undefined && aNdim === bNdim ? aNdim : undefined;
      }
      // nonneg only meaningful when definitely real
      const nonneg = !isComplex && a.nonneg && b.nonneg;
      const isLogical = a.isLogical && b.isLogical;
      return {
        kind: "tensor" as const,
        isComplex,
        ...(shape ? { shape } : {}),
        ...(ndim !== undefined && !shape ? { ndim } : {}),
        ...(nonneg ? { nonneg: true } : {}),
        ...(isLogical ? { isLogical: true } : {}),
      };
    }
    if (a.kind === "string" && b.kind === "string") {
      return {
        kind: "string",
        value: a.value != null && a.value === b.value ? a.value : undefined,
      };
    }
    if (a.kind === "char" && b.kind === "char") {
      return {
        kind: "char",
        value: a.value != null && a.value === b.value ? a.value : undefined,
      };
    }
    if (a.kind === "boolean" && b.kind === "boolean") {
      return {
        kind: "boolean",
        ...(a.value !== undefined && a.value === b.value
          ? { value: a.value }
          : {}),
      };
    }
    if (a.kind === "struct" && b.kind === "struct") {
      if (!a.fields || !b.fields) return { kind: "struct" };
      // Keep fields present in both, unify their types
      const fields: Record<string, JitType> = {};
      let hasFields = false;
      for (const key of Object.keys(a.fields)) {
        if (key in b.fields) {
          fields[key] = unifyJitTypes(a.fields[key], b.fields[key]);
          hasFields = true;
        }
      }
      return { kind: "struct", ...(hasFields ? { fields } : {}) };
    }
    if (a.kind === "struct_array" && b.kind === "struct_array") {
      if (!a.elemFields || !b.elemFields) return { kind: "struct_array" };
      const elemFields: Record<string, JitType> = {};
      let hasFields = false;
      for (const key of Object.keys(a.elemFields)) {
        if (key in b.elemFields) {
          elemFields[key] = unifyJitTypes(a.elemFields[key], b.elemFields[key]);
          hasFields = true;
        }
      }
      const length =
        a.length !== undefined && a.length === b.length ? a.length : undefined;
      return {
        kind: "struct_array",
        ...(hasFields ? { elemFields } : {}),
        ...(length !== undefined ? { length } : {}),
      };
    }
    if (a.kind === "class_instance" && b.kind === "class_instance") {
      if (a.className !== b.className) return { kind: "unknown" };
      const isHandleClass =
        a.isHandleClass === b.isHandleClass ? a.isHandleClass : undefined;
      if (!a.fields || !b.fields) {
        return {
          kind: "class_instance",
          className: a.className,
          ...(isHandleClass !== undefined ? { isHandleClass } : {}),
        };
      }
      const fields: Record<string, JitType> = {};
      let hasFields = false;
      for (const key of Object.keys(a.fields)) {
        if (key in b.fields) {
          fields[key] = unifyJitTypes(a.fields[key], b.fields[key]);
          hasFields = true;
        }
      }
      return {
        kind: "class_instance",
        className: a.className,
        ...(isHandleClass !== undefined ? { isHandleClass } : {}),
        ...(hasFields ? { fields } : {}),
      };
    }
    if (a.kind === "dictionary" && b.kind === "dictionary") {
      return { kind: "dictionary" };
    }
    if (a.kind === "function_handle" && b.kind === "function_handle") {
      return { kind: "function_handle" };
    }
    if (a.kind === "cell" && b.kind === "cell") {
      // Unify shapes: same→keep, different→drop
      let shape: number[] | undefined;
      if (a.shape && b.shape && a.shape.length === b.shape.length) {
        shape = a.shape.map((d, i) => (d === b.shape![i] ? d : -1));
      }
      return { kind: "cell", ...(shape ? { shape } : {}) };
    }
    if (a.kind === "sparse_matrix" && b.kind === "sparse_matrix") {
      return {
        kind: "sparse_matrix",
        isComplex: a.isComplex || b.isComplex,
        ...(a.m !== undefined && a.m === b.m ? { m: a.m } : {}),
        ...(a.n !== undefined && a.n === b.n ? { n: a.n } : {}),
      };
    }
    return { kind: "unknown" }; // unrecognized same-kind pair → bail
  }
  // Widen number + boolean → number (MATLAB treats logical as numeric)
  if (a.kind === "number" && b.kind === "boolean") {
    return { kind: "number", ...(a.isInteger ? { isInteger: true } : {}) };
  }
  if (a.kind === "boolean" && b.kind === "number") {
    return { kind: "number", ...(b.isInteger ? { isInteger: true } : {}) };
  }
  // Widen number/boolean → complex_or_number (every real is a valid complex)
  if (
    (a.kind === "complex_or_number" &&
      (b.kind === "number" || b.kind === "boolean")) ||
    ((a.kind === "number" || a.kind === "boolean") &&
      b.kind === "complex_or_number")
  ) {
    return { kind: "complex_or_number" };
  }
  return { kind: "unknown" };
}

export function isScalarType(t: JitType): boolean {
  return (
    t.kind === "number" ||
    t.kind === "boolean" ||
    t.kind === "complex_or_number" ||
    t.kind === "string" ||
    t.kind === "char"
  );
}

export function isNumericScalarType(t: JitType): boolean {
  return (
    t.kind === "number" ||
    t.kind === "boolean" ||
    t.kind === "complex_or_number"
  );
}

export function isTensorType(t: JitType): boolean {
  return t.kind === "tensor";
}

export function isComplexType(t: JitType): boolean {
  return (
    t.kind === "complex_or_number" ||
    (t.kind === "tensor" && t.isComplex === true)
  );
}

/** True when the type guarantees an integer value at runtime (safe for |0). */
export function isKnownInteger(t: JitType): boolean {
  return t.kind === "boolean" || (t.kind === "number" && t.isInteger === true);
}

/** Types that support arithmetic binary operations in the JIT. */
export function isArithmeticType(t: JitType): boolean {
  return (
    t.kind === "number" ||
    t.kind === "boolean" ||
    t.kind === "complex_or_number" ||
    t.kind === "tensor"
  );
}

function isVectorShape(shape: number[]): boolean {
  if (shape.length === 1) return shape[0] !== -1;
  if (shape.length === 2) {
    return (
      (shape[0] === 1 && shape[1] !== -1) || (shape[1] === 1 && shape[0] !== -1)
    );
  }
  return false;
}

export function shapeAfterReduction(
  shape: number[],
  dim?: number
): { scalar: true } | { scalar: false; shape: number[] } {
  if (dim !== undefined) {
    // Reduction along a dimension beyond ndims returns the shape unchanged
    // (MATLAB: sum(A, dim) where dim > ndims(A) returns A)
    if (dim > shape.length) {
      if (shape.every(d => d === 1)) return { scalar: true };
      return { scalar: false, shape: [...shape] };
    }
    const result = [...shape];
    result[dim - 1] = 1;
    while (result.length > 2 && result[result.length - 1] === 1) result.pop();
    if (result.every(d => d === 1)) return { scalar: true };
    return { scalar: false, shape: result };
  }
  if (isVectorShape(shape)) return { scalar: true };
  const firstNonSingleton = shape.findIndex(d => d !== 1);
  if (firstNonSingleton === -1) return { scalar: true };
  const result = [...shape];
  result[firstNonSingleton] = 1;
  while (result.length > 2 && result[result.length - 1] === 1) result.pop();
  return { scalar: false, shape: result };
}

// ── IR Nodes ────────────────────────────────────────────────────────────

export type JitExpr =
  | { tag: "NumberLiteral"; value: number; jitType: JitType }
  | { tag: "ImagLiteral"; jitType: JitType }
  | { tag: "Var"; name: string; jitType: JitType }
  | {
      tag: "Binary";
      op: BinaryOperation;
      left: JitExpr;
      right: JitExpr;
      jitType: JitType;
    }
  | { tag: "Unary"; op: UnaryOperation; operand: JitExpr; jitType: JitType }
  | { tag: "StringLiteral"; value: string; isChar: boolean; jitType: JitType }
  | { tag: "Call"; name: string; args: JitExpr[]; jitType: JitType }
  | {
      tag: "UserCall";
      jitName: string;
      name: string;
      args: JitExpr[];
      jitType: JitType;
    }
  | { tag: "Index"; base: JitExpr; indices: JitExpr[]; jitType: JitType }
  | {
      /**
       * Range slice read on a real tensor: `src(a:b)` producing a fresh
       * column-vector tensor of length `b - a + 1`. Stage 21 — unblocks
       * chunkie's chunk_nearparam Newton loop which splits a Legendre
       * expansion vector via `r0 = all0(1:dim)`.
       *
       * Codegen emits `$h.subarrayCopy1r(srcData, srcLen, start, end)`
       * using the hoisted data/length aliases. Per-iter allocation is
       * unavoidable without a range-alias extension to stage 5; small
       * slices are cheap in V8 young-gen.
       */
      tag: "RangeSliceRead";
      baseName: string;
      start: JitExpr;
      /** `null` when the range endpoint is the `end` keyword — codegen
       *  substitutes the hoisted `.data.length` alias. */
      end: JitExpr | null;
      jitType: JitType;
    }
  | {
      tag: "TensorLiteral";
      rows: JitExpr[][];
      nRows: number;
      nCols: number;
      jitType: JitType;
    }
  | {
      /**
       * Vertical concat growth: `[base; value]` where `base` is a real
       * column-vector (or empty) tensor and `value` is a numeric scalar.
       * Lowered from the chunkie grow-a-list pattern `it = [it; i]`.
       * Codegens to `$h.vconcatGrow1r(base, value)` which allocates a
       * fresh `(k+1, 1)` tensor and copies.
       */
      tag: "VConcatGrow";
      base: JitExpr;
      value: JitExpr;
      jitType: JitType;
    }
  | {
      /**
       * Scalar read of a struct field: `s.f` where `s` has a struct type
       * with a statically-known scalar field `f`. Lowered from `Member`
       * exprs with an `Ident` base. The codegen hoists each unique
       * `(baseName, fieldName)` pair to a local alias at function entry
       * so per-iter reads are bare local loads (no `Map.get` per use).
       * Mirrors the chunkie `opts.rho`, `chnkr.k`, `chnkr.nch` pattern.
       */
      tag: "MemberRead";
      baseName: string;
      fieldName: string;
      jitType: JitType;
    }
  | {
      /**
       * Chained struct array member read: `T.nodes(i).leaf` where
       * `T.nodes` is a struct array whose elements all share a field
       * `leaf`. Lowered from the parser shape
       * `Member(MethodCall(Ident(T), "nodes", [i]), "leaf")`.
       *
       * Codegen hoists `var $T_nodes_elements = T.fields.get("nodes").elements`
       * at function entry (one alias per unique (structVarName,
       * structArrayFieldName) pair) and emits
       * `$T_nodes_elements[Math.round(i) - 1].fields.get("leaf")` per use.
       *
       * `leafFieldJitType` records the result type: either a scalar
       * numeric type (number/boolean/complex_or_number) or a real tensor.
       * Tensor-valued leaves work because the existing per-Assign hoist
       * refresh picks up the newly-read tensor when the result flows
       * into `chld = T.nodes(i).chld` and the callers then use
       * `chld(k)` through the hoisted alias path.
       */
      tag: "StructArrayMemberRead";
      structVarName: string;
      structArrayFieldName: string;
      indexExpr: JitExpr;
      leafFieldName: string;
      jitType: JitType;
    }
  | {
      /**
       * Indirect call through a function handle variable: `f(a, b)` where
       * `f` has JitType `function_handle`. The codegen emits
       * `$h.callFuncHandle($rt, f, arg1, arg2, ...)` which invokes the
       * handle via its `jsFn` closure (or falls back to `rt.dispatch`).
       * The `returnType` is determined at lowering time by probing the
       * function handle's runtime value.
       */
      tag: "FuncHandleCall";
      name: string;
      args: JitExpr[];
      jitType: JitType;
    }
  | {
      /**
       * User function call that bailed recursive lowering (stage 24 —
       * soft-bail). Instead of bailing the containing loop, emit a
       * runtime dispatch through the interpreter. Return type is
       * determined at JIT compile time by probing (actually calling
       * the function once with representative args). The helper
       * `callUserFunc` verifies the actual return type on every call
       * and throws `JitFuncHandleBailError` on mismatch so the loop
       * runner can invalidate the cache entry and fall back to
       * interpretation.
       *
       * Use case: `oneintp(...)` / `lege.exev(...)` in chunkie's adap
       * inner loop — the callee body has tensor arithmetic the JIT
       * can't inline, but the outer loop (scalar stack + col-slice
       * writes + scalar compares) is a perfect JIT fit.
       */
      tag: "UserDispatchCall";
      name: string;
      args: JitExpr[];
      jitType: JitType;
    };

export type JitStmt =
  | { tag: "Assign"; name: string; expr: JitExpr }
  | {
      tag: "AssignIndex";
      /** Name of the tensor variable being written to (must be an Ident-based lvalue). */
      baseName: string;
      /** Scalar index expressions (1..3). */
      indices: JitExpr[];
      /** RHS scalar value to store. */
      value: JitExpr;
      /** Type of the base tensor (for codegen to choose the right helper). */
      baseType: JitType;
    }
  | {
      /**
       * Range-slice write of the form `dst(a:b) = src(c:d)` or
       * `dst(a:b) = src` (whole-tensor source). Both `dst` and `src` must
       * be real tensors, and the LHS must use exactly one (linear) range
       * index. Used by the loop JIT for the chunkie grow-and-copy pattern.
       *
       * When `srcStart` and `srcEnd` are both `null`, the source is used
       * in its entirety — codegen substitutes `1` and `srcLen` (the
       * source's hoisted length alias). This is stage 9's whole-tensor
       * RHS form: `isp(1:nn) = itemp` where `itemp` is a plain Var.
       */
      tag: "AssignIndexRange";
      baseName: string;
      baseType: JitType;
      dstStart: JitExpr;
      dstEnd: JitExpr;
      srcBaseName: string;
      srcType: JitType;
      srcStart: JitExpr | null;
      srcEnd: JitExpr | null;
    }
  | {
      /**
       * Multi-dim column slice write `dst(:, j) = src` where both `dst`
       * and `src` are real tensors. LHS must be `Index(Ident(dst),
       * [Colon, scalar_j])` with `dst` having statically 2-D shape.
       * RHS must be a plain Ident referencing a real-tensor var whose
       * total element count matches `dst.shape[0]` at runtime.
       *
       * Used by the loop JIT for chunkie's adapgausskerneval column
       * stack write: `vals(:, jj+1) = v2`. Also covers
       * chunkerinterior's `rss(:, jj) = rval(:, k)` after a slice
       * alias binds the RHS.
       */
      tag: "AssignIndexCol";
      baseName: string;
      baseType: JitType;
      colIndex: JitExpr;
      srcBaseName: string;
      srcType: JitType;
    }
  | {
      /**
       * Struct field assign lvalue `s.f = v`. When `needsPromote` is
       * true, the base is (re)initialized as a fresh empty struct
       * before the field is set — mirrors MATLAB's
       * `s = []; s.f = v` idiom that promotes the empty matrix to a
       * struct on first field write. `valueType` is the JIT type of
       * `value` so stage 12 can hoist subsequent reads through a
       * typed alias.
       */
      tag: "AssignMember";
      baseName: string;
      fieldName: string;
      value: JitExpr;
      needsPromote: boolean;
    }
  | {
      tag: "If";
      cond: JitExpr;
      thenBody: JitStmt[];
      elseifBlocks: { cond: JitExpr; body: JitStmt[] }[];
      elseBody: JitStmt[] | null;
    }
  | {
      tag: "For";
      varName: string;
      start: JitExpr;
      step: JitExpr | null;
      end: JitExpr;
      body: JitStmt[];
    }
  | { tag: "While"; cond: JitExpr; body: JitStmt[] }
  | { tag: "Break" }
  | { tag: "Continue" }
  | { tag: "Return" }
  | { tag: "ExprStmt"; expr: JitExpr }
  | {
      tag: "MultiAssign";
      names: (string | null)[];
      callName: string;
      args: JitExpr[];
      outputTypes: JitType[];
    }
  | { tag: "SetLoc"; line: number }
  | {
      /** C-JIT assertion: if this JitStmt survives to JS codegen at --opt 2,
       *  emit a runtime throw. The C-JIT codegen elides it. */
      tag: "AssertCJit";
    };

// (Also defined in JitExpr below)

// ── Scalar math builtins ────────────────────────────────────────────────

// ── Cache entry type ────────────────────────────────────────────────────

export interface JitCacheEntry {
  fn: (...args: unknown[]) => unknown;
  source: string;
}
