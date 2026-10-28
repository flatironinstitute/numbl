/**
 * JIT type system.
 *
 * Used by the executor classification phase and by the JIT type
 * adapter to map numbl runtime values onto JIT specializations.
 * The IR node definitions that previously lived alongside this file
 * (`JitExpr` / `JitStmt` / `JitCacheEntry`) belonged to the legacy
 * in-tree JS-JIT / C-JIT pipeline and were deleted with it.
 */

import {
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeSparseMatrix,
  isRuntimeStruct,
  isRuntimeStructArray,
  type RuntimeCell,
  type RuntimeChar,
  type RuntimeClassInstance,
  type RuntimeClassInstanceArray,
  type RuntimeSparseMatrix,
  type RuntimeStruct,
  type RuntimeStructArray,
  type RuntimeTensor,
  type RuntimeValue,
} from "./runtime/types.js";

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
  | {
      kind: "cell";
      shape?: number[];
      /** Per-index element types (1-based), tracked when literal-index
       *  writes happen with a known type. Reads with a literal index in
       *  the map return the tracked type instead of `unknown`, which
       *  unblocks chunkerfunc's `[out{1:3}] = fcurve(ts); r = out{1}; …`
       *  pattern. Sparse — missing keys mean "unknown". */
      elements?: Record<number, JitType>;
    }
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

function unifySign(
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
      // Unify per-index element types: keep only entries that are
      // present in both with a non-unknown unification.
      let elements: Record<number, JitType> | undefined;
      if (a.elements && b.elements) {
        for (const k of Object.keys(a.elements)) {
          const idx = Number(k);
          const bt = b.elements[idx];
          if (!bt) continue;
          const u = unifyJitTypes(a.elements[idx], bt);
          if (u.kind === "unknown") continue;
          (elements ??= {})[idx] = u;
        }
      }
      return {
        kind: "cell",
        ...(shape ? { shape } : {}),
        ...(elements ? { elements } : {}),
      };
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

// ── Inference from runtime values ───────────────────────────────────────

/** The JitType of a runtime value (executor classification). */
export function inferJitType(value: unknown): JitType {
  if (typeof value === "boolean") return { kind: "boolean", value };
  if (typeof value === "number") {
    const sign = signFromNumber(value);
    return { kind: "number", exact: value, ...(sign ? { sign } : {}) };
  }
  if (typeof value === "string") {
    return { kind: "string", value };
  }
  if (isRuntimeChar(value as RuntimeChar)) {
    return { kind: "char", value: (value as RuntimeChar).value };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "complex_number"
  ) {
    return { kind: "complex_or_number" };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "tensor"
  ) {
    const t = value as RuntimeTensor;
    const shape = t.shape.length >= 2 ? t.shape.slice() : [1, ...t.shape];
    if (t.imag) return { kind: "tensor", isComplex: true, shape };
    return {
      kind: "tensor",
      isComplex: false,
      shape,
      ...(t._isLogical ? { isLogical: true } : {}),
    };
  }
  if (isRuntimeStruct(value as RuntimeValue)) {
    const s = value as RuntimeStruct;
    const fields: Record<string, JitType> = {};
    for (const [k, v] of s.fields) {
      // Nested struct_array fields (e.g. `T.nodes`) get a real
      // `struct_array` JitType so the JIT loop lowering can chain
      // through `T.nodes(i).leaf`. Top-level struct_array values stay
      // `unknown`.
      if (isRuntimeStructArray(v)) {
        fields[k] = inferStructArrayType(v as RuntimeStructArray);
      } else {
        fields[k] = inferJitType(v);
      }
    }
    return { kind: "struct", fields };
  }
  if (isRuntimeSparseMatrix(value as RuntimeValue)) {
    const sp = value as RuntimeSparseMatrix;
    return { kind: "sparse_matrix", isComplex: !!sp.pi, m: sp.m, n: sp.n };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "cell"
  ) {
    const c = value as RuntimeCell;
    return { kind: "cell", shape: c.shape.slice() };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "dictionary"
  ) {
    return { kind: "dictionary" };
  }
  if (isRuntimeClassInstance(value as RuntimeValue)) {
    const ci = value as RuntimeClassInstance;
    const fields: Record<string, JitType> = {};
    for (const [k, v] of ci.fields) {
      fields[k] = inferJitType(v);
    }
    return {
      kind: "class_instance",
      className: ci.className,
      isHandleClass: ci.isHandleClass,
      fields,
    };
  }
  if (isRuntimeClassInstanceArray(value as RuntimeValue)) {
    // An object array dispatches like a class instance (e.g. an enumeration
    // member array passed to `uint32`/`double`); element access is what
    // distinguishes them, not the dispatch type.
    const arr = value as RuntimeClassInstanceArray;
    return {
      kind: "class_instance",
      className: arr.className,
      isHandleClass: false,
      fields: {},
    };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "function"
  ) {
    return { kind: "function_handle" };
  }
  return { kind: "unknown" };
}

/**
 * Infer a `struct_array` JitType from a runtime struct array. Walks
 * every element, collects per-field JIT types, and unifies them across
 * elements. Used only from the `struct` branch of `inferJitType`;
 * top-level struct array values are `{kind: "unknown"}`.
 */
function inferStructArrayType(sa: RuntimeStructArray): JitType {
  if (sa.elements.length === 0) {
    return { kind: "struct_array", length: 0 };
  }
  const elemFields: Record<string, JitType> = {};
  const first = sa.elements[0];
  for (const [k, v] of first.fields) {
    elemFields[k] = inferJitType(v);
  }
  for (let i = 1; i < sa.elements.length; i++) {
    const el = sa.elements[i];
    for (const k of Object.keys(elemFields)) {
      const v = el.fields.get(k);
      if (v === undefined) {
        delete elemFields[k];
        continue;
      }
      const before = elemFields[k];
      const vType = inferJitType(v);
      elemFields[k] = unifyStructArrayFieldTypes(before, vType);
    }
  }
  return {
    kind: "struct_array",
    elemFields,
    length: sa.elements.length,
  };
}

/**
 * Unify two element types for a struct array field. Unlike generic
 * `unifyJitTypes`, this widens mixed real-tensor / numeric-scalar
 * combinations to `tensor[?x?]` instead of bailing to `unknown`.
 *
 * Rationale: chunkie (and many MATLAB libraries) store 1-element vector
 * fields as bare numeric scalars (e.g. `T.nodes(i).xi = 87` when a leaf
 * has a single point) while other elements hold genuine row-vector
 * tensors. Treating the whole field as `unknown` bails the JIT on
 * `T.nodes(i).xi` reads even though the pattern is semantically sound
 * — a scalar is a 1x1 tensor in MATLAB. The JIT codegen for
 * `StructArrayMemberRead` with a tensor leaf wraps the element read in
 * `$h.asTensor(...)` so a runtime scalar is promoted to a 1x1 tensor
 * before the tensor-read fast paths see it.
 */
function unifyStructArrayFieldTypes(a: JitType, b: JitType): JitType {
  // Try the normal unification first — it handles same-kind merges
  // correctly (number+number, tensor+tensor, etc.)
  const normal = unifyJitTypes(a, b);
  if (normal.kind !== "unknown") return normal;
  // Normal unify returned unknown — this happens for mixed
  // tensor-vs-scalar combinations. In MATLAB, a scalar is a 1x1
  // tensor, so we widen to tensor[?x?] when one side is a real tensor
  // and the other is a numeric scalar.
  const isRealTensor = (t: JitType) =>
    t.kind === "tensor" && t.isComplex !== true;
  const isNumScalar = (t: JitType) =>
    t.kind === "number" || t.kind === "boolean";
  if (
    (isRealTensor(a) && isNumScalar(b)) ||
    (isNumScalar(a) && isRealTensor(b))
  ) {
    return { kind: "tensor", isComplex: false };
  }
  return normal;
}
