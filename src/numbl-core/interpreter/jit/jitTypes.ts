/**
 * JIT type system and IR node definitions.
 */

import type { BinaryOperation, UnaryOperation } from "../../parser/types.js";

// ── JIT Type System ─────────────────────────────────────────────────────

export type SignCategory = "positive" | "nonneg" | "nonpositive" | "negative";

export type JitType =
  | { kind: "number"; exact?: number; sign?: SignCategory }
  | { kind: "boolean"; value?: boolean }
  | { kind: "complex"; pureImaginary?: boolean }
  | {
      kind: "tensor";
      isComplex?: boolean;
      shape?: number[];
      ndim?: number;
      isLogical?: boolean;
      nonneg?: boolean;
    }
  | { kind: "string"; value?: string }
  | { kind: "char"; value?: string }
  | { kind: "struct"; fields?: Record<string, JitType> }
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
      return k;
    }
    case "boolean": {
      let k = "boolean";
      if (t.value !== undefined) k += `=${t.value}`;
      return k;
    }
    case "complex":
      return t.pureImaginary ? "complex:imag" : "complex";
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
      return {
        kind: "number",
        ...(exact !== undefined ? { exact } : {}),
        ...(sign ? { sign } : {}),
      };
    }
    if (a.kind === "complex" && b.kind === "complex") {
      return {
        kind: "complex",
        ...(a.pureImaginary && b.pureImaginary ? { pureImaginary: true } : {}),
      };
    }
    if (a.kind === "tensor" && b.kind === "tensor") {
      // Unify isComplex: same→keep, different→undefined
      const isComplex = a.isComplex === b.isComplex ? a.isComplex : undefined;
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
      const nonneg = isComplex !== true && a.nonneg && b.nonneg;
      const isLogical = a.isLogical && b.isLogical;
      return {
        kind: "tensor" as const,
        ...(isComplex !== undefined ? { isComplex } : {}),
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
    return a; // same kind, no flags to merge
  }
  return { kind: "unknown" };
}

export function isScalarType(t: JitType): boolean {
  return (
    t.kind === "number" ||
    t.kind === "boolean" ||
    t.kind === "complex" ||
    t.kind === "string" ||
    t.kind === "char"
  );
}

export function isTensorType(t: JitType): boolean {
  return t.kind === "tensor";
}

export function isComplexType(t: JitType): boolean {
  return t.kind === "complex" || (t.kind === "tensor" && t.isComplex === true);
}

export function isRealType(t: JitType): boolean {
  return (
    t.kind === "number" ||
    t.kind === "boolean" ||
    (t.kind === "tensor" && t.isComplex !== true)
  );
}

export function isVectorShape(shape: number[]): boolean {
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
  | { tag: "UserCall"; jitName: string; args: JitExpr[]; jitType: JitType }
  | { tag: "Index"; base: JitExpr; indices: JitExpr[]; jitType: JitType }
  | {
      tag: "TensorLiteral";
      rows: JitExpr[][];
      nRows: number;
      nCols: number;
      jitType: JitType;
    };

export type JitStmt =
  | { tag: "Assign"; name: string; expr: JitExpr }
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
    };

// ── Scalar math builtins ────────────────────────────────────────────────

export interface ScalarMathEntry {
  arity: number;
  /** Compute the result type given argument types */
  resultType: (argTypes: JitType[]) => JitType | null;
}

/** @deprecated All scalar math is now handled by IBuiltins (see ibuiltins.ts). */
/** @deprecated All scalar math is now handled by IBuiltins (see ibuiltins.ts). */
export const SCALAR_MATH: Record<string, ScalarMathEntry> = {};

// ── Cache entry type ────────────────────────────────────────────────────

export interface JitCacheEntry {
  fn: (...args: unknown[]) => unknown;
  source: string;
}
