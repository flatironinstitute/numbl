/**
 * JIT type system and IR node definitions.
 */

import type { BinaryOperation, UnaryOperation } from "../../parser/types.js";

// ── JIT Type System ─────────────────────────────────────────────────────

export type JitType =
  | { kind: "number"; nonneg?: boolean }
  | { kind: "logical" }
  | { kind: "complex" }
  | { kind: "realTensor"; nonneg?: boolean; isLogical?: boolean }
  | { kind: "complexTensor" }
  | { kind: "string"; value?: string }
  | { kind: "char"; value?: string }
  | { kind: "unknown" };

export function jitTypeKey(t: JitType): string {
  switch (t.kind) {
    case "number":
      return t.nonneg ? "number+" : "number";
    case "logical":
      return "logical";
    case "complex":
      return "complex";
    case "realTensor": {
      let k = t.nonneg ? "realTensor+" : "realTensor";
      if (t.isLogical) k += "L";
      return k;
    }
    case "complexTensor":
      return "complexTensor";
    case "string":
      return t.value != null ? `string:${t.value}` : "string";
    case "char":
      return t.value != null ? `char:${t.value}` : "char";
    case "unknown":
      return "unknown";
  }
}

export function computeJitCacheKey(
  nargout: number,
  argTypes: JitType[]
): string {
  return `${nargout}:${argTypes.map(jitTypeKey).join(":")}`;
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
      return { kind: "number", nonneg: a.nonneg && b.nonneg };
    }
    if (a.kind === "realTensor" && b.kind === "realTensor") {
      return {
        kind: "realTensor",
        nonneg: a.nonneg && b.nonneg,
        isLogical: a.isLogical && b.isLogical,
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
    return a; // same kind, no flags to merge
  }
  // logical is a subtype of number
  if (
    (a.kind === "logical" && b.kind === "number") ||
    (a.kind === "number" && b.kind === "logical")
  ) {
    return {
      kind: "number",
      nonneg:
        (a as { nonneg?: boolean }).nonneg &&
        (b as { nonneg?: boolean }).nonneg,
    };
  }
  return { kind: "unknown" };
}

export function isScalarType(t: JitType): boolean {
  return (
    t.kind === "number" ||
    t.kind === "logical" ||
    t.kind === "complex" ||
    t.kind === "string" ||
    t.kind === "char"
  );
}

export function isTensorType(t: JitType): boolean {
  return t.kind === "realTensor" || t.kind === "complexTensor";
}

export function isRealType(t: JitType): boolean {
  return t.kind === "number" || t.kind === "logical" || t.kind === "realTensor";
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
  | { tag: "ExprStmt"; expr: JitExpr };

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
