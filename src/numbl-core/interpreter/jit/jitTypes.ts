/**
 * JIT type system and IR node definitions.
 */

import type { BinaryOperation, UnaryOperation } from "../../parser/types.js";

// ── JIT Type System ─────────────────────────────────────────────────────

export type JitType =
  | { kind: "number"; nonneg?: boolean }
  | { kind: "complex" }
  | { kind: "realTensor"; nonneg?: boolean }
  | { kind: "complexTensor" }
  | { kind: "unknown" };

export function jitTypeKey(t: JitType): string {
  switch (t.kind) {
    case "number":
      return t.nonneg ? "number+" : "number";
    case "complex":
      return "complex";
    case "realTensor":
      return t.nonneg ? "realTensor+" : "realTensor";
    case "complexTensor":
      return "complexTensor";
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
  if (a.kind !== b.kind) return { kind: "unknown" };
  if (a.kind === "number" && b.kind === "number") {
    return { kind: "number", nonneg: a.nonneg && b.nonneg };
  }
  if (a.kind === "realTensor" && b.kind === "realTensor") {
    return { kind: "realTensor", nonneg: a.nonneg && b.nonneg };
  }
  return a; // same kind, no flags to merge
}

export function isScalarType(t: JitType): boolean {
  return t.kind === "number" || t.kind === "complex";
}

export function isTensorType(t: JitType): boolean {
  return t.kind === "realTensor" || t.kind === "complexTensor";
}

export function isRealType(t: JitType): boolean {
  return t.kind === "number" || t.kind === "realTensor";
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
  | { tag: "Call"; name: string; args: JitExpr[]; jitType: JitType }
  | { tag: "UserCall"; jitName: string; args: JitExpr[]; jitType: JitType };

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

function unaryNumberResult(
  compute: (nonneg: boolean) => boolean
): (argTypes: JitType[]) => JitType | null {
  return argTypes => {
    const a = argTypes[0];
    if (a.kind === "number")
      return { kind: "number", nonneg: compute(!!a.nonneg) };
    if (a.kind === "realTensor")
      return { kind: "realTensor", nonneg: compute(!!a.nonneg) };
    return null;
  };
}

/** Table of hard-coded scalar math functions and their type rules. */
export const SCALAR_MATH: Record<string, ScalarMathEntry> = {
  // Trig (output always loses nonneg)
  sin: { arity: 1, resultType: unaryNumberResult(() => false) },
  cos: { arity: 1, resultType: unaryNumberResult(() => false) },
  tan: { arity: 1, resultType: unaryNumberResult(() => false) },
  asin: { arity: 1, resultType: unaryNumberResult(() => false) },
  acos: { arity: 1, resultType: unaryNumberResult(() => false) },
  atan: { arity: 1, resultType: unaryNumberResult(() => false) },
  sinh: { arity: 1, resultType: unaryNumberResult(() => false) },
  cosh: { arity: 1, resultType: unaryNumberResult(() => false) },
  tanh: { arity: 1, resultType: unaryNumberResult(() => false) },

  // Always nonneg output
  abs: { arity: 1, resultType: unaryNumberResult(() => true) },
  exp: { arity: 1, resultType: unaryNumberResult(() => true) },

  // Requires nonneg input (negative → complex in MATLAB, can't JIT that)
  sqrt: {
    arity: 1,
    resultType: argTypes => {
      const a = argTypes[0];
      if (a.kind === "number" && a.nonneg)
        return { kind: "number", nonneg: true };
      if (a.kind === "realTensor" && a.nonneg)
        return { kind: "realTensor", nonneg: true };
      return null; // can't JIT sqrt of possibly-negative values
    },
  },
  floor: { arity: 1, resultType: unaryNumberResult(nn => nn) },
  ceil: { arity: 1, resultType: unaryNumberResult(nn => nn) },
  round: { arity: 1, resultType: unaryNumberResult(nn => nn) },
  fix: { arity: 1, resultType: unaryNumberResult(nn => nn) },

  // Log (loses nonneg)
  log: { arity: 1, resultType: unaryNumberResult(() => false) },
  log2: { arity: 1, resultType: unaryNumberResult(() => false) },
  log10: { arity: 1, resultType: unaryNumberResult(() => false) },

  // Sign (loses nonneg)
  sign: { arity: 1, resultType: unaryNumberResult(() => false) },

  // Binary math: scalar-only for now
  atan2: {
    arity: 2,
    resultType: argTypes => {
      if (argTypes[0].kind === "number" && argTypes[1].kind === "number")
        return { kind: "number" };
      return null;
    },
  },
  min: {
    arity: 2,
    resultType: argTypes => {
      if (argTypes[0].kind === "number" && argTypes[1].kind === "number")
        return {
          kind: "number",
          nonneg: !!argTypes[0].nonneg && !!argTypes[1].nonneg,
        };
      return null;
    },
  },
  max: {
    arity: 2,
    resultType: argTypes => {
      if (argTypes[0].kind === "number" && argTypes[1].kind === "number")
        return {
          kind: "number",
          nonneg: !!argTypes[0].nonneg && !!argTypes[1].nonneg,
        };
      return null;
    },
  },
  mod: {
    arity: 2,
    resultType: argTypes => {
      if (argTypes[0].kind === "number" && argTypes[1].kind === "number")
        return { kind: "number", nonneg: !!argTypes[1].nonneg };
      return null;
    },
  },
  rem: {
    arity: 2,
    resultType: argTypes => {
      if (argTypes[0].kind === "number" && argTypes[1].kind === "number")
        return { kind: "number" };
      return null;
    },
  },
  power: {
    arity: 2,
    resultType: argTypes => {
      if (argTypes[0].kind === "number" && argTypes[1].kind === "number")
        return { kind: "number" };
      return null;
    },
  },
};

// ── Cache entry type ────────────────────────────────────────────────────

export interface JitCacheEntry {
  fn: (...args: unknown[]) => unknown;
  source: string;
}
