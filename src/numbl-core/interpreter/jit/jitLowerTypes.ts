/**
 * Type-level helpers for the JIT lowering pass.
 *
 * Pure functions for:
 * - Sign algebra (how signs combine through arithmetic)
 * - Binary/unary operation result type inference
 * - Type environment management (clone, merge, equality)
 * - Known MATLAB constants
 *
 * These have no dependency on LowerCtx or the lowering state.
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import {
  type JitType,
  type SignCategory,
  unifyJitTypes,
  isScalarType,
  isNumericScalarType,
  isTensorType,
  isComplexType,
  isArithmeticType,
  flipSign,
} from "./jitTypes.js";

// ── Known constants ────────────────────────────────────────────────────

export const KNOWN_CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  inf: Infinity,
  Inf: Infinity,
  nan: NaN,
  NaN: NaN,
  eps: 2.220446049250313e-16,
  true: 1,
  false: 0,
};

// ── Type Environment ───────────────────────────────────────────────────

export type TypeEnv = Map<string, JitType>;

export function cloneEnv(env: TypeEnv): TypeEnv {
  return new Map(env);
}

/** Merge two type environments at a join point. Returns null if any type becomes unknown. */
export function mergeEnvs(a: TypeEnv, b: TypeEnv): TypeEnv | null {
  const result = cloneEnv(a);
  for (const [name, typeB] of b) {
    const typeA = result.get(name);
    if (typeA) {
      const unified = unifyJitTypes(typeA, typeB);
      if (unified.kind === "unknown") return null;
      result.set(name, unified);
    } else {
      result.set(name, typeB);
    }
  }
  return result;
}

/** Check if two type environments are identical (by JSON comparison). */
export function envsEqual(a: TypeEnv, b: TypeEnv): boolean {
  if (a.size !== b.size) return false;
  for (const [name, type] of a) {
    const other = b.get(name);
    if (!other || JSON.stringify(type) !== JSON.stringify(other)) return false;
  }
  return true;
}

// ── Sign algebra ───────────────────────────────────────────────────────

export function addSigns(
  a: SignCategory,
  b: SignCategory
): SignCategory | undefined {
  if (a === "positive" && b === "positive") return "positive";
  if (a === "negative" && b === "negative") return "negative";
  if (
    (a === "nonneg" || a === "positive") &&
    (b === "nonneg" || b === "positive")
  )
    return "nonneg";
  if (
    (a === "nonpositive" || a === "negative") &&
    (b === "nonpositive" || b === "negative")
  )
    return "nonpositive";
  return undefined;
}

export function mulSigns(
  a: SignCategory,
  b: SignCategory
): SignCategory | undefined {
  const aPos = a === "positive" || a === "nonneg";
  const aNeg = a === "negative" || a === "nonpositive";
  const bPos = b === "positive" || b === "nonneg";
  const bNeg = b === "negative" || b === "nonpositive";
  const aStrict = a === "positive" || a === "negative";
  const bStrict = b === "positive" || b === "negative";

  if ((aPos && bPos) || (aNeg && bNeg)) {
    return aStrict && bStrict ? "positive" : "nonneg";
  }
  if ((aPos && bNeg) || (aNeg && bPos)) {
    return aStrict && bStrict ? "negative" : "nonpositive";
  }
  return undefined;
}

export function combineSigns(
  a: SignCategory | undefined,
  b: SignCategory | undefined,
  op: BinaryOperation
): SignCategory | undefined {
  if (!a || !b) return undefined;
  switch (op) {
    case BinaryOperation.Add:
      return addSigns(a, b);
    case BinaryOperation.Sub:
      return addSigns(a, flipSign(b)!);
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return mulSigns(a, b);
    default:
      return undefined;
  }
}

// ── Binary operation result type ───────────────────────────────────────

export function binaryResultType(
  op: BinaryOperation,
  left: JitType,
  right: JitType
): JitType | null {
  if (!isArithmeticType(left) || !isArithmeticType(right)) return null;

  // Comparisons
  if (
    op === BinaryOperation.Equal ||
    op === BinaryOperation.NotEqual ||
    op === BinaryOperation.Less ||
    op === BinaryOperation.LessEqual ||
    op === BinaryOperation.Greater ||
    op === BinaryOperation.GreaterEqual
  ) {
    if (isScalarType(left) && isScalarType(right)) return { kind: "boolean" };
    const anyTensor = isTensorType(left) || isTensorType(right);
    const anyComplex = isComplexType(left) || isComplexType(right);
    if (anyTensor && !anyComplex) {
      const lt = isTensorType(left)
        ? (left as Extract<JitType, { kind: "tensor" }>)
        : undefined;
      const rt = isTensorType(right)
        ? (right as Extract<JitType, { kind: "tensor" }>)
        : undefined;
      const shape = lt?.shape ?? rt?.shape;
      const ndim = shape ? undefined : (lt?.ndim ?? rt?.ndim);
      return {
        kind: "tensor",
        isComplex: false,
        isLogical: true,
        ...(shape ? { shape } : {}),
        ...(ndim !== undefined ? { ndim } : {}),
      };
    }
    return null;
  }

  // Logical operators: scalar only
  if (op === BinaryOperation.AndAnd || op === BinaryOperation.OrOr) {
    if (isScalarType(left) && isScalarType(right)) return { kind: "boolean" };
    return null;
  }

  // Element-wise arithmetic ops only
  switch (op) {
    case BinaryOperation.Add:
    case BinaryOperation.Sub:
    case BinaryOperation.ElemMul:
    case BinaryOperation.ElemDiv:
      break;
    case BinaryOperation.Mul:
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    case BinaryOperation.Div:
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    case BinaryOperation.ElemPow:
      break;
    case BinaryOperation.Pow:
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    default:
      return null;
  }

  // Coerce logical to number for arithmetic
  const effLeft: JitType =
    left.kind === "boolean"
      ? { kind: "number", sign: "nonneg", isInteger: true }
      : left;
  const effRight: JitType =
    right.kind === "boolean"
      ? { kind: "number", sign: "nonneg", isInteger: true }
      : right;

  const anyComplex = isComplexType(effLeft) || isComplexType(effRight);
  const anyTensor = isTensorType(effLeft) || isTensorType(effRight);

  if (anyTensor) {
    const lt = isTensorType(effLeft)
      ? (effLeft as Extract<JitType, { kind: "tensor" }>)
      : undefined;
    const rt = isTensorType(effRight)
      ? (effRight as Extract<JitType, { kind: "tensor" }>)
      : undefined;
    const shape = lt?.shape ?? rt?.shape;
    const ndim = shape ? undefined : (lt?.ndim ?? rt?.ndim);
    const isComplex =
      anyComplex || (lt?.isComplex ?? false) || (rt?.isComplex ?? false);
    return {
      kind: "tensor",
      isComplex,
      ...(shape ? { shape } : {}),
      ...(ndim !== undefined ? { ndim } : {}),
    };
  }

  if (anyComplex) {
    if (op === BinaryOperation.Pow || op === BinaryOperation.ElemPow)
      return null;
    return { kind: "complex_or_number" };
  }

  if (effLeft.kind === "number" && effRight.kind === "number") {
    const sign = combineSigns(effLeft.sign, effRight.sign, op);
    // int ± int = int, int * int = int; division/power lose integer guarantee
    const isInteger =
      effLeft.isInteger &&
      effRight.isInteger &&
      (op === BinaryOperation.Add ||
        op === BinaryOperation.Sub ||
        op === BinaryOperation.Mul ||
        op === BinaryOperation.ElemMul);
    return {
      kind: "number",
      ...(sign ? { sign } : {}),
      ...(isInteger ? { isInteger: true } : {}),
    };
  }

  return null;
}

// ── Unary operation result type ────────────────────────────────────────

export function unaryResultType(
  op: UnaryOperation,
  operand: JitType
): JitType | null {
  switch (op) {
    case UnaryOperation.Plus:
      return operand;
    case UnaryOperation.Minus:
      if (operand.kind === "number") {
        const sign = flipSign(operand.sign);
        return {
          kind: "number",
          ...(sign ? { sign } : {}),
          ...(operand.isInteger ? { isInteger: true } : {}),
        };
      }
      if (operand.kind === "boolean")
        return { kind: "number", sign: "nonpositive" };
      if (operand.kind === "complex_or_number")
        return { kind: "complex_or_number" };
      if (operand.kind === "tensor")
        return {
          kind: "tensor",
          isComplex: operand.isComplex,
          shape: operand.shape,
          ndim: operand.ndim,
        };
      return null;
    case UnaryOperation.Not:
      if (isNumericScalarType(operand)) return { kind: "boolean" };
      return null;
    default:
      return null; // Transpose not supported
  }
}
