/**
 * Utility functions over IR nodes.
 */

import { BinaryOperation } from "../parser/index.js";
import { getConstantType } from "../builtins";
import { ItemType, isScalarType, isComplexType } from "./itemTypes.js";
import { IRExpr, IRExprKind } from "./nodes.js";

// Re-export for consumers that import from nodeUtils
export { isScalarType, isComplexType } from "./itemTypes.js";

/** Visit every expression in the tree (pre-order), calling fn for each. */
export function walkExpr(expr: IRExpr, fn: (e: IRExpr) => void): void {
  fn(expr);
  const k = expr.kind;
  switch (k.type) {
    case "Unary":
      walkExpr(k.operand, fn);
      break;
    case "Binary":
      walkExpr(k.left, fn);
      walkExpr(k.right, fn);
      break;
    case "Tensor":
    case "Cell":
      for (const row of k.rows) for (const e of row) walkExpr(e, fn);
      break;
    case "Range":
      walkExpr(k.start, fn);
      if (k.step) walkExpr(k.step, fn);
      walkExpr(k.end, fn);
      break;
    case "Member":
      walkExpr(k.base, fn);
      break;
    case "MemberDynamic":
      walkExpr(k.base, fn);
      walkExpr(k.nameExpr, fn);
      break;
    case "Index":
    case "IndexCell":
      walkExpr(k.base, fn);
      for (const idx of k.indices) walkExpr(idx, fn);
      break;
    case "MethodCall":
      walkExpr(k.base, fn);
      for (const a of k.args) walkExpr(a, fn);
      break;
    case "SuperConstructorCall":
    case "FuncCall":
      if (k.type === "FuncCall" && k.instanceBase) walkExpr(k.instanceBase, fn);
      for (const a of k.args) walkExpr(a, fn);
      break;
    case "ClassInstantiation":
      for (const a of k.args) walkExpr(a, fn);
      break;
    case "AnonFunc":
      walkExpr(k.body, fn);
      break;
  }
}

// Cache for computed expression types. Keyed on the IRExprKind object identity.
// Var and SuperConstructorCall are excluded because they depend on mutable
// variable.ty which can change during lowering via type unification.
const _typeCache = new WeakMap<object, ItemType>();

export const itemTypeForExprKind = (kind: IRExprKind): ItemType => {
  // Var and SuperConstructorCall depend on mutable variable.ty — never cache.
  if (kind.type === "Var") return kind.variable.ty || { kind: "Unknown" };
  if (kind.type === "SuperConstructorCall")
    return kind.objVar.ty || { kind: "Unknown" };

  const cached = _typeCache.get(kind);
  if (cached) return cached;

  const result = _computeItemType(kind);
  _typeCache.set(kind, result);
  return result;
};

function _computeItemType(
  kind: Exclude<IRExprKind, { type: "Var" } | { type: "SuperConstructorCall" }>
): ItemType {
  switch (kind.type) {
    case "Number": {
      return { kind: "Number" };
    }
    case "Char":
      return { kind: "Char" };
    case "String":
      return { kind: "String" };
    case "Constant":
      return getConstantType(kind.name) ?? { kind: "Unknown" };
    case "Unary": {
      const operandType = itemTypeForExprKind(kind.operand.kind);
      if (kind.op === "Not") {
        if (isScalarType(operandType)) return { kind: "Boolean" };
        if (operandType.kind === "Tensor") return operandType;
        return { kind: "Unknown" };
      }
      // Plus, Minus (without value), Transpose, NonConjugateTranspose: preserve operand type
      return operandType;
    }
    case "Binary": {
      const leftType = itemTypeForExprKind(kind.left.kind);
      const rightType = itemTypeForExprKind(kind.right.kind);
      // Comparison operators → Logical for scalars, Tensor for tensors
      if (
        kind.op === BinaryOperation.Equal ||
        kind.op === BinaryOperation.NotEqual ||
        kind.op === BinaryOperation.Less ||
        kind.op === BinaryOperation.LessEqual ||
        kind.op === BinaryOperation.Greater ||
        kind.op === BinaryOperation.GreaterEqual ||
        kind.op === BinaryOperation.BitAnd ||
        kind.op === BinaryOperation.BitOr
      ) {
        if (isScalarType(leftType) && isScalarType(rightType))
          return { kind: "Boolean" };
        if (leftType.kind === "Tensor" || rightType.kind === "Tensor") {
          // Comparison and logical operators always produce real logical tensors,
          // even when the operands are complex.
          return { kind: "Tensor", isLogical: true };
        }
        return { kind: "Unknown" };
      }
      // Short-circuit logical → always Logical
      if (kind.op === "AndAnd" || kind.op === "OrOr")
        return { kind: "Boolean" };

      // If either operand is Unknown or a ClassInstance, the result is
      // Unknown — the class may overload the operator and return anything.
      if (
        leftType.kind === "Unknown" ||
        rightType.kind === "Unknown" ||
        leftType.kind === "ClassInstance" ||
        rightType.kind === "ClassInstance"
      ) {
        return { kind: "Unknown" };
      }

      // Arithmetic with at least one Tensor operand → Tensor result.
      // Arithmetic on logical tensors produces numeric (non-logical) results,
      // so we never preserve isLogical here (unlike comparison operators above).
      if (leftType.kind === "Tensor" || rightType.kind === "Tensor") {
        const resultIsComplex =
          isComplexType(leftType) || isComplexType(rightType)
            ? true
            : undefined;
        return { kind: "Tensor", isComplex: resultIsComplex };
      }

      // Scalar arithmetic
      // Complex op Complex → Complex (or Num op Complex, etc.)
      if (
        leftType.kind === "ComplexNumber" ||
        rightType.kind === "ComplexNumber"
      ) {
        return { kind: "ComplexNumber" };
      }
      // Num op Num → Num
      if (leftType.kind === "Number" && rightType.kind === "Number")
        return { kind: "Number" };

      return { kind: "Unknown" };
    }
    case "Tensor": {
      // Check if any element is complex or a class instance / unknown.
      // When any element could be a class instance, the horzcat/vertcat
      // at runtime may dispatch to a class method and return a class
      // instance instead of a numeric tensor, so we must return Unknown.
      let hasComplexElement = false;
      let hasClassOrUnknown = false;
      for (const row of kind.rows) {
        for (const elem of row) {
          const elemType = itemTypeForExprKind(elem.kind);
          if (
            elemType.kind === "ClassInstance" ||
            elemType.kind === "Unknown"
          ) {
            hasClassOrUnknown = true;
            break;
          }
          if (
            elemType.kind === "ComplexNumber" ||
            (elemType.kind === "Tensor" && elemType.isComplex)
          ) {
            hasComplexElement = true;
          }
        }
        if (hasClassOrUnknown) break;
      }
      if (hasClassOrUnknown) return { kind: "Unknown" };
      return {
        kind: "Tensor",
        isComplex: hasComplexElement || undefined,
      };
    }
    case "Cell":
      // TODO
      return { kind: "Cell", elementType: "unknown", length: "unknown" };
    case "Index": {
      const baseType = itemTypeForExprKind(kind.base.kind);
      if (baseType.kind === "Tensor" || baseType.kind === "Number") {
        // All scalar indices → element access, otherwise → Tensor slice
        const allScalar = kind.indices.every(idx => {
          const t = itemTypeForExprKind(idx.kind);
          return t.kind === "Number";
        });
        if (allScalar) {
          // Scalar indexing: if base is complex tensor, result could be complex
          if (baseType.kind === "Tensor" && baseType.isComplex) {
            return { kind: "ComplexNumber" };
          }
          return { kind: "Number" };
        }
        // Slice indexing: preserve isComplex flag
        const isComplex =
          baseType.kind === "Tensor" ? baseType.isComplex : undefined;
        return { kind: "Tensor", isComplex };
      }
      if (baseType.kind === "String") return { kind: "String" };
      if (baseType.kind === "Cell") return baseType;
      if (baseType.kind === "Function") return baseType.returns;
      return { kind: "Unknown" };
    }
    case "IndexCell": {
      const cellType = itemTypeForExprKind(kind.base.kind);
      if (cellType.kind === "Cell" && cellType.elementType !== "unknown") {
        return cellType.elementType;
      }
      return { kind: "Unknown" };
    }
    case "Range":
      return { kind: "Tensor" };
    case "Colon":
      return { kind: "Tensor" };
    case "End":
      return { kind: "Number" };
    case "Member": {
      const baseType = itemTypeForExprKind(kind.base.kind);
      if (baseType.kind === "Struct" && kind.name in baseType.knownFields) {
        return baseType.knownFields[kind.name];
      }
      return { kind: "Unknown" };
    }
    case "MemberDynamic":
      return { kind: "Unknown" };
    case "MethodCall":
      return kind.returnType;
    case "FuncCall": {
      return kind.returnType;
    }
    case "AnonFunc": {
      const paramTypes = kind.params.map(
        p => p.ty ?? ({ kind: "Unknown" } as ItemType)
      );
      const returnType = itemTypeForExprKind(kind.body.kind);
      return { kind: "Function", params: paramTypes, returns: returnType };
    }
    case "FuncHandle":
      return { kind: "Function", params: [], returns: { kind: "Unknown" } };
    case "MetaClass":
      // TODO
      return { kind: "Unknown" };
    case "ClassInstantiation":
      return { kind: "ClassInstance", className: kind.className };
    case "RuntimeError":
      return { kind: "Unknown" };
  }
}
