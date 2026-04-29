/**
 * Shared codegen helpers for the element-wise C-JIT executors
 * (`c-jit-fuse`, `c-jit-chain`).
 *
 * Both executors lower the same AST subset (the one accepted by
 * `elemwiseStructural.isElemwiseStructuralExpr`) to C; they only
 * differ in how an `Ident` leaf maps to a C expression (tensor pointer
 * indexing vs. local-name vs. scalar parameter). This module owns the
 * shared structural emitter and a few utility helpers; the per-executor
 * codegen is reduced to a thin wrapper that supplies the leaf mapping.
 */

import {
  BinaryOperation,
  type Expr,
  UnaryOperation,
} from "../../parser/types.js";
import { cBuiltinName } from "./builtins.js";

/** Emit a C `double` literal that's unambiguous to the compiler.
 *  Handles NaN, ±Inf, integer-valued doubles. */
export function formatDouble(v: number): string {
  if (Number.isNaN(v)) return "((double)NAN)";
  if (v === Infinity) return "((double)INFINITY)";
  if (v === -Infinity) return "(-(double)INFINITY)";
  if (Number.isInteger(v)) return `${v}.0`;
  const s = String(v);
  if (/[.eE]/.test(s)) return s;
  return `${s}.0`;
}

/** Walk `e` and add every `Ident` name to `out`. Whitelisted to the
 *  element-wise AST subset — anything outside (e.g. `Index`) is a
 *  programming error (the structural classifier should have rejected). */
export function collectIdents(e: Expr, out: Set<string>): void {
  switch (e.type) {
    case "Ident":
      out.add(e.name);
      return;
    case "Number":
      return;
    case "Binary":
      collectIdents(e.left, out);
      collectIdents(e.right, out);
      return;
    case "Unary":
      collectIdents(e.operand, out);
      return;
    case "FuncCall":
      for (const a of e.args) collectIdents(a, out);
      return;
  }
}

/** Lower an element-wise AST expression to a C string. The caller
 *  supplies `identToC` to map each `Ident` name to its in-loop C form
 *  (e.g. `t0[i]`, `in_x[i]`, `s2`, or a bare local name). */
export function emitElemwiseExpr(
  e: Expr,
  identToC: (name: string) => string
): string {
  switch (e.type) {
    case "Number":
      return formatDouble(parseFloat(e.value));
    case "Ident":
      return identToC(e.name);
    case "Binary":
      return emitBinary(e, identToC);
    case "Unary": {
      const x = emitElemwiseExpr(e.operand, identToC);
      switch (e.op) {
        case UnaryOperation.Plus:
          return `(+${x})`;
        case UnaryOperation.Minus:
          return `(-${x})`;
        default:
          throw new Error(`elemwise codegen: unsupported unary op ${e.op}`);
      }
    }
    case "FuncCall": {
      const name = cBuiltinName(e.name);
      const arg = emitElemwiseExpr(e.args[0], identToC);
      return `${name}(${arg})`;
    }
    default:
      throw new Error(
        `elemwise codegen: unsupported expr ${(e as { type: string }).type}`
      );
  }
}

function emitBinary(
  e: Expr & { type: "Binary" },
  identToC: (name: string) => string
): string {
  const l = emitElemwiseExpr(e.left, identToC);
  const r = emitElemwiseExpr(e.right, identToC);
  switch (e.op) {
    case BinaryOperation.Add:
      return `(${l} + ${r})`;
    case BinaryOperation.Sub:
      return `(${l} - ${r})`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `(${l} * ${r})`;
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return `(${l} / ${r})`;
    default:
      throw new Error(`elemwise codegen: unsupported binary op ${e.op}`);
  }
}
