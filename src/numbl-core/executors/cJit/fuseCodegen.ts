/**
 * c-jit-fuse codegen — single-pass element-wise C kernel.
 *
 * Emits one C function whose body is a single `for (long i = 0; i < n; i++)`
 * loop with `out[i] = <fused expression>;`. The fused expression is
 * the AST RHS, with each `Ident` mapped to either `tN[i]` (tensor)
 * or `sN` (scalar) and each builtin call mapped to its math.h
 * counterpart.
 *
 * ABI:
 *   void <fnName>(double *out, long n,
 *                 const double *t0, const double *t1, ...,
 *                 double s0, double s1, ...);
 */

import {
  BinaryOperation,
  type Expr,
  UnaryOperation,
} from "../../parser/types.js";
import type { FuseClassification } from "./fuseAnalyze.js";

/** Emit a complete C source file for a fuse classification. */
export function generateFuseCSource(
  fnName: string,
  cls: FuseClassification
): string {
  const tensorIndex = new Map<string, number>();
  cls.tensorInputs.forEach((n, i) => tensorIndex.set(n, i));
  const scalarIndex = new Map<string, number>();
  cls.scalarInputs.forEach((n, i) => scalarIndex.set(n, i));

  const params: string[] = [];
  // Inputs are NOT restrict-qualified: two distinct names can refer
  // to the same underlying Float64Array (MATLAB copy-on-write makes
  // `b = a` share buffers), so we must allow input aliasing. `out`
  // IS restrict-qualified: the executor always allocates a fresh
  // Float64Array, so no input pointer can alias it. That alone is
  // usually enough to unlock vectorization.
  for (let i = 0; i < cls.tensorInputs.length; i++) {
    params.push(`const double *t${i}`);
  }
  for (let i = 0; i < cls.scalarInputs.length; i++) {
    params.push(`double s${i}`);
  }

  const body = emitExpr(cls.rhs, tensorIndex, scalarIndex);

  const lines: string[] = [];
  lines.push(`#include <math.h>`);
  lines.push(``);
  lines.push(
    `void ${fnName}(double *restrict out, long n${params.length > 0 ? ", " + params.join(", ") : ""}) {`
  );
  // `#pragma omp simd` asks the compiler to vectorize the loop even
  // when it would otherwise be conservative (e.g., due to function-
  // call boundaries on math.h calls). Combined with `restrict out`,
  // GCC/Clang will emit straight-line SIMD code on -march=native.
  lines.push(`  #pragma omp simd`);
  lines.push(`  for (long i = 0; i < n; i++) {`);
  lines.push(`    out[i] = ${body};`);
  lines.push(`  }`);
  lines.push(`}`);
  return lines.join("\n") + "\n";
}

function emitExpr(
  e: Expr,
  tensorIndex: ReadonlyMap<string, number>,
  scalarIndex: ReadonlyMap<string, number>
): string {
  switch (e.type) {
    case "Number":
      return formatDouble(parseFloat(e.value));
    case "Ident": {
      const ti = tensorIndex.get(e.name);
      if (ti !== undefined) return `t${ti}[i]`;
      const si = scalarIndex.get(e.name);
      if (si !== undefined) return `s${si}`;
      throw new Error(`fuse codegen: unknown identifier ${e.name}`);
    }
    case "Binary":
      return emitBinary(e, tensorIndex, scalarIndex);
    case "Unary": {
      const x = emitExpr(e.operand, tensorIndex, scalarIndex);
      switch (e.op) {
        case UnaryOperation.Plus:
          return `(+${x})`;
        case UnaryOperation.Minus:
          return `(-${x})`;
        default:
          throw new Error(`fuse codegen: unsupported unary op ${e.op}`);
      }
    }
    case "FuncCall": {
      const name = e.name === "abs" ? "fabs" : e.name;
      const arg = emitExpr(e.args[0], tensorIndex, scalarIndex);
      return `${name}(${arg})`;
    }
    default:
      throw new Error(
        `fuse codegen: unsupported expr ${(e as { type: string }).type}`
      );
  }
}

function emitBinary(
  e: Expr & { type: "Binary" },
  tensorIndex: ReadonlyMap<string, number>,
  scalarIndex: ReadonlyMap<string, number>
): string {
  const l = emitExpr(e.left, tensorIndex, scalarIndex);
  const r = emitExpr(e.right, tensorIndex, scalarIndex);
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
      throw new Error(`fuse codegen: unsupported binary op ${e.op}`);
  }
}

function formatDouble(v: number): string {
  if (Number.isNaN(v)) return "((double)NAN)";
  if (v === Infinity) return "((double)INFINITY)";
  if (v === -Infinity) return "(-(double)INFINITY)";
  if (Number.isInteger(v)) return `${v}.0`;
  const s = String(v);
  if (/[.eE]/.test(s)) return s;
  return `${s}.0`;
}
