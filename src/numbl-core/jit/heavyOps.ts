/**
 * Shared heavy-op heuristic for fused-loop emitters (e1 and e2).
 *
 * Counts the number of "expensive" math operations in a JitExpr — the
 * kind of work that's heavy enough per element that OpenMP thread-
 * spawn overhead pays off at N >= 100k. Arithmetic-only chains skip
 * the parallel-for pragma because threads slow them down: the body
 * becomes memory-bandwidth-bound and adding threads only adds overhead.
 */

import type { JitExpr } from "./jitTypes.js";
import { BinaryOperation } from "../parser/types.js";

const HEAVY_OPS = new Set([
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "acosh",
  "atanh",
  "exp",
  "expm1",
  "log",
  "log1p",
  "log2",
  "log10",
  "pow",
  "atan2",
  "hypot",
]);

export function countHeavyOps(expr: JitExpr): number {
  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
    case "ImagLiteral":
      return 0;
    case "Binary":
      return (
        (expr.op === BinaryOperation.Pow || expr.op === BinaryOperation.ElemPow
          ? 1
          : 0) +
        countHeavyOps(expr.left) +
        countHeavyOps(expr.right)
      );
    case "Unary":
      return countHeavyOps(expr.operand);
    case "Call":
      return (
        (HEAVY_OPS.has(expr.name) ? 1 : 0) +
        expr.args.reduce((n, a) => n + countHeavyOps(a), 0)
      );
    default:
      return 0;
  }
}

/** Minimum element count before `#pragma omp parallel for simd` kicks
 *  in. Below this the thread-spawn cost dominates the work.
 *  Overridable via `NUMBL_OMP_THRESHOLD` for benchmarks. */
export function ompParallelThreshold(): number {
  return parseInt(process.env.NUMBL_OMP_THRESHOLD || "", 10) || 100_000;
}
