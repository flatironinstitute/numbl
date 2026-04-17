/**
 * Fusion analysis for the C-JIT codegen.
 *
 * Scans a statement list for runs of tensor element-wise assigns that
 * can be collapsed into a single per-element `for` loop. Each such run
 * is a "fusible chain."
 *
 * A chain breaks on:
 *   - control flow (If/For/While)
 *   - any non-Assign statement
 *   - a tensor assign whose RHS references a tensor that is NOT an input
 *     param and NOT previously assigned within the same chain
 *   - a scalar assign (left for the per-op emitter)
 *
 * An optional **trailing reduction** is absorbed when the statement
 * immediately after a tensor chain is of the form
 *   `acc = acc + reduce(lastChainVar)`   or
 *   `acc = reduce(lastChainVar)`
 * where `reduce` is sum/prod/max/min/mean/any/all. Absorbing the
 * reduction lets the fused loop emit an inline accumulator instead of
 * materialising the intermediate buffer.
 */

import type { JitExpr, JitStmt } from "../jitTypes.js";
import { BinaryOperation } from "../../../parser/types.js";
import { C_TENSOR_UNARY_OPS, C_TENSOR_REDUCTION_OPS } from "./cFeasibility.js";

// ── Public types ──────────────────────────────────────────���──────────

/** One tensor assign inside a fusible chain. */
export interface FusedAssign {
  /** Destination tensor variable name. */
  destName: string;
  /** RHS expression tree (all tensor ops are element-wise). */
  expr: JitExpr;
}

/** A trailing reduction absorbed into the fused loop. */
export interface FusedReduction {
  /** Scalar accumulator variable name (e.g. `chain_acc`). */
  accName: string;
  /** Reduction builtin name (e.g. `sum`). */
  reduceName: string;
  /** The tensor variable being reduced (last chain dest). */
  tensorName: string;
  /**
   * When true, the scalar statement is `acc = acc OP reduce(tensor)`,
   * and `accOp` says which binary op combines the old accumulator with
   * the reduction result. When false, it's a plain `acc = reduce(tensor)`.
   */
  hasAccumulate: boolean;
  accOp?: BinaryOperation;
}

/** Describes one fusible chain found in a statement list. */
export interface FusibleChain {
  /** Index of the first statement in the chain (within the parent list). */
  startIdx: number;
  /** Number of statements consumed (tensor assigns + optional reduction). */
  length: number;
  /** The tensor assigns to fuse. */
  assigns: FusedAssign[];
  /** Optional trailing reduction. */
  reduction?: FusedReduction;
}

// ── Analysis ─────────────────────────────────────────────────────────

/**
 * Returns true when the expression tree is a pure tensor element-wise
 * computation: every tensor reference is either a known variable from
 * `knownTensors` or an input param from `paramTensors`, and every op
 * is element-wise (Binary +−×÷ / comparisons, Unary ±, tensor Call).
 */
function isPureElementwise(
  expr: JitExpr,
  paramTensors: ReadonlySet<string>,
  knownTensors: ReadonlySet<string>
): boolean {
  switch (expr.tag) {
    case "NumberLiteral":
      return true;

    case "Var":
      if (expr.jitType.kind === "tensor") {
        return paramTensors.has(expr.name) || knownTensors.has(expr.name);
      }
      // scalar var — ok
      return true;

    case "Binary": {
      // All binary ops that feasibility already accepted are fine for
      // fusion as long as both children are pure.
      return (
        isPureElementwise(expr.left, paramTensors, knownTensors) &&
        isPureElementwise(expr.right, paramTensors, knownTensors)
      );
    }

    case "Unary":
      return isPureElementwise(expr.operand, paramTensors, knownTensors);

    case "Call": {
      // Tensor unary calls (exp, sin, etc.)
      if (expr.jitType.kind === "tensor" && expr.name in C_TENSOR_UNARY_OPS) {
        return expr.args.every(a =>
          isPureElementwise(a, paramTensors, knownTensors)
        );
      }
      // Scalar math calls (sin of a scalar, etc.)
      if (expr.jitType.kind !== "tensor") {
        return expr.args.every(a =>
          isPureElementwise(a, paramTensors, knownTensors)
        );
      }
      return false;
    }

    default:
      return false;
  }
}

/**
 * Attempt to recognise a trailing reduction that can be absorbed into
 * the fused loop. Patterns:
 *   `acc = acc + sum(tensorVar)`
 *   `acc = sum(tensorVar)`
 *
 * `tensorVar` must be the last tensor assigned in the chain.
 */
function tryMatchReduction(
  stmt: JitStmt,
  lastChainDest: string
): FusedReduction | null {
  if (stmt.tag !== "Assign") return null;
  const expr = stmt.expr;

  // Pattern 1: acc = reduce(tensor)
  if (
    expr.tag === "Call" &&
    expr.name in C_TENSOR_REDUCTION_OPS &&
    expr.args.length === 1 &&
    expr.args[0].tag === "Var" &&
    expr.args[0].name === lastChainDest
  ) {
    return {
      accName: stmt.name,
      reduceName: expr.name,
      tensorName: lastChainDest,
      hasAccumulate: false,
    };
  }

  // Pattern 2: acc = acc OP reduce(tensor)
  if (expr.tag === "Binary") {
    // Check left = Var(acc), right = Call(reduce, tensor)
    if (
      expr.left.tag === "Var" &&
      expr.left.name === stmt.name &&
      expr.right.tag === "Call" &&
      expr.right.name in C_TENSOR_REDUCTION_OPS &&
      expr.right.args.length === 1 &&
      expr.right.args[0].tag === "Var" &&
      expr.right.args[0].name === lastChainDest
    ) {
      return {
        accName: stmt.name,
        reduceName: expr.right.name,
        tensorName: lastChainDest,
        hasAccumulate: true,
        accOp: expr.op,
      };
    }
    // Check right = Var(acc), left = Call(reduce, tensor)
    if (
      expr.right.tag === "Var" &&
      expr.right.name === stmt.name &&
      expr.left.tag === "Call" &&
      expr.left.name in C_TENSOR_REDUCTION_OPS &&
      expr.left.args.length === 1 &&
      expr.left.args[0].tag === "Var" &&
      expr.left.args[0].name === lastChainDest
    ) {
      return {
        accName: stmt.name,
        reduceName: expr.left.name,
        tensorName: lastChainDest,
        hasAccumulate: true,
        accOp: expr.op,
      };
    }
  }

  return null;
}

/**
 * Scan a statement list and return all fusible chains.
 *
 * `paramTensors` is the set of tensor parameter names (input data that
 * will be read via `data[i]` in the fused loop).
 * `allTensorVars` is the full set of tensor-typed variables (params +
 * locals + outputs).
 */
export function findFusibleChains(
  stmts: JitStmt[],
  paramTensors: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): FusibleChain[] {
  const chains: FusibleChain[] = [];
  let i = 0;

  while (i < stmts.length) {
    const chainAssigns: FusedAssign[] = [];
    const chainStart = i;
    // Tensor vars produced so far in this chain.
    const produced = new Set<string>();

    while (i < stmts.length) {
      const s = stmts[i];
      // Skip SetLoc (line-number tracking) — it's a no-op.
      if (s.tag === "SetLoc") {
        i++;
        continue;
      }
      if (s.tag !== "Assign") break;
      if (!allTensorVars.has(s.name)) break;
      if (s.expr.jitType.kind !== "tensor") break;
      if (!isPureElementwise(s.expr, paramTensors, produced)) break;

      chainAssigns.push({ destName: s.name, expr: s.expr });
      produced.add(s.name);
      i++;
    }

    if (chainAssigns.length >= 2) {
      // Check for trailing reduction (skip any SetLoc first).
      while (i < stmts.length && stmts[i].tag === "SetLoc") i++;
      let reduction: FusedReduction | undefined;
      const lastDest = chainAssigns[chainAssigns.length - 1].destName;
      if (i < stmts.length) {
        const r = tryMatchReduction(stmts[i], lastDest);
        if (r) {
          reduction = r;
          i++;
        }
      }

      chains.push({
        startIdx: chainStart,
        length: i - chainStart,
        assigns: chainAssigns,
        reduction,
      });
    } else {
      // Not enough to fuse — skip one statement.
      if (chainAssigns.length === 0) i++;
      // If chainAssigns.length === 1, `i` already advanced past it;
      // it will be emitted by the per-op path.
    }
  }

  return chains;
}
