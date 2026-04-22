/**
 * Fusion analysis for JIT backends (shared by JS-JIT and C-JIT).
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

import type { JitExpr, JitStmt } from "./jitTypes.js";
import { BinaryOperation } from "../parser/types.js";
import {
  FUSIBLE_TENSOR_UNARY_OPS,
  FUSIBLE_TENSOR_BINARY_OPS,
  FUSIBLE_TENSOR_REDUCTION_OPS,
} from "./fusionOps.js";

// ── Public types ─────────────────────────────────────────────────────

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

/**
 * Return true when two tensor shapes are statically known to produce
 * the same number of elements. Unknown dims (`-1`) are treated as
 * incompatible — a fused chain with an unknown-length assign can't
 * soundly share a single `lenVar` across every op. A missing shape
 * (`undefined`) is also incompatible.
 */
function fusibleShapesCompatible(
  a: number[] | undefined,
  b: number[] | undefined
): boolean {
  if (!a || !b) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === -1 || b[i] === -1) return false;
    if (a[i] !== b[i]) return false;
  }
  return true;
}

// ── Analysis ─────────────────────────────────────────────────────────

/**
 * Count the number of tensor-level operations in an expression tree.
 * A tensor op is a Binary/Unary node whose result is a tensor, or a
 * Call node that is a tensor unary builtin. This lets us decide whether
 * a single-assign chain is worth fusing: with 2+ tensor ops the fused
 * loop eliminates at least one intermediate buffer allocation.
 */
function countTensorOps(
  expr: JitExpr,
  allowedUnaryOps: ReadonlySet<string>
): number {
  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
      return 0;
    case "Binary": {
      const childOps =
        countTensorOps(expr.left, allowedUnaryOps) +
        countTensorOps(expr.right, allowedUnaryOps);
      return expr.jitType.kind === "tensor" ? childOps + 1 : childOps;
    }
    case "Unary": {
      const childOps = countTensorOps(expr.operand, allowedUnaryOps);
      return expr.jitType.kind === "tensor" ? childOps + 1 : childOps;
    }
    case "Call": {
      const childOps = expr.args.reduce(
        (n, a) => n + countTensorOps(a, allowedUnaryOps),
        0
      );
      if (
        expr.jitType.kind === "tensor" &&
        (allowedUnaryOps.has(expr.name) ||
          FUSIBLE_TENSOR_BINARY_OPS.has(expr.name))
      )
        return childOps + 1;
      return childOps;
    }
    default:
      return 0;
  }
}

/**
 * Returns true when the expression tree is a pure tensor element-wise
 * computation: every tensor reference is either a known variable from
 * `knownTensors` or an input param from `paramTensors`, and every op
 * is element-wise (Binary +−×÷ / comparisons, Unary ±, tensor Call).
 */
function isPureElementwise(
  expr: JitExpr,
  paramTensors: ReadonlySet<string>,
  knownTensors: ReadonlySet<string>,
  allowedUnaryOps: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): boolean {
  switch (expr.tag) {
    case "NumberLiteral":
      return true;

    case "ImagLiteral":
      // Complex-tensor chains route through emitComplexFusedChain, which
      // reads ImagLiteral as the (0.0, 1.0) per-element pair. For a real
      // chain that references `1i`, the chain is rejected at the result-
      // type level (destName goes complex) before we reach here.
      return true;

    case "Var":
      if (expr.jitType.kind === "tensor") {
        // Any tensor Var in scope is acceptable: params carry the caller's
        // buffer; outer locals already assigned before the chain start have
        // a valid buffer too (classify.ts + the emit prelude guarantee
        // every tensor local has a pointer by the time any read happens).
        // Chain-produced intermediates are tracked in `knownTensors`. The
        // tensor's shape / length feeds the loop via tensorLen(). We still
        // intersect with allTensorVars as a safety net against misclassified
        // non-tensor names reaching this branch.
        return (
          paramTensors.has(expr.name) ||
          knownTensors.has(expr.name) ||
          allTensorVars.has(expr.name)
        );
      }
      // scalar var — ok
      return true;

    case "Binary": {
      // Complex-result tensor Binary: only the ops emitComplexPerElem
      // handles (Add/Sub/Mul/ElemMul). Div would need Smith's method with
      // branches, which doesn't vectorize; comparisons / logicals don't
      // apply to complex values. Reject so fusion falls back to per-op.
      if (expr.jitType.kind === "tensor" && expr.jitType.isComplex === true) {
        if (
          expr.op !== BinaryOperation.Add &&
          expr.op !== BinaryOperation.Sub &&
          expr.op !== BinaryOperation.Mul &&
          expr.op !== BinaryOperation.ElemMul
        ) {
          return false;
        }
      }
      return (
        isPureElementwise(
          expr.left,
          paramTensors,
          knownTensors,
          allowedUnaryOps,
          allTensorVars
        ) &&
        isPureElementwise(
          expr.right,
          paramTensors,
          knownTensors,
          allowedUnaryOps,
          allTensorVars
        )
      );
    }

    case "Unary":
      return isPureElementwise(
        expr.operand,
        paramTensors,
        knownTensors,
        allowedUnaryOps,
        allTensorVars
      );

    case "Call": {
      // Complex-tensor calls fusible by emitComplexPerElem: conj/real/imag.
      const COMPLEX_FUSED_UNARY = new Set(["conj", "real", "imag"]);
      if (
        expr.jitType.kind === "tensor" &&
        expr.jitType.isComplex === true &&
        COMPLEX_FUSED_UNARY.has(expr.name) &&
        expr.args.length === 1
      ) {
        return isPureElementwise(
          expr.args[0],
          paramTensors,
          knownTensors,
          allowedUnaryOps,
          allTensorVars
        );
      }
      // Tensor unary calls (exp, sin, etc.) — the real per-element emitter
      // only knows how to read `v_name_data[__i]`, so reject when any arg
      // is a complex tensor (abs(z), sin(z), ...). Those fall back to per-
      // op emission where the complex kernel in numbl_ops handles them.
      if (expr.jitType.kind === "tensor" && allowedUnaryOps.has(expr.name)) {
        const anyComplexArg = expr.args.some(
          a => a.jitType.kind === "tensor" && a.jitType.isComplex === true
        );
        if (anyComplexArg) return false;
        return expr.args.every(a =>
          isPureElementwise(
            a,
            paramTensors,
            knownTensors,
            allowedUnaryOps,
            allTensorVars
          )
        );
      }
      // Tensor binary calls (max, min, mod, rem, atan2, hypot)
      if (
        expr.jitType.kind === "tensor" &&
        FUSIBLE_TENSOR_BINARY_OPS.has(expr.name) &&
        expr.args.length === 2
      ) {
        const anyComplexArg = expr.args.some(
          a => a.jitType.kind === "tensor" && a.jitType.isComplex === true
        );
        if (anyComplexArg) return false;
        return expr.args.every(a =>
          isPureElementwise(
            a,
            paramTensors,
            knownTensors,
            allowedUnaryOps,
            allTensorVars
          )
        );
      }
      // Scalar math calls (sin of a scalar, etc.)
      if (expr.jitType.kind !== "tensor") {
        return expr.args.every(a =>
          isPureElementwise(
            a,
            paramTensors,
            knownTensors,
            allowedUnaryOps,
            allTensorVars
          )
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
    FUSIBLE_TENSOR_REDUCTION_OPS.has(expr.name) &&
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
      FUSIBLE_TENSOR_REDUCTION_OPS.has(expr.right.name) &&
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
      FUSIBLE_TENSOR_REDUCTION_OPS.has(expr.left.name) &&
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

/** Synthetic temp name for inline-reduction fusion (no user variable). */
const INLINE_RED_TMP = "__red_tmp";

/**
 * Attempt to recognise a standalone inline reduction — a scalar assign
 * whose RHS is a reduction of a purely element-wise tensor expression:
 *   `acc = sum(x .* y + 0.5)`
 *   `acc = acc + sum(exp(-x .* x))`
 *
 * Returns a FusibleChain with a synthetic single assign (the expression
 * inside the reduction) plus the reduction itself, so the existing
 * codegen handles it without special-casing.
 */
function tryMatchInlineReduction(
  stmt: JitStmt,
  paramTensors: ReadonlySet<string>,
  allowedUnaryOps: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): FusibleChain | null {
  if (stmt.tag !== "Assign") return null;
  const expr = stmt.expr;

  // Helper: check if a Call node is a reduction of a pure element-wise
  // tensor expression (not just a named variable).
  const tryExtract = (
    call: JitExpr & { tag: "Call" }
  ): { reduceName: string; innerExpr: JitExpr } | null => {
    if (!FUSIBLE_TENSOR_REDUCTION_OPS.has(call.name)) return null;
    if (call.args.length !== 1) return null;
    const arg = call.args[0];
    // Only match when the argument is an expression, not a bare variable
    // (bare-variable reductions are handled by tryMatchReduction on a
    // preceding chain).
    if (arg.tag === "Var") return null;
    if (arg.jitType.kind !== "tensor") return null;
    // Complex reductions can't be absorbed (scalar accumulator is a double,
    // not a ComplexPair). Reject early.
    if (arg.jitType.isComplex === true) return null;
    const empty = new Set<string>();
    if (
      !isPureElementwise(
        arg,
        paramTensors,
        empty,
        allowedUnaryOps,
        allTensorVars
      )
    )
      return null;
    return { reduceName: call.name, innerExpr: arg };
  };

  // Pattern 1: acc = reduce(expr)
  if (expr.tag === "Call") {
    const m = tryExtract(expr as JitExpr & { tag: "Call" });
    if (m) {
      return {
        startIdx: -1, // filled in by caller
        length: 1,
        assigns: [{ destName: INLINE_RED_TMP, expr: m.innerExpr }],
        reduction: {
          accName: stmt.name,
          reduceName: m.reduceName,
          tensorName: INLINE_RED_TMP,
          hasAccumulate: false,
        },
      };
    }
  }

  // Pattern 2: acc = acc OP reduce(expr)  or  acc = reduce(expr) OP acc
  if (expr.tag === "Binary") {
    let call: (JitExpr & { tag: "Call" }) | null = null;
    if (
      expr.left.tag === "Var" &&
      expr.left.name === stmt.name &&
      expr.right.tag === "Call"
    ) {
      call = expr.right as JitExpr & { tag: "Call" };
    } else if (
      expr.right.tag === "Var" &&
      expr.right.name === stmt.name &&
      expr.left.tag === "Call"
    ) {
      call = expr.left as JitExpr & { tag: "Call" };
    }
    if (call) {
      const m = tryExtract(call);
      if (m) {
        return {
          startIdx: -1,
          length: 1,
          assigns: [{ destName: INLINE_RED_TMP, expr: m.innerExpr }],
          reduction: {
            accName: stmt.name,
            reduceName: m.reduceName,
            tensorName: INLINE_RED_TMP,
            hasAccumulate: true,
            accOp: expr.op,
          },
        };
      }
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
 * `allowedUnaryOps` optionally restricts which tensor unary Call names
 * are fusible. Defaults to `FUSIBLE_TENSOR_UNARY_OPS` (full set).
 * The JS backend passes a restricted set that excludes transcendentals
 * (V8 can't vectorize them, so fusing them is slower than per-op calls).
 */
export function findFusibleChains(
  stmts: JitStmt[],
  paramTensors: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  allowedUnaryOps: ReadonlySet<string> = FUSIBLE_TENSOR_UNARY_OPS
): FusibleChain[] {
  const chains: FusibleChain[] = [];
  let i = 0;

  while (i < stmts.length) {
    const chainAssigns: FusedAssign[] = [];
    const chainStart = i;
    // Tensor vars produced so far in this chain.
    const produced = new Set<string>();
    // Static shape the chain's per-element loop iterates over. The fused
    // emitters pick a single `lenVar` shared across every assign, so a
    // chain like `a = x1; b = x2` with mismatched x1/x2 lengths would
    // truncate the longer one. Track the first assign's jitType.shape
    // and break before adding any assign whose shape disagrees.
    let chainShape: number[] | undefined;

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
      if (
        !isPureElementwise(
          s.expr,
          paramTensors,
          produced,
          allowedUnaryOps,
          allTensorVars
        )
      )
        break;
      const thisShape = s.expr.jitType.shape;
      if (
        chainAssigns.length > 0 &&
        !fusibleShapesCompatible(chainShape, thisShape)
      ) {
        break;
      }

      chainAssigns.push({ destName: s.name, expr: s.expr });
      produced.add(s.name);
      if (chainAssigns.length === 1) chainShape = thisShape;
      i++;
    }

    // Determine if this chain is worth fusing.
    // Multi-assign chains (>= 2) always qualify.
    // Single-assign chains qualify when the expression tree has 2+ tensor
    // ops (otherwise the per-op path is fine — no intermediate to save).
    const worthFusing =
      chainAssigns.length >= 2 ||
      (chainAssigns.length === 1 &&
        countTensorOps(chainAssigns[0].expr, allowedUnaryOps) >= 2);

    if (worthFusing) {
      // Check for trailing reduction (skip any SetLoc first). Skip reduction
      // absorption when the chain's final dest is a complex tensor — the
      // fused-loop accumulator is a scalar double, which can't hold a
      // complex-scalar reduction like `sum(complex_tensor)`. Per-op code
      // handles the reduction separately (still emits the fused chain body).
      while (i < stmts.length && stmts[i].tag === "SetLoc") i++;
      let reduction: FusedReduction | undefined;
      const lastAssign = chainAssigns[chainAssigns.length - 1];
      const lastDest = lastAssign.destName;
      const lastIsComplex =
        lastAssign.expr.jitType.kind === "tensor" &&
        lastAssign.expr.jitType.isComplex === true;
      if (i < stmts.length && !lastIsComplex) {
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
      // Not a tensor chain — check for standalone inline reduction.
      // This handles `s = sum(x .* y + 0.5)` and
      // `s = s + sum(exp(-x .* x))` patterns.
      if (chainAssigns.length === 0) {
        const s = stmts[i];
        const inlineChain = tryMatchInlineReduction(
          s,
          paramTensors,
          allowedUnaryOps,
          allTensorVars
        );
        if (inlineChain) {
          inlineChain.startIdx = i;
          chains.push(inlineChain);
          i++;
        } else {
          i++;
        }
      }
      // If chainAssigns.length === 1, `i` already advanced past it;
      // it will be emitted by the per-op path.
    }
  }

  return chains;
}
