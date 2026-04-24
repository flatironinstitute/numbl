/**
 * Shared per-element scalar-expression emission for fused loops.
 *
 * Both the JS-JIT and C-JIT fused-chain emitters walk the chain's
 * expression trees and emit each sub-expression in "per-element"
 * form — tensor Vars become `data[__i]` reads (or a scalar local for
 * chain-produced intermediates), Binary/Unary/Call map to scalar
 * operations that will run once per element of the fused loop.
 *
 * The walk itself is identical between the two backends; only the
 * leaf syntax differs (JS `Math.sin` vs C `sin`, integer literal
 * formatting, mangling prefix). A backend supplies a `FusedTarget`
 * describing those leaves and a value-form `ScalarOpTarget` for the
 * arithmetic/comparison/logical switches.
 *
 * Note: the op target used here must emit comparison / logical ops
 * in *numeric* form (result is a double 0.0/1.0 suitable for tensor
 * write-back). For C this coincides with the regular value target;
 * for JS a second target instance is needed because value-form
 * comparisons return a JS boolean.
 */

import type { JitExpr } from "./jitTypes.js";
import type { FusibleChain } from "./fusion.js";
import {
  type ScalarOpTarget,
  emitScalarBinaryOp,
  emitScalarUnaryOp,
} from "./scalarEmit.js";

/** Scalar local name for a chain-produced tensor intermediate. */
export function fusedLocal(name: string): string {
  return `__f_${name}`;
}

export interface FusedTarget {
  /** Format a numeric literal (e.g. `1` for JS, `1.0` for C). */
  formatNumber(v: number): string;
  /** Mangle a scalar variable reference (non-tensor). */
  mangle(name: string): string;
  /**
   * Emit a per-element read of tensor var `name` — i.e. the expression
   * that yields `data[__i]` for that tensor. The backend decides how
   * the data pointer is named and whether it's aliased locally.
   */
  tensorElemRead(name: string): string;
  /**
   * Emit a read of tensor `name` at a runtime 1-based scalar index
   * `idxC` — i.e. `data[(int64_t)idx - 1]`. Used by the e2 whole-loop
   * kernel (scalar-context access; elemwise backends can leave this
   * undefined, the emitter will throw on an Index node). Returns `null`
   * to reject.
   */
  tensorScalarIndexRead?(name: string, idxC: string): string | null;
  /**
   * Emit a call to a scalar math builtin. The backend decides which
   * builtins it supports and how they map to library functions (e.g.
   * JS `Math.sin` vs C `sin`). Return `null` to reject.
   *
   * `name` is the builtin name (e.g. `"sin"`, `"mod"`, `"rem"`);
   * `args` are already-emitted per-element scalar expressions.
   */
  emitBuiltinCall(name: string, args: string[]): string | null;
}

/** Shared walker: emit a JitExpr as a per-element scalar expression. */
export function emitFusedScalarExpr(
  expr: JitExpr,
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  opTarget: ScalarOpTarget,
  fusedTarget: FusedTarget
): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return fusedTarget.formatNumber(expr.value);

    case "Var": {
      if (expr.jitType.kind === "tensor" || allTensorVars.has(expr.name)) {
        if (chainLocals.has(expr.name)) return fusedLocal(expr.name);
        return fusedTarget.tensorElemRead(expr.name);
      }
      return fusedTarget.mangle(expr.name);
    }

    case "Binary": {
      const l = emitFusedScalarExpr(
        expr.left,
        chainLocals,
        allTensorVars,
        opTarget,
        fusedTarget
      );
      const r = emitFusedScalarExpr(
        expr.right,
        chainLocals,
        allTensorVars,
        opTarget,
        fusedTarget
      );
      return emitScalarBinaryOp(expr.op, l, r, opTarget);
    }

    case "Unary": {
      const operand = emitFusedScalarExpr(
        expr.operand,
        chainLocals,
        allTensorVars,
        opTarget,
        fusedTarget
      );
      return emitScalarUnaryOp(expr.op, operand, opTarget);
    }

    case "Call": {
      const args = expr.args.map(a =>
        emitFusedScalarExpr(
          a,
          chainLocals,
          allTensorVars,
          opTarget,
          fusedTarget
        )
      );
      const result = fusedTarget.emitBuiltinCall(expr.name, args);
      if (result === null) {
        throw new Error(
          `fused scalar emitter: unsupported call in fused chain: ${expr.name}`
        );
      }
      return result;
    }

    case "Index": {
      // Scalar-context index read of a tensor (e.g. `x(i)` in a scalar
      // loop). Only 1-index form is supported today; the base must be
      // a tensor Var. Backend decides how to emit; it returns null to
      // reject.
      if (expr.base.tag !== "Var" || expr.base.jitType.kind !== "tensor") {
        throw new Error(
          `fused scalar emitter: Index requires a tensor Var base`
        );
      }
      if (expr.indices.length !== 1) {
        throw new Error(
          `fused scalar emitter: multi-index tensor access not supported`
        );
      }
      const idxC = emitFusedScalarExpr(
        expr.indices[0],
        chainLocals,
        allTensorVars,
        opTarget,
        fusedTarget
      );
      const out = fusedTarget.tensorScalarIndexRead?.(expr.base.name, idxC);
      if (!out) {
        throw new Error(
          `fused scalar emitter: backend doesn't support scalar tensor index`
        );
      }
      return out;
    }

    default:
      throw new Error(
        `fused scalar emitter: unsupported expr in fused chain: ${expr.tag}`
      );
  }
}

/**
 * Find the first tensor-param name referenced in a chain's assigns.
 * Used by both backends to pick the length-determining tensor.
 */
export function findTensorParamInChain(
  chain: FusibleChain,
  paramTensors: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): string | null {
  for (const a of chain.assigns) {
    const found = findTensorParamInExpr(a.expr, paramTensors, allTensorVars);
    if (found) return found;
  }
  return null;
}

function findTensorParamInExpr(
  expr: JitExpr,
  paramTensors: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): string | null {
  if (expr.tag === "Var" && allTensorVars.has(expr.name)) {
    if (paramTensors.has(expr.name)) return expr.name;
    return null;
  }
  if (expr.tag === "Binary") {
    return (
      findTensorParamInExpr(expr.left, paramTensors, allTensorVars) ??
      findTensorParamInExpr(expr.right, paramTensors, allTensorVars)
    );
  }
  if (expr.tag === "Unary") {
    return findTensorParamInExpr(expr.operand, paramTensors, allTensorVars);
  }
  if (expr.tag === "Call") {
    for (const a of expr.args) {
      const f = findTensorParamInExpr(a, paramTensors, allTensorVars);
      if (f) return f;
    }
  }
  return null;
}

/**
 * Collect distinct tensor names referenced in the chain's expression
 * trees that are NOT produced by the chain itself (i.e. read from
 * outside: params or pre-existing locals). Both backends need this to
 * pick a length-reference tensor when no formal param is in the chain.
 */
export function collectInputTensors(
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>
): Set<string> {
  const result = new Set<string>();
  const chainDests = new Set(chain.assigns.map(a => a.destName));
  for (const a of chain.assigns) {
    walkForTensors(a.expr, allTensorVars, chainDests, result);
  }
  return result;
}

function walkForTensors(
  expr: JitExpr,
  allTensorVars: ReadonlySet<string>,
  chainDests: ReadonlySet<string>,
  out: Set<string>
): void {
  if (expr.tag === "Var" && allTensorVars.has(expr.name)) {
    if (!chainDests.has(expr.name)) out.add(expr.name);
    return;
  }
  if (expr.tag === "Binary") {
    walkForTensors(expr.left, allTensorVars, chainDests, out);
    walkForTensors(expr.right, allTensorVars, chainDests, out);
  } else if (expr.tag === "Unary") {
    walkForTensors(expr.operand, allTensorVars, chainDests, out);
  } else if (expr.tag === "Call") {
    for (const a of expr.args)
      walkForTensors(a, allTensorVars, chainDests, out);
  }
}
