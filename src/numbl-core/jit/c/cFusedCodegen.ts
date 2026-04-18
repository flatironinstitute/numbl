/**
 * Fused per-element loop emission for the C-JIT.
 *
 * Given a FusibleChain (from fusion.ts), emits a single
 *   `for (int64_t __i = 0; __i < N; __i++) { ... }`
 * loop that evaluates all the chain's tensor assigns as inline scalar
 * expressions per element — no libnumbl_ops calls, no intermediate
 * buffers.
 *
 * Scalar expressions (number literals, scalar vars, scalar math calls)
 * pass through unchanged. Tensor var references become either:
 *   - `v_name_data[__i]`  for input params / pre-existing tensors
 *   - `__f_name`           for chain-produced intermediates (scalar local)
 *
 * The optional trailing reduction is absorbed as an inline accumulator
 * (`__f_acc += expr`) inside the same loop, eliminating the need to
 * materialise the tensor result at all when it is only consumed by the
 * reduction.
 */

import { BinaryOperation } from "../../parser/types.js";
import type { JitExpr, JitType } from "../jitTypes.js";
import type { FusibleChain } from "../fusion.js";
import {
  type FusedTarget,
  emitFusedScalarExpr,
  fusedLocal,
  findTensorParamInChain,
} from "../fusedScalarEmit.js";
import {
  C_SCALAR_TARGET,
  formatNumberLiteral,
  mangle,
  tensorData,
  tensorLen,
} from "./jitCodegenC.js";
import { getIBuiltin } from "../../interpreter/builtins/types.js";
import {
  C_REDUCTION_LITERALS,
  accumulateOp,
  determineWriteBack,
  reductionCombine,
  reductionInit,
} from "../fusedChainHelpers.js";

/** Minimum element count before `#pragma omp parallel for` kicks in.
 *  Below this, thread-spawn overhead dominates. */
const OMP_PARALLEL_THRESHOLD =
  parseInt(process.env.NUMBL_OMP_THRESHOLD || "", 10) || 100_000;

/** Builtins that are expensive enough per element to justify thread-spawn overhead. */
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

/** Count heavy (transcendental) ops in an expression tree. */
function countHeavyOps(expr: JitExpr): number {
  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
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

// ── Fused target (per-element leaves) ────────────────────────────────
//
// The op switches (binary/unary) reuse C_SCALAR_TARGET — C's value
// form is already numeric (booleans are `(double)` casts), so the
// same target works in both value and per-element contexts.
//
// Function-call leaves consult each builtin's own `jitEmitC` so the
// C function-name mapping lives with the builtin, not in a central
// table here. Fused context is per-element, so argTypes are
// fabricated as scalar numbers — all fusible ops (per fusionOps.ts)
// are element-wise, and none carry a `requireNonneg` guard.

const C_FUSED_TARGET: FusedTarget = {
  formatNumber: formatNumberLiteral,
  mangle,
  tensorElemRead: name => `${tensorData(name)}[__i]`,
  emitBuiltinCall: (name, args) => {
    const ib = getIBuiltin(name);
    if (!ib?.jitEmitC) return null;
    const argTypes: JitType[] = args.map(() => ({ kind: "number" }));
    return ib.jitEmitC(args, argTypes);
  },
};

// ── Public API ────────────────────────────────────────────────────────

/**
 * Emit a fused per-element loop for the given chain.
 *
 * Appends C source lines to `lines`. All scalar math helpers the inner
 * body may reference (mod, sign, ...) live in jit_runtime.a, so this
 * function no longer reports back "helpers needed" — the emitter simply
 * calls them as library symbols.
 *
 * `allTensorVars` is the full set of tensor-typed variable names.
 * `paramTensors` is the subset that are input parameters.
 * `outputTensorNames` is the subset that are function outputs.
 * `localTensorNames` is the subset that are non-param, non-output locals.
 */
export function emitFusedChain(
  lines: string[],
  indent: string,
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  paramTensors: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>,
  localTensorNames: ReadonlySet<string>,
  openmp?: boolean
): void {
  // Determine the length variable — use the first tensor param referenced.
  const refParam = findTensorParamInChain(chain, paramTensors, allTensorVars);
  const lenVar = refParam
    ? tensorLen(refParam)
    : tensorLen(chain.assigns[0].destName);

  // Determine which dest names need a write-back to their buffer
  // (shared with JS codegen).
  const { writeBack } = determineWriteBack(chain, outputTensorNames);

  // For dests that need write-back and are local tensors, ensure buffer
  // is allocated before the loop.
  for (const d of writeBack) {
    if (localTensorNames.has(d)) {
      lines.push(`${indent}${tensorLen(d)} = ${lenVar};`);
      lines.push(
        `${indent}if (!${tensorData(d)}) ${tensorData(d)} = (double *)malloc((size_t)${lenVar} * sizeof(double));`
      );
    }
  }

  // Emit reduction accumulator init.
  const reduceAccLocal = "__f_reduce_acc";
  if (chain.reduction) {
    lines.push(
      `${indent}double ${reduceAccLocal} = ${reductionInit(chain.reduction.reduceName, C_REDUCTION_LITERALS)};`
    );
  }

  // Track which tensor vars have been produced by earlier assigns in
  // the chain — these are read via scalar locals, not array reads.
  const chainLocals = new Set<string>();

  // Open the fused loop.
  // Conditions for parallel-for:
  //  1. Writes to output/param tensors (not just local temporaries
  //     consumed by subsequent per-op code — parallelizing those scatters
  //     data across caches and hurts the sequential consumer).
  //  2. Chain body has transcendental ops (sin, exp, etc.) so the
  //     per-element compute justifies thread-spawn overhead.
  const writesToOutput = [...writeBack].some(
    d => outputTensorNames.has(d) || paramTensors.has(d)
  );
  const heavyOps = chain.assigns.reduce((n, a) => n + countHeavyOps(a.expr), 0);
  if (!chain.reduction) {
    if (openmp && writesToOutput && heavyOps > 0) {
      lines.push(
        `${indent}#pragma omp parallel for simd if(${lenVar} >= ${OMP_PARALLEL_THRESHOLD})`
      );
    } else {
      lines.push(`${indent}#pragma omp simd`);
    }
  }
  lines.push(`${indent}for (int64_t __i = 0; __i < ${lenVar}; __i++) {`);
  const inner = indent + "  ";

  for (const assign of chain.assigns) {
    const rhs = emitFusedScalarExpr(
      assign.expr,
      chainLocals,
      allTensorVars,
      C_SCALAR_TARGET,
      C_FUSED_TARGET
    );

    // First assignment to this dest in the loop → declare the scalar local.
    // Subsequent assignments → just reassign.
    if (!chainLocals.has(assign.destName)) {
      lines.push(`${inner}double ${fusedLocal(assign.destName)} = ${rhs};`);
      chainLocals.add(assign.destName);
    } else {
      lines.push(`${inner}${fusedLocal(assign.destName)} = ${rhs};`);
    }
  }

  // Write-back to buffers.
  for (const d of writeBack) {
    lines.push(`${inner}${tensorData(d)}[__i] = ${fusedLocal(d)};`);
  }

  // Inline reduction accumulate.
  if (chain.reduction) {
    const valueExpr = fusedLocal(chain.reduction.tensorName);
    lines.push(
      `${inner}${reductionCombine(chain.reduction.reduceName, reduceAccLocal, valueExpr, C_REDUCTION_LITERALS)}`
    );
  }

  // Close the loop.
  lines.push(`${indent}}`);

  // Update tensor lengths for write-back dests.
  for (const d of writeBack) {
    lines.push(`${indent}${tensorLen(d)} = ${lenVar};`);
  }

  // Post-loop: apply mean division if needed, then store reduction result.
  if (chain.reduction) {
    if (chain.reduction.reduceName === "mean") {
      lines.push(`${indent}${reduceAccLocal} /= (double)${lenVar};`);
    }
    const acc = mangle(chain.reduction.accName);
    if (chain.reduction.hasAccumulate && chain.reduction.accOp !== undefined) {
      lines.push(
        `${indent}${accumulateOp(chain.reduction.accOp, acc, reduceAccLocal)}`
      );
    } else {
      lines.push(`${indent}${acc} = ${reduceAccLocal};`);
    }
  }
}
