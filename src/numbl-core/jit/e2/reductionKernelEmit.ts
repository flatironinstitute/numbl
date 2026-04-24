/**
 * e2 — reduction kernel emission.
 *
 * Handles two related patterns in a single emitter:
 *
 *   (A) Standalone reduction:
 *           acc = [acc OP] reduce(elemwiseExpr)
 *       Empty chain prefix; the kernel walks the inputs once and
 *       accumulates `reduce(per-element-expr)` into a scalar buffer.
 *
 *   (B) Chain + trailing reduction:
 *           lhs1 = ...; lhs2 = ...; ...; lhsK = ...;
 *           acc = [acc OP] reduce(lhsK)
 *       The chain runs in the same per-element loop; lhsK is purely
 *       chain-local (never materialized) — the kernel accumulates
 *       reduce(lhsK) into the scalar buffer. Other chain LHSs may
 *       still escape (extra `out_<name>` outputs).
 *
 * Both cases use the same kernel shape:
 *
 *     void e2r_<hash>(int64_t n,
 *                     ..in_*.., ..in_lhs_input.., ..s_*..,
 *                     ..out_escape.., double *out_acc)
 *     {
 *         double acc = <init>;
 *         #pragma omp simd
 *         for (int64_t i = 0; i < n; i++) {
 *             double <chain_lhs1>, ..., <chain_lhsK>;
 *             <chain_lhs1> = <stmt0_rhs_C>;
 *             ...
 *             <chain_lhsK> = <stmtK_rhs_C>;
 *             out_<escape>[i] = <escape>;
 *             <reduce-combine>(acc, <reduce_value_expr>);
 *         }
 *         *out_acc = acc;
 *     }
 *
 * For "mean": JS combines `acc /= n` after reading the buffer back.
 * For "max"/"min": uses if-update inside the loop (works under
 * `-ffast-math` + `#pragma omp simd`).
 */

import type { JitExpr } from "../jitTypes.js";
import { emitFusedScalarExpr } from "../fusedScalarEmit.js";
import {
  C_REDUCTION_LITERALS,
  reductionInit,
  reductionCombine,
} from "../fusedChainHelpers.js";
import { fnv1a64Hex } from "../e1/hash.js";
import { countHeavyOps, ompParallelThreshold } from "../heavyOps.js";
import {
  allTensorVarsFor,
  buildKoffiParts,
  buildParamList,
  cOutputPtr,
  emitChainAssignLines,
  makeFusedTarget,
  uniqueLhsOrdered,
  type ChainAssignSpec,
  type KernelInputs,
  E2_C_PROLOGUE,
  E2_C_SCALAR_TARGET,
} from "./emitShared.js";

export interface ReductionEmitSpec {
  /** Chain prefix (length 0 for standalone-reduction). */
  chain: ChainAssignSpec[];
  /** Reduction op name: sum, prod, max, min, mean, any, all. */
  reduceName: string;
  /** Per-element value expression to feed the reduction.
   *  - For (A) standalone: the elemwise expression `reduce(...)` was
   *    given.
   *  - For (B) chain + trailing: a `Var(lastChainLhsName)` JitExpr —
   *    the emitter resolves it to the stack-local. */
  reduceValueExpr: JitExpr;
  inputs: KernelInputs;
}

export interface E2ReductionEmitResult {
  kernelName: string;
  cSource: string;
  koffiSig: string;
  hash: string;
  inputTensors: string[];
  inputLhsNames: string[];
  inputScalars: string[];
  escapeLhsNames: string[];
  /** True when the kernel produces a scalar reduction output (always
   *  true for this emitter; here for symmetry with other entries). */
  hasReductionOutput: true;
  reduceName: string;
  chainLength: number;
}

/** OpenMP `reduction(...)` clause for a given reduction op. Returns
 *  null for ops we can't express with a clause (`any`/`all` use
 *  if-update patterns, not natural reduction operators on doubles). */
function ompReductionClause(reduceName: string, accVar: string): string | null {
  switch (reduceName) {
    case "sum":
    case "mean":
      return `reduction(+:${accVar})`;
    case "prod":
      return `reduction(*:${accVar})`;
    case "max":
      return `reduction(max:${accVar})`;
    case "min":
      return `reduction(min:${accVar})`;
    default:
      return null;
  }
}

export function emitE2ReductionKernel(
  spec: ReductionEmitSpec,
  par: boolean = false
): E2ReductionEmitResult {
  const locallyAssigned = new Set<string>();
  const allTensorVars = allTensorVarsFor(spec.inputs, spec.chain);
  const allLhsOrdered = uniqueLhsOrdered(spec.chain);
  const ft = makeFusedTarget(locallyAssigned);

  const bodyLines: string[] = [];
  if (allLhsOrdered.length > 0) {
    bodyLines.push(`        double ${allLhsOrdered.join(", ")};`);
  }
  bodyLines.push(
    ...emitChainAssignLines(spec.chain, allTensorVars, ft, locallyAssigned)
  );
  for (const e of spec.inputs.escapeLhsNames) {
    bodyLines.push(`        ${cOutputPtr(e)}[i] = ${e};`);
  }
  const reduceValueC = emitFusedScalarExpr(
    spec.reduceValueExpr,
    new Set(),
    allTensorVars,
    E2_C_SCALAR_TARGET,
    ft
  );
  bodyLines.push(
    `        ${reductionCombine(spec.reduceName, "acc", reduceValueC, C_REDUCTION_LITERALS)}`
  );

  const paramList = buildParamList(spec.inputs);
  paramList.push(`double *out_acc`);

  const initC = reductionInit(spec.reduceName, C_REDUCTION_LITERALS);

  // `--par` upgrades the loop to `parallel for simd reduction(...)`
  // with an `if(n >= T)` gate. Requires both an OpenMP-expressible
  // reduction op (sum/mean/prod/max/min) AND enough per-element work
  // to outweigh thread-spawn overhead. Heavy-op count covers the
  // chain RHSs PLUS the reduce-value expression.
  const heavyOps =
    spec.chain.reduce((n, a) => n + countHeavyOps(a.rhs), 0) +
    countHeavyOps(spec.reduceValueExpr);
  const reductionClause = ompReductionClause(spec.reduceName, "acc");
  const useParallel = par && heavyOps > 0 && reductionClause !== null;
  const pragma = useParallel
    ? `    #pragma omp parallel for simd ${reductionClause} if(n >= ${ompParallelThreshold()})`
    : `    #pragma omp simd`;

  const loopLines = [
    `    double acc = ${initC};`,
    pragma,
    "    for (int64_t i = 0; i < n; i++) {",
    ...bodyLines,
    "    }",
    `    *out_acc = acc;`,
  ];

  const prologue = E2_C_PROLOGUE;
  const bodyTemplate =
    `void __KERNEL_NAME__(${paramList.join(", ")})\n` +
    `{\n${loopLines.join("\n")}\n}\n`;
  const cSourceTemplate = prologue + bodyTemplate;

  const h = fnv1a64Hex(cSourceTemplate);
  const kernelName = `e2r_${h}`;
  const cSource = cSourceTemplate.replace("__KERNEL_NAME__", kernelName);

  const koffiParts = buildKoffiParts(spec.inputs);
  koffiParts.push("double *");
  const koffiSig = `void ${kernelName}(${koffiParts.join(", ")})`;

  return {
    kernelName,
    cSource,
    koffiSig,
    hash: h,
    inputTensors: spec.inputs.tensorNames.slice(),
    inputLhsNames: spec.inputs.inputLhsNames.slice(),
    inputScalars: spec.inputs.scalarNames.slice(),
    escapeLhsNames: spec.inputs.escapeLhsNames.slice(),
    hasReductionOutput: true,
    reduceName: spec.reduceName,
    chainLength: spec.chain.length,
  };
}
