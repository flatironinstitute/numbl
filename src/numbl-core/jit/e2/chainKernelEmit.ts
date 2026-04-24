/**
 * e2 — multi-LHS fused chain C kernel emission.
 *
 * Given a sequence of `JitExpr` RHSs each writing to a (possibly
 * distinct) chain LHS, produces one C function that runs every assign
 * in a single per-element loop. Each chain LHS becomes a stack-local
 * `double <name>` declared once at the top of the loop body. Within
 * the body, references to a chain-LHS name resolve to the stack-local
 * once the corresponding assign has run; before that point they
 * resolve to `in_<name>[i]` (so the kernel signature includes
 * `in_<lhsName>` for any chain LHS that's read before being written).
 *
 * After the per-iter assigns, every "escape" LHS (one that's actually
 * referenced by the rest of the function body) gets written to its
 * `out_<name>[i]` pointer. Chain-locals (only used inside the chain)
 * are dropped at the end of the iteration with no buffer materialized.
 *
 *     void e2c_<hash>(int64_t n,
 *                     const double *in_<input1>, ...,
 *                     [const double *in_<lhs_needing_input>, ...,]
 *                     double s_<scalar1>, ...,
 *                     double *out_<escape_lhs1>, ...)
 *     {
 *         #pragma omp simd
 *         for (int64_t i = 0; i < n; i++) {
 *             double <chain_lhs1>, <chain_lhs2>, ...;
 *             <chain_lhs1> = <stmt0_rhs_C>;
 *             <chain_lhs2> = <stmt1_rhs_C>;
 *             ...
 *             out_<escape_lhs1>[i] = <escape_lhs1>;
 *             out_<escape_lhs2>[i] = <escape_lhs2>;
 *         }
 *     }
 */

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
} from "./emitShared.js";

export type { ChainAssignSpec } from "./emitShared.js";

export interface E2ChainEmitResult {
  kernelName: string;
  cSource: string;
  koffiSig: string;
  hash: string;
  /** Tensor input names in signature order — does NOT include any
   *  in_<lhs> entries. */
  inputTensors: string[];
  /** Chain LHS names that appear as `in_<name>` in the signature, in
   *  order. */
  inputLhsNames: string[];
  /** Scalar input names in signature order. */
  inputScalars: string[];
  /** Chain LHS names that appear as `out_<name>` in the signature,
   *  in order. */
  escapeLhsNames: string[];
  chainLength: number;
}

export function emitE2ChainKernel(
  assigns: ChainAssignSpec[],
  inputs: KernelInputs,
  par: boolean = false
): E2ChainEmitResult {
  if (assigns.length === 0) {
    throw new Error("emitE2ChainKernel: empty chain");
  }

  const locallyAssigned = new Set<string>();
  const allTensorVars = allTensorVarsFor(inputs, assigns);
  const allLhsOrdered = uniqueLhsOrdered(assigns);
  const ft = makeFusedTarget(locallyAssigned);

  const bodyLines: string[] = [];
  bodyLines.push(`        double ${allLhsOrdered.join(", ")};`);
  bodyLines.push(
    ...emitChainAssignLines(assigns, allTensorVars, ft, locallyAssigned)
  );
  for (const e of inputs.escapeLhsNames) {
    bodyLines.push(`        ${cOutputPtr(e)}[i] = ${e};`);
  }

  // `--par` upgrades tensor-writeback loops to `parallel for simd` when
  // the per-element body has heavy transcendentals — thread-spawn
  // overhead only pays off past ~100 cycles of work. Arithmetic-only
  // chains stick to plain `simd` because threads add overhead exceeding
  // the memory-bandwidth-bound body's time. Same heuristic as e1.
  const heavyOps = assigns.reduce((n, a) => n + countHeavyOps(a.rhs), 0);
  const useParallel = par && inputs.escapeLhsNames.length > 0 && heavyOps > 0;
  const pragma = useParallel
    ? `    #pragma omp parallel for simd if(n >= ${ompParallelThreshold()})`
    : "    #pragma omp simd";
  const loopLines = [
    pragma,
    "    for (int64_t i = 0; i < n; i++) {",
    ...bodyLines,
    "    }",
  ];

  const paramList = buildParamList(inputs);
  const prologue = E2_C_PROLOGUE;
  const bodyTemplate =
    `void __KERNEL_NAME__(${paramList.join(", ")})\n` +
    `{\n${loopLines.join("\n")}\n}\n`;
  const cSourceTemplate = prologue + bodyTemplate;

  const h = fnv1a64Hex(cSourceTemplate);
  const kernelName = `e2c_${h}`;
  const cSource = cSourceTemplate.replace("__KERNEL_NAME__", kernelName);

  const koffiSig = `void ${kernelName}(${buildKoffiParts(inputs).join(", ")})`;

  return {
    kernelName,
    cSource,
    koffiSig,
    hash: h,
    inputTensors: inputs.tensorNames.slice(),
    inputLhsNames: inputs.inputLhsNames.slice(),
    inputScalars: inputs.scalarNames.slice(),
    escapeLhsNames: inputs.escapeLhsNames.slice(),
    chainLength: assigns.length,
  };
}
