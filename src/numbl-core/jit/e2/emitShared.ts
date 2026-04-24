/**
 * e2 — shared kernel-emission helpers used by both the chain emitter
 * and the reduction emitter. The two emitters build the same kernel
 * shape up to a few trailing differences (reduction init / combine /
 * out_acc output), so everything that's identical lives here.
 */

import type { JitExpr, JitType } from "../jitTypes.js";
import { emitFusedScalarExpr, type FusedTarget } from "../fusedScalarEmit.js";
import { C_SCALAR_TARGET, formatNumberLiteral } from "../c/context.js";
import type { ScalarOpTarget } from "../scalarEmit.js";
import { getIBuiltin } from "../../interpreter/builtins/index.js";

/**
 * C helper included in every e2 kernel prologue.
 *
 * `-ffast-math` implies `-ffinite-math-only`, which lets the compiler
 * assume no NaN/Inf values and constant-fold `x != x` to 0 and `x == x`
 * to 1. The NaN-detection idiom `mask = x ~= x` would silently return all
 * zeros. `numbl_is_nan_fp` inspects the IEEE-754 bit pattern directly —
 * the optimizer cannot look through `memcpy` to apply finite-math
 * assumptions, so this survives the flag.
 */
export const E2_C_PROLOGUE =
  "#include <math.h>\n" +
  "#include <stdint.h>\n" +
  "#include <string.h>\n" +
  "\n" +
  "static inline int numbl_is_nan_fp(double x) {\n" +
  "    uint64_t b; memcpy(&b, &x, sizeof(b));\n" +
  "    return (b & 0x7FFFFFFFFFFFFFFFULL) > 0x7FF0000000000000ULL;\n" +
  "}\n\n";

/**
 * e2-specific scalar op target.
 *
 * Identical to `C_SCALAR_TARGET` except `binEq` / `binNe`: when both
 * operand strings are the same C expression (i.e. the source was `x == x`
 * or `x ~= x`), the standard `==` / `!=` forms are constant-folded to
 * 1 / 0 by `-ffinite-math-only`. Replace them with the bit-pattern NaN
 * helper so the self-comparison gives the correct IEEE 754 result.
 */
export const E2_C_SCALAR_TARGET: ScalarOpTarget = {
  ...C_SCALAR_TARGET,
  binEq: (l, r) =>
    l === r
      ? `((double)(!numbl_is_nan_fp(${l})))`
      : `(((double)((${l}) == (${r}))))`,
  binNe: (l, r) =>
    l === r
      ? `((double)(numbl_is_nan_fp(${l})))`
      : `(((double)((${l}) != (${r}))))`,
};

export interface ChainAssignSpec {
  lhsName: string;
  rhs: JitExpr;
}

export interface KernelInputs {
  /** Regular env input tensor names (NOT chain LHSs). */
  tensorNames: string[];
  /** Scalar env input names. */
  scalarNames: string[];
  /** Chain LHS names that need `in_<name>` because they're read before
   *  being written. */
  inputLhsNames: string[];
  /** Chain LHS names that escape the chain (materialized via
   *  `out_<name>`). Does NOT include a reduce-target name — that one
   *  is always chain-local by construction. */
  escapeLhsNames: string[];
}

export const cInputPtr = (name: string): string => `in_${name}`;
export const cOutputPtr = (name: string): string => `out_${name}`;
export const cScalarParam = (name: string): string => `s_${name}`;

/** FusedTarget for the per-element body. Resolves Var reads to either
 *  the chain-local stack name (once the corresponding assign has run)
 *  or `in_<name>[i]` (before that point), mangles scalar param names,
 *  and dispatches whitelisted builtins through their `jitEmitC`. */
export function makeFusedTarget(
  locallyAssigned: ReadonlySet<string>
): FusedTarget {
  return {
    formatNumber: formatNumberLiteral,
    mangle: cScalarParam,
    tensorElemRead: name =>
      locallyAssigned.has(name) ? name : cInputPtr(name) + "[i]",
    emitBuiltinCall: (name, args) => {
      const ib = getIBuiltin(name);
      if (!ib?.jitEmitC) return null;
      const argTypes: JitType[] = args.map(() => ({ kind: "number" }));
      return ib.jitEmitC(args, argTypes);
    },
  };
}

/** Unique chain LHS names in source order — for the leading
 *  `double <a>, <b>;` declaration. */
export function uniqueLhsOrdered(chain: ChainAssignSpec[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const a of chain) {
    if (!seen.has(a.lhsName)) {
      seen.add(a.lhsName);
      out.push(a.lhsName);
    }
  }
  return out;
}

/** All tensor-typed names visible to emitFusedScalarExpr: regular env
 *  tensors, `in_<lhs>` tensors, and chain LHSs. */
export function allTensorVarsFor(
  inputs: KernelInputs,
  chain: ChainAssignSpec[]
): Set<string> {
  return new Set<string>([
    ...inputs.tensorNames,
    ...inputs.inputLhsNames,
    ...chain.map(a => a.lhsName),
  ]);
}

/** Emit one `<lhs> = <rhsC>;` line per chain assign, growing
 *  `locallyAssigned` as we go so later stmts resolve earlier LHSs to
 *  the stack-local. */
export function emitChainAssignLines(
  chain: ChainAssignSpec[],
  allTensorVars: ReadonlySet<string>,
  ft: FusedTarget,
  locallyAssigned: Set<string>
): string[] {
  const out: string[] = [];
  for (const a of chain) {
    const rhsC = emitFusedScalarExpr(
      a.rhs,
      new Set(),
      allTensorVars,
      E2_C_SCALAR_TARGET,
      ft
    );
    out.push(`        ${a.lhsName} = ${rhsC};`);
    locallyAssigned.add(a.lhsName);
  }
  return out;
}

/** Kernel param list (tensor → inputLhs → scalar → escapeLhs). Callers
 *  append any trailing params (e.g. `double *out_acc`). */
export function buildParamList(inputs: KernelInputs): string[] {
  const params: string[] = ["int64_t n"];
  for (const t of inputs.tensorNames)
    params.push(`const double *${cInputPtr(t)}`);
  for (const t of inputs.inputLhsNames)
    params.push(`const double *${cInputPtr(t)}`);
  for (const s of inputs.scalarNames) params.push(`double ${cScalarParam(s)}`);
  for (const e of inputs.escapeLhsNames)
    params.push(`double *${cOutputPtr(e)}`);
  return params;
}

/** koffi type list in the same order as `buildParamList`. Callers
 *  append any trailing entries. */
export function buildKoffiParts(inputs: KernelInputs): string[] {
  const parts: string[] = ["int64_t"];
  for (let k = 0; k < inputs.tensorNames.length; k++) parts.push("double *");
  for (let k = 0; k < inputs.inputLhsNames.length; k++) parts.push("double *");
  for (let k = 0; k < inputs.scalarNames.length; k++) parts.push("double");
  for (let k = 0; k < inputs.escapeLhsNames.length; k++) parts.push("double *");
  return parts;
}
