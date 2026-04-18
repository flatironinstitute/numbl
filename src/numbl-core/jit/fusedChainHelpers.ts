/**
 * Chain-level helpers shared by the JS and C fused-codegen backends.
 *
 * The per-element scalar expression walker lives in `fusedScalarEmit.ts`;
 * this module covers the surrounding logic that decides which chain dests
 * need a write-back to their tensor buffer, and the reduction-accumulator
 * init/combine snippets for inline reductions.
 *
 * Reductions are parameterized over a small `ReductionLiterals` record so
 * each backend supplies its own spelling of `0` vs `0.0`, `===` vs `==`,
 * `-Infinity` vs `(-1.0/0.0)`, etc. — the control structure is identical.
 */

import { BinaryOperation } from "../parser/types.js";
import type { FusibleChain } from "./fusion.js";

/**
 * Compute the set of distinct dest names in a fused chain and which of
 * them require a write-back into their tensor buffer.
 *
 * A dest normally needs write-back; the exception is the chain's last
 * tensor if it is fully consumed by a trailing reduction (in which case
 * the scalar reduction accumulator is the only output — materialising
 * the tensor buffer would be wasted work). If that last-dest tensor is
 * ALSO a named output of the enclosing function, the write-back is kept
 * so the caller sees the updated buffer.
 */
export function determineWriteBack(
  chain: FusibleChain,
  outputTensorNames: ReadonlySet<string>
): {
  destNames: Set<string>;
  writeBack: Set<string>;
  reductionConsumes: boolean;
} {
  const lastDest = chain.assigns[chain.assigns.length - 1].destName;
  const reductionConsumes =
    chain.reduction !== undefined && chain.reduction.tensorName === lastDest;

  const destNames = new Set<string>();
  for (const a of chain.assigns) destNames.add(a.destName);

  const writeBack = new Set<string>();
  for (const d of destNames) {
    if (reductionConsumes && d === lastDest) {
      if (outputTensorNames.has(d)) writeBack.add(d);
    } else {
      writeBack.add(d);
    }
  }

  return { destNames, writeBack, reductionConsumes };
}

// ── Reduction init / combine ─────────────────────────────────────────

/**
 * Target-specific literal spellings used by the reduction helpers.
 *
 * The structure of the reduction snippets is identical between JS and
 * C, but the literals differ: JS uses `1`, `-Infinity`, `===`/`!==`,
 * while C uses `1.0`, `(-1.0/0.0)`, `==`/`!=`. The caller picks a
 * record for its target and reuses it.
 */
export interface ReductionLiterals {
  /** Additive identity (`0` for JS, `0.0` for C). */
  zero: string;
  /** Multiplicative identity / truthy (`1` or `1.0`). */
  one: string;
  /** Positive infinity literal (`Infinity` or `(1.0/0.0)`). */
  posInf: string;
  /** Negative infinity literal (`-Infinity` or `(-1.0/0.0)`). */
  negInf: string;
  /** Strict-equality operator (`===` for JS, `==` for C). */
  eq: string;
  /** Strict-inequality operator (`!==` for JS, `!=` for C). */
  neq: string;
}

export const JS_REDUCTION_LITERALS: ReductionLiterals = {
  zero: "0",
  one: "1",
  posInf: "Infinity",
  negInf: "-Infinity",
  eq: "===",
  neq: "!==",
};

export const C_REDUCTION_LITERALS: ReductionLiterals = {
  zero: "0.0",
  one: "1.0",
  posInf: "(1.0/0.0)",
  negInf: "(-1.0/0.0)",
  eq: "==",
  neq: "!=",
};

/** Initial value expression for a reduction accumulator. */
export function reductionInit(
  reduceName: string,
  lits: ReductionLiterals
): string {
  switch (reduceName) {
    case "sum":
    case "mean":
      return lits.zero;
    case "prod":
      return lits.one;
    case "max":
      return lits.negInf;
    case "min":
      return lits.posInf;
    case "any":
      return lits.zero;
    case "all":
      return lits.one;
    default:
      throw new Error(`fusedChainHelpers: unknown reduction ${reduceName}`);
  }
}

/** Statement that folds a per-element `valueExpr` into the accumulator. */
export function reductionCombine(
  reduceName: string,
  accVar: string,
  valueExpr: string,
  lits: ReductionLiterals
): string {
  switch (reduceName) {
    case "sum":
    case "mean":
      return `${accVar} += ${valueExpr};`;
    case "prod":
      return `${accVar} *= ${valueExpr};`;
    case "max":
      return `if (${valueExpr} > ${accVar}) ${accVar} = ${valueExpr};`;
    case "min":
      return `if (${valueExpr} < ${accVar}) ${accVar} = ${valueExpr};`;
    case "any":
      return `if (${valueExpr} ${lits.neq} ${lits.zero}) ${accVar} = ${lits.one};`;
    case "all":
      return `if (${valueExpr} ${lits.eq} ${lits.zero}) ${accVar} = ${lits.zero};`;
    default:
      throw new Error(`fusedChainHelpers: unknown reduction ${reduceName}`);
  }
}

/**
 * Statement that folds a per-chain `val` into an enclosing accumulator
 * `dest` via the outer-loop op (e.g. `ir_acc = ir_acc + sum(...)`).
 *
 * Target-neutral: `+=` / `-=` / `*=` have identical syntax in JS and C.
 */
export function accumulateOp(
  op: BinaryOperation,
  dest: string,
  val: string
): string {
  switch (op) {
    case BinaryOperation.Add:
      return `${dest} += ${val};`;
    case BinaryOperation.Sub:
      return `${dest} -= ${val};`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `${dest} *= ${val};`;
    default:
      return `${dest} = ${dest} + ${val};`;
  }
}
