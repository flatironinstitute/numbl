/**
 * c-jit-chain AST transformer.
 *
 * Walks a stmt list once and identifies maximal contiguous runs of
 * adjacent stmts that are individually candidates for fusion (today:
 * tensor element-wise Assigns whose RHS is a fusable expression
 * tree). Wraps each non-trivial run (length >= 2) in a `Synth` node
 * with `tag: "c-jit-chain"`; the matching `cJitChainExecutor` will
 * compile the whole chain to one C kernel at dispatch time.
 *
 * Singletons (chains of length 1) are left as the original Assign so
 * the existing `c-jit-fuse` per-stmt path handles them. That keeps
 * the win path narrow: chains exist only where multiple adjacent
 * fusable Assigns can amortize per-call overhead and share memory
 * traversal.
 *
 * Purely structural — no env access. Runtime decisions (size,
 * type-shape match) happen in the executor's `propose()`.
 *
 * Recursive descent: For/While/If bodies are transformed lazily via
 * `Registry.transformStmts` when the interpreter walks them. This
 * pass just operates on the immediate stmt list.
 */

import { type Expr, type Stmt } from "../../parser/types.js";
import { isElemwiseStructuralExpr } from "./elemwiseStructural.js";

/** Per-stmt classification used by the chain executor. The data
 *  carried on each Synth node is `ChainAnalysis`. */
export interface ChainAnalysis {
  /** Original Assigns in source order. */
  readonly assigns: readonly (Stmt & { type: "Assign" })[];
  /** Stable shape hash — combines per-stmt RHS structures and LHS
   *  names, used by the executor's cacheKey. Sizes / runtime types
   *  are NOT included; those affect cost, not codegen shape. */
  readonly cacheKey: string;
}

/**
 * Transform a stmt list, wrapping maximal fusable runs (length >= 2)
 * in `Synth` nodes. Returns a new list when any transformation
 * happens; returns the input unchanged otherwise.
 */
export function chainPass(stmts: readonly Stmt[]): Stmt[] {
  let result: Stmt[] | null = null;
  let i = 0;
  while (i < stmts.length) {
    if (isFusableAssign(stmts[i])) {
      // Extend run while consecutive stmts are also fusable Assigns.
      let j = i;
      while (j < stmts.length && isFusableAssign(stmts[j])) j++;
      const runLen = j - i;
      if (runLen >= 2) {
        if (!result) result = stmts.slice(0, i);
        const assigns = stmts.slice(i, j) as (Stmt & { type: "Assign" })[];
        result.push(buildSynth(assigns));
        i = j;
        continue;
      }
    }
    if (result) result.push(stmts[i]);
    i++;
  }
  return result ?? (stmts as Stmt[]);
}

function buildSynth(assigns: readonly (Stmt & { type: "Assign" })[]): Stmt {
  const data: ChainAnalysis = {
    assigns,
    cacheKey: buildCacheKey(assigns),
  };
  return {
    type: "Synth",
    tag: "c-jit-chain",
    subStmts: [...assigns],
    data,
    span: assigns[0].span,
  };
}

function isFusableAssign(stmt: Stmt): boolean {
  if (stmt.type !== "Assign") return false;
  return isElemwiseStructuralExpr(stmt.expr);
}

/** Stable-hashable string that encodes the AST shape of every Assign
 *  in the chain. Matches the existing `fuseAnalyze` style — used by
 *  the executor's cacheKey so two runs of the same chain get the same
 *  cached compiled artifact. */
function buildCacheKey(
  assigns: readonly (Stmt & { type: "Assign" })[]
): string {
  const parts: string[] = [];
  for (const a of assigns) {
    parts.push(`A:${a.name}`);
    emitExprKey(a.expr, parts);
  }
  return parts.join("|");
}

function emitExprKey(e: Expr, out: string[]): void {
  switch (e.type) {
    case "Number":
      out.push(`N:${e.value}`);
      return;
    case "Ident":
      out.push(`V:${e.name}`);
      return;
    case "Binary":
      out.push(`B:${e.op}`);
      emitExprKey(e.left, out);
      emitExprKey(e.right, out);
      return;
    case "Unary":
      out.push(`U:${e.op}`);
      emitExprKey(e.operand, out);
      return;
    case "FuncCall":
      out.push(`C:${e.name}`);
      for (const a of e.args) emitExprKey(a, out);
      return;
    default:
      out.push(`?:${e.type}`);
  }
}
