/**
 * e2 — multi-reduction driver.
 *
 * Handles a scalar `Assign` whose RHS contains TWO or more reduction
 * calls (`sum`, `prod`, `max`, `min`, `mean`) over the same single
 * tensor variable, e.g.
 *
 *     red_acc = red_acc + sum(x) + mean(x) + max(x) + min(x);
 *
 * The default interpreter path makes one pass through the tensor per
 * reduction (4× the memory traffic of the optimal). The e2 driver
 * detects the pattern, compiles ONE kernel that computes every
 * requested reduction in a single pass, and substitutes the reduction
 * subtrees in the RHS with the kernel's scalar outputs before
 * evaluating the residual expression.
 *
 * Reuses [e1/multiReductionKernel.ts](../e1/multiReductionKernel.ts)
 * for the C emission (same shape works for both backends).
 */

import type { Expr, Stmt, Span } from "../../parser/types.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import { RuntimeError } from "../../runtime/error.js";
import { type RuntimeTensor, isRuntimeTensor } from "../../runtime/types.js";
import {
  emitMultiReductionKernel,
  type MultiReduceOp,
} from "../../jit/multiReductionKernel.js";
import { getE2CompileFn, e2MinElems } from "./compileFn.js";
import { isOpenmpAvailable } from "../../jit/openmpFlag.js";

/** Reductions handled by the multi-reduction kernel. `any` / `all` are
 *  excluded for the same reason e1's kernel excludes them: their
 *  short-circuit semantics would interfere with running the other
 *  accumulators on the same loop. */
const MULTI_REDUCE_OPS: ReadonlySet<string> = new Set([
  "sum",
  "prod",
  "max",
  "min",
  "mean",
]);

interface ReductionSite {
  /** The original AST subtree (FuncCall node). */
  subtree: Expr;
  reduceName: MultiReduceOp;
  tensorName: string;
}

/** Walk `expr` and collect all top-level reduction-of-single-Var
 *  subtrees. Stops descending into a subtree once we've matched it
 *  (a nested `sum(sum(x))` doesn't make sense for our purpose). */
function findReductions(expr: Expr, out: ReductionSite[]): void {
  if (
    expr.type === "FuncCall" &&
    MULTI_REDUCE_OPS.has(expr.name) &&
    expr.args.length === 1 &&
    expr.args[0].type === "Ident"
  ) {
    out.push({
      subtree: expr,
      reduceName: expr.name as MultiReduceOp,
      tensorName: expr.args[0].name,
    });
    return; // don't descend further into an already-matched call
  }
  switch (expr.type) {
    case "Binary":
      findReductions(expr.left, out);
      findReductions(expr.right, out);
      return;
    case "Unary":
      findReductions(expr.operand, out);
      return;
    case "FuncCall":
      for (const a of expr.args) findReductions(a, out);
      return;
    case "Range":
      findReductions(expr.start, out);
      if (expr.step) findReductions(expr.step, out);
      findReductions(expr.end, out);
      return;
    case "Index":
    case "IndexCell":
      findReductions(expr.base, out);
      for (const i of expr.indices) findReductions(i, out);
      return;
    default:
      return;
  }
}

/** Build a NumberLiteral AST node for substitution. `String(v)` →
 *  `parseFloat(...)` round-trips finite doubles exactly; it also
 *  round-trips `NaN` and `±Infinity`, which covers every value our
 *  reduction kernel can produce. The span is reused from the reduction
 *  call so source-mapped errors point at the original site. */
function makeNumberNode(value: number, span: Span): Expr {
  return { type: "Number", value: String(value), span };
}

/** Substitute every Expr subtree found in `replacements` with the
 *  corresponding NumberLiteral. Returns a new Expr tree (does not
 *  mutate the input). */
function substituteExpr(expr: Expr, replacements: Map<Expr, number>): Expr {
  if (replacements.has(expr)) {
    return makeNumberNode(replacements.get(expr)!, expr.span);
  }
  switch (expr.type) {
    case "Binary":
      return {
        ...expr,
        left: substituteExpr(expr.left, replacements),
        right: substituteExpr(expr.right, replacements),
      };
    case "Unary":
      return { ...expr, operand: substituteExpr(expr.operand, replacements) };
    case "FuncCall":
      return {
        ...expr,
        args: expr.args.map(a => substituteExpr(a, replacements)),
      };
    case "Range":
      return {
        ...expr,
        start: substituteExpr(expr.start, replacements),
        step: expr.step ? substituteExpr(expr.step, replacements) : null,
        end: substituteExpr(expr.end, replacements),
      };
    case "Index":
    case "IndexCell":
      return {
        ...expr,
        base: substituteExpr(expr.base, replacements),
        indices: expr.indices.map(i => substituteExpr(i, replacements)),
      };
    default:
      return expr;
  }
}

/** Per-Stmt cache: Map<sig, { fn, slotNames, hasAnyNonNan, ops }>. */
interface CacheEntry {
  fn: (...args: unknown[]) => unknown;
  slotNames: string[];
  hasAnyNonNan: boolean;
  /** Distinct reduction ops in the kernel (insertion order). */
  ops: MultiReduceOp[];
}
const stmtCache = new WeakMap<Stmt, Map<string, CacheEntry | "BAILED">>();

export function tryE2MultiReduction(
  interp: Interpreter,
  stmt: Stmt & { type: "Assign" }
): boolean {
  const reductions: ReductionSite[] = [];
  findReductions(stmt.expr, reductions);
  if (reductions.length < 2) return false;

  // All reductions must reduce over the SAME single tensor variable.
  const tensorName = reductions[0].tensorName;
  for (let k = 1; k < reductions.length; k++) {
    if (reductions[k].tensorName !== tensorName) return false;
  }

  // Validate the tensor at runtime.
  const tensorVal = interp.env.get(tensorName);
  if (!tensorVal || !isRuntimeTensor(tensorVal)) return false;
  const t = tensorVal as RuntimeTensor;
  if (t.imag) return false;
  if (!(t.data instanceof Float64Array)) return false;
  const n = t.data.length;
  if (n < e2MinElems()) return false;

  // Distinct ops in source order.
  const seen = new Set<MultiReduceOp>();
  const ops: MultiReduceOp[] = [];
  for (const r of reductions) {
    if (!seen.has(r.reduceName)) {
      seen.add(r.reduceName);
      ops.push(r.reduceName);
    }
  }
  // Need at least 2 distinct ops to win over the per-op interpreter
  // path. (4× sum(x) is one pass either way.)
  if (ops.length < 2) return false;

  const par = interp.par && isOpenmpAvailable();
  const sig = `t=${tensorName}|ops=${ops.slice().sort().join(",")}|par=${par ? "1" : "0"}`;
  let bucket = stmtCache.get(stmt);
  if (!bucket) {
    bucket = new Map();
    stmtCache.set(stmt, bucket);
  }
  let entry = bucket.get(sig);
  if (entry === "BAILED") return false;

  if (!entry) {
    const info = emitMultiReductionKernel(ops, par);
    const compile = getE2CompileFn();
    let fn;
    try {
      fn = compile(info.cSource, info.koffiSig, info.kernelName, msg =>
        process.stderr.write(`[e2] ${msg}\n`)
      );
    } catch (e) {
      throw new RuntimeError(
        `--opt e2: multi-reduction kernel compilation failed for ${stmt.name}: ${
          e instanceof Error ? e.message : String(e)
        }`
      );
    }
    if (!fn) {
      throw new RuntimeError(
        `--opt e2: multi-reduction kernel compilation failed for ${stmt.name}`
      );
    }
    entry = {
      fn,
      slotNames: info.slotNames,
      hasAnyNonNan: info.hasAnyNonNan,
      ops,
    };
    bucket.set(sig, entry);

    const file = stmt.span?.file ?? interp.currentFile;
    const line = interp.rt.$line ?? 0;
    interp.onCCompile?.(
      [
        `e2 kernel: multi-reduction ${ops.join("+")}(${tensorName}) into ${stmt.name} @ ${file}:${line}`,
        `inputs:  ${tensorName}: tensor double[${n}]`,
        `outputs: ${ops.length} scalars (one per reduction)`,
      ].join("\n * "),
      info.cSource
    );
  }

  // Run the kernel. Output buffer holds one Float64 slot per slotName.
  const buf = new Float64Array(entry.slotNames.length);
  entry.fn(n, t.data, buf);

  const slotIdx = new Map<string, number>();
  for (let i = 0; i < entry.slotNames.length; i++) {
    slotIdx.set(entry.slotNames[i], i);
  }
  const allNan = entry.hasAnyNonNan && buf[slotIdx.get("any_non_nan")!] === 0;

  // Map each ORIGINAL reduction subtree to its scalar value.
  const replacements = new Map<Expr, number>();
  for (const r of reductions) {
    let val: number;
    switch (r.reduceName) {
      case "sum":
        val = buf[slotIdx.get("sum")!];
        break;
      case "mean":
        val = buf[slotIdx.get("sum")!] / n;
        break;
      case "prod":
        val = buf[slotIdx.get("prod")!];
        break;
      case "max":
        val = allNan ? NaN : buf[slotIdx.get("max")!];
        // The kernel can't produce non-finite mid-loop values without
        // -ffast-math caveats; the ALL-NaN case overrides to NaN.
        break;
      case "min":
        val = allNan ? NaN : buf[slotIdx.get("min")!];
        break;
    }
    replacements.set(r.subtree, val);
  }

  // Substitute the reduction subtrees and let the interpreter evaluate
  // the residual scalar expression. This handles arbitrary surrounding
  // arithmetic (e.g. `red_acc + (...) - 1.0 * (...)`) without any
  // duplication of the interpreter's eval logic.
  const newRhs = substituteExpr(stmt.expr, replacements);
  const raw = interp.evalExpr(newRhs);
  const val = Array.isArray(raw) ? raw[0] : raw;
  const rv = ensureRuntimeValue(val) as RuntimeValue;
  interp.env.set(stmt.name, rv);
  interp.ans = rv;
  if (!stmt.suppressed) {
    interp.rt.displayAssign(stmt.name, rv);
  }
  return true;
}
