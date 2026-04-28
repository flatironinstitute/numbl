/**
 * JS-side multi-reduction block emission.
 *
 * Detection + dispatch for scalar Assigns of the form
 *
 *   acc = <expr tree with >=2 reductions over a single tensor Var>
 *
 * On match, the JS codegen emits a block that runs a single-pass
 * inline JS reduction loop, stashes each reduction result in a local,
 * and then emits the Assign's RHS with each reduction Call substituted
 * by the local that holds its value. The RHS-substitution contract
 * lives in `jitCodegen._multiReductionSubst`, which this module
 * installs and clears around the `emitExpr` call the caller provides.
 *
 * Limited to:
 *   - sum / prod / max / min / mean (short-circuit any/all excluded)
 *   - a single tensor Var argument to every reduction
 *   - all reductions read the same tensor name
 *   - the tensor is real-typed
 *   - at least two reductions in the RHS
 *
 * NaN semantics for max/min are matched to MATLAB omit-NaN behaviour:
 * IEEE unordered compare already skips NaN, and an `any_non_nan` flag
 * drives the all-NaN → NaN fallback.
 */

import type { JitExpr, JitStmt } from "../../../jitTypes.js";

type MultiReduceOp = "sum" | "prod" | "max" | "min" | "mean";

/** Fusable reduction names. `any` / `all` stay on the default path —
 *  their short-circuit `break` doesn't fit the multi-accumulator loop. */
const FUSABLE_REDUCE_OPS = new Set<MultiReduceOp>([
  "sum",
  "prod",
  "max",
  "min",
  "mean",
]);

export interface MultiReductionMatch {
  /** The scalar Assign being emitted (target + original RHS). */
  stmt: JitStmt & { tag: "Assign" };
  /** The tensor variable name all reductions read. */
  tensorName: string;
  /** Distinct ops that actually appear, in first-occurrence order. Used
   *  to size the kernel's output buffer and decide which accumulators
   *  to declare on the JS fallback path. */
  ops: MultiReduceOp[];
  /** Every reduction Call node found in the RHS, paired with its op.
   *  The JS emitter builds a substitution map from each Call to the
   *  local that holds its result. */
  sites: { call: JitExpr & { tag: "Call" }; op: MultiReduceOp }[];
}

/** Walk the RHS and collect every qualifying reduction Call. Returns
 *  null on the first disqualifying shape (mixed tensor names, complex,
 *  non-Var arg, non-fusable op, etc.). */
function walkForReductions(
  expr: JitExpr,
  state: {
    tensorName: string | null;
    sites: MultiReductionMatch["sites"];
    seenOps: Set<MultiReduceOp>;
    orderedOps: MultiReduceOp[];
  }
): boolean {
  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
    case "ImagLiteral":
      return true;
    case "Binary":
      return (
        walkForReductions(expr.left, state) &&
        walkForReductions(expr.right, state)
      );
    case "Unary":
      return walkForReductions(expr.operand, state);
    case "Call": {
      if (FUSABLE_REDUCE_OPS.has(expr.name as MultiReduceOp)) {
        if (expr.args.length !== 1) return false;
        const a = expr.args[0];
        if (
          a.tag !== "Var" ||
          a.jitType.kind !== "tensor" ||
          a.jitType.isComplex === true
        ) {
          return false;
        }
        if (state.tensorName === null) state.tensorName = a.name;
        else if (state.tensorName !== a.name) return false;
        const op = expr.name as MultiReduceOp;
        state.sites.push({ call: expr, op });
        if (!state.seenOps.has(op)) {
          state.seenOps.add(op);
          state.orderedOps.push(op);
        }
        return true;
      }
      // Non-reduction call — fine, but recurse in case args sneak one.
      for (const arg of expr.args) {
        if (!walkForReductions(arg, state)) return false;
      }
      return true;
    }
    default:
      // Unknown node types disable fusion conservatively; the default
      // per-op path will handle the statement.
      return false;
  }
}

/** Try to match the multi-reduction pattern on a single statement. */
export function tryMatchMultiReduction(
  stmt: JitStmt
): MultiReductionMatch | null {
  if (stmt.tag !== "Assign") return null;
  // Target must be a scalar — tensor targets take a different path.
  if (stmt.expr.jitType.kind === "tensor") return null;
  if (stmt.expr.jitType.kind === "complex_or_number") return null;

  const state = {
    tensorName: null as string | null,
    sites: [] as MultiReductionMatch["sites"],
    seenOps: new Set<MultiReduceOp>(),
    orderedOps: [] as MultiReduceOp[],
  };
  if (!walkForReductions(stmt.expr, state)) return null;
  if (state.sites.length < 2) return null;
  if (state.tensorName === null) return null;

  return {
    stmt,
    tensorName: state.tensorName,
    ops: state.orderedOps,
    sites: state.sites,
  };
}

/**
 * Emit the multi-reduction block. Writes into `lines`.
 *
 * The block:
 *   1. Aliases `<tensorName>.data` to a local and reads its length.
 *   2. Emits a single-pass JS loop that updates each accumulator.
 *   3. Post-loop: `mean = sum / n`, and a NaN fixup for max/min when
 *      every input element was NaN.
 *   4. Installs `_multiReductionSubst` pointing each reduction Call at
 *      its local, emits the Assign's RHS via the caller-provided
 *      `emitExpr`, and writes the final `<target> = <rhs>;`.
 */
export function emitMultiReductionBlock(
  lines: string[],
  indent: string,
  match: MultiReductionMatch,
  mangleName: (n: string) => string,
  emitExprWithSubst: (expr: JitExpr, subst: Map<JitExpr, string>) => string
): void {
  const inner = indent + "  ";
  const tensorName = match.tensorName;
  const mangledTensor = mangleName(tensorName);

  // Pick unique names per emit site so two multi-reductions in the same
  // function don't collide.
  const id = ++_blockCounter;
  const dataAlias = `__mr${id}_data`;
  const lenAlias = `__mr${id}_n`;
  const anyFlag = `__mr${id}_any`;

  const hasSum = match.ops.includes("sum");
  const hasMean = match.ops.includes("mean");
  const hasProd = match.ops.includes("prod");
  const hasMax = match.ops.includes("max");
  const hasMin = match.ops.includes("min");
  const needSumAcc = hasSum || hasMean;
  const hasAnyNonNan = hasMax || hasMin;

  // Locals that hold each reduction's final value (including the
  // derived `mean`). These are the substitution targets.
  const locals: Record<MultiReduceOp, string> = {
    sum: `__mr${id}_sum`,
    prod: `__mr${id}_prod`,
    max: `__mr${id}_max`,
    min: `__mr${id}_min`,
    mean: `__mr${id}_mean`,
  };

  lines.push(`${indent}{`);
  lines.push(`${inner}const ${lenAlias} = ${mangledTensor}.data.length;`);
  lines.push(`${inner}const ${dataAlias} = ${mangledTensor}.data;`);

  // Predeclare the result locals so they're in scope after the
  // conditional branch merges.
  const declared: string[] = [];
  if (needSumAcc) declared.push(`${locals.sum} = 0`);
  if (hasProd) declared.push(`${locals.prod} = 1`);
  if (hasMax) declared.push(`${locals.max} = -Infinity`);
  if (hasMin) declared.push(`${locals.min} = Infinity`);
  if (hasAnyNonNan) declared.push(`${anyFlag} = 0`);
  lines.push(`${inner}let ${declared.join(", ")};`);

  // Inline JS single-pass reduction loop.
  emitJsFallbackLoop(lines, inner, dataAlias, lenAlias, anyFlag, locals, {
    needSumAcc,
    hasProd,
    hasMax,
    hasMin,
    hasAnyNonNan,
  });

  // Post-loop: NaN fixup for max/min when every element was NaN.
  if (hasMax) {
    lines.push(`${inner}if (!${anyFlag}) ${locals.max} = NaN;`);
  }
  if (hasMin) {
    lines.push(`${inner}if (!${anyFlag}) ${locals.min} = NaN;`);
  }
  // Derive mean from sum/n. Empty vector → NaN (MATLAB: mean([]) is NaN).
  if (hasMean) {
    lines.push(
      `${inner}const ${locals.mean} = ${lenAlias} > 0 ? ${locals.sum} / ${lenAlias} : NaN;`
    );
  }

  // Build the substitution map: each reduction Call in the RHS points
  // at its local. Multiple Calls to the same op (e.g. `sum(x) + sum(x)`)
  // all map to the same local.
  const subst = new Map<JitExpr, string>();
  for (const site of match.sites) subst.set(site.call, locals[site.op]);

  const lhs = mangleName(match.stmt.name);
  const rhs = emitExprWithSubst(match.stmt.expr, subst);
  lines.push(`${inner}${lhs} = ${rhs};`);
  lines.push(`${indent}}`);
}

/** Monotonic counter so two multi-reduction blocks in the same JS
 *  output have distinct local-name prefixes. Reset between compiles
 *  via `resetMultiReductionState`. */
let _blockCounter = 0;

/** Reset per-function counter so generated names stay stable between
 *  compiles of the same IR. Called from `generateJS`. */
export function resetMultiReductionState(): void {
  _blockCounter = 0;
}

function emitJsFallbackLoop(
  lines: string[],
  indent: string,
  dataAlias: string,
  lenAlias: string,
  anyFlag: string,
  locals: Record<MultiReduceOp, string>,
  flags: {
    needSumAcc: boolean;
    hasProd: boolean;
    hasMax: boolean;
    hasMin: boolean;
    hasAnyNonNan: boolean;
  }
): void {
  const loopInner = indent + "  ";
  lines.push(`${indent}for (let __i = 0; __i < ${lenAlias}; __i++) {`);
  lines.push(`${loopInner}const __v = ${dataAlias}[__i];`);
  if (flags.needSumAcc) lines.push(`${loopInner}${locals.sum} += __v;`);
  if (flags.hasProd) lines.push(`${loopInner}${locals.prod} *= __v;`);
  if (flags.hasAnyNonNan) {
    // `v === v` is false iff v is NaN. V8 doesn't fold it away (JS has
    // no `-ffast-math`), so this is the cheapest NaN test.
    lines.push(`${loopInner}if (__v === __v) {`);
    if (flags.hasMax)
      lines.push(`${loopInner}  if (__v > ${locals.max}) ${locals.max} = __v;`);
    if (flags.hasMin)
      lines.push(`${loopInner}  if (__v < ${locals.min}) ${locals.min} = __v;`);
    lines.push(`${loopInner}  ${anyFlag} = 1;`);
    lines.push(`${loopInner}}`);
  }
  lines.push(`${indent}}`);
}
