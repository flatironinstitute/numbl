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

import { BinaryOperation, UnaryOperation } from "../../../parser/types.js";
import type { JitExpr } from "../jitTypes.js";
import type { FusibleChain } from "../fusion.js";
import { FUSIBLE_TENSOR_UNARY_OPS } from "../fusionOps.js";

const MANGLE_PREFIX = "v_";

function mangle(name: string): string {
  return `${MANGLE_PREFIX}${name}`;
}

function tensorData(name: string): string {
  return `${mangle(name)}_data`;
}

function tensorLen(name: string): string {
  return `${mangle(name)}_len`;
}

/** Format a number literal for C source. */
function fmtNum(v: number): string {
  if (!Number.isFinite(v)) {
    if (Number.isNaN(v)) return "(0.0/0.0)";
    return v > 0 ? "(1.0/0.0)" : "(-1.0/0.0)";
  }
  if (Number.isInteger(v)) return `${v}.0`;
  return `${v}`;
}

/** Scalar C name for a chain-produced tensor intermediate. */
function fusedLocal(name: string): string {
  return `__f_${name}`;
}

// ── Scalar math builtins (mirrors jitCodegenC.ts) ────────────────────

const BUILTIN_TO_C: Record<string, string> = {
  sin: "sin",
  cos: "cos",
  tan: "tan",
  asin: "asin",
  acos: "acos",
  atan: "atan",
  sinh: "sinh",
  cosh: "cosh",
  tanh: "tanh",
  asinh: "asinh",
  acosh: "acosh",
  atanh: "atanh",
  exp: "exp",
  log: "log",
  log2: "log2",
  log10: "log10",
  sqrt: "sqrt",
  abs: "fabs",
  floor: "floor",
  ceil: "ceil",
  fix: "trunc",
  round: "round",
  atan2: "atan2",
  hypot: "hypot",
  rem: "fmod",
  expm1: "expm1",
  log1p: "log1p",
  pow: "pow",
  mod: "__numbl_mod",
  sign: "__numbl_sign",
};

// ── Expression emission (per-element scalar form) ────────────────────

/**
 * Emit a JitExpr as a C scalar expression for element `__i`.
 *
 * `chainLocals` is the set of tensor variable names that have already
 * been assigned within the current fused chain — these are read via
 * the scalar local `__f_name` rather than `v_name_data[__i]`.
 *
 * `allTensorVars` is the full set of tensor-typed variables so we can
 * distinguish tensor reads from scalar reads.
 *
 * `helpersNeeded` collects names of scalar helpers that need to be
 * emitted (e.g. __numbl_mod).
 */
function emitScalarExpr(
  expr: JitExpr,
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  helpersNeeded: Set<string>
): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return fmtNum(expr.value);

    case "Var": {
      if (expr.jitType.kind === "tensor" || allTensorVars.has(expr.name)) {
        // Chain-produced intermediate → scalar local
        if (chainLocals.has(expr.name)) return fusedLocal(expr.name);
        // Input param / pre-existing tensor → array read
        return `${tensorData(expr.name)}[__i]`;
      }
      return mangle(expr.name);
    }

    case "Binary":
      return emitBinaryScalar(expr, chainLocals, allTensorVars, helpersNeeded);

    case "Unary":
      return emitUnaryScalar(expr, chainLocals, allTensorVars, helpersNeeded);

    case "Call":
      return emitCallScalar(expr, chainLocals, allTensorVars, helpersNeeded);

    default:
      throw new Error(
        `cFusedCodegen: unsupported expr in fused chain: ${expr.tag}`
      );
  }
}

function emitBinaryScalar(
  expr: JitExpr & { tag: "Binary" },
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  helpersNeeded: Set<string>
): string {
  const l = emitScalarExpr(
    expr.left,
    chainLocals,
    allTensorVars,
    helpersNeeded
  );
  const r = emitScalarExpr(
    expr.right,
    chainLocals,
    allTensorVars,
    helpersNeeded
  );

  switch (expr.op) {
    case BinaryOperation.Add:
      return `(${l} + ${r})`;
    case BinaryOperation.Sub:
      return `(${l} - ${r})`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `(${l} * ${r})`;
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return `(${l} / ${r})`;
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return `pow(${l}, ${r})`;
    case BinaryOperation.Equal:
      return `((double)((${l}) == (${r})))`;
    case BinaryOperation.NotEqual:
      return `((double)((${l}) != (${r})))`;
    case BinaryOperation.Less:
      return `((double)((${l}) < (${r})))`;
    case BinaryOperation.LessEqual:
      return `((double)((${l}) <= (${r})))`;
    case BinaryOperation.Greater:
      return `((double)((${l}) > (${r})))`;
    case BinaryOperation.GreaterEqual:
      return `((double)((${l}) >= (${r})))`;
    case BinaryOperation.AndAnd:
      return `((double)(((${l}) != 0.0) && ((${r}) != 0.0)))`;
    case BinaryOperation.OrOr:
      return `((double)(((${l}) != 0.0) || ((${r}) != 0.0)))`;
    default:
      throw new Error(`cFusedCodegen: unsupported binary op ${expr.op}`);
  }
}

function emitUnaryScalar(
  expr: JitExpr & { tag: "Unary" },
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  helpersNeeded: Set<string>
): string {
  const operand = emitScalarExpr(
    expr.operand,
    chainLocals,
    allTensorVars,
    helpersNeeded
  );
  switch (expr.op) {
    case UnaryOperation.Plus:
      return `(+${operand})`;
    case UnaryOperation.Minus:
      return `(-${operand})`;
    case UnaryOperation.Not:
      return `((double)((${operand}) == 0.0))`;
    default:
      throw new Error(`cFusedCodegen: unsupported unary op ${expr.op}`);
  }
}

function emitCallScalar(
  expr: JitExpr & { tag: "Call" },
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  helpersNeeded: Set<string>
): string {
  // Tensor unary call → becomes scalar math call on the per-element value
  if (
    (expr.jitType.kind === "tensor" &&
      FUSIBLE_TENSOR_UNARY_OPS.has(expr.name)) ||
    expr.name in BUILTIN_TO_C
  ) {
    const cName = BUILTIN_TO_C[expr.name];
    if (!cName) {
      throw new Error(`cFusedCodegen: unmapped builtin ${expr.name}`);
    }
    if (expr.name === "mod" || expr.name === "sign") {
      helpersNeeded.add(expr.name);
    }
    const args = expr.args.map(a =>
      emitScalarExpr(a, chainLocals, allTensorVars, helpersNeeded)
    );
    return `${cName}(${args.join(", ")})`;
  }
  throw new Error(
    `cFusedCodegen: unsupported call in fused chain: ${expr.name}`
  );
}

// ── Reduction init / combine helpers ─────────────────────────────────

function reductionInit(reduceName: string): string {
  switch (reduceName) {
    case "sum":
    case "mean":
      return "0.0";
    case "prod":
      return "1.0";
    case "max":
      return "(-1.0/0.0)"; // -Inf
    case "min":
      return "(1.0/0.0)"; // +Inf
    case "any":
      return "0.0";
    case "all":
      return "1.0";
    default:
      throw new Error(`cFusedCodegen: unknown reduction ${reduceName}`);
  }
}

function reductionCombine(
  reduceName: string,
  accVar: string,
  valueExpr: string
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
      return `if (${valueExpr} != 0.0) ${accVar} = 1.0;`;
    case "all":
      return `if (${valueExpr} == 0.0) ${accVar} = 0.0;`;
    default:
      throw new Error(`cFusedCodegen: unknown reduction ${reduceName}`);
  }
}

function accumulateOp(op: BinaryOperation, dest: string, val: string): string {
  switch (op) {
    case BinaryOperation.Add:
      return `${dest} += ${val};`;
    case BinaryOperation.Sub:
      return `${dest} -= ${val};`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `${dest} *= ${val};`;
    default:
      return `${dest} = ${dest} + ${val};`; // fallback
  }
}

// ── Public API ────────────────────────────────────────────────────────

/**
 * Emit a fused per-element loop for the given chain.
 *
 * Appends C source lines to `lines`. Returns the set of helper
 * function names needed (e.g. "__numbl_mod") so the caller can emit
 * their definitions.
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
  localTensorNames: ReadonlySet<string>
): Set<string> {
  const helpersNeeded = new Set<string>();

  // Determine the length variable — use the first tensor param referenced.
  const lenVar = findLenVar(chain, paramTensors, allTensorVars);

  // Determine which dest names need a write-back to their buffer.
  // A dest needs write-back if:
  //   1. It's an output or param-output tensor, OR
  //   2. It's read after the chain (we conservatively always write back
  //      unless the reduction fully consumes it)
  // For simplicity: write back all distinct dest names EXCEPT one that
  // is consumed solely by the trailing reduction.
  const lastDest = chain.assigns[chain.assigns.length - 1].destName;
  const reductionConsumes =
    chain.reduction && chain.reduction.tensorName === lastDest;

  // Collect distinct dest names that appear in the chain.
  const destNames = new Set<string>();
  for (const a of chain.assigns) destNames.add(a.destName);

  // Determine which dests need write-back.
  const writeBack = new Set<string>();
  for (const d of destNames) {
    if (reductionConsumes && d === lastDest) {
      // Check if this dest is ALSO an output — if so, still write back.
      if (outputTensorNames.has(d)) {
        writeBack.add(d);
      }
      // Otherwise skip — the reduction consumes it.
    } else {
      writeBack.add(d);
    }
  }

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
      `${indent}double ${reduceAccLocal} = ${reductionInit(chain.reduction.reduceName)};`
    );
  }

  // Track which tensor vars have been produced by earlier assigns in
  // the chain — these are read via scalar locals, not array reads.
  const chainLocals = new Set<string>();

  // Open the fused loop.
  if (!chain.reduction) {
    lines.push(`${indent}#pragma omp simd`);
  }
  lines.push(`${indent}for (int64_t __i = 0; __i < ${lenVar}; __i++) {`);
  const inner = indent + "  ";

  for (const assign of chain.assigns) {
    const rhs = emitScalarExpr(
      assign.expr,
      chainLocals,
      allTensorVars,
      helpersNeeded
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
      `${inner}${reductionCombine(chain.reduction.reduceName, reduceAccLocal, valueExpr)}`
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

  return helpersNeeded;
}

/**
 * Find a tensor length C expression to use as the loop bound.
 * Walks the first chain assign's expression tree to find a tensor param
 * reference, then returns its `_len` variable.
 */
function findLenVar(
  chain: FusibleChain,
  paramTensors: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): string {
  // Try to find a param tensor reference in the chain's expressions.
  for (const a of chain.assigns) {
    const found = findTensorParamInExpr(a.expr, paramTensors, allTensorVars);
    if (found) return tensorLen(found);
  }
  // Fallback: use the first dest's length (it must be set from somewhere).
  return tensorLen(chain.assigns[0].destName);
}

function findTensorParamInExpr(
  expr: JitExpr,
  paramTensors: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): string | null {
  if (expr.tag === "Var" && allTensorVars.has(expr.name)) {
    if (paramTensors.has(expr.name)) return expr.name;
    return null; // chain-produced, doesn't have a _len
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
