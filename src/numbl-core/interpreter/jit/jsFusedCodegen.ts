/**
 * Fused per-element loop emission for the JS-JIT.
 *
 * Given a FusibleChain (from fusion.ts), emits a single block-scoped
 * JavaScript `for` loop that evaluates all the chain's tensor assigns
 * as inline scalar expressions per element — no $h.tAdd / $h.tMul
 * helper calls, no intermediate tensor allocations.
 *
 * Tensor var references become either:
 *   - `__<name>_data[__i]`  for input params / pre-existing tensors
 *   - `__f_<name>`           for chain-produced intermediates (scalar local)
 *
 * The optional trailing reduction is absorbed as an inline accumulator
 * inside the same loop.
 */

import { BinaryOperation } from "../../parser/types.js";
import type { JitExpr } from "./jitTypes.js";
import type { FusibleChain } from "./fusion.js";
import type { ScalarOpTarget } from "./scalarEmit.js";
import {
  type FusedTarget,
  emitFusedScalarExpr,
  fusedLocal,
  findTensorParamInChain,
} from "./fusedScalarEmit.js";

// ── JS math builtin mapping ──────────────────────────────────────────

const BUILTIN_TO_JS: Record<string, string> = {
  sin: "Math.sin",
  cos: "Math.cos",
  tan: "Math.tan",
  asin: "Math.asin",
  acos: "Math.acos",
  atan: "Math.atan",
  sinh: "Math.sinh",
  cosh: "Math.cosh",
  tanh: "Math.tanh",
  asinh: "Math.asinh",
  acosh: "Math.acosh",
  atanh: "Math.atanh",
  exp: "Math.exp",
  log: "Math.log",
  log2: "Math.log2",
  log10: "Math.log10",
  sqrt: "Math.sqrt",
  abs: "Math.abs",
  floor: "Math.floor",
  ceil: "Math.ceil",
  fix: "Math.trunc",
  round: "Math.round",
  sign: "Math.sign",
  atan2: "Math.atan2",
  hypot: "Math.hypot",
  pow: "Math.pow",
  expm1: "Math.expm1",
  log1p: "Math.log1p",
  max: "Math.max",
  min: "Math.min",
};

/** Data alias for a tensor variable inside the fused block. */
function dataAlias(name: string, mangle: (n: string) => string): string {
  return `__${mangle(name)}_data`;
}

// ── Per-element op target (numeric form) ─────────────────────────────
//
// Fused bodies write results into a Float64Array. Comparisons and
// logicals must emit a numeric 0/1 (not a JS boolean) so the V8
// JIT keeps the loop body in a double-typed shape.

const JS_FUSED_OP_TARGET: ScalarOpTarget = {
  binAdd: (l, r) => `(${l} + ${r})`,
  binSub: (l, r) => `(${l} - ${r})`,
  binMul: (l, r) => `(${l} * ${r})`,
  binDiv: (l, r) => `(${l} / ${r})`,
  binPow: (l, r) => `Math.pow(${l}, ${r})`,
  binEq: (l, r) => `((${l}) === (${r}) ? 1 : 0)`,
  binNe: (l, r) => `((${l}) !== (${r}) ? 1 : 0)`,
  binLt: (l, r) => `((${l}) < (${r}) ? 1 : 0)`,
  binLe: (l, r) => `((${l}) <= (${r}) ? 1 : 0)`,
  binGt: (l, r) => `((${l}) > (${r}) ? 1 : 0)`,
  binGe: (l, r) => `((${l}) >= (${r}) ? 1 : 0)`,
  binAnd: (l, r) => `(((${l}) !== 0) && ((${r}) !== 0) ? 1 : 0)`,
  binOr: (l, r) => `(((${l}) !== 0) || ((${r}) !== 0) ? 1 : 0)`,
  unaryPlus: o => `(+${o})`,
  unaryMinus: o => `(-${o})`,
  unaryNot: o => `((${o}) === 0 ? 1 : 0)`,
  // Truthiness hooks unused in fused context — provide safe fallbacks.
  toTruthy: v => `((${v}) !== 0)`,
  condEq: (l, r) => `(${l}) === (${r})`,
  condNe: (l, r) => `(${l}) !== (${r})`,
  condLt: (l, r) => `(${l}) < (${r})`,
  condLe: (l, r) => `(${l}) <= (${r})`,
  condGt: (l, r) => `(${l}) > (${r})`,
  condGe: (l, r) => `(${l}) >= (${r})`,
  condNot: t => `!(${t})`,
  condAnd: (l, r) => `(${l}) && (${r})`,
  condOr: (l, r) => `(${l}) || (${r})`,
};

/** Build the JS fused target bound to a specific mangle function. */
function makeJsFusedTarget(mangle: (n: string) => string): FusedTarget {
  return {
    formatNumber: v => String(v),
    mangle,
    tensorElemRead: name => `${dataAlias(name, mangle)}[__i]`,
    emitBuiltinCall: (name, args) => {
      if (name in BUILTIN_TO_JS) {
        return `${BUILTIN_TO_JS[name]}(${args.join(", ")})`;
      }
      if (name === "mod") return `$h.mod(${args.join(", ")})`;
      if (name === "rem") return `((${args[0]}) % (${args[1]}))`;
      return null;
    },
  };
}

// ── Reduction helpers ────────────────────────────────────────────────

function reductionInit(reduceName: string): string {
  switch (reduceName) {
    case "sum":
    case "mean":
      return "0";
    case "prod":
      return "1";
    case "max":
      return "-Infinity";
    case "min":
      return "Infinity";
    case "any":
      return "0";
    case "all":
      return "1";
    default:
      throw new Error(`jsFusedCodegen: unknown reduction ${reduceName}`);
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
      return `if (${valueExpr} !== 0) ${accVar} = 1;`;
    case "all":
      return `if (${valueExpr} === 0) ${accVar} = 0;`;
    default:
      throw new Error(`jsFusedCodegen: unknown reduction ${reduceName}`);
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
      return `${dest} = ${dest} + ${val};`;
  }
}

// ── Public API ────────────────────────────────────────────────────────

/**
 * Collect all distinct tensor names referenced in the chain's expression
 * trees (params and chain-produced locals). We need data aliases for all
 * non-chain-produced tensors (i.e. params / pre-existing vars).
 */
function collectInputTensors(
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

/**
 * Emit a fused per-element loop for the given chain (JS backend).
 *
 * Emits a block-scoped `{ ... }` section containing:
 * 1. Data aliases for input tensors (`const __x_data = x.data;`)
 * 2. Output buffer allocation with reuse check
 * 3. A single `for` loop with inline scalar computation
 * 4. Result wrapping (`x = $h.wrapF64(__x_data, refParam.shape);`)
 */
export function emitJsFusedChain(
  lines: string[],
  indent: string,
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  paramTensors: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>,
  _localTensorNames: ReadonlySet<string>,
  mangle: (n: string) => string
): void {
  // Find a reference param for length and shape.
  const refParam = findTensorParamInChain(chain, paramTensors, allTensorVars);
  if (!refParam) return; // shouldn't happen — bail silently

  const refMangled = mangle(refParam);

  // Build the fused target bound to this backend's mangle.
  const fusedTarget = makeJsFusedTarget(mangle);

  // Determine write-back dests (same logic as C codegen).
  const lastDest = chain.assigns[chain.assigns.length - 1].destName;
  const reductionConsumes =
    chain.reduction && chain.reduction.tensorName === lastDest;

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

  // Collect input tensor names (params/pre-existing vars read by the chain).
  const inputTensors = collectInputTensors(chain, allTensorVars);

  // Open block scope.
  lines.push(`${indent}{`);
  const inner = indent + "  ";
  const loopInner = inner + "  ";

  // Length from reference param.
  lines.push(`${inner}const __len = ${refMangled}.data.length;`);

  // Data aliases for input tensors.
  for (const name of inputTensors) {
    lines.push(
      `${inner}const ${dataAlias(name, mangle)} = ${mangle(name)}.data;`
    );
  }

  // Output buffer allocation with reuse check for write-back dests.
  for (const d of writeBack) {
    const m = mangle(d);
    const da = dataAlias(d, mangle);
    lines.push(
      `${inner}const ${da} = (${m} && ${m}._rc === 1 && ${m}.data instanceof Float64Array && ${m}.data.length === __len) ? ${m}.data : $h.uninit(__len);`
    );
  }

  // Reduction accumulator init.
  const reduceAccLocal = "__f_reduce_acc";
  if (chain.reduction) {
    lines.push(
      `${inner}let ${reduceAccLocal} = ${reductionInit(chain.reduction.reduceName)};`
    );
  }

  // Track chain-produced locals.
  const chainLocals = new Set<string>();

  // Open the fused loop.
  lines.push(`${inner}for (let __i = 0; __i < __len; __i++) {`);

  for (const assign of chain.assigns) {
    const rhs = emitFusedScalarExpr(
      assign.expr,
      chainLocals,
      allTensorVars,
      JS_FUSED_OP_TARGET,
      fusedTarget
    );

    if (!chainLocals.has(assign.destName)) {
      lines.push(`${loopInner}let ${fusedLocal(assign.destName)} = ${rhs};`);
      chainLocals.add(assign.destName);
    } else {
      lines.push(`${loopInner}${fusedLocal(assign.destName)} = ${rhs};`);
    }
  }

  // Write-back to buffers.
  for (const d of writeBack) {
    lines.push(`${loopInner}${dataAlias(d, mangle)}[__i] = ${fusedLocal(d)};`);
  }

  // Inline reduction accumulate.
  if (chain.reduction) {
    const valueExpr = fusedLocal(chain.reduction.tensorName);
    lines.push(
      `${loopInner}${reductionCombine(chain.reduction.reduceName, reduceAccLocal, valueExpr)}`
    );
  }

  // Close the loop.
  lines.push(`${inner}}`);

  // Post-loop: mean division.
  if (chain.reduction && chain.reduction.reduceName === "mean") {
    lines.push(`${inner}${reduceAccLocal} /= __len;`);
  }

  // Wrap write-back buffers as RuntimeTensors.
  for (const d of writeBack) {
    lines.push(
      `${inner}${mangle(d)} = $h.wrapF64(${dataAlias(d, mangle)}, ${refMangled}.shape);`
    );
  }

  // Store reduction result.
  if (chain.reduction) {
    const accMangled = mangle(chain.reduction.accName);
    if (chain.reduction.hasAccumulate && chain.reduction.accOp !== undefined) {
      lines.push(
        `${inner}${accumulateOp(chain.reduction.accOp, accMangled, reduceAccLocal)}`
      );
    } else {
      lines.push(`${inner}${accMangled} = ${reduceAccLocal};`);
    }
  }

  // Close block scope.
  lines.push(`${indent}}`);
}
