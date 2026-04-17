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

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { JitExpr } from "./jitTypes.js";
import type { FusibleChain } from "./fusion.js";
import {
  FUSIBLE_TENSOR_UNARY_OPS,
  FUSIBLE_TENSOR_BINARY_OPS,
} from "./fusionOps.js";

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

/** Scalar local name for a chain-produced tensor intermediate. */
function fusedLocal(name: string): string {
  return `__f_${name}`;
}

/** Data alias for a tensor variable inside the fused block. */
function dataAlias(name: string, mangle: (n: string) => string): string {
  return `__${mangle(name)}_data`;
}

// ── Expression emission (per-element scalar form) ────────────────────

function emitScalarExpr(
  expr: JitExpr,
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  mangle: (n: string) => string
): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return String(expr.value);

    case "Var": {
      if (expr.jitType.kind === "tensor" || allTensorVars.has(expr.name)) {
        if (chainLocals.has(expr.name)) return fusedLocal(expr.name);
        return `${dataAlias(expr.name, mangle)}[__i]`;
      }
      return mangle(expr.name);
    }

    case "Binary":
      return emitBinaryScalar(expr, chainLocals, allTensorVars, mangle);

    case "Unary":
      return emitUnaryScalar(expr, chainLocals, allTensorVars, mangle);

    case "Call":
      return emitCallScalar(expr, chainLocals, allTensorVars, mangle);

    default:
      throw new Error(
        `jsFusedCodegen: unsupported expr in fused chain: ${expr.tag}`
      );
  }
}

function emitBinaryScalar(
  expr: JitExpr & { tag: "Binary" },
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  mangle: (n: string) => string
): string {
  const l = emitScalarExpr(expr.left, chainLocals, allTensorVars, mangle);
  const r = emitScalarExpr(expr.right, chainLocals, allTensorVars, mangle);

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
      return `Math.pow(${l}, ${r})`;
    case BinaryOperation.Equal:
      return `((${l}) === (${r}) ? 1 : 0)`;
    case BinaryOperation.NotEqual:
      return `((${l}) !== (${r}) ? 1 : 0)`;
    case BinaryOperation.Less:
      return `((${l}) < (${r}) ? 1 : 0)`;
    case BinaryOperation.LessEqual:
      return `((${l}) <= (${r}) ? 1 : 0)`;
    case BinaryOperation.Greater:
      return `((${l}) > (${r}) ? 1 : 0)`;
    case BinaryOperation.GreaterEqual:
      return `((${l}) >= (${r}) ? 1 : 0)`;
    case BinaryOperation.AndAnd:
      return `(((${l}) !== 0) && ((${r}) !== 0) ? 1 : 0)`;
    case BinaryOperation.OrOr:
      return `(((${l}) !== 0) || ((${r}) !== 0) ? 1 : 0)`;
    default:
      throw new Error(`jsFusedCodegen: unsupported binary op ${expr.op}`);
  }
}

function emitUnaryScalar(
  expr: JitExpr & { tag: "Unary" },
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  mangle: (n: string) => string
): string {
  const operand = emitScalarExpr(
    expr.operand,
    chainLocals,
    allTensorVars,
    mangle
  );
  switch (expr.op) {
    case UnaryOperation.Plus:
      return `(+${operand})`;
    case UnaryOperation.Minus:
      return `(-${operand})`;
    case UnaryOperation.Not:
      return `((${operand}) === 0 ? 1 : 0)`;
    default:
      throw new Error(`jsFusedCodegen: unsupported unary op ${expr.op}`);
  }
}

function emitCallScalar(
  expr: JitExpr & { tag: "Call" },
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  mangle: (n: string) => string
): string {
  if (
    (expr.jitType.kind === "tensor" &&
      (FUSIBLE_TENSOR_UNARY_OPS.has(expr.name) ||
        FUSIBLE_TENSOR_BINARY_OPS.has(expr.name))) ||
    expr.name in BUILTIN_TO_JS
  ) {
    const jsName = BUILTIN_TO_JS[expr.name];
    if (!jsName) {
      throw new Error(`jsFusedCodegen: unmapped builtin ${expr.name}`);
    }
    const args = expr.args.map(a =>
      emitScalarExpr(a, chainLocals, allTensorVars, mangle)
    );
    return `${jsName}(${args.join(", ")})`;
  }
  // mod and rem are special — not in Math.*
  if (expr.name === "mod") {
    const args = expr.args.map(a =>
      emitScalarExpr(a, chainLocals, allTensorVars, mangle)
    );
    return `$h.mod(${args.join(", ")})`;
  }
  if (expr.name === "rem") {
    const args = expr.args.map(a =>
      emitScalarExpr(a, chainLocals, allTensorVars, mangle)
    );
    return `((${args[0]}) % (${args[1]}))`;
  }
  throw new Error(
    `jsFusedCodegen: unsupported call in fused chain: ${expr.name}`
  );
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
 * Find the first tensor param name referenced in the chain's expressions.
 * Used to determine the loop length and shape for output wrapping.
 */
function findRefParam(
  chain: FusibleChain,
  paramTensors: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): string | null {
  for (const a of chain.assigns) {
    const found = findTensorParamInExpr(a.expr, paramTensors, allTensorVars);
    if (found) return found;
  }
  return null;
}

function findTensorParamInExpr(
  expr: JitExpr,
  paramTensors: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>
): string | null {
  if (expr.tag === "Var" && allTensorVars.has(expr.name)) {
    if (paramTensors.has(expr.name)) return expr.name;
    return null;
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
  const refParam = findRefParam(chain, paramTensors, allTensorVars);
  if (!refParam) return; // shouldn't happen — bail silently

  const refMangled = mangle(refParam);

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
    const rhs = emitScalarExpr(assign.expr, chainLocals, allTensorVars, mangle);

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
