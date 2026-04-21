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

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { JitExpr, JitType } from "../jitTypes.js";
import type { FusibleChain } from "../fusion.js";
import {
  type FusedTarget,
  emitFusedScalarExpr,
  fusedLocal,
  findTensorParamInChain,
  collectInputTensors,
} from "../fusedScalarEmit.js";
import {
  C_SCALAR_TARGET,
  formatNumberLiteral,
  mangle,
  mangleIm,
  tensorD0,
  tensorD1,
  tensorData,
  tensorDataIm,
  tensorLen,
} from "./jitCodegenC.js";
import { shapeExprsFor } from "./emit.js";
import { getIBuiltin } from "../../interpreter/builtins/types.js";
import {
  C_REDUCTION_LITERALS,
  accumulateOp,
  determineWriteBack,
  reductionCombine,
  reductionInit,
} from "../fusedChainHelpers.js";

/** Minimum element count before `#pragma omp parallel for` kicks in.
 *  Below this, thread-spawn overhead dominates. */
const OMP_PARALLEL_THRESHOLD =
  parseInt(process.env.NUMBL_OMP_THRESHOLD || "", 10) || 100_000;

/** Builtins that are expensive enough per element to justify thread-spawn overhead. */
const HEAVY_OPS = new Set([
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "acosh",
  "atanh",
  "exp",
  "expm1",
  "log",
  "log1p",
  "log2",
  "log10",
  "pow",
  "atan2",
  "hypot",
]);

/** Count heavy (transcendental) ops in an expression tree. */
function countHeavyOps(expr: JitExpr): number {
  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
      return 0;
    case "Binary":
      return (
        (expr.op === BinaryOperation.Pow || expr.op === BinaryOperation.ElemPow
          ? 1
          : 0) +
        countHeavyOps(expr.left) +
        countHeavyOps(expr.right)
      );
    case "Unary":
      return countHeavyOps(expr.operand);
    case "Call":
      return (
        (HEAVY_OPS.has(expr.name) ? 1 : 0) +
        expr.args.reduce((n, a) => n + countHeavyOps(a), 0)
      );
    default:
      return 0;
  }
}

// ── Fused target (per-element leaves) ────────────────────────────────
//
// The op switches (binary/unary) reuse C_SCALAR_TARGET — C's value
// form is already numeric (booleans are `(double)` casts), so the
// same target works in both value and per-element contexts.
//
// Function-call leaves consult each builtin's own `jitEmitC` so the
// C function-name mapping lives with the builtin, not in a central
// table here. Fused context is per-element, so argTypes are
// fabricated as scalar numbers — all fusible ops (per fusionOps.ts)
// are element-wise, and none carry a `requireNonneg` guard.

const C_FUSED_TARGET: FusedTarget = {
  formatNumber: formatNumberLiteral,
  mangle,
  tensorElemRead: name => `${tensorData(name)}[__i]`,
  emitBuiltinCall: (name, args) => {
    const ib = getIBuiltin(name);
    if (!ib?.jitEmitC) return null;
    const argTypes: JitType[] = args.map(() => ({ kind: "number" }));
    return ib.jitEmitC(args, argTypes);
  },
};

// ── Public API ────────────────────────────────────────────────────────

/**
 * Emit a fused per-element loop for the given chain.
 *
 * Appends C source lines to `lines`. All scalar math helpers the inner
 * body may reference (mod, sign, ...) live in jit_runtime.a, so this
 * function no longer reports back "helpers needed" — the emitter simply
 * calls them as library symbols.
 *
 * Dispatches to the real or complex per-element emitter based on whether
 * any assign in the chain produces a complex tensor. Both paths share
 * the outer shell (lenVar, writeBack, parallel-for decision, loop open);
 * only the buffer sizing, per-element emission, and write-back differ.
 *
 * `allTensorVars` is the full set of tensor-typed variable names.
 * `paramTensors` is the subset that are input parameters.
 * `outputTensorNames` is the subset that are function outputs.
 * `localTensorNames` is the subset that are non-param, non-output locals.
 * `complexTensorNames` is the subset whose tensor has a paired imag buffer.
 * `complexScalarVars` is the set of scalar vars that hold complex values
 * (pair-of-doubles `v_name` / `__im_v_name`).
 */
export function emitFusedChain(
  lines: string[],
  indent: string,
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  paramTensors: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>,
  localTensorNames: ReadonlySet<string>,
  dynamicOutputNames: ReadonlySet<string>,
  complexTensorNames: ReadonlySet<string>,
  complexScalarVars: ReadonlySet<string>,
  openmp?: boolean
): void {
  if (chainIsComplex(chain, complexTensorNames)) {
    emitComplexFusedChain(
      lines,
      indent,
      chain,
      allTensorVars,
      paramTensors,
      outputTensorNames,
      localTensorNames,
      dynamicOutputNames,
      complexTensorNames,
      complexScalarVars
    );
    return;
  }
  emitRealFusedChain(
    lines,
    indent,
    chain,
    allTensorVars,
    paramTensors,
    outputTensorNames,
    localTensorNames,
    dynamicOutputNames,
    openmp
  );
}

/** True when any assign in the chain writes (or reads through Var) a
 *  complex tensor. The chain was already detected as pure element-wise,
 *  so we only need to check the destNames and Var references. */
function chainIsComplex(
  chain: FusibleChain,
  complexTensorNames: ReadonlySet<string>
): boolean {
  for (const a of chain.assigns) {
    if (complexTensorNames.has(a.destName)) return true;
    if (a.expr.jitType.kind === "tensor" && a.expr.jitType.isComplex === true)
      return true;
  }
  return false;
}

function emitRealFusedChain(
  lines: string[],
  indent: string,
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  paramTensors: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>,
  localTensorNames: ReadonlySet<string>,
  dynamicOutputNames: ReadonlySet<string>,
  openmp?: boolean
): void {
  // Determine the length variable — prefer a formal param tensor; if
  // the chain reads no params (e.g. function bodies with only local
  // tensors), fall back to any source tensor read by the chain. Using
  // a destination's length would read 0 on first entry and produce an
  // empty loop.
  const refTensor =
    findTensorParamInChain(chain, paramTensors, allTensorVars) ??
    [...collectInputTensors(chain, allTensorVars)][0];
  const lenVar = tensorLen(refTensor ?? chain.assigns[0].destName);

  // Determine which dest names need a write-back to their buffer
  // (shared with JS codegen).
  const { writeBack } = determineWriteBack(chain, outputTensorNames);

  // For each write-back dest, size its buffer to `lenVar` before the loop.
  // Locals and dynamic outputs need free+malloc so a later chain firing
  // (with different `lenVar`) doesn't overflow a stale buffer. Fixed-size
  // outputs share the caller's buffer — we must not free it.
  for (const d of writeBack) {
    const needsRealloc = localTensorNames.has(d) || dynamicOutputNames.has(d);
    lines.push(`${indent}${tensorLen(d)} = ${lenVar};`);
    if (needsRealloc) {
      lines.push(`${indent}if (${tensorData(d)}) free(${tensorData(d)});`);
      lines.push(
        `${indent}${tensorData(d)} = (${tensorLen(d)} > 0) ? (double *)malloc((size_t)${tensorLen(d)} * sizeof(double)) : NULL;`
      );
    }
  }

  // Emit reduction accumulator init.
  const reduceAccLocal = "__f_reduce_acc";
  if (chain.reduction) {
    lines.push(
      `${indent}double ${reduceAccLocal} = ${reductionInit(chain.reduction.reduceName, C_REDUCTION_LITERALS)};`
    );
  }

  // Track which tensor vars have been produced by earlier assigns in
  // the chain — these are read via scalar locals, not array reads.
  const chainLocals = new Set<string>();

  // Open the fused loop.
  // Conditions for parallel-for:
  //  1. Writes to output/param tensors (not just local temporaries
  //     consumed by subsequent per-op code — parallelizing those scatters
  //     data across caches and hurts the sequential consumer).
  //  2. Chain body has transcendental ops (sin, exp, etc.) so the
  //     per-element compute justifies thread-spawn overhead.
  const writesToOutput = [...writeBack].some(
    d => outputTensorNames.has(d) || paramTensors.has(d)
  );
  const heavyOps = chain.assigns.reduce((n, a) => n + countHeavyOps(a.expr), 0);
  if (!chain.reduction) {
    if (openmp && writesToOutput && heavyOps > 0) {
      lines.push(
        `${indent}#pragma omp parallel for simd if(${lenVar} >= ${OMP_PARALLEL_THRESHOLD})`
      );
    } else {
      lines.push(`${indent}#pragma omp simd`);
    }
  }
  lines.push(`${indent}for (int64_t __i = 0; __i < ${lenVar}; __i++) {`);
  const inner = indent + "  ";

  for (const assign of chain.assigns) {
    const rhs = emitFusedScalarExpr(
      assign.expr,
      chainLocals,
      allTensorVars,
      C_SCALAR_TARGET,
      C_FUSED_TARGET
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
      `${inner}${reductionCombine(chain.reduction.reduceName, reduceAccLocal, valueExpr, C_REDUCTION_LITERALS)}`
    );
  }

  // Close the loop.
  lines.push(`${indent}}`);

  // Update tensor lengths for write-back dests.
  for (const d of writeBack) {
    lines.push(`${indent}${tensorLen(d)} = ${lenVar};`);
  }

  // Propagate dynamic-output shape so size(y) ≠ [0 0].
  for (const d of writeBack) {
    if (!dynamicOutputNames.has(d)) continue;
    let shapeSrc: JitExpr | undefined;
    for (const a of chain.assigns) if (a.destName === d) shapeSrc = a.expr;
    const [d0Expr, d1Expr] = shapeExprsFor(shapeSrc, lenVar);
    lines.push(`${indent}${tensorD0(d)} = ${d0Expr};`);
    lines.push(`${indent}${tensorD1(d)} = ${d1Expr};`);
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
}

// ── Complex-tensor fused chain ───────────────────────────────────────
//
// Mirrors emitRealFusedChain but threads paired (re, im) buffers
// through every step. Reductions are not absorbed — fusion.ts drops the
// trailing reduction for complex chains because the scalar accumulator
// in the fused-loop helpers can't hold a complex value. The caller
// emits the reduction via per-op code after the fused loop runs.

/** Per-element complex value: two C scalar expression strings. */
interface ComplexPerElem {
  re: string;
  im: string;
}

function emitComplexFusedChain(
  lines: string[],
  indent: string,
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  paramTensors: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>,
  localTensorNames: ReadonlySet<string>,
  dynamicOutputNames: ReadonlySet<string>,
  complexTensorNames: ReadonlySet<string>,
  complexScalarVars: ReadonlySet<string>
): void {
  // See emitRealFusedChain for rationale — prefer a formal param, then
  // any source tensor; destination lengths start at 0.
  const refTensor =
    findTensorParamInChain(chain, paramTensors, allTensorVars) ??
    [...collectInputTensors(chain, allTensorVars)][0];
  const lenVar = tensorLen(refTensor ?? chain.assigns[0].destName);

  const { writeBack } = determineWriteBack(chain, outputTensorNames);

  // Size paired re+im buffers for each write-back dest. Same size-match
  // guard as emitEnsureComplexTensorBuf in emit.ts (skip free+malloc
  // when length is unchanged across iterations — the common hot-loop case).
  for (const d of writeBack) {
    const needsRealloc = localTensorNames.has(d) || dynamicOutputNames.has(d);
    if (needsRealloc) {
      const inner2 = indent + "  ";
      lines.push(`${indent}{`);
      lines.push(`${inner2}int64_t __need = ${lenVar};`);
      lines.push(`${inner2}if (__need != ${tensorLen(d)}) {`);
      lines.push(`${inner2}  if (${tensorData(d)}) free(${tensorData(d)});`);
      lines.push(
        `${inner2}  if (${tensorDataIm(d)}) free(${tensorDataIm(d)});`
      );
      lines.push(
        `${inner2}  ${tensorData(d)} = (__need > 0) ? (double *)malloc((size_t)__need * sizeof(double)) : NULL;`
      );
      lines.push(
        `${inner2}  ${tensorDataIm(d)} = (__need > 0) ? (double *)malloc((size_t)__need * sizeof(double)) : NULL;`
      );
      lines.push(`${inner2}  ${tensorLen(d)} = __need;`);
      lines.push(`${inner2}}`);
      lines.push(`${indent}}`);
    } else {
      lines.push(`${indent}${tensorLen(d)} = ${lenVar};`);
    }
  }

  // Open the loop. No parallel-for yet — the complex elemwise body is
  // compute-thin (6 flops per element) and thread-spawn overhead would
  // dominate. #pragma omp simd is enough for the compiler to vectorize.
  lines.push(`${indent}#pragma omp simd`);
  lines.push(`${indent}for (int64_t __i = 0; __i < ${lenVar}; __i++) {`);
  const inner = indent + "  ";

  // Chain-produced intermediates: tracked in a set so later reads emit
  // `__f_NAME_re` / `__f_NAME_im` instead of a buffer read.
  const chainLocals = new Set<string>();
  const tmpN = { n: 0 };

  for (const assign of chain.assigns) {
    const materializations: string[] = [];
    const pair = emitComplexPerElem(
      assign.expr,
      chainLocals,
      allTensorVars,
      complexTensorNames,
      complexScalarVars,
      materializations,
      inner,
      tmpN
    );
    for (const m of materializations) lines.push(m);
    if (!chainLocals.has(assign.destName)) {
      lines.push(
        `${inner}double ${fusedLocal(assign.destName)}_re = ${pair.re};`
      );
      lines.push(
        `${inner}double ${fusedLocal(assign.destName)}_im = ${pair.im};`
      );
      chainLocals.add(assign.destName);
    } else {
      // Reassignment aliasing: pair.im may reference `_re` (the prior
      // value); writing _re first would make pair.im read the new one.
      // Stage through temporaries so both reads see the old pair.
      tmpN.n++;
      const tre = `__fr${tmpN.n}_re`;
      const tim = `__fr${tmpN.n}_im`;
      lines.push(`${inner}double ${tre} = ${pair.re};`);
      lines.push(`${inner}double ${tim} = ${pair.im};`);
      lines.push(`${inner}${fusedLocal(assign.destName)}_re = ${tre};`);
      lines.push(`${inner}${fusedLocal(assign.destName)}_im = ${tim};`);
    }
  }

  for (const d of writeBack) {
    lines.push(`${inner}${tensorData(d)}[__i] = ${fusedLocal(d)}_re;`);
    lines.push(`${inner}${tensorDataIm(d)}[__i] = ${fusedLocal(d)}_im;`);
  }

  lines.push(`${indent}}`);

  for (const d of writeBack) {
    lines.push(`${indent}${tensorLen(d)} = ${lenVar};`);
  }

  // Propagate dynamic-output shape so size(y) ≠ [0 0].
  for (const d of writeBack) {
    if (!dynamicOutputNames.has(d)) continue;
    let shapeSrc: JitExpr | undefined;
    for (const a of chain.assigns) if (a.destName === d) shapeSrc = a.expr;
    const [d0Expr, d1Expr] = shapeExprsFor(shapeSrc, lenVar);
    lines.push(`${indent}${tensorD0(d)} = ${d0Expr};`);
    lines.push(`${indent}${tensorD1(d)} = ${d1Expr};`);
  }
}

/** Walk a complex element-wise expression tree into per-element re/im
 *  C expression strings. Analogous to emitFusedScalarExpr but returns a
 *  ComplexPerElem and pre-declares scratch `__fm{n}_{re,im}` locals when
 *  operands would otherwise be re-evaluated (Binary Mul / Div / Sub with
 *  reused sides). Pre-decls are appended to `materializations` in the
 *  order they must appear before the main assignment. */
function emitComplexPerElem(
  expr: JitExpr,
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  complexTensorNames: ReadonlySet<string>,
  complexScalarVars: ReadonlySet<string>,
  materializations: string[],
  indent: string,
  tmpN: { n: number }
): ComplexPerElem {
  switch (expr.tag) {
    case "NumberLiteral":
      return { re: formatNumberLiteral(expr.value), im: "0.0" };

    case "ImagLiteral":
      return { re: "0.0", im: "1.0" };

    case "Var": {
      const name = expr.name;
      if (expr.jitType.kind === "tensor" || allTensorVars.has(name)) {
        if (chainLocals.has(name)) {
          return {
            re: `${fusedLocal(name)}_re`,
            im: `${fusedLocal(name)}_im`,
          };
        }
        if (complexTensorNames.has(name)) {
          return {
            re: `${tensorData(name)}[__i]`,
            im: `${tensorDataIm(name)}[__i]`,
          };
        }
        // Real tensor in complex context: widen with im = 0.
        return { re: `${tensorData(name)}[__i]`, im: "0.0" };
      }
      // Scalar var.
      if (complexScalarVars.has(name)) {
        return { re: mangle(name), im: mangleIm(name) };
      }
      return { re: mangle(name), im: "0.0" };
    }

    case "Unary": {
      if (expr.op === UnaryOperation.Plus) {
        return emitComplexPerElem(
          expr.operand,
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          indent,
          tmpN
        );
      }
      if (expr.op === UnaryOperation.Minus) {
        const o = emitComplexPerElem(
          expr.operand,
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          indent,
          tmpN
        );
        return { re: `(-(${o.re}))`, im: `(-(${o.im}))` };
      }
      throw new Error(
        `complex fused: unsupported unary op ${expr.op} in fused chain`
      );
    }

    case "Binary": {
      const l = emitComplexPerElem(
        expr.left,
        chainLocals,
        allTensorVars,
        complexTensorNames,
        complexScalarVars,
        materializations,
        indent,
        tmpN
      );
      const r = emitComplexPerElem(
        expr.right,
        chainLocals,
        allTensorVars,
        complexTensorNames,
        complexScalarVars,
        materializations,
        indent,
        tmpN
      );
      switch (expr.op) {
        case BinaryOperation.Add:
          return { re: `(${l.re} + ${r.re})`, im: `(${l.im} + ${r.im})` };
        case BinaryOperation.Sub:
          return { re: `(${l.re} - ${r.re})`, im: `(${l.im} - ${r.im})` };
        case BinaryOperation.Mul:
        case BinaryOperation.ElemMul: {
          // Both sides used twice in the per-component formula. Reuse of a
          // leaf (Var, number) is free; reuse of a compound sub-expression
          // would 4x-expand the source. Materialize eagerly — the C
          // compiler's CSE cleans up the trivial cases.
          const lm = materializePair(l, materializations, indent, tmpN);
          const rm = materializePair(r, materializations, indent, tmpN);
          return {
            re: `(${lm.re} * ${rm.re} - ${lm.im} * ${rm.im})`,
            im: `(${lm.re} * ${rm.im} + ${lm.im} * ${rm.re})`,
          };
        }
        default:
          throw new Error(
            `complex fused: unsupported binary op ${expr.op} in fused chain`
          );
      }
    }

    case "Call": {
      if (expr.name === "conj" && expr.args.length === 1) {
        const o = emitComplexPerElem(
          expr.args[0],
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          indent,
          tmpN
        );
        return { re: o.re, im: `(-(${o.im}))` };
      }
      if (expr.name === "real" && expr.args.length === 1) {
        const o = emitComplexPerElem(
          expr.args[0],
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          indent,
          tmpN
        );
        return { re: o.re, im: "0.0" };
      }
      if (expr.name === "imag" && expr.args.length === 1) {
        const o = emitComplexPerElem(
          expr.args[0],
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          indent,
          tmpN
        );
        return { re: o.im, im: "0.0" };
      }
      throw new Error(
        `complex fused: unsupported call '${expr.name}' in fused chain`
      );
    }

    default:
      throw new Error(
        `complex fused: unsupported expr ${expr.tag} in fused chain`
      );
  }
}

/** Materialize a ComplexPerElem into `__fm{n}_re` / `__fm{n}_im` locals
 *  declared just above the current per-element statement. Skips the
 *  materialization when the operand is already a simple identifier or
 *  `ident[__i]` array read — no reason to bounce those through a temp. */
function materializePair(
  pair: ComplexPerElem,
  materializations: string[],
  indent: string,
  tmpN: { n: number }
): ComplexPerElem {
  const simple = /^(-?\d+(\.\d+)?|[A-Za-z_]\w*(\[__i\])?)$/;
  if (simple.test(pair.re) && simple.test(pair.im)) return pair;
  const n = ++tmpN.n;
  const reVar = `__fm${n}_re`;
  const imVar = `__fm${n}_im`;
  materializations.push(`${indent}double ${reVar} = ${pair.re};`);
  materializations.push(`${indent}double ${imVar} = ${pair.im};`);
  return { re: reVar, im: imVar };
}
