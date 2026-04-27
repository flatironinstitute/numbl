/**
 * e2 — complex multi-LHS fused chain C kernel emission (paired-buffer).
 *
 * Sister to [chainKernelEmit.ts](./chainKernelEmit.ts) for chains that
 * produce at least one complex tensor. Mirrors the codegen shape of
 * [e1/complexKernelEmit.ts](../e1/complexKernelEmit.ts) and uses the
 * same fusion envelope (+ - * .* unary +/- conj real imag, real/complex
 * widening, ImagLiteral). Anything outside that subset (`./`,
 * `abs(complex)`, transcendentals on complex) is rejected at the e2
 * lowerer level, which causes the driver to bail to the interpreter
 * per-op complex path — matching e1's fusion fallthrough behavior.
 *
 *     void e2cc_<hash>(int64_t n,
 *                      const double *in_<cta>_re, const double *in_<cta>_im,
 *                      const double *in_<rtb>,
 *                      [in_<lhs_input>_re/_im or in_<lhs_input> ...,]
 *                      double s_<csc>_re, double s_<csc>_im,
 *                      double s_<rsc>,
 *                      [out_<lhs>_re, out_<lhs>_im or out_<lhs> ...])
 *     {
 *         #pragma omp simd
 *         for (int64_t i = 0; i < n; i++) {
 *             double <clhs1>_re, <clhs1>_im, ..., <rlhs1>, ...;
 *             <clhs1>_re = ...; <clhs1>_im = ...;
 *             <rlhs1> = ...;
 *             out_<clhs>_re[i] = <clhs>_re; out_<clhs>_im[i] = <clhs>_im;
 *             out_<rlhs>[i] = <rlhs>;
 *         }
 *     }
 *
 * Complex chains deliberately stick to `#pragma omp simd` regardless of
 * `--par`: per-element bodies are ~6 flops spread across paired re/im
 * buffers (memory-bandwidth-bound), and thread-spawn overhead dominates
 * the compute win at realistic N. Matches e1's stance.
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { JitExpr } from "../../jit/jitTypes.js";
import { formatNumberLiteral } from "../../jit/c/context.js";
import { fnv1a64Hex } from "../../jit/e1/hash.js";
import type { ChainAssignSpec } from "./emitShared.js";

export interface E2ComplexKernelInputs {
  /** Env tensor names, split by complex-ness. */
  complexTensorNames: string[];
  realTensorNames: string[];
  /** Env scalar names, split by complex-ness. */
  complexScalarNames: string[];
  realScalarNames: string[];
  /** Chain LHS names that need `in_<name>` because they're read before
   *  being written. Split by complex-ness of the LHS. */
  complexInputLhsNames: string[];
  realInputLhsNames: string[];
  /** Chain LHS names that escape the chain. Split by complex-ness. */
  complexEscapeLhsNames: string[];
  realEscapeLhsNames: string[];
}

export interface E2ComplexChainEmitResult {
  kernelName: string;
  cSource: string;
  koffiSig: string;
  hash: string;
  /** In signature order: complex tensors, real tensors. */
  complexInputTensors: string[];
  realInputTensors: string[];
  /** In signature order: complex input LHSs, real input LHSs. */
  complexInputLhsNames: string[];
  realInputLhsNames: string[];
  /** In signature order: complex scalars, real scalars. */
  complexInputScalars: string[];
  realInputScalars: string[];
  /** In signature order: complex escape LHSs, real escape LHSs. */
  complexEscapeLhsNames: string[];
  realEscapeLhsNames: string[];
  chainLength: number;
}

// ── Kernel-local naming ─────────────────────────────────────────────────

const cInputRe = (n: string): string => `in_${n}_re`;
const cInputIm = (n: string): string => `in_${n}_im`;
const cInputReal = (n: string): string => `in_${n}`;
const cOutputRe = (n: string): string => `out_${n}_re`;
const cOutputIm = (n: string): string => `out_${n}_im`;
const cOutputReal = (n: string): string => `out_${n}`;
const cScalarRe = (n: string): string => `s_${n}_re`;
const cScalarIm = (n: string): string => `s_${n}_im`;
const cScalarReal = (n: string): string => `s_${n}`;

interface ComplexPair {
  re: string;
  im: string;
}

// ── Per-element walker (paired re/im) ──────────────────────────────────

interface WalkCtx {
  complexChainLocals: ReadonlySet<string>;
  realChainLocals: ReadonlySet<string>;
  complexTensorVars: ReadonlySet<string>;
  realTensorVars: ReadonlySet<string>;
  complexScalarVars: ReadonlySet<string>;
  materializations: string[];
  tmpN: { n: number };
}

function emitPerElem(expr: JitExpr, ctx: WalkCtx): ComplexPair {
  switch (expr.tag) {
    case "NumberLiteral":
      return { re: formatNumberLiteral(expr.value), im: "0.0" };

    case "ImagLiteral":
      return { re: "0.0", im: "1.0" };

    case "Var": {
      const name = expr.name;
      // Chain locals: prefer the complex pair if this LHS was written
      // as complex in the chain.
      if (ctx.complexChainLocals.has(name)) {
        return { re: `${name}_re`, im: `${name}_im` };
      }
      if (ctx.realChainLocals.has(name)) {
        return { re: name, im: "0.0" };
      }
      // Env tensors.
      if (ctx.complexTensorVars.has(name)) {
        return { re: `${cInputRe(name)}[i]`, im: `${cInputIm(name)}[i]` };
      }
      if (ctx.realTensorVars.has(name)) {
        return { re: `${cInputReal(name)}[i]`, im: "0.0" };
      }
      // Env scalars.
      if (ctx.complexScalarVars.has(name)) {
        return { re: cScalarRe(name), im: cScalarIm(name) };
      }
      return { re: cScalarReal(name), im: "0.0" };
    }

    case "Unary": {
      if (expr.op === UnaryOperation.Plus) {
        return emitPerElem(expr.operand, ctx);
      }
      if (expr.op === UnaryOperation.Minus) {
        const o = emitPerElem(expr.operand, ctx);
        return { re: `(-(${o.re}))`, im: `(-(${o.im}))` };
      }
      throw new Error(
        `e2 complex kernel: unsupported unary op ${expr.op} in fused chain`
      );
    }

    case "Binary": {
      const l = emitPerElem(expr.left, ctx);
      const r = emitPerElem(expr.right, ctx);
      switch (expr.op) {
        case BinaryOperation.Add:
          return { re: `(${l.re} + ${r.re})`, im: `(${l.im} + ${r.im})` };
        case BinaryOperation.Sub:
          return { re: `(${l.re} - ${r.re})`, im: `(${l.im} - ${r.im})` };
        case BinaryOperation.Mul:
        case BinaryOperation.ElemMul: {
          const lm = materializePair(l, ctx);
          const rm = materializePair(r, ctx);
          return {
            re: `(${lm.re} * ${rm.re} - ${lm.im} * ${rm.im})`,
            im: `(${lm.re} * ${rm.im} + ${lm.im} * ${rm.re})`,
          };
        }
        default:
          throw new Error(
            `e2 complex kernel: unsupported binary op ${expr.op} in fused chain`
          );
      }
    }

    case "Call": {
      if (expr.name === "conj" && expr.args.length === 1) {
        const o = emitPerElem(expr.args[0], ctx);
        return { re: o.re, im: `(-(${o.im}))` };
      }
      if (expr.name === "real" && expr.args.length === 1) {
        const o = emitPerElem(expr.args[0], ctx);
        return { re: o.re, im: "0.0" };
      }
      if (expr.name === "imag" && expr.args.length === 1) {
        const o = emitPerElem(expr.args[0], ctx);
        return { re: o.im, im: "0.0" };
      }
      throw new Error(
        `e2 complex kernel: unsupported call '${expr.name}' in fused chain`
      );
    }

    default:
      throw new Error(
        `e2 complex kernel: unsupported expr '${(expr as { tag: string }).tag}' in fused chain`
      );
  }
}

/** True when `expr`'s JitExpr tree references `name` as a `Var`. Used
 *  to detect self-alias on reassigns (so the emitter knows when to
 *  stage the paired pair through per-iter temps). */
function rhsReferencesVar(expr: JitExpr, name: string): boolean {
  switch (expr.tag) {
    case "Var":
      return expr.name === name;
    case "Binary":
      return (
        rhsReferencesVar(expr.left, name) || rhsReferencesVar(expr.right, name)
      );
    case "Unary":
      return rhsReferencesVar(expr.operand, name);
    case "Call":
      return expr.args.some(a => rhsReferencesVar(a, name));
    default:
      return false;
  }
}

/** Stage compound operand pairs into `__fm<n>_re` / `__fm<n>_im` locals
 *  so the Mul formula doesn't re-evaluate them. Leaf reads (identifier,
 *  number literal, `in_x_re[i]`) pass through unchanged. Mirrors the
 *  same optimization in e1/complexKernelEmit.ts. */
function materializePair(pair: ComplexPair, ctx: WalkCtx): ComplexPair {
  const simple = /^(-?\d+(\.\d+)?|[A-Za-z_]\w*(\[i\])?)$/;
  if (simple.test(pair.re) && simple.test(pair.im)) return pair;
  const n = ++ctx.tmpN.n;
  const reVar = `__fm${n}_re`;
  const imVar = `__fm${n}_im`;
  ctx.materializations.push(`        double ${reVar} = ${pair.re};`);
  ctx.materializations.push(`        double ${imVar} = ${pair.im};`);
  return { re: reVar, im: imVar };
}

// ── Main entry point ──────────────────────────────────────────────────

/** Per-assign LHS info — complex-ness of the RHS determines whether
 *  this stmt emits a paired (re/im) local or a single real local. */
export interface ComplexChainAssignSpec extends ChainAssignSpec {
  /** True when THIS stmt's RHS is complex. Chain-LHS type can differ
   *  per reassignment; we track per-stmt so a `a = real; a = complex;`
   *  sequence sees `a` as complex only after the second assign. */
  rhsIsComplex: boolean;
}

export function emitE2ComplexChainKernel(
  assigns: ComplexChainAssignSpec[],
  inputs: E2ComplexKernelInputs
): E2ComplexChainEmitResult {
  if (assigns.length === 0) {
    throw new Error("emitE2ComplexChainKernel: empty chain");
  }

  // Track which chain LHSs are complex vs real within the chain body.
  // A chain LHS becomes complex on its first complex assign and stays
  // that way for subsequent reads. This mirrors how envTypes was built
  // up in the driver.
  const complexChainLocals = new Set<string>();
  const realChainLocals = new Set<string>();

  // All chain LHS names in order of first appearance.
  const allLhsOrdered: string[] = [];
  const seenLhs = new Set<string>();
  for (const a of assigns) {
    if (!seenLhs.has(a.lhsName)) {
      seenLhs.add(a.lhsName);
      allLhsOrdered.push(a.lhsName);
    }
  }

  // Complex-ness per LHS name (OR across all writes to that LHS).
  const lhsIsComplex = new Map<string, boolean>();
  for (const a of assigns) {
    const prev = lhsIsComplex.get(a.lhsName) ?? false;
    lhsIsComplex.set(a.lhsName, prev || a.rhsIsComplex);
  }

  // Tensor / scalar var sets for the walker.
  const complexTensorVars = new Set<string>([
    ...inputs.complexTensorNames,
    ...inputs.complexInputLhsNames,
  ]);
  const realTensorVars = new Set<string>([
    ...inputs.realTensorNames,
    ...inputs.realInputLhsNames,
  ]);
  const complexScalarVars = new Set<string>(inputs.complexScalarNames);

  const bodyLines: string[] = [];

  // Declare all chain-local doubles upfront. Complex LHSs get re/im
  // pair, real LHSs get a single var. These accumulate across the
  // chain — any LHS ever complex gets a pair regardless of which
  // specific assign wrote it.
  const declTokens: string[] = [];
  for (const name of allLhsOrdered) {
    if (lhsIsComplex.get(name)) {
      declTokens.push(`${name}_re`, `${name}_im`);
    } else {
      declTokens.push(name);
    }
  }
  if (declTokens.length > 0) {
    bodyLines.push(`        double ${declTokens.join(", ")};`);
  }

  const tmpN = { n: 0 };
  const assignedSoFar = new Set<string>();

  for (const a of assigns) {
    const thisLhsFinalIsComplex = lhsIsComplex.get(a.lhsName)!;
    // Walker context: a previously-assigned LHS is available as a
    // chain-local read. Its complex-ness is the CUMULATIVE type.
    const complexLocalsNow = new Set<string>();
    const realLocalsNow = new Set<string>();
    for (const prior of assignedSoFar) {
      if (lhsIsComplex.get(prior)) complexLocalsNow.add(prior);
      else realLocalsNow.add(prior);
    }
    const materializations: string[] = [];
    const ctx: WalkCtx = {
      complexChainLocals: complexLocalsNow,
      realChainLocals: realLocalsNow,
      complexTensorVars,
      realTensorVars,
      complexScalarVars,
      materializations,
      tmpN,
    };
    const pair = emitPerElem(a.rhs, ctx);
    bodyLines.push(...materializations);

    // Self-alias detection: if this is a REASSIGN of the LHS and the
    // RHS references the LHS as a chain-local, the `im = ...` update
    // would read the newly-written `re`. Stage through per-iter temps
    // to preserve original-value semantics. Otherwise (first write, or
    // reassign that doesn't read the LHS), emit a direct assign — keeps
    // the generated C lean enough for cc to vectorize cleanly.
    const isReassign = assignedSoFar.has(a.lhsName);
    const needsStage = isReassign && rhsReferencesVar(a.rhs, a.lhsName);

    if (thisLhsFinalIsComplex) {
      if (needsStage) {
        tmpN.n++;
        const tre = `__fr${tmpN.n}_re`;
        const tim = `__fr${tmpN.n}_im`;
        bodyLines.push(`        double ${tre} = ${pair.re};`);
        bodyLines.push(`        double ${tim} = ${pair.im};`);
        bodyLines.push(`        ${a.lhsName}_re = ${tre};`);
        bodyLines.push(`        ${a.lhsName}_im = ${tim};`);
      } else {
        bodyLines.push(`        ${a.lhsName}_re = ${pair.re};`);
        bodyLines.push(`        ${a.lhsName}_im = ${pair.im};`);
      }
      complexChainLocals.add(a.lhsName);
    } else {
      // Real LHS: only re is meaningful. No cross-component aliasing
      // possible, so staging is never needed.
      bodyLines.push(`        ${a.lhsName} = ${pair.re};`);
      realChainLocals.add(a.lhsName);
    }
    assignedSoFar.add(a.lhsName);
  }

  // Write-backs.
  for (const name of inputs.complexEscapeLhsNames) {
    if (lhsIsComplex.get(name)) {
      bodyLines.push(`        ${cOutputRe(name)}[i] = ${name}_re;`);
      bodyLines.push(`        ${cOutputIm(name)}[i] = ${name}_im;`);
    } else {
      // A LHS that's declared complex-escape but never written complex.
      // Shouldn't occur given how the driver classifies, but emit im=0
      // defensively so the output buffer has valid data.
      bodyLines.push(`        ${cOutputRe(name)}[i] = ${name};`);
      bodyLines.push(`        ${cOutputIm(name)}[i] = 0.0;`);
    }
  }
  for (const name of inputs.realEscapeLhsNames) {
    bodyLines.push(`        ${cOutputReal(name)}[i] = ${name};`);
  }

  // Build C parameter list. Order mirrors e1: complex tensors first
  // (re,im pairs), then real tensors, complex input LHSs (re,im pairs),
  // real input LHSs, complex scalars (re,im), real scalars, complex
  // outputs (re,im), real outputs. This grouping simplifies marshaling
  // on the JS side.
  const paramList: string[] = ["int64_t n"];
  for (const t of inputs.complexTensorNames) {
    paramList.push(`const double *${cInputRe(t)}`);
    paramList.push(`const double *${cInputIm(t)}`);
  }
  for (const t of inputs.realTensorNames) {
    paramList.push(`const double *${cInputReal(t)}`);
  }
  for (const t of inputs.complexInputLhsNames) {
    paramList.push(`const double *${cInputRe(t)}`);
    paramList.push(`const double *${cInputIm(t)}`);
  }
  for (const t of inputs.realInputLhsNames) {
    paramList.push(`const double *${cInputReal(t)}`);
  }
  for (const s of inputs.complexScalarNames) {
    paramList.push(`double ${cScalarRe(s)}`);
    paramList.push(`double ${cScalarIm(s)}`);
  }
  for (const s of inputs.realScalarNames) {
    paramList.push(`double ${cScalarReal(s)}`);
  }
  for (const t of inputs.complexEscapeLhsNames) {
    paramList.push(`double *${cOutputRe(t)}`);
    paramList.push(`double *${cOutputIm(t)}`);
  }
  for (const t of inputs.realEscapeLhsNames) {
    paramList.push(`double *${cOutputReal(t)}`);
  }

  const loopLines = [
    "    #pragma omp simd",
    "    for (int64_t i = 0; i < n; i++) {",
    ...bodyLines,
    "    }",
  ];

  const prologue = "#include <math.h>\n#include <stdint.h>\n\n";
  const bodyTemplate =
    `void __KERNEL_NAME__(${paramList.join(", ")})\n` +
    `{\n${loopLines.join("\n")}\n}\n`;
  const cSourceTemplate = prologue + bodyTemplate;

  const h = fnv1a64Hex(cSourceTemplate);
  const kernelName = `e2cc_${h}`;
  const cSource = cSourceTemplate.replace("__KERNEL_NAME__", kernelName);

  // koffi signature in the same order.
  const koffiParts: string[] = ["int64_t"];
  for (let k = 0; k < inputs.complexTensorNames.length; k++) {
    koffiParts.push("double *");
    koffiParts.push("double *");
  }
  for (let k = 0; k < inputs.realTensorNames.length; k++)
    koffiParts.push("double *");
  for (let k = 0; k < inputs.complexInputLhsNames.length; k++) {
    koffiParts.push("double *");
    koffiParts.push("double *");
  }
  for (let k = 0; k < inputs.realInputLhsNames.length; k++)
    koffiParts.push("double *");
  for (let k = 0; k < inputs.complexScalarNames.length; k++) {
    koffiParts.push("double");
    koffiParts.push("double");
  }
  for (let k = 0; k < inputs.realScalarNames.length; k++)
    koffiParts.push("double");
  for (let k = 0; k < inputs.complexEscapeLhsNames.length; k++) {
    koffiParts.push("double *");
    koffiParts.push("double *");
  }
  for (let k = 0; k < inputs.realEscapeLhsNames.length; k++)
    koffiParts.push("double *");
  const koffiSig = `void ${kernelName}(${koffiParts.join(", ")})`;

  return {
    kernelName,
    cSource,
    koffiSig,
    hash: h,
    complexInputTensors: inputs.complexTensorNames.slice(),
    realInputTensors: inputs.realTensorNames.slice(),
    complexInputLhsNames: inputs.complexInputLhsNames.slice(),
    realInputLhsNames: inputs.realInputLhsNames.slice(),
    complexInputScalars: inputs.complexScalarNames.slice(),
    realInputScalars: inputs.realScalarNames.slice(),
    complexEscapeLhsNames: inputs.complexEscapeLhsNames.slice(),
    realEscapeLhsNames: inputs.realEscapeLhsNames.slice(),
    chainLength: assigns.length,
  };
}
