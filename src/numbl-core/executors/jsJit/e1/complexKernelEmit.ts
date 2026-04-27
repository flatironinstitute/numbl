/**
 * e1 (experimental) — complex-tensor standalone C-kernel emission.
 *
 * Sister module to `kernelEmit.ts` (which handles real-tensor chains).
 * Given a FusibleChain that produces at least one complex tensor, emit
 * a paired-buffer C kernel of the form
 *
 *   void k_<hash>(int64_t n,
 *                 const double *in_<a>_re, const double *in_<a>_im,  // complex tensor
 *                 const double *in_<b>,                              // real tensor (widened)
 *                 double s_<c>_re, double s_<c>_im,                  // complex scalar
 *                 double s_<d>,                                      // real scalar
 *                 double *out_<y>_re, double *out_<y>_im)            // complex output
 *   {
 *       #pragma omp simd
 *       for (int64_t i = 0; i < n; i++) {
 *           double __f_y_re = ...;
 *           double __f_y_im = ...;
 *           out_<y>_re[i] = __f_y_re;
 *           out_<y>_im[i] = __f_y_im;
 *       }
 *   }
 *
 * Supports the same fusion envelope as emitComplexPerElem in
 * `c/emit/fused.ts`:
 *   - Binary: + - * .*
 *   - Unary: + -
 *   - Call: conj, real, imag
 *   - Operand widening: real tensor / real scalar read with im = 0
 *   - ImagLiteral: (0.0, 1.0) pair
 *
 * Complex chains do NOT carry a trailing reduction — `fusion.ts` drops
 * the absorption for complex chains because the inline scalar
 * accumulator can't hold a complex value. Kernels emitted here have no
 * reduction output.
 */

import { BinaryOperation, UnaryOperation } from "../../../parser/types.js";
import type { JitExpr } from "../../../jit/jitTypes.js";
import type { FusibleChain } from "../../../jit/fusion.js";
import { collectInputTensors } from "../../../jit/fusedScalarEmit.js";
import { formatNumberLiteral } from "../../../jit/c/context.js";
import { determineWriteBack } from "../../../jit/fusedChainHelpers.js";
import type { KernelEmitResult } from "./kernelEmit.js";
import { fnv1a64Hex } from "../../../jit/hash.js";

// ── Kernel-local naming ─────────────────────────────────────────────────

function cInputRe(name: string): string {
  return `in_${name}_re`;
}
function cInputIm(name: string): string {
  return `in_${name}_im`;
}
function cInputReal(name: string): string {
  return `in_${name}`;
}
function cOutputRe(name: string): string {
  return `out_${name}_re`;
}
function cOutputIm(name: string): string {
  return `out_${name}_im`;
}
function cScalarRe(name: string): string {
  return `s_${name}_re`;
}
function cScalarIm(name: string): string {
  return `s_${name}_im`;
}
function cScalarReal(name: string): string {
  return `s_${name}`;
}
function localRe(name: string): string {
  return `__f_${name}_re`;
}
function localIm(name: string): string {
  return `__f_${name}_im`;
}

interface ComplexPerElem {
  re: string;
  im: string;
}

// ── Scalar / tensor liveness walks ───────────────────────────────────

function collectInputScalars(
  chain: FusibleChain,
  complexScalarVars: ReadonlySet<string>
): { complex: string[]; real: string[] } {
  const chainDests = new Set(chain.assigns.map(a => a.destName));
  const complex: string[] = [];
  const real: string[] = [];
  const seen = new Set<string>();
  function walk(expr: JitExpr): void {
    switch (expr.tag) {
      case "Var":
        if (expr.jitType.kind === "tensor") return;
        if (chainDests.has(expr.name)) return;
        if (seen.has(expr.name)) return;
        seen.add(expr.name);
        if (
          complexScalarVars.has(expr.name) ||
          expr.jitType.kind === "complex_or_number"
        ) {
          complex.push(expr.name);
        } else {
          real.push(expr.name);
        }
        return;
      case "Binary":
        walk(expr.left);
        walk(expr.right);
        return;
      case "Unary":
        walk(expr.operand);
        return;
      case "Call":
        for (const a of expr.args) walk(a);
        return;
      default:
        return;
    }
  }
  for (const a of chain.assigns) walk(a.expr);
  return { complex, real };
}

// ── Per-element walker (paired re/im) ───────────────────────────────

function emitPerElem(
  expr: JitExpr,
  chainLocals: ReadonlySet<string>,
  allTensorVars: ReadonlySet<string>,
  complexTensorNames: ReadonlySet<string>,
  complexScalarVars: ReadonlySet<string>,
  materializations: string[],
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
          return { re: localRe(name), im: localIm(name) };
        }
        if (complexTensorNames.has(name)) {
          return {
            re: `${cInputRe(name)}[i]`,
            im: `${cInputIm(name)}[i]`,
          };
        }
        // Real tensor in complex context: widen with im = 0.
        return { re: `${cInputReal(name)}[i]`, im: "0.0" };
      }
      // Scalar var.
      if (complexScalarVars.has(name)) {
        return { re: cScalarRe(name), im: cScalarIm(name) };
      }
      return { re: cScalarReal(name), im: "0.0" };
    }

    case "Unary": {
      if (expr.op === UnaryOperation.Plus) {
        return emitPerElem(
          expr.operand,
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          tmpN
        );
      }
      if (expr.op === UnaryOperation.Minus) {
        const o = emitPerElem(
          expr.operand,
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          tmpN
        );
        return { re: `(-(${o.re}))`, im: `(-(${o.im}))` };
      }
      throw new Error(
        `complex e1 kernel: unsupported unary op ${expr.op} in fused chain`
      );
    }

    case "Binary": {
      const l = emitPerElem(
        expr.left,
        chainLocals,
        allTensorVars,
        complexTensorNames,
        complexScalarVars,
        materializations,
        tmpN
      );
      const r = emitPerElem(
        expr.right,
        chainLocals,
        allTensorVars,
        complexTensorNames,
        complexScalarVars,
        materializations,
        tmpN
      );
      switch (expr.op) {
        case BinaryOperation.Add:
          return { re: `(${l.re} + ${r.re})`, im: `(${l.im} + ${r.im})` };
        case BinaryOperation.Sub:
          return { re: `(${l.re} - ${r.re})`, im: `(${l.im} - ${r.im})` };
        case BinaryOperation.Mul:
        case BinaryOperation.ElemMul: {
          const lm = materializePair(l, materializations, tmpN);
          const rm = materializePair(r, materializations, tmpN);
          return {
            re: `(${lm.re} * ${rm.re} - ${lm.im} * ${rm.im})`,
            im: `(${lm.re} * ${rm.im} + ${lm.im} * ${rm.re})`,
          };
        }
        default:
          throw new Error(
            `complex e1 kernel: unsupported binary op ${expr.op} in fused chain`
          );
      }
    }

    case "Call": {
      if (expr.name === "conj" && expr.args.length === 1) {
        const o = emitPerElem(
          expr.args[0],
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          tmpN
        );
        return { re: o.re, im: `(-(${o.im}))` };
      }
      if (expr.name === "real" && expr.args.length === 1) {
        const o = emitPerElem(
          expr.args[0],
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          tmpN
        );
        return { re: o.re, im: "0.0" };
      }
      if (expr.name === "imag" && expr.args.length === 1) {
        const o = emitPerElem(
          expr.args[0],
          chainLocals,
          allTensorVars,
          complexTensorNames,
          complexScalarVars,
          materializations,
          tmpN
        );
        return { re: o.im, im: "0.0" };
      }
      throw new Error(
        `complex e1 kernel: unsupported call '${expr.name}' in fused chain`
      );
    }

    default:
      throw new Error(
        `complex e1 kernel: unsupported expr ${expr.tag} in fused chain`
      );
  }
}

/** Stage compound operand pairs into `__fm{n}_re` / `__fm{n}_im` locals
 *  so the Mul formula doesn't re-evaluate them. Leaf reads (identifier,
 *  number literal, `in_x_re[i]`) pass through unchanged. */
function materializePair(
  pair: ComplexPerElem,
  materializations: string[],
  tmpN: { n: number }
): ComplexPerElem {
  const simple = /^(-?\d+(\.\d+)?|[A-Za-z_]\w*(\[i\])?)$/;
  if (simple.test(pair.re) && simple.test(pair.im)) return pair;
  const n = ++tmpN.n;
  const reVar = `__fm${n}_re`;
  const imVar = `__fm${n}_im`;
  materializations.push(`        double ${reVar} = ${pair.re};`);
  materializations.push(`        double ${imVar} = ${pair.im};`);
  return { re: reVar, im: imVar };
}

// ── Main entry point ────────────────────────────────────────────────

/**
 * Emit a complex-tensor fused chain as a standalone C kernel.
 *
 * Returns null when the chain contains an expression the per-element
 * walker doesn't support (abs, complex divide, transcendental on
 * complex, etc.) — the caller falls back to the JS-JIT per-op path.
 */
// Note: no `par` parameter — complex chains deliberately ignore --par
// (6-flop per-element bodies are too compute-thin to amortize thread
// spawn overhead). The caller in jsFusedCodegen.ts knows not to pass
// it.
export function emitComplexChainKernel(
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  complexTensorNames: ReadonlySet<string>,
  complexScalarVars: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>
): KernelEmitResult | null {
  // Complex chains never carry a reduction (fusion.ts drops it).
  if (chain.reduction) return null;

  const { writeBack } = determineWriteBack(chain, outputTensorNames);
  if (writeBack.size === 0) return null;

  const inputTensorsSet = collectInputTensors(chain, allTensorVars);

  // Reject self-read dests for v1 (same as real kernel).
  for (const d of writeBack) {
    if (inputTensorsSet.has(d)) return null;
  }

  const complexInputs: string[] = [];
  const realInputs: string[] = [];
  for (const t of [...inputTensorsSet].sort()) {
    if (complexTensorNames.has(t)) complexInputs.push(t);
    else realInputs.push(t);
  }
  const writeBackOrdered = [...writeBack].sort();
  const { complex: complexScalars, real: realScalars } = collectInputScalars(
    chain,
    complexScalarVars
  );
  complexScalars.sort();
  realScalars.sort();

  // Walk chain assigns to build per-element body.
  const bodyLines: string[] = [];
  const chainLocals = new Set<string>();
  const tmpN = { n: 0 };

  try {
    for (const assign of chain.assigns) {
      const materializations: string[] = [];
      const pair = emitPerElem(
        assign.expr,
        chainLocals,
        allTensorVars,
        complexTensorNames,
        complexScalarVars,
        materializations,
        tmpN
      );
      for (const m of materializations) bodyLines.push(m);
      if (!chainLocals.has(assign.destName)) {
        bodyLines.push(
          `        double ${localRe(assign.destName)} = ${pair.re};`
        );
        bodyLines.push(
          `        double ${localIm(assign.destName)} = ${pair.im};`
        );
        chainLocals.add(assign.destName);
      } else {
        // Reassignment aliasing: stage through temporaries so pair.im's
        // reference to the prior _re value sees the old one.
        tmpN.n++;
        const tre = `__fr${tmpN.n}_re`;
        const tim = `__fr${tmpN.n}_im`;
        bodyLines.push(`        double ${tre} = ${pair.re};`);
        bodyLines.push(`        double ${tim} = ${pair.im};`);
        bodyLines.push(`        ${localRe(assign.destName)} = ${tre};`);
        bodyLines.push(`        ${localIm(assign.destName)} = ${tim};`);
      }
    }
  } catch {
    return null;
  }

  // Write-back to output buffers.
  for (const d of writeBackOrdered) {
    bodyLines.push(`        ${cOutputRe(d)}[i] = ${localRe(d)};`);
    bodyLines.push(`        ${cOutputIm(d)}[i] = ${localIm(d)};`);
  }

  // Build C parameter list.
  const paramList: string[] = ["int64_t n"];
  for (const t of complexInputs) {
    paramList.push(`const double *${cInputRe(t)}`);
    paramList.push(`const double *${cInputIm(t)}`);
  }
  for (const t of realInputs) {
    paramList.push(`const double *${cInputReal(t)}`);
  }
  for (const s of complexScalars) {
    paramList.push(`double ${cScalarRe(s)}`);
    paramList.push(`double ${cScalarIm(s)}`);
  }
  for (const s of realScalars) {
    paramList.push(`double ${cScalarReal(s)}`);
  }
  for (const d of writeBackOrdered) {
    paramList.push(`double *${cOutputRe(d)}`);
    paramList.push(`double *${cOutputIm(d)}`);
  }

  // Complex chains deliberately stick to `#pragma omp simd` regardless
  // of `--par`. The per-element body is ~6 flops, memory-bandwidth-
  // bound across paired re/im buffers; thread-spawn overhead dominates
  // the compute win at realistic N. Mirrors the C-JIT's complex
  // emitter, which also opts out of threading.
  const bodyStr = [
    "    #pragma omp simd",
    "    for (int64_t i = 0; i < n; i++) {",
    ...bodyLines,
    "    }",
  ].join("\n");

  const prologue = `#include <math.h>\n#include <stdint.h>\n\n`;
  const bodyTemplate =
    `void __KERNEL_NAME__(${paramList.join(", ")})\n` + `{\n${bodyStr}\n}\n`;
  const cSourceTemplate = prologue + bodyTemplate;

  const h = fnv1a64Hex(cSourceTemplate);
  const kernelName = `cnk_${h}`;
  const cSource = cSourceTemplate.replace("__KERNEL_NAME__", kernelName);

  // Build koffi signature.
  const koffiParams: string[] = ["int64_t"];
  for (let k = 0; k < complexInputs.length; k++) {
    koffiParams.push("double *");
    koffiParams.push("double *");
  }
  for (let k = 0; k < realInputs.length; k++) koffiParams.push("double *");
  for (let k = 0; k < complexScalars.length; k++) {
    koffiParams.push("double");
    koffiParams.push("double");
  }
  for (let k = 0; k < realScalars.length; k++) koffiParams.push("double");
  for (let k = 0; k < writeBackOrdered.length; k++) {
    koffiParams.push("double *");
    koffiParams.push("double *");
  }
  const koffiSig = `void ${kernelName}(${koffiParams.join(", ")})`;

  return {
    kernelName,
    cSource,
    koffiSig,
    hash: h,
    jsCallArgs: buildCallArgSlotTags(
      complexInputs,
      realInputs,
      complexScalars,
      realScalars,
      writeBackOrdered
    ),
  };
}

/**
 * Encode kernel call-arg slots as structured tags the JS codegen fills in.
 * Tags:
 *   "n"        — length
 *   "tcre:X"   — complex tensor X, real buffer
 *   "tcim:X"   — complex tensor X, imag buffer
 *   "t:X"      — real tensor X (widened)
 *   "scre:X"   — complex scalar X, real part
 *   "scim:X"   — complex scalar X, imag part
 *   "s:X"      — real scalar X
 *   "ocre:X"   — complex output X, real buffer
 *   "ocim:X"   — complex output X, imag buffer
 */
function buildCallArgSlotTags(
  complexInputs: string[],
  realInputs: string[],
  complexScalars: string[],
  realScalars: string[],
  outputs: string[]
): string[] {
  const out: string[] = ["n"];
  for (const t of complexInputs) {
    out.push(`tcre:${t}`);
    out.push(`tcim:${t}`);
  }
  for (const t of realInputs) out.push(`t:${t}`);
  for (const s of complexScalars) {
    out.push(`scre:${s}`);
    out.push(`scim:${s}`);
  }
  for (const s of realScalars) out.push(`s:${s}`);
  for (const o of outputs) {
    out.push(`ocre:${o}`);
    out.push(`ocim:${o}`);
  }
  return out;
}
