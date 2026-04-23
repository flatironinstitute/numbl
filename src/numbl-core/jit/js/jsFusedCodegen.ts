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

import type { FusibleChain } from "../fusion.js";
import type { JitExpr } from "../jitTypes.js";
import type { ScalarOpTarget } from "../scalarEmit.js";
import {
  type FusedTarget,
  emitFusedScalarExpr,
  fusedLocal,
  findTensorParamInChain,
  collectInputTensors,
} from "../fusedScalarEmit.js";
import {
  JS_REDUCTION_LITERALS,
  accumulateOp,
  determineWriteBack,
  reductionCombine,
  reductionInit,
} from "../fusedChainHelpers.js";
import { emitChainKernel } from "../e1/kernelEmit.js";
import { emitComplexChainKernel } from "../e1/complexKernelEmit.js";

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
  // round: MATLAB rounds half-away-from-zero; Math.round rounds half-
  // toward-+Inf, so `round(-1.5)` disagrees (-2 vs -1). Route through
  // `$h.round` which matches the interpreter and C-JIT paths.
  round: "$h.round",
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

/** Collect write-back dests that are also read in the chain BEFORE they
 *  are first written. These need the existing tensor's values copied
 *  into a freshly-allocated buffer so the per-element body can read
 *  them; pure-output dests (written but never read in the chain) skip
 *  the copy.
 *
 *  A dest `d` counts as self-read when any assign *at or before d's
 *  first write* references `d` in its RHS. That covers both the direct
 *  self-alias (`y = y .* x + 3`) and the cross-assign case
 *  (`z = y + x; y = z + 1` — y's first write is the 2nd assign, but
 *  the 1st assign's RHS reads y). */
function collectSelfReadDests(chain: FusibleChain): Set<string> {
  const out = new Set<string>();
  const writtenSoFar = new Set<string>();
  for (const a of chain.assigns) {
    // First, for each name referenced in this assign's RHS, if that
    // name is a future chain dest (i.e. not yet written), mark it as
    // needing its input preserved.
    const refs = new Set<string>();
    collectExprVars(a.expr, refs);
    for (const name of refs) {
      if (!writtenSoFar.has(name) && isChainDest(chain, name)) {
        out.add(name);
      }
    }
    writtenSoFar.add(a.destName);
  }
  return out;
}

function isChainDest(chain: FusibleChain, name: string): boolean {
  for (const a of chain.assigns) if (a.destName === name) return true;
  return false;
}

function collectExprVars(expr: JitExpr, out: Set<string>): void {
  if (expr.tag === "Var") {
    out.add(expr.name);
    return;
  }
  if (expr.tag === "Binary") {
    collectExprVars(expr.left, out);
    collectExprVars(expr.right, out);
    return;
  }
  if (expr.tag === "Unary") {
    collectExprVars(expr.operand, out);
    return;
  }
  if (expr.tag === "Call") {
    for (const a of expr.args) collectExprVars(a, out);
  }
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

// ── Public API ────────────────────────────────────────────────────────

/**
 * Emit a fused per-element loop for the given chain (JS backend).
 *
 * Emits a block-scoped `{ ... }` section containing:
 * 1. Data aliases for input tensors (`const __x_data = x.data;`)
 * 2. Output buffer allocation with reuse check
 * 3. A single `for` loop with inline scalar computation
 * 4. Result wrapping (`x = $h.wrapF64(__x_data, refParam.shape);`)
 */
/** Emit the plain inline JS fused loop — one `for` over `__len`, with
 *  per-element scalar expressions, write-backs, and optional trailing
 *  reduction accumulate. Shared between the `--fuse`-only path and the
 *  `else` branch of the e1 size dispatch. */
function emitInlineLoop(
  lines: string[],
  inner: string,
  loopInner: string,
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  writeBack: ReadonlySet<string>,
  fusedTarget: FusedTarget,
  mangle: (n: string) => string,
  reduceAccLocal: string
): void {
  const chainLocals = new Set<string>();

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

  for (const d of writeBack) {
    lines.push(`${loopInner}${dataAlias(d, mangle)}[__i] = ${fusedLocal(d)};`);
  }

  if (chain.reduction) {
    const valueExpr = fusedLocal(chain.reduction.tensorName);
    lines.push(
      `${loopInner}${reductionCombine(chain.reduction.reduceName, reduceAccLocal, valueExpr, JS_REDUCTION_LITERALS)}`
    );
  }

  lines.push(`${inner}}`);
}

/** Minimum `__len` at which the e1 kernel dispatch prefers the C path.
 *  40 comes from benchmarks/koffi_overhead_bench.md — the break-even
 *  for a single `exp(1+sqrt(x))` kernel. Longer chains can tolerate a
 *  lower threshold since the fixed koffi cost amortizes across more
 *  per-element work, but 40 is a safe universal default for v1. */
const E1_SIZE_THRESHOLD = 40;

export function emitJsFusedChain(
  lines: string[],
  indent: string,
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  paramTensors: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>,
  _localTensorNames: ReadonlySet<string>,
  complexTensorNames: ReadonlySet<string>,
  complexScalarVars: ReadonlySet<string>,
  mangle: (n: string) => string,
  experimental?: string,
  par?: boolean
): void {
  // Pick a tensor to use for length + shape. Prefer a formal param
  // (stable binding, can't be reassigned inside the chain), but fall
  // back to any input tensor read by the chain — same pattern as
  // `emitRealFusedChain` in c/emit/fused.ts. Without this fallback,
  // chains whose inputs are all locals (e.g. `z = a + a*1i;` where
  // `a` is a local real tensor) silently bail and their assignments
  // get dropped, since the enclosing loop still advances past them.
  const refTensor =
    findTensorParamInChain(chain, paramTensors, allTensorVars) ??
    [...collectInputTensors(chain, allTensorVars)][0];
  if (!refTensor) {
    // Truly no input tensor — rare (e.g. a chain that only writes
    // a dest from scalars/literals). Skip emission rather than
    // silently dropping; the caller will still advance past the
    // chain, but the non-fused per-op path can't produce this
    // shape either, so there's nothing to do.
    return;
  }

  const refMangled = mangle(refTensor);

  // Complex chain routing (e1 only). Delegates to the paired-buffer
  // kernel emitter; on null (unsupported op) bail so the surrounding
  // JS-JIT per-op path handles it.
  if (experimental === "e1") {
    const hasComplexDest = chain.assigns.some(a =>
      complexTensorNames.has(a.destName)
    );
    const hasComplexResult = chain.assigns.some(
      a => a.expr.jitType.kind === "tensor" && a.expr.jitType.isComplex === true
    );
    if (hasComplexDest || hasComplexResult) {
      emitComplexChainBlock(
        lines,
        indent,
        chain,
        allTensorVars,
        paramTensors,
        outputTensorNames,
        complexTensorNames,
        complexScalarVars,
        mangle,
        refTensor,
        refMangled
      );
      return;
    }
  }

  // Build the fused target bound to this backend's mangle.
  const fusedTarget = makeJsFusedTarget(mangle);

  // Determine write-back dests (shared with C codegen).
  const { writeBack } = determineWriteBack(chain, outputTensorNames);

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
  //
  // When a dest is also read in the chain BEFORE it is first written
  // (e.g. `y = y .* x + 3`), the fresh-alloc branch must copy the
  // existing values in — otherwise the loop reads uninitialized memory.
  // Detect this per-dest; non-self-read dests (pure outputs) skip the
  // copy to avoid a wasted memcpy.
  const selfReadDests = collectSelfReadDests(chain);
  for (const d of writeBack) {
    const m = mangle(d);
    const da = dataAlias(d, mangle);
    if (selfReadDests.has(d)) {
      lines.push(
        `${inner}const ${da} = (${m} && ${m}._rc === 1 && ${m}.data instanceof Float64Array && ${m}.data.length === __len) ? ${m}.data : $h.uninitCopy(${m}.data, __len);`
      );
    } else {
      lines.push(
        `${inner}const ${da} = (${m} && ${m}._rc === 1 && ${m}.data instanceof Float64Array && ${m}.data.length === __len) ? ${m}.data : $h.uninit(__len);`
      );
    }
  }

  // Reduction accumulator init.
  const reduceAccLocal = "__f_reduce_acc";
  if (chain.reduction) {
    lines.push(
      `${inner}let ${reduceAccLocal} = ${reductionInit(chain.reduction.reduceName, JS_REDUCTION_LITERALS)};`
    );
  }

  // e1 experimental: when active and the chain is kernelizable, emit a
  // runtime size dispatch that calls the compiled C kernel at large N
  // and falls back to the inline JS loop at small N. emitChainKernel
  // now supports trailing reductions, so we don't pre-gate on
  // `!chain.reduction` — if the kernel emitter rejects (e.g. unsupported
  // builtin in the chain), it returns null and we fall through.
  const kernelInfo =
    experimental === "e1"
      ? emitChainKernel(chain, allTensorVars, outputTensorNames, par ?? false)
      : null;

  if (kernelInfo) {
    // Per-chain scratch for the reduction result, if any. The kernel
    // writes its final accumulator value here and we copy it into the
    // JS-side `reduceAccLocal` in place of the inline-loop accumulate.
    const reduceScratch = "__f_reduce_buf";
    if (chain.reduction) {
      lines.push(`${inner}const ${reduceScratch} = new Float64Array(1);`);
    }

    // Resolve each kernel call-arg slot to the JS expression that the
    // surrounding fused block already has on hand (data aliases for
    // tensors, mangled name for scalars, the reduction scratch buf).
    const callArgExprs = kernelInfo.jsCallArgs.map(slot => {
      if (slot === "n") return "__len";
      if (slot === "r") return reduceScratch;
      const colonIdx = slot.indexOf(":");
      const kind = slot.slice(0, colonIdx);
      const name = slot.slice(colonIdx + 1);
      if (kind === "t" || kind === "o") return dataAlias(name, mangle);
      if (kind === "s") return mangle(name);
      throw new Error(`emitJsFusedChain: unknown kernel slot "${slot}"`);
    });

    // Kernel source embedded inline so --dump-js shows everything.
    // JSON.stringify handles the escaping of the C string.
    const kernelKey = JSON.stringify(kernelInfo.kernelName);
    lines.push(
      `${inner}$h.$kernels[${kernelKey}] ??= $h.compileKernel(${JSON.stringify(kernelInfo.cSource)}, ${JSON.stringify(kernelInfo.koffiSig)});`
    );
    lines.push(`${inner}if (__len >= ${E1_SIZE_THRESHOLD}) {`);
    lines.push(
      `${loopInner}$h.$kernels[${kernelKey}](${callArgExprs.join(", ")});`
    );
    // Copy the kernel's reduction result into `reduceAccLocal` so the
    // post-block mean division + final accumulator write path can stay
    // branch-agnostic (same code for both C-kernel and JS-fallback
    // branches once we're past this block).
    if (chain.reduction) {
      lines.push(`${loopInner}${reduceAccLocal} = ${reduceScratch}[0];`);
    }
    lines.push(`${inner}} else {`);
    emitInlineLoop(
      lines,
      inner,
      loopInner,
      chain,
      allTensorVars,
      writeBack,
      fusedTarget,
      mangle,
      reduceAccLocal
    );
    lines.push(`${inner}}`);
  } else {
    // Non-e1 or non-kernelizable: plain inline JS fused loop only.
    emitInlineLoop(
      lines,
      inner,
      loopInner,
      chain,
      allTensorVars,
      writeBack,
      fusedTarget,
      mangle,
      reduceAccLocal
    );
  }

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

/** Data-alias pair for a complex tensor (real/imag buffers). */
function complexInputReAlias(
  name: string,
  mangle: (n: string) => string
): string {
  return `__${mangle(name)}_re`;
}
function complexInputImAlias(
  name: string,
  mangle: (n: string) => string
): string {
  return `__${mangle(name)}_im`;
}
function complexOutputReAlias(
  name: string,
  mangle: (n: string) => string
): string {
  return `__${mangle(name)}_out_re`;
}
function complexOutputImAlias(
  name: string,
  mangle: (n: string) => string
): string {
  return `__${mangle(name)}_out_im`;
}

/**
 * Emit the e1 complex fused-chain block: allocate paired re/im output
 * buffers, marshal paired input buffers + scalar re/im pairs, call the
 * compiled C kernel, then wrap the outputs as complex RuntimeTensors.
 *
 * No size-dispatch fallback: there is no JS-JIT complex fused loop. If
 * the kernel emitter rejects the chain (unsupported op), this function
 * throws and the caller lets the surrounding per-op codegen handle it.
 */
function emitComplexChainBlock(
  lines: string[],
  indent: string,
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  _paramTensors: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>,
  complexTensorNames: ReadonlySet<string>,
  complexScalarVars: ReadonlySet<string>,
  mangle: (n: string) => string,
  _refTensor: string,
  refMangled: string
): void {
  const kernelInfo = emitComplexChainKernel(
    chain,
    allTensorVars,
    complexTensorNames,
    complexScalarVars,
    outputTensorNames
  );
  if (!kernelInfo) {
    // Fall back: emit nothing here. This produces an incorrect result
    // (the chain's per-op statements would have been covered by the
    // fused block but we emitted nothing). In practice, if the chain
    // was rejected by the kernel emitter it was also rejected by
    // fusion.ts's own filter for complex chains, so we shouldn't get
    // here — but bail loudly if we do.
    throw new Error(
      "emitJsFusedChain: complex chain detected but complexKernelEmit rejected it"
    );
  }

  const { writeBack } = determineWriteBack(chain, outputTensorNames);
  const inputTensors = collectInputTensors(chain, allTensorVars);

  lines.push(`${indent}{`);
  const inner = indent + "  ";
  // `.data.length` works for both real and complex tensors — the
  // `.data` buffer is the real part either way.
  lines.push(`${inner}const __len = ${refMangled}.data.length;`);

  // Data aliases for input tensors (paired for complex, single for real).
  for (const name of [...inputTensors].sort()) {
    if (complexTensorNames.has(name)) {
      const reA = complexInputReAlias(name, mangle);
      const imA = complexInputImAlias(name, mangle);
      lines.push(`${inner}const ${reA} = ${mangle(name)}.data;`);
      // A complex tensor without an imag buffer (widened from real in
      // another flow) would yield undefined here; we assume complex
      // tensors always carry `.imag`. The constructor (`makeTensor`) and
      // complex ops uphold that invariant.
      lines.push(`${inner}const ${imA} = ${mangle(name)}.imag;`);
    } else {
      // Real tensor read into a complex chain: widened with im=0. Only
      // the real buffer is marshaled.
      lines.push(
        `${inner}const ${dataAlias(name, mangle)} = ${mangle(name)}.data;`
      );
    }
  }

  // Output buffer allocation. Complex outputs get paired re/im Float64
  // buffers; self-read dests aren't supported (kernel emitter rejected
  // them), so we always use `$h.uninit`.
  for (const d of writeBack) {
    const m = mangle(d);
    if (complexTensorNames.has(d)) {
      const reA = complexOutputReAlias(d, mangle);
      const imA = complexOutputImAlias(d, mangle);
      lines.push(
        `${inner}const ${reA} = (${m} && ${m}._rc === 1 && ${m}.data instanceof Float64Array && ${m}.data.length === __len && ${m}.imag instanceof Float64Array && ${m}.imag.length === __len) ? ${m}.data : $h.uninit(__len);`
      );
      lines.push(
        `${inner}const ${imA} = (${m} && ${m}._rc === 1 && ${m}.data instanceof Float64Array && ${m}.data.length === __len && ${m}.imag instanceof Float64Array && ${m}.imag.length === __len) ? ${m}.imag : $h.uninit(__len);`
      );
    } else {
      // Real output in a complex chain — unlikely (chain is complex
      // throughout) but emit a plain buffer just in case.
      lines.push(
        `${inner}const ${dataAlias(d, mangle)} = (${m} && ${m}._rc === 1 && ${m}.data instanceof Float64Array && ${m}.data.length === __len) ? ${m}.data : $h.uninit(__len);`
      );
    }
  }

  // Resolve call-arg slots.
  const callArgExprs = kernelInfo.jsCallArgs.map(slot => {
    if (slot === "n") return "__len";
    const colonIdx = slot.indexOf(":");
    const kind = slot.slice(0, colonIdx);
    const name = slot.slice(colonIdx + 1);
    switch (kind) {
      case "t":
        return dataAlias(name, mangle);
      case "tcre":
        return complexInputReAlias(name, mangle);
      case "tcim":
        return complexInputImAlias(name, mangle);
      case "s":
        return mangle(name);
      case "scre":
        return `$h.re(${mangle(name)})`;
      case "scim":
        return `$h.im(${mangle(name)})`;
      case "ocre":
        return complexOutputReAlias(name, mangle);
      case "ocim":
        return complexOutputImAlias(name, mangle);
      default:
        throw new Error(`emitComplexChainBlock: unknown kernel slot "${slot}"`);
    }
  });

  // Emit kernel source + call. No size-dispatch fallback.
  const kernelKey = JSON.stringify(kernelInfo.kernelName);
  lines.push(
    `${inner}$h.$kernels[${kernelKey}] ??= $h.compileKernel(${JSON.stringify(kernelInfo.cSource)}, ${JSON.stringify(kernelInfo.koffiSig)});`
  );
  lines.push(`${inner}$h.$kernels[${kernelKey}](${callArgExprs.join(", ")});`);

  // Wrap outputs.
  for (const d of writeBack) {
    if (complexTensorNames.has(d)) {
      lines.push(
        `${inner}${mangle(d)} = $h.wrapF64c(${complexOutputReAlias(d, mangle)}, ${complexOutputImAlias(d, mangle)}, ${refMangled}.shape);`
      );
    } else {
      lines.push(
        `${inner}${mangle(d)} = $h.wrapF64(${dataAlias(d, mangle)}, ${refMangled}.shape);`
      );
    }
  }

  lines.push(`${indent}}`);
}
