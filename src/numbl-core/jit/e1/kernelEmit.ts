/**
 * e1 (experimental) — standalone C-kernel emission for fusible tensor
 * chains used by the JS-JIT.
 *
 * Given a `FusibleChain` the normal JS fused codegen would emit, this
 * module produces an equivalent standalone C function of the form
 *
 *   void k_<hash>(int64_t n,
 *                 const double *in_<x>, ...,
 *                 double s_<scalar>, ...,
 *                 double *out_<y>, ...)
 *   {
 *       #pragma omp simd
 *       for (int64_t i = 0; i < n; i++) {
 *           double f_tmp1 = <expr>;
 *           ...
 *           out_<y>[i] = <final>;
 *       }
 *   }
 *
 * It returns the full C source, a koffi signature string, a content-
 * addressed hash, and the ordered list of JS expressions the generated
 * code should pass as arguments — everything the JS codegen needs to
 * emit a `$h.compileKernel(source, sig); kernel(n, x_data, y_data)`
 * dispatch.
 *
 * The prototype deliberately handles only the common real-tensor chain
 * shape: no reductions, no complex tensors, no dynamic-shape outputs.
 * Any chain that falls outside that envelope causes `emitChainKernel`
 * to return `null`, which signals the caller to fall back to the plain
 * inline JS fused loop.
 */

import { BinaryOperation } from "../../parser/types.js";
import type { JitExpr, JitType } from "../jitTypes.js";
import type { FusibleChain } from "../fusion.js";
import {
  emitFusedScalarExpr,
  fusedLocal,
  collectInputTensors,
} from "../fusedScalarEmit.js";
import { C_SCALAR_TARGET, formatNumberLiteral } from "../c/context.js";
import {
  determineWriteBack,
  reductionCombine,
  reductionInit,
  C_REDUCTION_LITERALS,
} from "../fusedChainHelpers.js";
import type { FusedTarget } from "../fusedScalarEmit.js";
import type { ScalarOpTarget } from "../scalarEmit.js";
import { getIBuiltin } from "../../interpreter/builtins/index.js";

/** Transcendentals and `pow` — expensive enough per element that
 *  thread-spawn overhead pays off at N >= 100k. Arithmetic-only
 *  chains stick to plain `simd` because threads slow them down.
 *  Mirrors the HEAVY_OPS set in `c/emit/fused.ts`. */
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

function countHeavyOps(expr: JitExpr): number {
  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
    case "ImagLiteral":
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

// ── Public types ──────────────────────────────────────────────────────

/**
 * A fused chain compiled to a standalone C kernel.  The caller (the JS
 * codegen) combines this with a runtime size threshold to emit
 *
 *     if (n >= THRESHOLD) $h.<kernelName>(n, x_data, y_data)
 *     else <plain JS fused loop>
 */
export interface KernelEmitResult {
  /** Hash-derived C function name, e.g. `nk_3a7f81b2`. */
  kernelName: string;
  /** Full C source: `#include` + function definition. */
  cSource: string;
  /** koffi function signature, e.g. `"void nk_3a7f81b2(int64_t, ...)"`. */
  koffiSig: string;
  /** Content hash over the final C source (stable id for caching). */
  hash: string;
  /** Ordered list of JS expressions to pass as call arguments. The
   *  caller emits something like `$h.<kernelName>(${jsCallArgs.join(",")})`. */
  jsCallArgs: string[];
}

// ── Scalar/fused target for the C kernel body ─────────────────────────

/** Kernel-local data pointer name for an input tensor, e.g.
 *  `in_x` for a chain input named `x`. The `in_` prefix makes the
 *  generated C easy to scan in `--dump-js`. */
function cInputPtr(name: string): string {
  return `in_${name}`;
}
/** Kernel-local data pointer name for a write-back dest. */
function cOutputPtr(name: string): string {
  return `out_${name}`;
}
/** Kernel-local scalar parameter name. */
function cScalarParam(name: string): string {
  return `s_${name}`;
}

/** Build the FusedTarget that emits C for the per-element body.
 *
 *  Mirrors the C-JIT's `C_FUSED_TARGET` (see
 *  `jit/c/emit/fused.ts`): each builtin's own `jitEmitC` owns the
 *  C-name mapping, so e1 picks up `max`, `min`, `mod`, `rem`, `atan2`,
 *  `hypot`, etc. automatically without a parallel whitelist. */
function makeCFusedTarget(writeBack: ReadonlySet<string>): FusedTarget {
  return {
    formatNumber: formatNumberLiteral,
    mangle: cScalarParam,
    tensorElemRead: name =>
      (writeBack.has(name) ? cOutputPtr(name) : cInputPtr(name)) + "[i]",
    emitBuiltinCall: (name, args) => {
      const ib = getIBuiltin(name);
      if (!ib?.jitEmitC) return null;
      // Fused context is per-element, so argTypes are fabricated as
      // scalar numbers — all fusible ops are element-wise and none
      // carry a `requireNonneg` guard on scalar args.
      const argTypes: JitType[] = args.map(() => ({ kind: "number" }));
      return ib.jitEmitC(args, argTypes);
    },
  };
}

// ── Input / scalar liveness walk ──────────────────────────────────────

/**
 * Collect scalar variable names read by the chain's expression trees
 * that are NOT produced by the chain (i.e. live-in scalars the kernel
 * needs as parameters). Returns them in **encounter order** so the
 * generated C signature is stable across runs.
 */
function collectInputScalars(chain: FusibleChain): string[] {
  const chainDests = new Set(chain.assigns.map(a => a.destName));
  const order: string[] = [];
  const seen = new Set<string>();
  for (const a of chain.assigns) {
    walkForScalars(a.expr, chainDests, seen, order);
  }
  return order;
}

function walkForScalars(
  expr: JitExpr,
  chainDests: ReadonlySet<string>,
  seen: Set<string>,
  order: string[]
): void {
  switch (expr.tag) {
    case "Var":
      // Scalar live-in: non-tensor-typed Var that isn't a chain dest.
      if (expr.jitType.kind === "tensor") return;
      if (chainDests.has(expr.name)) return;
      if (!seen.has(expr.name)) {
        seen.add(expr.name);
        order.push(expr.name);
      }
      return;
    case "Binary":
      walkForScalars(expr.left, chainDests, seen, order);
      walkForScalars(expr.right, chainDests, seen, order);
      return;
    case "Unary":
      walkForScalars(expr.operand, chainDests, seen, order);
      return;
    case "Call":
      for (const a of expr.args) walkForScalars(a, chainDests, seen, order);
      return;
    default:
      return;
  }
}

// ── Main entry point ──────────────────────────────────────────────────

/** Minimum element count before `#pragma omp parallel for simd` kicks
 *  in. Below this the thread-spawn cost dominates the work — same
 *  threshold the C-JIT uses, overridable via `NUMBL_OMP_THRESHOLD`. */
function ompParallelThreshold(): number {
  return parseInt(process.env.NUMBL_OMP_THRESHOLD || "", 10) || 100_000;
}

export function emitChainKernel(
  chain: FusibleChain,
  allTensorVars: ReadonlySet<string>,
  outputTensorNames: ReadonlySet<string>,
  par: boolean
): KernelEmitResult | null {
  const { writeBack } = determineWriteBack(chain, outputTensorNames);
  // A kernel must produce *something* — either tensor write-back or a
  // reduction result. If both are empty, there's nothing to do.
  if (writeBack.size === 0 && !chain.reduction) return null;

  const inputTensorsSet = collectInputTensors(chain, allTensorVars);
  const inputTensors = [...inputTensorsSet].sort();
  const writeBackOrdered = [...writeBack].sort();
  const inputScalars = collectInputScalars(chain);

  // Cross-check: a write-back dest read by the chain before it's written
  // (`y = y .* x + 3`) currently isn't covered by this kernel ABI —
  // we'd need to pass `y` as both an input and an output pointer, and
  // the body would read through the output pointer. Straightforward to
  // add, but out of scope for the v1 prototype.
  for (const d of writeBackOrdered) {
    if (inputTensorsSet.has(d)) return null;
  }

  // Build the per-element body by walking each assign with the C
  // fused target. The fused-scalar emitter returns `null` via an
  // exception if it hits an op without a C mapping — catch and bail.
  const fusedTarget = makeCFusedTarget(writeBack);
  const scalarTarget: ScalarOpTarget = C_SCALAR_TARGET;
  const chainLocals = new Set<string>();
  const bodyLines: string[] = [];
  try {
    for (const a of chain.assigns) {
      const rhs = emitFusedScalarExpr(
        a.expr,
        chainLocals,
        allTensorVars,
        scalarTarget,
        fusedTarget
      );
      if (!chainLocals.has(a.destName)) {
        bodyLines.push(`        double ${fusedLocal(a.destName)} = ${rhs};`);
        chainLocals.add(a.destName);
      } else {
        bodyLines.push(`        ${fusedLocal(a.destName)} = ${rhs};`);
      }
    }
  } catch {
    return null;
  }
  for (const d of writeBackOrdered) {
    bodyLines.push(`        ${cOutputPtr(d)}[i] = ${fusedLocal(d)};`);
  }

  // Trailing reduction: accumulate `<reduceName>(<lastDest>)` into an
  // internal scalar during the loop, then write it to an out-pointer
  // after the loop. The JS side reads back from the buffer, applies
  // any mean division, and handles the final `acc = reduceAccLocal`
  // or `acc OP= reduceAccLocal` write. We deliberately do NOT do the
  // mean division inside the kernel — JS already has a post-block
  // divide for the fallback path, and duplicating it would be fragile.
  const reduceAccC = "__f_reduce_acc";
  if (chain.reduction) {
    const valueExpr = fusedLocal(chain.reduction.tensorName);
    bodyLines.push(
      `        ${reductionCombine(chain.reduction.reduceName, reduceAccC, valueExpr, C_REDUCTION_LITERALS)}`
    );
  }

  // Build the C function signature. Preliminary name with a zero hash;
  // we compute the real hash over the final source (minus the name
  // itself, which gets substituted in afterwards).
  const paramList: string[] = [];
  paramList.push("int64_t n");
  for (const t of inputTensors) paramList.push(`const double *${cInputPtr(t)}`);
  for (const s of inputScalars) paramList.push(`double ${cScalarParam(s)}`);
  for (const d of writeBackOrdered) paramList.push(`double *${cOutputPtr(d)}`);
  if (chain.reduction) paramList.push(`double *out_acc`);

  const prologueLines: string[] = [];
  if (chain.reduction) {
    prologueLines.push(
      `    double ${reduceAccC} = ${reductionInit(chain.reduction.reduceName, C_REDUCTION_LITERALS)};`
    );
  }
  const postLines: string[] = [];
  if (chain.reduction) {
    postLines.push(`    *out_acc = ${reduceAccC};`);
  }

  // Choose the loop pragma. `--par` upgrades tensor-writeback loops to
  // `parallel for simd` when the body has heavy transcendentals — the
  // spawn overhead only pays off past ~100 cycles of per-element work.
  // Arithmetic-only chains stick to plain `simd` because threads would
  // add overhead that exceeds the memory-bandwidth-bound body's time.
  // Reduction chains also stick to `simd` since the serial scalar acc
  // can't be parallelized without a `reduction(...)` clause.
  const heavyOps = chain.assigns.reduce(
    (n, a) => n + countHeavyOps(a.expr),
    0
  );
  const useParallel =
    par && !chain.reduction && writeBack.size > 0 && heavyOps > 0;
  const pragma = useParallel
    ? `    #pragma omp parallel for simd if(n >= ${ompParallelThreshold()})`
    : "    #pragma omp simd";

  const bodyStr = [
    ...prologueLines,
    pragma,
    "    for (int64_t i = 0; i < n; i++) {",
    ...bodyLines,
    "    }",
    ...postLines,
  ].join("\n");

  // Two-pass assembly: emit with "__KERNEL_NAME__" placeholder, hash
  // the resulting text, substitute the final name in.
  const prologue = `#include <math.h>\n#include <stdint.h>\n\n`;
  const bodyTemplate =
    `void __KERNEL_NAME__(${paramList.join(", ")})\n` + `{\n${bodyStr}\n}\n`;
  const cSourceTemplate = prologue + bodyTemplate;

  const h = fnv1a64Hex(cSourceTemplate);
  const kernelName = `nk_${h}`;
  const cSource = cSourceTemplate.replace("__KERNEL_NAME__", kernelName);

  // Build koffi signature. Order matches the C param list exactly.
  const koffiParams: string[] = [];
  koffiParams.push("int64_t");
  for (let k = 0; k < inputTensors.length; k++) koffiParams.push("double *");
  for (let k = 0; k < inputScalars.length; k++) koffiParams.push("double");
  for (let k = 0; k < writeBackOrdered.length; k++)
    koffiParams.push("double *");
  if (chain.reduction) koffiParams.push("double *");
  const koffiSig = `void ${kernelName}(${koffiParams.join(", ")})`;

  return {
    kernelName,
    cSource,
    koffiSig,
    hash: h,
    // jsCallArgs is returned as a list of *kernel-relative* arg slot
    // descriptors: the JS codegen knows the mangling/aliasing scheme
    // of its surrounding function, so it fills these in at the call
    // site. We encode the slots as structured placeholders so the
    // codegen can map each one to the right expression.
    jsCallArgs: buildCallArgSlotTags(
      inputTensors,
      inputScalars,
      writeBackOrdered,
      chain.reduction !== undefined
    ),
  };
}

/**
 * Encode kernel call-arg slots as structured tags the JS codegen fills
 * in. Tags look like `n`, `t:<name>`, `s:<name>`, `o:<name>`, `r` — a thin
 * wire format that keeps the kernel emitter agnostic of which specific
 * JS expression names the caller uses. The reduction slot carries no
 * name since there's at most one per chain.
 */
function buildCallArgSlotTags(
  inputTensors: string[],
  inputScalars: string[],
  outputTensors: string[],
  hasReduction: boolean
): string[] {
  const out: string[] = ["n"];
  for (const t of inputTensors) out.push(`t:${t}`);
  for (const s of inputScalars) out.push(`s:${s}`);
  for (const o of outputTensors) out.push(`o:${o}`);
  if (hasReduction) out.push("r");
  return out;
}

// 64-bit FNV-1a over the source's UTF-8 code units, returned as 16 hex
// chars. Deterministic and fully self-contained — avoids pulling Node's
// `crypto` module into the browser bundle (the e1 path is Node-only at
// runtime, but this file is reachable from the JS-JIT module graph that
// Vite bundles for the web REPL). Cryptographic strength isn't needed:
// the hash is just a content-addressed kernel-name suffix.
function fnv1a64Hex(s: string): string {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < s.length; i++) {
    h = ((h ^ BigInt(s.charCodeAt(i))) * prime) & mask;
  }
  return h.toString(16).padStart(16, "0");
}
