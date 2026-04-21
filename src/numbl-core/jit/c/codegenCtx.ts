/**
 * Shared types, constants, and helpers used by both emit.ts (the per-
 * function emitter) and jitCodegenC.ts (the orchestration that builds
 * the final signature / prelude / epilogue).
 *
 * Contains no emit logic itself — pulling this out of the main file
 * breaks the otherwise-circular dependency between jitCodegenC.ts and
 * the emit helpers.
 */
import { BinaryOperation } from "../../parser/types.js";
import type { ScalarOpTarget } from "../scalarEmit.js";
import type { ClassificationResult } from "./classify.js";
import type { CParamDesc, COutputDesc } from "./abi.js";
import { getIBuiltin } from "../../interpreter/builtins/types.js";

/** Per-callee ABI info the outer emitter uses to marshal its call sites.
 *  `emitUserCall` reads `paramDescs` to know what slots each param
 *  contributes (data + len + optional d0/d1 for tensors) and
 *  `outputDescs` to know whether the return uses the dynamic-output ABI. */
export interface CalleeAbi {
  paramDescs: CParamDesc[];
  outputDescs: COutputDesc[];
}

const MANGLE_PREFIX = "v_";
const IM_SUFFIX = "_im";

/**
 * Minimum NUMBL_JIT_RT_VERSION the emitter needs at link time.
 *
 * Bump this in lockstep with NUMBL_JIT_RT_VERSION in
 * native/jit_runtime/jit_runtime.h whenever we add a helper the emitter
 * calls. The emitted C asserts `NUMBL_JIT_RT_VERSION >= N` so a stale
 * archive fails the per-JIT compile step with a clear message instead
 * of a cryptic linker "undefined reference" error.
 *
 * Version log:
 *   1 — initial: idx1r, mod, sign, reduce_flat, tic/toc/monotonic_time.
 *   2 — set1r_h (scalar linear Index write with soft-bail on OOB).
 *   3 — idx2r / idx3r / set2r_h / set3r_h (multi-index Index read/write).
 *   4 — setRange1r_h / setCol2r_h / copyRange1r (range/col slice r/w).
 *   5 — is_nan / is_inf / is_finite (predicates that survive -ffast-math
 *       by inspecting IEEE-754 bit patterns).
 */
export const NUMBL_JIT_RT_REQUIRED_VERSION = 5;

export function mangle(name: string): string {
  return `${MANGLE_PREFIX}${name}`;
}

/** Imaginary part of a complex scalar local / param. Paired with
 *  `mangle(name)` (the real part). */
export function mangleIm(name: string): string {
  return `${MANGLE_PREFIX}${name}${IM_SUFFIX}`;
}

/** Join a C type and an identifier with a space unless the type already
 *  ends in `*` (pointer types get no space between `*` and the name). */
export function spaceBeforeName(cType: string): string {
  return cType.endsWith("*") ? "" : " ";
}

// ── Scalar op target (value form + truthiness form) ─────────────────────
//
// C coerces numeric results of comparisons/logicals to `double` so the
// JIT type system can continue treating them as numbers. In condition
// context (if/while/&&/||), the cast is dropped because C's control
// flow already accepts scalar numeric tests.
export const C_SCALAR_TARGET: ScalarOpTarget = {
  binAdd: (l, r) => `(${l} + ${r})`,
  binSub: (l, r) => `(${l} - ${r})`,
  binMul: (l, r) => `(${l} * ${r})`,
  binDiv: (l, r) => `(${l} / ${r})`,
  binPow: (l, r) => `pow(${l}, ${r})`,
  binEq: (l, r) => `(((double)((${l}) == (${r}))))`,
  binNe: (l, r) => `(((double)((${l}) != (${r}))))`,
  binLt: (l, r) => `(((double)((${l}) < (${r}))))`,
  binLe: (l, r) => `(((double)((${l}) <= (${r}))))`,
  binGt: (l, r) => `(((double)((${l}) > (${r}))))`,
  binGe: (l, r) => `(((double)((${l}) >= (${r}))))`,
  binAnd: (l, r) => `((double)(((${l}) != 0.0) && ((${r}) != 0.0)))`,
  binOr: (l, r) => `((double)(((${l}) != 0.0) || ((${r}) != 0.0)))`,
  unaryPlus: o => `(+${o})`,
  unaryMinus: o => `(-${o})`,
  unaryNot: o => `((double)((${o}) == 0.0))`,
  toTruthy: v => `((${v}) != 0.0)`,
  condEq: (l, r) => `((${l}) == (${r}))`,
  condNe: (l, r) => `((${l}) != (${r}))`,
  condLt: (l, r) => `((${l}) < (${r}))`,
  condLe: (l, r) => `((${l}) <= (${r}))`,
  condGt: (l, r) => `((${l}) > (${r}))`,
  condGe: (l, r) => `((${l}) >= (${r}))`,
  condNot: t => `(!(${t}))`,
  condAnd: (l, r) => `((${l}) && (${r}))`,
  condOr: (l, r) => `((${l}) || (${r}))`,
};

// Binary op → libnumbl_ops opcode enum name.
export const TENSOR_BIN_OP: Partial<Record<BinaryOperation, string>> = {
  [BinaryOperation.Add]: "NUMBL_REAL_BIN_ADD",
  [BinaryOperation.Sub]: "NUMBL_REAL_BIN_SUB",
  [BinaryOperation.Mul]: "NUMBL_REAL_BIN_MUL",
  [BinaryOperation.ElemMul]: "NUMBL_REAL_BIN_MUL",
  [BinaryOperation.Div]: "NUMBL_REAL_BIN_DIV",
  [BinaryOperation.ElemDiv]: "NUMBL_REAL_BIN_DIV",
};

// Binary comparison op → libnumbl_ops opcode enum name.
export const TENSOR_CMP_OP: Partial<Record<BinaryOperation, string>> = {
  [BinaryOperation.Equal]: "NUMBL_CMP_EQ",
  [BinaryOperation.NotEqual]: "NUMBL_CMP_NE",
  [BinaryOperation.Less]: "NUMBL_CMP_LT",
  [BinaryOperation.LessEqual]: "NUMBL_CMP_LE",
  [BinaryOperation.Greater]: "NUMBL_CMP_GT",
  [BinaryOperation.GreaterEqual]: "NUMBL_CMP_GE",
};

/**
 * Tensor-op dispatch helpers. Each reads `IBuiltin.jitCapabilities` for
 * the named builtin and returns the corresponding libnumbl_ops opcode
 * enum name / C function name — or `undefined` when the builtin has no
 * C-JIT tensor-op routing. These helpers are the single source of truth
 * for both the C feasibility check (cFeasibility.ts) and the C emitter
 * (emit.ts); registering a new tensor op is a single edit on the
 * IBuiltin, not three parallel table updates.
 */

/** libnumbl_ops unary opcode enum name for a tensor-unary builtin. */
export function getTensorUnaryOp(name: string): string | undefined {
  return getIBuiltin(name)?.jitCapabilities?.tensorUnaryOp;
}

/** C function name for a 2-arg element-wise tensor binary builtin. */
export function getTensorBinaryFn(name: string): string | undefined {
  return getIBuiltin(name)?.jitCapabilities?.tensorBinaryFn;
}

/** libnumbl_ops reduce opcode enum name for a tensor-reduction builtin. */
export function getTensorReductionOp(name: string): string | undefined {
  return getIBuiltin(name)?.jitCapabilities?.tensorReductionOp;
}

export function formatNumberLiteral(v: number): string {
  if (!Number.isFinite(v)) {
    if (Number.isNaN(v)) return "(0.0/0.0)";
    return v > 0 ? "(1.0/0.0)" : "(-1.0/0.0)";
  }
  if (Number.isInteger(v)) return `${v}.0`;
  return `${v}`;
}

// ── Tensor data/len accessors ─────────────────────────────────────────

export function tensorData(name: string): string {
  return `${mangle(name)}_data`;
}
/** Imaginary data pointer for a complex tensor. Pairs with
 *  `tensorData(name)` — both locals carry the same length and shape
 *  info. The kernels in numbl_ops accept NULL for the imag pointer as
 *  "all zero", so a nominally-complex tensor whose `.imag` is undefined
 *  can still be passed efficiently without allocating a zero buffer. */
export function tensorDataIm(name: string): string {
  return `${mangle(name)}_data_im`;
}
export function tensorLen(name: string): string {
  return `${mangle(name)}_len`;
}

/** Row count (shape[0]) for a 2D/3D-indexed tensor param. Also reused
 *  as the mutable row-count local for fresh-alloc tensors (TensorLiteral/
 *  zeros/ones/VConcatGrow targets). */
export function tensorD0(name: string): string {
  return `${mangle(name)}_d0`;
}
/** Column count. For 3D tensor params this is shape[1]; for fresh-alloc
 *  tensors it's shape[1] (i.e., the column count of a matrix or 1 for
 *  a column vector). */
export function tensorD1(name: string): string {
  return `${mangle(name)}_d1`;
}

// ── Per-function emit context ─────────────────────────────────────────

export interface EmitCtx {
  /** Shared classification table: every tensor-name decision reads from
   *  here (kind, maxIndexDim, hasFreshAlloc, needsUnshare,
   *  isDynamicOutput, ...). See `classify.ts`. */
  cls: ClassificationResult;
  /** Counter for scratch buffer slots. Each tensor sub-expression that
   *  doesn't have a top-level dest gets a scratch double* + int64_t pair. */
  scratchCount: number;
  /** Scratch indices whose tensor expression was complex — the prelude
   *  declares a paired `__sN_data_im` for these, and epilogue frees both
   *  buffers. Real scratches stay single-buffer. */
  complexScratch: Set<number>;
  /** Counter for for-loop step temps. */
  tmp: { n: number };
  /** Set of scratch indices that were actually used. */
  usedScratch: Set<number>;
  /** When set, expression emission can prepend statements (e.g. for
   *  reductions of complex tensor expressions that need scratch buffers). */
  pendingStmts?: { lines: string[]; indent: string };
  /** Emit fused per-element loops for tensor chains (--fuse). */
  fuse: boolean;
  /** Set when tic or toc is used — triggers __tic_state parameter. */
  needsTicState: boolean;
  /** Set when any Index read is emitted — triggers __err_flag parameter
   *  and the __numbl_idx1r helper. JS wrapper checks the flag after the
   *  call and throws "Index exceeds array bounds" if set. */
  needsErrorFlag: boolean;
  /** Emit `#pragma omp parallel for` on fused non-reduction loops. */
  openmp: boolean;
  /** ABI of every reachable UserCall callee (keyed by `jitName`). Populated
   *  by `generateC` as each callee is emitted, then handed to the outer
   *  emitter so `emitUserCall` can walk the callee's paramDescs to marshal
   *  tensor args (and decide scalar-vs-tensor return shape) at the call
   *  site. Absent during tests that only emit a single function. */
  calleeAbi?: Map<string, CalleeAbi>;
  /** Names (params + locals + outputs) that hold `complex_or_number`
   *  scalar values and thus have paired `v_name` / `v_name_im` locals. */
  complexScalarVars: Set<string>;
}

// ── Classification-lookup shortcuts ───────────────────────────────────
//
// Keep the emit sites short: `isTensorVar(ctx, n)` reads better than
// `ctx.cls.tensorVars.has(n)`, and it gives us one place to add an
// assertion later if a caller queries an unknown name.

export function isTensorVar(ctx: EmitCtx, name: string): boolean {
  return ctx.cls.tensorVars.has(name);
}
export function isComplexTensorVar(ctx: EmitCtx, name: string): boolean {
  return !!ctx.cls.meta.get(name)?.isComplex;
}
export function isComplexScalarVar(ctx: EmitCtx, name: string): boolean {
  return ctx.complexScalarVars.has(name);
}
export function hasFreshAlloc(ctx: EmitCtx, name: string): boolean {
  return !!ctx.cls.meta.get(name)?.hasFreshAlloc;
}
export function isDynamicOutput(ctx: EmitCtx, name: string): boolean {
  return !!ctx.cls.meta.get(name)?.isDynamicOutput;
}
export function isLocalTensor(ctx: EmitCtx, name: string): boolean {
  return ctx.cls.localTensorNames.has(name);
}
export function isOutputTensor(ctx: EmitCtx, name: string): boolean {
  return ctx.cls.outputTensorNames.has(name);
}
export function tensorMaxDim(ctx: EmitCtx, name: string): number {
  return ctx.cls.meta.get(name)?.maxIndexDim ?? 0;
}

// ── Scratch buffer helpers ────────────────────────────────────────────

/** Allocate a scratch buffer pair (__s{n}_data, __s{n}_len). */
export function allocScratch(ctx: EmitCtx): number {
  ctx.scratchCount += 1;
  ctx.usedScratch.add(ctx.scratchCount);
  return ctx.scratchCount;
}

/** Allocate a complex scratch triple (__s{n}_data, __s{n}_data_im,
 *  __s{n}_len) — the emitter tags the index so the prelude declares
 *  and the epilogue frees the imag companion. */
export function allocComplexScratch(ctx: EmitCtx): number {
  const n = allocScratch(ctx);
  ctx.complexScratch.add(n);
  return n;
}

export function scratchData(n: number): string {
  return `__s${n}_data`;
}
/** Imaginary companion for a complex scratch buffer. Only declared in
 *  the prelude when the scratch index appears in ctx.complexScratch. */
export function scratchDataIm(n: number): string {
  return `__s${n}_data_im`;
}
export function scratchLen(n: number): string {
  return `__s${n}_len`;
}
