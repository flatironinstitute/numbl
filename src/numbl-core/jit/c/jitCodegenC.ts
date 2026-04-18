/**
 * JIT IR -> pure C code generation (koffi path).
 *
 * Emits plain C functions with raw `double*` / `int64_t` parameters —
 * no N-API, no `napi_value`, no `napi_env`. koffi passes Float64Array
 * directly as `double*` on the JS side; the C function works with raw
 * pointers and libnumbl_ops calls.
 *
 * Tensor variables are represented as `(double *_data, int64_t _len)`
 * pairs. Buffer allocation and reuse are handled on the JS side
 * (cJitInstall.ts); the C function only reads from / writes into
 * caller-provided buffers.
 *
 * Scalar outputs for multi-output use `double*` out-pointers.
 * Tensor outputs use pre-allocated `double*` buffers passed in.
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import {
  isKnownInteger,
  type JitExpr,
  type JitStmt,
  type JitType,
} from "../jitTypes.js";
import {
  C_TENSOR_BINARY_BUILTINS,
  C_TENSOR_REDUCTION_OPS,
  C_TENSOR_UNARY_OPS,
} from "./cFeasibility.js";
import { findFusibleChains } from "../fusion.js";
import { emitFusedChain } from "./cFusedCodegen.js";
import {
  type ScalarOpTarget,
  emitScalarBinaryOp,
  emitScalarUnaryOp,
  emitScalarTruthiness,
} from "../scalarEmit.js";
import { getIBuiltin } from "../../interpreter/builtins/types.js";
import { collectTensorUsage } from "../js/jitCodegenHoist.js";

const MANGLE_PREFIX = "v_";

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
 */
const NUMBL_JIT_RT_REQUIRED_VERSION = 4;

export function mangle(name: string): string {
  return `${MANGLE_PREFIX}${name}`;
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
const TENSOR_BIN_OP: Partial<Record<BinaryOperation, string>> = {
  [BinaryOperation.Add]: "NUMBL_REAL_BIN_ADD",
  [BinaryOperation.Sub]: "NUMBL_REAL_BIN_SUB",
  [BinaryOperation.Mul]: "NUMBL_REAL_BIN_MUL",
  [BinaryOperation.ElemMul]: "NUMBL_REAL_BIN_MUL",
  [BinaryOperation.Div]: "NUMBL_REAL_BIN_DIV",
  [BinaryOperation.ElemDiv]: "NUMBL_REAL_BIN_DIV",
};

// Binary comparison op → libnumbl_ops opcode enum name.
const TENSOR_CMP_OP: Partial<Record<BinaryOperation, string>> = {
  [BinaryOperation.Equal]: "NUMBL_CMP_EQ",
  [BinaryOperation.NotEqual]: "NUMBL_CMP_NE",
  [BinaryOperation.Less]: "NUMBL_CMP_LT",
  [BinaryOperation.LessEqual]: "NUMBL_CMP_LE",
  [BinaryOperation.Greater]: "NUMBL_CMP_GT",
  [BinaryOperation.GreaterEqual]: "NUMBL_CMP_GE",
};

// Unary builtin name → libnumbl_ops unary opcode enum name.
const TENSOR_UNARY_OP: Record<string, string> = {
  exp: "NUMBL_UNARY_EXP",
  abs: "NUMBL_UNARY_ABS",
  floor: "NUMBL_UNARY_FLOOR",
  ceil: "NUMBL_UNARY_CEIL",
  round: "NUMBL_UNARY_ROUND",
  fix: "NUMBL_UNARY_TRUNC",
  sin: "NUMBL_UNARY_SIN",
  cos: "NUMBL_UNARY_COS",
  tan: "NUMBL_UNARY_TAN",
  atan: "NUMBL_UNARY_ATAN",
  sinh: "NUMBL_UNARY_SINH",
  cosh: "NUMBL_UNARY_COSH",
  tanh: "NUMBL_UNARY_TANH",
  sign: "NUMBL_UNARY_SIGN",
};

// Reduction builtin → libnumbl_ops reduce opcode enum name.
const TENSOR_REDUCE_OP: Record<string, string> = {
  sum: "NUMBL_REDUCE_SUM",
  prod: "NUMBL_REDUCE_PROD",
  max: "NUMBL_REDUCE_MAX",
  min: "NUMBL_REDUCE_MIN",
  any: "NUMBL_REDUCE_ANY",
  all: "NUMBL_REDUCE_ALL",
  mean: "NUMBL_REDUCE_MEAN",
};

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

interface EmitCtx {
  tensorVars: Set<string>;
  /** Tensor outputs whose buffers are pre-allocated by JS. */
  outputTensorNames: Set<string>;
  /** Counter for scratch buffer slots. Each tensor sub-expression that
   *  doesn't have a top-level dest gets a scratch double* + int64_t pair. */
  scratchCount: number;
  /** Counter for for-loop step temps. */
  tmp: { n: number };
  /** Set of scratch indices that were actually used. */
  usedScratch: Set<number>;
  /** Statements to add at function epilogue (free scratch buffers). */
  freeStmts: string[];
  /** Non-output tensor locals that need to be freed at epilogue. */
  localTensorNames: Set<string>;
  /** When set, expression emission can prepend statements (e.g. for
   *  reductions of complex tensor expressions that need scratch buffers). */
  pendingStmts?: { lines: string[]; indent: string };
  /** Emit fused per-element loops for tensor chains (--fuse). */
  fuse: boolean;
  /** Tensor parameter names (for fusion analysis). */
  paramTensorNames: Set<string>;
  /** Set when tic or toc is used — triggers __tic_state parameter. */
  needsTicState: boolean;
  /** Set when any Index read is emitted — triggers __err_flag parameter
   *  and the __numbl_idx1r helper. JS wrapper checks the flag after the
   *  call and throws "Index exceeds array bounds" if set. */
  needsErrorFlag: boolean;
  /** Emit `#pragma omp parallel for` on fused non-reduction loops. */
  openmp: boolean;
  /** Per-tensor-var max indexing arity (1/2/3). Tensors with arity >= 2
   *  receive a `_d0` ABI param; arity == 3 also gets `_d1`. Tensor locals
   *  and vars that aren't indexed are omitted. */
  tensorMaxDim: Map<string, number>;
  /** Tensor names that get a fresh-tensor RHS assignment (TensorLiteral /
   *  zeros / ones / VConcatGrow). These have `v_<n>_d0` / `v_<n>_d1`
   *  tracked as mutable C locals so the emitter can both set the shape
   *  on each fresh-alloc and read it downstream (e.g., in `length(v)`
   *  or the output ABI). Shape-only: buffer ownership is still tracked
   *  via `localTensorNames` / `outputTensorNames` / `dynamicOutputs`. */
  freshAllocTensors: Set<string>;
  /** Tensor outputs that use the new dynamic-output ABI: the JS wrapper
   *  passes a `double **` slot that the C function fills via
   *  `*out = v_<n>_data`, transferring ownership of a C-malloc'd buffer
   *  to the caller. Each such output also receives `_d0_out` / `_d1_out`
   *  int64_t slots to report the runtime shape. Subset of
   *  `outputTensorNames`. */
  dynamicOutputs: Set<string>;
}

// ── Expression emission ───────────────────────────────────────────────

function isTensorExpr(expr: JitExpr): boolean {
  return expr.jitType.kind === "tensor";
}

/** Allocate a scratch buffer pair (__s{n}_data, __s{n}_len). */
function allocScratch(ctx: EmitCtx): number {
  ctx.scratchCount += 1;
  ctx.usedScratch.add(ctx.scratchCount);
  return ctx.scratchCount;
}

function scratchData(n: number): string {
  return `__s${n}_data`;
}
function scratchLen(n: number): string {
  return `__s${n}_len`;
}

/** Emit a value-expression. For scalars, returns a C `double` expression.
 *  For tensors, returns the data-variable name (the caller knows to also
 *  access the corresponding _len variable). */
function emitExpr(expr: JitExpr, ctx: EmitCtx): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return formatNumberLiteral(expr.value);

    case "Var":
      if (ctx.tensorVars.has(expr.name)) return tensorData(expr.name);
      return mangle(expr.name);

    case "Binary":
      return emitBinary(expr, ctx);

    case "Unary":
      return emitUnary(expr, ctx);

    case "Call":
      return emitCall(expr, ctx);

    case "Index":
      return emitIndex(expr, ctx);

    case "RangeSliceRead": {
      // Result is a fresh tensor — must be emitted in statement context.
      if (!ctx.pendingStmts) {
        throw new Error(
          `C-JIT codegen: RangeSliceRead outside statement context`
        );
      }
      const result = emitRangeSliceReadToStmts(
        ctx.pendingStmts.lines,
        ctx.pendingStmts.indent,
        expr,
        ctx
      );
      return result.data;
    }

    default:
      throw new Error(`C-JIT codegen: unsupported expr ${expr.tag}`);
  }
}

function emitIndex(expr: JitExpr & { tag: "Index" }, ctx: EmitCtx): string {
  if (expr.base.tag !== "Var" || !ctx.tensorVars.has(expr.base.name)) {
    throw new Error("C-JIT codegen: Index base must be a tensor Var");
  }
  const n = expr.indices.length;
  if (n < 1 || n > 3) {
    throw new Error(
      `C-JIT codegen: Index arity ${n} unsupported (only 1D/2D/3D)`
    );
  }
  ctx.needsErrorFlag = true;
  const name = expr.base.name;
  const data = tensorData(name);
  const len = tensorLen(name);
  const idxCodes = expr.indices.map(idx => {
    let s = emitExpr(idx, ctx);
    if (!isKnownInteger(idx.jitType)) s = `round(${s})`;
    return s;
  });
  if (n === 1) {
    return `numbl_idx1r(${data}, (size_t)${len}, ${idxCodes[0]}, __err_flag)`;
  }
  if (n === 2) {
    return `numbl_idx2r(${data}, (size_t)${len}, (size_t)${tensorD0(name)}, ${idxCodes[0]}, ${idxCodes[1]}, __err_flag)`;
  }
  return `numbl_idx3r(${data}, (size_t)${len}, (size_t)${tensorD0(name)}, (size_t)${tensorD1(name)}, ${idxCodes[0]}, ${idxCodes[1]}, ${idxCodes[2]}, __err_flag)`;
}

/** For tensor binary/unary, we need multi-statement emission. This helper
 *  emits the operation as statements into `lines` and returns the result
 *  data pointer variable name. */
function emitTensorBinaryStmts(
  lines: string[],
  indent: string,
  expr: JitExpr & { tag: "Binary" },
  ctx: EmitCtx,
  destDataVar: string,
  destLenVar: string
): void {
  const binOp = TENSOR_BIN_OP[expr.op];
  const cmpOp = TENSOR_CMP_OP[expr.op];
  const isCmp = !!cmpOp;
  const opEnum = binOp || cmpOp;
  if (!opEnum) {
    throw new Error(`C-JIT codegen: tensor binary op ${expr.op} has no opcode`);
  }

  // Get operand data/len. For tensor operands, emit sub-expressions first.
  const leftIsTensor = isTensorExpr(expr.left);
  const rightIsTensor = isTensorExpr(expr.right);

  let leftData: string, leftLen: string;
  let rightData: string, rightLen: string;

  if (leftIsTensor) {
    const lResult = emitTensorExprToStmts(lines, indent, expr.left, ctx);
    leftData = lResult.data;
    leftLen = lResult.len;
  } else {
    leftData = emitExpr(expr.left, ctx);
    leftLen = "";
  }

  if (rightIsTensor) {
    const rResult = emitTensorExprToStmts(lines, indent, expr.right, ctx);
    rightData = rResult.data;
    rightLen = rResult.len;
  } else {
    rightData = emitExpr(expr.right, ctx);
    rightLen = "";
  }

  if (leftIsTensor && rightIsTensor) {
    // tensor-tensor: use the first operand's length for output
    lines.push(`${indent}${destLenVar} = ${leftLen};`);
    const fn = isCmp ? "numbl_real_comparison" : "numbl_real_binary_elemwise";
    lines.push(
      `${indent}${fn}(${opEnum}, (size_t)${destLenVar}, ${leftData}, ${rightData}, ${destDataVar});`
    );
  } else if (leftIsTensor) {
    // tensor-scalar
    lines.push(`${indent}${destLenVar} = ${leftLen};`);
    const fn = isCmp
      ? "numbl_real_scalar_comparison"
      : "numbl_real_scalar_binary_elemwise";
    lines.push(
      `${indent}${fn}(${opEnum}, (size_t)${destLenVar}, ${rightData}, ${leftData}, 0, ${destDataVar});`
    );
  } else if (rightIsTensor) {
    // scalar-tensor
    lines.push(`${indent}${destLenVar} = ${rightLen};`);
    const fn = isCmp
      ? "numbl_real_scalar_comparison"
      : "numbl_real_scalar_binary_elemwise";
    lines.push(
      `${indent}${fn}(${opEnum}, (size_t)${destLenVar}, ${leftData}, ${rightData}, 1, ${destDataVar});`
    );
  }
}

/** Emit a tensor expression as statements, returning the data/len vars. */
function emitTensorExprToStmts(
  lines: string[],
  indent: string,
  expr: JitExpr,
  ctx: EmitCtx
): { data: string; len: string } {
  if (expr.tag === "Var" && ctx.tensorVars.has(expr.name)) {
    return { data: tensorData(expr.name), len: tensorLen(expr.name) };
  }

  if (expr.tag === "RangeSliceRead") {
    return emitRangeSliceReadToStmts(lines, indent, expr, ctx);
  }

  // Need a scratch buffer for the result.
  const sIdx = allocScratch(ctx);
  const sData = scratchData(sIdx);
  const sLen = scratchLen(sIdx);

  if (expr.tag === "Binary" && isTensorExpr(expr)) {
    const tensorLen0 = findTensorLenExpr(expr, ctx);
    lines.push(`${indent}${sLen} = ${tensorLen0};`);
    lines.push(
      `${indent}if (!${sData}) ${sData} = (double *)malloc((size_t)${sLen} * sizeof(double));`
    );
    emitTensorBinaryStmts(lines, indent, expr, ctx, sData, sLen);
    return { data: sData, len: sLen };
  }

  if (expr.tag === "Unary" && isTensorExpr(expr)) {
    if (expr.op === UnaryOperation.Plus) {
      return emitTensorExprToStmts(lines, indent, expr.operand, ctx);
    }
    if (expr.op === UnaryOperation.Minus) {
      const operand = emitTensorExprToStmts(lines, indent, expr.operand, ctx);
      lines.push(`${indent}${sLen} = ${operand.len};`);
      lines.push(
        `${indent}if (!${sData}) ${sData} = (double *)malloc((size_t)${sLen} * sizeof(double));`
      );
      lines.push(
        `${indent}numbl_real_scalar_binary_elemwise(NUMBL_REAL_BIN_MUL, (size_t)${sLen}, -1.0, ${operand.data}, 1, ${sData});`
      );
      return { data: sData, len: sLen };
    }
  }

  if (
    expr.tag === "Call" &&
    isTensorExpr(expr) &&
    expr.name in C_TENSOR_UNARY_OPS
  ) {
    const opEnum = TENSOR_UNARY_OP[expr.name];
    const arg = emitTensorExprToStmts(lines, indent, expr.args[0], ctx);
    lines.push(`${indent}${sLen} = ${arg.len};`);
    lines.push(
      `${indent}if (!${sData}) ${sData} = (double *)malloc((size_t)${sLen} * sizeof(double));`
    );
    lines.push(
      `${indent}numbl_real_unary_elemwise(${opEnum}, (size_t)${sLen}, ${arg.data}, ${sData});`
    );
    return { data: sData, len: sLen };
  }

  throw new Error(
    `C-JIT codegen: cannot emit tensor expr ${expr.tag} to scratch`
  );
}

/** Emit `dstData = src(a:b) copy` pattern into caller-provided dst data /
 *  len vars. Frees any existing buffer in dstData, malloc's the exact
 *  range size, and calls numbl_copyRange1r. Used by both the direct
 *  Assign path (local tensor buffer) and the scratch-buffer path. */
function emitRangeSliceReadToBuf(
  lines: string[],
  indent: string,
  expr: JitExpr & { tag: "RangeSliceRead" },
  ctx: EmitCtx,
  destData: string,
  destLen: string
): void {
  if (!ctx.tensorVars.has(expr.baseName)) {
    throw new Error(
      `C-JIT codegen: RangeSliceRead base '${expr.baseName}' is not a tensor var`
    );
  }
  ctx.needsErrorFlag = true;
  const srcData = tensorData(expr.baseName);
  const srcLen = tensorLen(expr.baseName);

  let startCode = emitExpr(expr.start, ctx);
  if (!isKnownInteger(expr.start.jitType)) startCode = `round(${startCode})`;

  let endCode: string;
  if (expr.end === null) {
    endCode = `(double)${srcLen}`;
  } else {
    endCode = emitExpr(expr.end, ctx);
    if (!isKnownInteger(expr.end.jitType)) endCode = `round(${endCode})`;
  }

  lines.push(`${indent}{`);
  lines.push(`${indent}  double __rs = ${startCode};`);
  lines.push(`${indent}  double __re = ${endCode};`);
  lines.push(`${indent}  int64_t __rn = (int64_t)(__re - __rs + 1.0);`);
  lines.push(`${indent}  if (__rn < 0) __rn = 0;`);
  lines.push(`${indent}  if (${destData}) free(${destData});`);
  lines.push(
    `${indent}  ${destData} = (__rn > 0) ? (double *)malloc((size_t)__rn * sizeof(double)) : NULL;`
  );
  lines.push(`${indent}  ${destLen} = __rn;`);
  lines.push(
    `${indent}  numbl_copyRange1r(${srcData}, (size_t)${srcLen}, __rs, __re, ${destData}, __err_flag);`
  );
  lines.push(`${indent}}`);
}

/** Emit a `src(a:b)` RangeSliceRead into a fresh scratch buffer,
 *  returning the {data, len} pair for the result. */
function emitRangeSliceReadToStmts(
  lines: string[],
  indent: string,
  expr: JitExpr & { tag: "RangeSliceRead" },
  ctx: EmitCtx
): { data: string; len: string } {
  const sIdx = allocScratch(ctx);
  const sData = scratchData(sIdx);
  const sLen = scratchLen(sIdx);
  emitRangeSliceReadToBuf(lines, indent, expr, ctx, sData, sLen);
  return { data: sData, len: sLen };
}

/** Find a tensor-length expression from a tensor expr tree (for pre-allocating
 *  scratch buffers). Returns a C expression for the length. */
function findTensorLenExpr(expr: JitExpr, ctx: EmitCtx): string {
  if (expr.tag === "Var" && ctx.tensorVars.has(expr.name)) {
    return tensorLen(expr.name);
  }
  if (expr.tag === "Binary") {
    if (isTensorExpr(expr.left)) return findTensorLenExpr(expr.left, ctx);
    if (isTensorExpr(expr.right)) return findTensorLenExpr(expr.right, ctx);
  }
  if (expr.tag === "Unary") {
    return findTensorLenExpr(expr.operand, ctx);
  }
  if (expr.tag === "Call" && expr.args.length > 0) {
    return findTensorLenExpr(expr.args[0], ctx);
  }
  throw new Error(`C-JIT codegen: cannot find tensor length in ${expr.tag}`);
}

function emitBinary(expr: JitExpr & { tag: "Binary" }, ctx: EmitCtx): string {
  // Tensor-result binary: handled in statement context (emitTensorAssign).
  // If we reach here, it's a scalar result.
  if (isTensorExpr(expr)) {
    throw new Error(
      "C-JIT codegen: tensor binary must be emitted via statement context"
    );
  }

  const l = emitExpr(expr.left, ctx);
  const r = emitExpr(expr.right, ctx);
  return emitScalarBinaryOp(expr.op, l, r, C_SCALAR_TARGET);
}

function emitUnary(expr: JitExpr & { tag: "Unary" }, ctx: EmitCtx): string {
  if (isTensorExpr(expr)) {
    throw new Error(
      "C-JIT codegen: tensor unary must be emitted via statement context"
    );
  }

  const operand = emitExpr(expr.operand, ctx);
  return emitScalarUnaryOp(expr.op, operand, C_SCALAR_TARGET);
}

function emitCall(expr: JitExpr & { tag: "Call" }, ctx: EmitCtx): string {
  // Tensor unary: must be emitted in statement context.
  if (isTensorExpr(expr) && expr.name in C_TENSOR_UNARY_OPS) {
    throw new Error(
      "C-JIT codegen: tensor unary call must be emitted via statement context"
    );
  }
  // Tensor reduction: result is scalar. Emit reduction inline.
  if (!isTensorExpr(expr) && expr.name in C_TENSOR_REDUCTION_OPS) {
    const opEnum = TENSOR_REDUCE_OP[expr.name];
    if (!opEnum) {
      throw new Error(
        `C-JIT codegen: tensor reduction ${expr.name} has no opcode`
      );
    }
    const arg = expr.args[0];
    if (arg.tag === "Var" && ctx.tensorVars.has(arg.name)) {
      return `numbl_reduce_flat(${opEnum}, ${tensorData(arg.name)}, ${tensorLen(arg.name)})`;
    }
    // Non-Var tensor arg: emit the tensor expression to a scratch,
    // then reduce. Requires statement-level context — use pendingStmts.
    if (ctx.pendingStmts) {
      const tensorResult = emitTensorExprToStmts(
        ctx.pendingStmts.lines,
        ctx.pendingStmts.indent,
        arg,
        ctx
      );
      return `numbl_reduce_flat(${opEnum}, ${tensorResult.data}, ${tensorResult.len})`;
    }
    throw new Error(
      `C-JIT codegen: reduction of complex tensor expr outside statement context`
    );
  }
  // tic/toc: timer builtins via __tic_state (helpers live in jit_runtime.a).
  if (expr.name === "tic" && expr.args.length === 0) {
    ctx.needsTicState = true;
    return `numbl_tic(__tic_state)`;
  }
  if (expr.name === "toc" && expr.args.length === 0) {
    ctx.needsTicState = true;
    return `numbl_toc(__tic_state)`;
  }
  // length / isempty on a tensor Var. The common case is a 1-D vector
  // where `length == data.length` and `isempty == (length == 0)`; for
  // multi-D tensors the JIT's type system currently lowers through
  // `$h.ib_length` / `$h.ib_isempty`, so this fast path only fires
  // for vectors via the tensor Var shape plumbing. Keep it simple: use
  // `_len` for length, and emit the == 0 compare for isempty.
  if (
    (expr.name === "length" || expr.name === "isempty") &&
    expr.args.length === 1 &&
    expr.args[0].tag === "Var" &&
    ctx.tensorVars.has((expr.args[0] as JitExpr & { tag: "Var" }).name)
  ) {
    const name = (expr.args[0] as JitExpr & { tag: "Var" }).name;
    const lenCode = tensorLen(name);
    if (expr.name === "length") {
      if (ctx.freshAllocTensors.has(name)) {
        // Dynamic tensors carry d0/d1 locals; use max(d0, d1) so
        // length on a matrix reports the larger dim (MATLAB convention).
        return `((double)((${tensorD0(name)} > ${tensorD1(name)}) ? ${tensorD0(name)} : ${tensorD1(name)}))`;
      }
      return `((double)${lenCode})`;
    }
    // isempty
    return `((double)(${lenCode} == 0))`;
  }
  // Scalar math builtin — query the builtin's own C emission hook.
  const ib = getIBuiltin(expr.name);
  if (ib?.jitEmitC) {
    const args = expr.args.map(a => emitExpr(a, ctx));
    const argTypes = expr.args.map(a => a.jitType);
    const fast = ib.jitEmitC(args, argTypes);
    if (fast) return fast;
  }
  throw new Error(`C-JIT codegen: unmapped builtin ${expr.name}`);
}

function emitTruthiness(expr: JitExpr, ctx: EmitCtx): string {
  // Tensor-valued Binary/Unary in condition context: skip the shared
  // walker's comparison/logical switches (they would emit a garbage
  // pointer compare) and route to value-form so emitExpr raises the
  // "must be emitted via statement context" error.
  if (
    (expr.tag === "Binary" && isTensorExpr(expr)) ||
    (expr.tag === "Unary" &&
      expr.op === UnaryOperation.Not &&
      isTensorExpr(expr.operand))
  ) {
    return `((${emitExpr(expr, ctx)}) != 0.0)`;
  }
  return emitScalarTruthiness(expr, e => emitExpr(e, ctx), C_SCALAR_TARGET);
}

// ── Statement emission ────────────────────────────────────────────────

function emitStmts(
  lines: string[],
  stmts: JitStmt[],
  indent: string,
  ctx: EmitCtx
): void {
  if (!ctx.fuse) {
    for (const s of stmts) emitStmt(lines, s, indent, ctx);
    return;
  }

  // Fusion enabled: find fusible chains, emit them as fused loops,
  // emit everything else via the per-op path.
  const chains = findFusibleChains(stmts, ctx.paramTensorNames, ctx.tensorVars);

  // Build a set of statement indices covered by chains.
  const coveredByChain = new Map<
    number,
    { chain: ReturnType<typeof findFusibleChains>[0] }
  >();
  for (const chain of chains) {
    coveredByChain.set(chain.startIdx, { chain });
  }

  let i = 0;
  while (i < stmts.length) {
    const entry = coveredByChain.get(i);
    if (entry) {
      emitFusedChain(
        lines,
        indent,
        entry.chain,
        ctx.tensorVars,
        ctx.paramTensorNames,
        ctx.outputTensorNames,
        ctx.localTensorNames,
        ctx.openmp
      );
      i += entry.chain.length;
    } else {
      emitStmt(lines, stmts[i], indent, ctx);
      i++;
    }
  }
}

/** For local tensor dests (not output, not param), ensure the buffer
 *  is allocated before writing. Emits a realloc-like pattern. */
function emitEnsureLocalBuf(
  lines: string[],
  indent: string,
  destName: string,
  lenExpr: string,
  ctx: EmitCtx
): void {
  if (!ctx.localTensorNames.has(destName)) return;
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  lines.push(`${indent}${dLen} = ${lenExpr};`);
  lines.push(
    `${indent}if (!${dData}) ${dData} = (double *)malloc((size_t)${dLen} * sizeof(double));`
  );
}

/** Emit a fresh-alloc pattern: free old buffer, malloc new, fill with
 *  the cells of a TensorLiteral. `destName` identifies the tracked
 *  `v_<dest>_data / _len / _d0 / _d1` locals. */
function emitTensorLiteralAssign(
  lines: string[],
  indent: string,
  destName: string,
  expr: JitExpr & { tag: "TensorLiteral" },
  ctx: EmitCtx
): void {
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  const dD0 = tensorD0(destName);
  const dD1 = tensorD1(destName);
  const nRows = expr.nRows;
  const nCols = expr.nCols;
  if (nRows === 0 || nCols === 0) {
    lines.push(`${indent}if (${dData}) free(${dData});`);
    lines.push(`${indent}${dData} = NULL;`);
    lines.push(`${indent}${dLen} = 0;`);
    lines.push(`${indent}${dD0} = 0;`);
    lines.push(`${indent}${dD1} = 0;`);
    return;
  }
  const nLen = nRows * nCols;
  lines.push(`${indent}{`);
  const inner = indent + "  ";
  // Evaluate all cells into a scratch C array first. Use the nearest
  // enclosing statement context so nested fresh-alloc sub-exprs can
  // emit their own preliminaries.
  lines.push(
    `${inner}double *__tl = (double *)malloc(${nLen} * sizeof(double));`
  );
  // Column-major fill: iterate columns, then rows.
  for (let c = 0; c < nCols; c++) {
    for (let r = 0; r < nRows; r++) {
      const cell = expr.rows[r][c];
      const cellCode = emitExpr(cell, ctx);
      lines.push(`${inner}__tl[${c * nRows + r}] = ${cellCode};`);
    }
  }
  lines.push(`${inner}if (${dData}) free(${dData});`);
  lines.push(`${inner}${dData} = __tl;`);
  lines.push(`${inner}${dLen} = ${nLen};`);
  lines.push(`${inner}${dD0} = ${nRows};`);
  lines.push(`${inner}${dD1} = ${nCols};`);
  lines.push(`${indent}}`);
}

/** Emit zeros(n) / zeros(n, m) / ones(...) into the tracked locals.
 *  1-arg produces an NxN matrix (MATLAB semantics). 2-arg is NxM. */
function emitZerosOnesAssign(
  lines: string[],
  indent: string,
  destName: string,
  expr: JitExpr & { tag: "Call" },
  ctx: EmitCtx
): void {
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  const dD0 = tensorD0(destName);
  const dD1 = tensorD1(destName);
  const fill = expr.name === "ones" ? "1.0" : "0.0";
  // Round shape args like MATLAB (any fractional is rounded to nearest).
  const roundArg = (a: JitExpr): string => {
    let s = emitExpr(a, ctx);
    if (!isKnownInteger(a.jitType)) s = `(int64_t)round(${s})`;
    else s = `(int64_t)(${s})`;
    return s;
  };
  const arg0 = roundArg(expr.args[0]);
  const arg1 = expr.args.length === 2 ? roundArg(expr.args[1]) : arg0;
  lines.push(`${indent}{`);
  const inner = indent + "  ";
  lines.push(`${inner}int64_t __zr = ${arg0};`);
  lines.push(`${inner}int64_t __zc = ${arg1};`);
  lines.push(`${inner}if (__zr < 0) __zr = 0;`);
  lines.push(`${inner}if (__zc < 0) __zc = 0;`);
  lines.push(`${inner}int64_t __zn = __zr * __zc;`);
  lines.push(`${inner}double *__zb = NULL;`);
  lines.push(`${inner}if (__zn > 0) {`);
  if (expr.name === "zeros") {
    lines.push(
      `${inner}  __zb = (double *)calloc((size_t)__zn, sizeof(double));`
    );
  } else {
    lines.push(
      `${inner}  __zb = (double *)malloc((size_t)__zn * sizeof(double));`
    );
    lines.push(
      `${inner}  for (int64_t __i = 0; __i < __zn; __i++) __zb[__i] = ${fill};`
    );
  }
  lines.push(`${inner}}`);
  lines.push(`${inner}if (${dData}) free(${dData});`);
  lines.push(`${inner}${dData} = __zb;`);
  lines.push(`${inner}${dLen} = __zn;`);
  lines.push(`${inner}${dD0} = __zr;`);
  lines.push(`${inner}${dD1} = __zc;`);
  lines.push(`${indent}}`);
}

/** Emit VConcatGrow `dest = [base; value]`. Allocates a `(old_len + 1, 1)`
 *  buffer, copies base's contents, appends value, and transfers into the
 *  tracked locals. Handles the self-grow case (base == dest): the memcpy
 *  completes before the old buffer is freed. */
function emitVConcatGrowAssign(
  lines: string[],
  indent: string,
  destName: string,
  expr: JitExpr & { tag: "VConcatGrow" },
  ctx: EmitCtx
): void {
  if (expr.base.tag !== "Var") {
    throw new Error("C-JIT codegen: VConcatGrow base must be a Var");
  }
  const baseName = expr.base.name;
  if (!ctx.tensorVars.has(baseName)) {
    throw new Error(
      `C-JIT codegen: VConcatGrow base '${baseName}' is not a tensor var`
    );
  }
  const bData = tensorData(baseName);
  const bLen = tensorLen(baseName);
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  const dD0 = tensorD0(destName);
  const dD1 = tensorD1(destName);
  const valueCode = emitExpr(expr.value, ctx);
  lines.push(`${indent}{`);
  const inner = indent + "  ";
  lines.push(`${inner}int64_t __vo = ${bLen};`);
  lines.push(`${inner}int64_t __vn = __vo + 1;`);
  lines.push(
    `${inner}double *__vb = (double *)malloc((size_t)__vn * sizeof(double));`
  );
  lines.push(
    `${inner}if (__vo > 0) memcpy(__vb, ${bData}, (size_t)__vo * sizeof(double));`
  );
  lines.push(`${inner}__vb[__vo] = ${valueCode};`);
  // Free old dest buffer AFTER memcpy (handles self-grow where base == dest).
  lines.push(`${inner}if (${dData}) free(${dData});`);
  lines.push(`${inner}${dData} = __vb;`);
  lines.push(`${inner}${dLen} = __vn;`);
  lines.push(`${inner}${dD0} = __vn;`);
  lines.push(`${inner}${dD1} = 1;`);
  lines.push(`${indent}}`);
}

/** Emit a tensor-result Assign: handles Binary, Unary, Call on tensors. */
function emitTensorAssign(
  lines: string[],
  indent: string,
  destName: string,
  expr: JitExpr,
  ctx: EmitCtx
): void {
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  const needsAlloc = ctx.localTensorNames.has(destName);

  if (expr.tag === "TensorLiteral") {
    emitTensorLiteralAssign(lines, indent, destName, expr, ctx);
    return;
  }

  if (expr.tag === "VConcatGrow") {
    emitVConcatGrowAssign(lines, indent, destName, expr, ctx);
    return;
  }

  if (
    expr.tag === "Call" &&
    (expr.name === "zeros" || expr.name === "ones") &&
    isTensorExpr(expr)
  ) {
    emitZerosOnesAssign(lines, indent, destName, expr, ctx);
    return;
  }

  if (expr.tag === "RangeSliceRead") {
    // `r = src(a:b)` into a tensor var. Two cases:
    //  - Local tensor: the local owns its buffer. free-then-malloc
    //    the exact range length, numbl_copyRange1r fills in-place.
    //  - Output tensor: v_<name>_data IS the preallocated JS output
    //    buffer (sized from firstTensorLen in the JS wrapper). Emit
    //    into a scratch first, then memcpy into the output buffer.
    //    The output buffer is large enough whenever the slice length
    //    doesn't exceed the first tensor input's length, which is the
    //    JS-JIT subarrayCopy1r contract by construction.
    if (ctx.localTensorNames.has(destName)) {
      emitRangeSliceReadToBuf(lines, indent, expr, ctx, dData, dLen);
      return;
    }
    if (ctx.outputTensorNames.has(destName)) {
      const scratch = emitRangeSliceReadToStmts(lines, indent, expr, ctx);
      lines.push(`${indent}${dLen} = ${scratch.len};`);
      lines.push(
        `${indent}if (${dLen} > 0) memcpy(${dData}, ${scratch.data}, (size_t)${dLen} * sizeof(double));`
      );
      return;
    }
    throw new Error(
      `C-JIT codegen: RangeSliceRead assign to tensor param '${destName}' unsupported`
    );
  }

  if (expr.tag === "Var" && ctx.tensorVars.has(expr.name)) {
    const srcName = expr.name;
    const srcData = tensorData(srcName);
    const srcLen = tensorLen(srcName);
    const destIsDynamic = ctx.freshAllocTensors.has(destName);
    const destIsFixedOutput =
      ctx.outputTensorNames.has(destName) && !ctx.dynamicOutputs.has(destName);
    if (destIsDynamic) {
      // Deep-copy: the source's buffer may be freed later in the body
      // (e.g. the test_loop_slice_write `tmp_pt = out_pt; out_pt =
      // zeros(...)` pattern). A pointer alias would read freed memory.
      // Free the old dest buffer, malloc a fresh one, copy, update
      // shape locals. Guarded by a self-alias check since `x = x` is
      // a no-op.
      const dD0 = tensorD0(destName);
      const dD1 = tensorD1(destName);
      const srcD0Expr = ctx.freshAllocTensors.has(srcName)
        ? tensorD0(srcName)
        : (ctx.tensorMaxDim.get(srcName) ?? 0) >= 2
          ? tensorD0(srcName)
          : srcLen;
      const srcD1Expr = ctx.freshAllocTensors.has(srcName)
        ? tensorD1(srcName)
        : (ctx.tensorMaxDim.get(srcName) ?? 0) >= 3
          ? tensorD1(srcName)
          : "1";
      lines.push(`${indent}if (${dData} != ${srcData}) {`);
      lines.push(`${indent}  if (${dData}) free(${dData});`);
      lines.push(`${indent}  ${dLen} = ${srcLen};`);
      lines.push(
        `${indent}  ${dData} = (${dLen} > 0) ? (double *)malloc((size_t)${dLen} * sizeof(double)) : NULL;`
      );
      lines.push(
        `${indent}  if (${dLen} > 0) memcpy(${dData}, ${srcData}, (size_t)${dLen} * sizeof(double));`
      );
      lines.push(`${indent}}`);
      lines.push(`${indent}${dD0} = ${srcD0Expr};`);
      lines.push(`${indent}${dD1} = ${srcD1Expr};`);
      return;
    }
    if (destIsFixedOutput) {
      // Fixed output: the preallocated _buf must hold the result data
      // when the C function returns. Pointer-swap would leave the buf
      // untouched, so memcpy the source contents into it instead.
      // (Skip the copy when src and dest already share the buffer, e.g.
      // a param-output reassigned to itself.)
      lines.push(`${indent}${dLen} = ${srcLen};`);
      lines.push(
        `${indent}if (${dData} != ${srcData}) memcpy(${dData}, ${srcData}, (size_t)${dLen} * sizeof(double));`
      );
    } else {
      lines.push(`${indent}${dData} = ${srcData};`);
      lines.push(`${indent}${dLen} = ${srcLen};`);
    }
    return;
  }

  if (expr.tag === "Binary" && isTensorExpr(expr)) {
    // For local tensors, we need to allocate before the binary op writes.
    // Find the tensor length from the operands first.
    if (needsAlloc) {
      const lenExpr = findTensorLenExpr(expr, ctx);
      emitEnsureLocalBuf(lines, indent, destName, lenExpr, ctx);
    }
    emitTensorBinaryStmts(lines, indent, expr, ctx, dData, dLen);
    return;
  }

  if (expr.tag === "Unary" && isTensorExpr(expr)) {
    if (expr.op === UnaryOperation.Plus) {
      emitTensorAssign(lines, indent, destName, expr.operand, ctx);
      return;
    }
    if (expr.op === UnaryOperation.Minus) {
      const operand = emitTensorExprToStmts(lines, indent, expr.operand, ctx);
      if (needsAlloc) {
        emitEnsureLocalBuf(lines, indent, destName, operand.len, ctx);
      } else {
        lines.push(`${indent}${dLen} = ${operand.len};`);
      }
      lines.push(
        `${indent}numbl_real_scalar_binary_elemwise(NUMBL_REAL_BIN_MUL, (size_t)${dLen}, -1.0, ${operand.data}, 1, ${dData});`
      );
      return;
    }
  }

  if (
    expr.tag === "Call" &&
    isTensorExpr(expr) &&
    expr.name in C_TENSOR_UNARY_OPS
  ) {
    const opEnum = TENSOR_UNARY_OP[expr.name];
    const arg = emitTensorExprToStmts(lines, indent, expr.args[0], ctx);
    if (needsAlloc) {
      emitEnsureLocalBuf(lines, indent, destName, arg.len, ctx);
    } else {
      lines.push(`${indent}${dLen} = ${arg.len};`);
    }
    lines.push(
      `${indent}numbl_real_unary_elemwise(${opEnum}, (size_t)${dLen}, ${arg.data}, ${dData});`
    );
    return;
  }

  // Two-arg tensor binary builtin (max, min, atan2, hypot, mod, rem):
  // emit a per-element loop calling the C math function.
  if (
    expr.tag === "Call" &&
    isTensorExpr(expr) &&
    C_TENSOR_BINARY_BUILTINS.has(expr.name) &&
    expr.args.length === 2
  ) {
    const BINARY_BUILTIN_TO_C: Record<string, string> = {
      max: "fmax",
      min: "fmin",
      atan2: "atan2",
      hypot: "hypot",
      mod: "numbl_mod",
      rem: "fmod",
    };
    const cFn = BINARY_BUILTIN_TO_C[expr.name];
    if (!cFn) {
      throw new Error(`C-JIT codegen: unmapped binary builtin: ${expr.name}`);
    }

    const left = expr.args[0];
    const right = expr.args[1];
    const leftIsTensor = left.jitType.kind === "tensor";
    const rightIsTensor = right.jitType.kind === "tensor";

    if (leftIsTensor && rightIsTensor) {
      const lArg = emitTensorExprToStmts(lines, indent, left, ctx);
      const rArg = emitTensorExprToStmts(lines, indent, right, ctx);
      if (needsAlloc)
        emitEnsureLocalBuf(lines, indent, destName, lArg.len, ctx);
      else lines.push(`${indent}${dLen} = ${lArg.len};`);
      lines.push(`${indent}for (int64_t __i = 0; __i < ${dLen}; __i++)`);
      lines.push(
        `${indent}  ${dData}[__i] = ${cFn}(${lArg.data}[__i], ${rArg.data}[__i]);`
      );
    } else if (leftIsTensor) {
      const lArg = emitTensorExprToStmts(lines, indent, left, ctx);
      const rScalar = emitExpr(right, ctx);
      if (needsAlloc)
        emitEnsureLocalBuf(lines, indent, destName, lArg.len, ctx);
      else lines.push(`${indent}${dLen} = ${lArg.len};`);
      lines.push(`${indent}for (int64_t __i = 0; __i < ${dLen}; __i++)`);
      lines.push(
        `${indent}  ${dData}[__i] = ${cFn}(${lArg.data}[__i], ${rScalar});`
      );
    } else {
      const lScalar = emitExpr(left, ctx);
      const rArg = emitTensorExprToStmts(lines, indent, right, ctx);
      if (needsAlloc)
        emitEnsureLocalBuf(lines, indent, destName, rArg.len, ctx);
      else lines.push(`${indent}${dLen} = ${rArg.len};`);
      lines.push(`${indent}for (int64_t __i = 0; __i < ${dLen}; __i++)`);
      lines.push(
        `${indent}  ${dData}[__i] = ${cFn}(${lScalar}, ${rArg.data}[__i]);`
      );
    }
    return;
  }

  throw new Error(`C-JIT codegen: unhandled tensor assign RHS: ${expr.tag}`);
}

/** Emit an Assign where the RHS is a reduction on a complex tensor expression
 *  (not just a Var). This needs statement-level emission for the tensor
 *  sub-expression, then a reduction call on the scratch buffer. */
function emitReductionOfTensorExpr(
  lines: string[],
  indent: string,
  destName: string,
  callExpr: JitExpr & { tag: "Call" },
  ctx: EmitCtx
): void {
  const opEnum = TENSOR_REDUCE_OP[callExpr.name];
  const arg = callExpr.args[0];
  const tensorResult = emitTensorExprToStmts(lines, indent, arg, ctx);
  lines.push(
    `${indent}${mangle(destName)} = numbl_reduce_flat(${opEnum}, ${tensorResult.data}, ${tensorResult.len});`
  );
}

function emitStmt(
  lines: string[],
  stmt: JitStmt,
  indent: string,
  ctx: EmitCtx
): void {
  switch (stmt.tag) {
    case "Assign": {
      // Tensor-result assign: needs statement-level emission.
      if (
        stmt.expr.jitType.kind === "tensor" &&
        ctx.tensorVars.has(stmt.name)
      ) {
        emitTensorAssign(lines, indent, stmt.name, stmt.expr, ctx);
        return;
      }
      // Reduction of a complex tensor expression (not just a Var).
      if (
        stmt.expr.tag === "Call" &&
        stmt.expr.name in C_TENSOR_REDUCTION_OPS &&
        stmt.expr.args[0]?.jitType.kind === "tensor" &&
        stmt.expr.args[0].tag !== "Var"
      ) {
        emitReductionOfTensorExpr(lines, indent, stmt.name, stmt.expr, ctx);
        return;
      }
      ctx.pendingStmts = { lines, indent };
      const rhs = emitExpr(stmt.expr, ctx);
      ctx.pendingStmts = undefined;
      lines.push(`${indent}${mangle(stmt.name)} = ${rhs};`);
      return;
    }

    case "ExprStmt":
      ctx.pendingStmts = { lines, indent };
      lines.push(`${indent}(void)(${emitExpr(stmt.expr, ctx)});`);
      ctx.pendingStmts = undefined;
      return;

    case "AssignIndex": {
      // Feasibility already rejected non-real-tensor / non-scalar-value
      // cases and arity > 3. The emitter routes to numbl_set1r_h /
      // numbl_set2r_h / numbl_set3r_h based on index count.
      const n = stmt.indices.length;
      if (n < 1 || n > 3) {
        throw new Error(
          `C-JIT codegen: AssignIndex arity ${n} unsupported (only 1D/2D/3D)`
        );
      }
      if (!ctx.tensorVars.has(stmt.baseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndex base '${stmt.baseName}' is not a tensor var`
        );
      }
      ctx.needsErrorFlag = true;
      const name = stmt.baseName;
      const data = tensorData(name);
      const len = tensorLen(name);
      const idxCodes = stmt.indices.map(idx => {
        let s = emitExpr(idx, ctx);
        if (!isKnownInteger(idx.jitType)) s = `round(${s})`;
        return s;
      });
      const v = emitExpr(stmt.value, ctx);
      if (n === 1) {
        lines.push(
          `${indent}numbl_set1r_h(${data}, (size_t)${len}, ${idxCodes[0]}, ${v}, __err_flag);`
        );
      } else if (n === 2) {
        lines.push(
          `${indent}numbl_set2r_h(${data}, (size_t)${len}, (size_t)${tensorD0(name)}, ${idxCodes[0]}, ${idxCodes[1]}, ${v}, __err_flag);`
        );
      } else {
        lines.push(
          `${indent}numbl_set3r_h(${data}, (size_t)${len}, (size_t)${tensorD0(name)}, (size_t)${tensorD1(name)}, ${idxCodes[0]}, ${idxCodes[1]}, ${idxCodes[2]}, ${v}, __err_flag);`
        );
      }
      return;
    }

    case "AssignIndexRange": {
      // `dst(dStart:dEnd) = src(sStart:sEnd)` or `dst(dStart:dEnd) = src`
      // (whole-tensor RHS form — srcStart/srcEnd are both null; we
      // substitute `1` and srcLen, matching the JS-JIT's codegen).
      // Feasibility already required both base and src to be tensor
      // params, so the data/len ABI is guaranteed. Emit numbl_setRange1r_h
      // with __err_flag — OOB (1.0) and length-mismatch (3.0) are both
      // reported back through the shared flag and translated in the JS
      // wrapper.
      if (!ctx.tensorVars.has(stmt.baseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexRange base '${stmt.baseName}' is not a tensor var`
        );
      }
      if (!ctx.tensorVars.has(stmt.srcBaseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexRange src '${stmt.srcBaseName}' is not a tensor var`
        );
      }
      ctx.needsErrorFlag = true;
      const dData = tensorData(stmt.baseName);
      const dLen = tensorLen(stmt.baseName);
      const sData = tensorData(stmt.srcBaseName);
      const sLen = tensorLen(stmt.srcBaseName);
      const emitRI = (e: JitExpr): string => {
        let code = emitExpr(e, ctx);
        if (!isKnownInteger(e.jitType)) code = `round(${code})`;
        return code;
      };
      const dStart = emitRI(stmt.dstStart);
      const dEnd = emitRI(stmt.dstEnd);
      const srcStart = stmt.srcStart !== null ? emitRI(stmt.srcStart) : `1.0`;
      const srcEnd =
        stmt.srcEnd !== null ? emitRI(stmt.srcEnd) : `(double)${sLen}`;
      lines.push(
        `${indent}numbl_setRange1r_h(${dData}, (size_t)${dLen}, ${dStart}, ${dEnd}, ${sData}, (size_t)${sLen}, ${srcStart}, ${srcEnd}, __err_flag);`
      );
      return;
    }

    case "AssignIndexCol": {
      // `dst(:, j) = src` — dst must have arity 2 so tensorMaxDim
      // guarantees the `_d0` (row count) ABI param is present. Size
      // mismatch (srcLen != dstRows) reports err_flag 3.0; OOB col
      // that would require dst growth reports 2.0 (soft-bail).
      if (!ctx.tensorVars.has(stmt.baseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexCol base '${stmt.baseName}' is not a tensor var`
        );
      }
      if (!ctx.tensorVars.has(stmt.srcBaseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexCol src '${stmt.srcBaseName}' is not a tensor var`
        );
      }
      ctx.needsErrorFlag = true;
      const dData = tensorData(stmt.baseName);
      const dLen = tensorLen(stmt.baseName);
      const dRows = tensorD0(stmt.baseName);
      const sData = tensorData(stmt.srcBaseName);
      const sLen = tensorLen(stmt.srcBaseName);
      let colCode = emitExpr(stmt.colIndex, ctx);
      if (!isKnownInteger(stmt.colIndex.jitType)) colCode = `round(${colCode})`;
      lines.push(
        `${indent}numbl_setCol2r_h(${dData}, (size_t)${dRows}, (size_t)${dLen}, ${colCode}, ${sData}, (size_t)${sLen}, __err_flag);`
      );
      return;
    }

    case "If": {
      lines.push(`${indent}if (${emitTruthiness(stmt.cond, ctx)}) {`);
      emitStmts(lines, stmt.thenBody, indent + "  ", ctx);
      for (const eib of stmt.elseifBlocks) {
        lines.push(`${indent}} else if (${emitTruthiness(eib.cond, ctx)}) {`);
        emitStmts(lines, eib.body, indent + "  ", ctx);
      }
      if (stmt.elseBody) {
        lines.push(`${indent}} else {`);
        emitStmts(lines, stmt.elseBody, indent + "  ", ctx);
      }
      lines.push(`${indent}}`);
      return;
    }

    case "For": {
      const v = mangle(stmt.varName);
      const t = `__t${++ctx.tmp.n}`;
      const start = emitExpr(stmt.start, ctx);
      const end = emitExpr(stmt.end, ctx);
      const step = stmt.step ? emitExpr(stmt.step, ctx) : "1.0";
      if (stmt.step) {
        lines.push(
          `${indent}for (double ${t} = ${start}; (${step}) != 0.0 && ((${step}) > 0.0 ? ${t} <= (${end}) : ${t} >= (${end})); ${t} += (${step})) {`
        );
      } else {
        lines.push(
          `${indent}for (double ${t} = ${start}; ${t} <= (${end}); ${t} += 1.0) {`
        );
      }
      lines.push(`${indent}  ${v} = ${t};`);
      emitStmts(lines, stmt.body, indent + "  ", ctx);
      lines.push(`${indent}}`);
      return;
    }

    case "While":
      lines.push(`${indent}while (${emitTruthiness(stmt.cond, ctx)}) {`);
      emitStmts(lines, stmt.body, indent + "  ", ctx);
      lines.push(`${indent}}`);
      return;

    case "Break":
      lines.push(`${indent}break;`);
      return;

    case "Continue":
      lines.push(`${indent}continue;`);
      return;

    case "Return":
      // No-op: return is handled by the epilogue.
      return;

    case "SetLoc":
      return;

    case "AssertCJit":
      // C-JIT codegen reached → assertion satisfied, elide.
      return;

    default:
      throw new Error(`C-JIT codegen: unsupported stmt ${stmt.tag}`);
  }
}

// ── Top-level ─────────────────────────────────────────────────────────

function findTensorLocals(body: JitStmt[], out: Set<string>): void {
  const visitStmt = (s: JitStmt): void => {
    switch (s.tag) {
      case "Assign":
        if (s.expr.jitType.kind === "tensor") out.add(s.name);
        break;
      case "If":
        s.thenBody.forEach(visitStmt);
        s.elseifBlocks.forEach(eb => eb.body.forEach(visitStmt));
        if (s.elseBody) s.elseBody.forEach(visitStmt);
        break;
      case "For":
        s.body.forEach(visitStmt);
        break;
      case "While":
        s.body.forEach(visitStmt);
        break;
      default:
        break;
    }
  };
  body.forEach(visitStmt);
}

/** Is `expr` a fresh-tensor-producing RHS? TensorLiteral, VConcatGrow,
 *  or a call to `zeros` / `ones`. Each of these allocates a brand-new
 *  buffer; when the Assign dest is a tensor output, that output must
 *  use the dynamic-output ABI (double **out) rather than the legacy
 *  pre-allocated-buffer ABI. */
function isFreshTensorRhs(expr: JitExpr): boolean {
  if (expr.tag === "TensorLiteral") return true;
  if (expr.tag === "VConcatGrow") return true;
  if (expr.tag === "Call" && (expr.name === "zeros" || expr.name === "ones")) {
    return expr.jitType.kind === "tensor";
  }
  return false;
}

/** Walk body to collect tensor names that, directly or transitively,
 *  hold a freshly-allocated (C-owned) buffer whose size is not tied to
 *  any input tensor. Seeded with names receiving a direct fresh-tensor
 *  RHS (TensorLiteral / VConcatGrow / zeros / ones); propagates through
 *  Var-alias assigns since `dst = src` makes `dst` share `src`'s buffer.
 *  These names need `_d0` / `_d1` tracked as C locals so the shape can
 *  change mid-function, and — for tensor outputs — the dynamic-output
 *  ABI (the JS wrapper's first-tensor-input-sized preallocated buffer
 *  can't fit them). */
function findFreshTensorAssignNames(body: JitStmt[], out: Set<string>): void {
  const visitStmt = (s: JitStmt): void => {
    switch (s.tag) {
      case "Assign":
        if (s.expr.jitType.kind === "tensor" && isFreshTensorRhs(s.expr)) {
          out.add(s.name);
        }
        break;
      case "If":
        s.thenBody.forEach(visitStmt);
        s.elseifBlocks.forEach(eb => eb.body.forEach(visitStmt));
        if (s.elseBody) s.elseBody.forEach(visitStmt);
        break;
      case "For":
        s.body.forEach(visitStmt);
        break;
      case "While":
        s.body.forEach(visitStmt);
        break;
      default:
        break;
    }
  };
  body.forEach(visitStmt);

  // Propagate through Var-alias: `dst = src` aliases src's buffer into
  // dst, so dst is dynamic iff src is. RangeSliceRead also produces a
  // fresh allocation whose size is derived from runtime endpoints —
  // its destination joins the dynamic set too. Run to fixed point so
  // chains like `a = zeros(n); b = a; c = b` all propagate.
  let changed = true;
  while (changed) {
    changed = false;
    const propagate = (s: JitStmt): void => {
      switch (s.tag) {
        case "Assign":
          if (s.expr.jitType.kind === "tensor" && !out.has(s.name)) {
            const e = s.expr;
            if (e.tag === "Var" && out.has(e.name)) {
              out.add(s.name);
              changed = true;
            } else if (e.tag === "RangeSliceRead") {
              out.add(s.name);
              changed = true;
            }
          }
          break;
        case "If":
          s.thenBody.forEach(propagate);
          s.elseifBlocks.forEach(eb => eb.body.forEach(propagate));
          if (s.elseBody) s.elseBody.forEach(propagate);
          break;
        case "For":
          s.body.forEach(propagate);
          break;
        case "While":
          s.body.forEach(propagate);
          break;
        default:
          break;
      }
    };
    body.forEach(propagate);
  }
}

/** Collect tensor-param names that are written through AssignIndex /
 *  AssignIndexRange / AssignIndexCol. All three shape the same
 *  MATLAB-value-semantics problem: the write must not leak to the
 *  caller's buffer, so pure-input tensor params need an unshare-at-
 *  entry copy. */
function findAssignIndexTargets(body: JitStmt[], out: Set<string>): void {
  const visitStmt = (s: JitStmt): void => {
    switch (s.tag) {
      case "AssignIndex":
      case "AssignIndexRange":
      case "AssignIndexCol":
        out.add(s.baseName);
        break;
      case "If":
        s.thenBody.forEach(visitStmt);
        s.elseifBlocks.forEach(eb => eb.body.forEach(visitStmt));
        if (s.elseBody) s.elseBody.forEach(visitStmt);
        break;
      case "For":
        s.body.forEach(visitStmt);
        break;
      case "While":
        s.body.forEach(visitStmt);
        break;
      default:
        break;
    }
  };
  body.forEach(visitStmt);
}

export interface CParamDesc {
  name: string;
  kind: "scalar" | "tensor";
  /** For tensor params: max indexing arity the body uses (1, 2, or 3).
   *  Drives the extra `_d0` / `_d1` shape args the JS wrapper must
   *  marshal. `undefined` means the tensor is only used in whole-tensor
   *  ops (legacy data/len ABI). */
  ndim?: number;
}

/** Per-output descriptor. Tells the JS wrapper how to marshal outputs. */
export interface COutputDesc {
  name: string;
  kind: "scalar" | "boolean" | "tensor";
  /** True for tensor outputs using the dynamic-output ABI: the C code
   *  malloc's the buffer and transfers ownership via `double **` and
   *  extra d0/d1 out-slots. The JS wrapper decodes the pointer, copies
   *  into a fresh Float64Array, and frees the C allocation. */
  dynamic?: boolean;
}

export interface GenerateCResult {
  cSource: string;
  cFnName: string;
  paramDescs: CParamDesc[];
  outputDescs: COutputDesc[];
  /** True when any tensor is involved (params, locals, or outputs). */
  usesTensors: boolean;
  /** koffi function signature string for declaring the C function. */
  koffiSignature: string;
  /** True when tic/toc are used — the function has an extra `double*` param. */
  needsTicState: boolean;
  /** True when any Index read was emitted — the function has an extra
   *  `double *__err_flag` trailing param. */
  needsErrorFlag: boolean;
}

export function generateC(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  argTypes: JitType[],
  _outputType: JitType | null,
  outputTypes: JitType[],
  fnName: string,
  fuse?: boolean,
  openmp?: boolean
): GenerateCResult {
  if (params.length !== argTypes.length) {
    throw new Error("C-JIT codegen: params/argTypes length mismatch");
  }

  const tensorVars = new Set<string>();
  for (let i = 0; i < params.length; i++) {
    if (argTypes[i].kind === "tensor") tensorVars.add(params[i]);
  }
  findTensorLocals(body, tensorVars);

  const effectiveOutputs = outputs.slice(0, nargout || 1);

  // Classify outputs.
  const outputDescs: COutputDesc[] = effectiveOutputs.map((name, i) => ({
    name,
    kind:
      outputTypes[i]?.kind === "tensor"
        ? "tensor"
        : outputTypes[i]?.kind === "boolean"
          ? "boolean"
          : "scalar",
  }));

  // Note: `dynamic` on each tensor output desc is set below once the
  // `dynamicOutputs` set is computed from the body walk.

  const outputTensorNames = new Set<string>(
    outputDescs.filter(od => od.kind === "tensor").map(od => od.name)
  );
  const paramTensorNames = new Set<string>(
    params.filter((_, i) => argTypes[i].kind === "tensor")
  );
  const localTensorNames = new Set<string>(
    [...tensorVars].filter(
      v => !paramTensorNames.has(v) && !outputTensorNames.has(v)
    )
  );

  // Tensor params that receive an AssignIndex write but aren't also a named
  // output. MATLAB call-by-value requires the write to land on a buffer
  // independent of the caller's data; we emit an unshare-at-entry malloc +
  // memcpy for these in the prelude and a matching free in the epilogue.
  // (Param-outputs already get a seeded output buffer from the JS wrapper.)
  const assignIndexTargets = new Set<string>();
  findAssignIndexTargets(body, assignIndexTargets);
  const unshareTensorParams = new Set<string>();
  for (const name of assignIndexTargets) {
    if (paramTensorNames.has(name) && !outputTensorNames.has(name)) {
      unshareTensorParams.add(name);
    }
  }

  // Per-tensor max indexing arity (for multi-index Index / AssignIndex).
  // Reuses the JS hoist analyzer: any tensor Var that appears as an
  // Index base or an AssignIndex target contributes its `indices.length`
  // to the max for that name. Tensors that only participate in whole-
  // tensor ops (or don't appear at all) don't get an entry, and keep
  // the old (data, len) ABI.
  const tensorMaxDim = new Map<string, number>();
  {
    const usage = collectTensorUsage(body);
    for (const [name, u] of usage) {
      if (!u.isReal) continue;
      const d = Math.max(u.maxReadDim, u.maxWriteDim);
      if (d > 0) tensorMaxDim.set(name, d);
    }
  }

  // Fresh-tensor assignments (TensorLiteral / zeros / ones / VConcatGrow).
  // Every name hit here participates in shape-changing reassignment: emit
  // `v_<n>_d0` / `v_<n>_d1` as mutable locals initialized from whatever
  // path seeded the buffer, and update them on each fresh-alloc site.
  const freshAllocTensors = new Set<string>();
  findFreshTensorAssignNames(body, freshAllocTensors);

  // Tensor outputs that need the dynamic-output ABI. Any tensor output
  // whose value can be reassigned to a fresh-allocated buffer (directly
  // via TensorLiteral/zeros/ones/VConcatGrow) can't fit the "JS preallocates
  // a fixed buf" contract — the C code allocates the buffer itself.
  const dynamicOutputs = new Set<string>();
  for (const name of outputTensorNames) {
    if (freshAllocTensors.has(name)) dynamicOutputs.add(name);
  }
  for (const od of outputDescs) {
    if (od.kind === "tensor" && dynamicOutputs.has(od.name)) od.dynamic = true;
  }

  const ctx: EmitCtx = {
    tensorVars,
    outputTensorNames,
    scratchCount: 0,
    tmp: { n: 0 },
    usedScratch: new Set(),
    freeStmts: [],
    localTensorNames,
    fuse: fuse ?? false,
    paramTensorNames,
    needsTicState: false,
    needsErrorFlag: false,
    openmp: openmp ?? false,
    tensorMaxDim,
    freshAllocTensors,
    dynamicOutputs,
  };

  const indent = "  ";
  const bodyLines: string[] = [];

  // Tensor params that are also outputs. MATLAB's `function x = foo(x, ...)`
  // call-by-value + local-mutation semantics: the callee starts with a
  // copy of the caller's x. For fixed outputs the JS wrapper already
  // passes a seeded buffer as `_buf`; for dynamic outputs the C code
  // unshare-copies the caller's data itself (the seeded buffer travels
  // through `_buf` as const input, not as output).
  const paramOutputTensors = new Set<string>();
  for (let i = 0; i < params.length; i++) {
    if (
      argTypes[i].kind === "tensor" &&
      outputDescs.some(od => od.name === params[i] && od.kind === "tensor")
    ) {
      paramOutputTensors.add(params[i]);
    }
  }

  // Tensor params reassigned via a fresh-alloc RHS (not as a param-output)
  // need their own writable local too — treat them as unshared-at-entry
  // so that first read before reassignment sees the caller's data but
  // the free-on-reassign path stays correct.
  for (const p of paramTensorNames) {
    if (
      freshAllocTensors.has(p) &&
      !outputTensorNames.has(p) &&
      !unshareTensorParams.has(p)
    ) {
      unshareTensorParams.add(p);
    }
  }

  emitStmts(bodyLines, body, indent, ctx);

  // Names for which we emit `_d0` / `_d1` mutable locals: anything that
  // either receives a fresh-alloc (shape reassignment) or is referenced
  // as a tensor with multi-index arity. For non-dynamic tensors the _d0/
  // _d1 locals are initialized from the ABI `_in` params (existing
  // behavior). For fresh-alloc tensors they start from the same source
  // but get overwritten on each TensorLiteral / zeros / ones / VConcatGrow.
  const needsShapeLocals = (name: string): boolean => {
    if (freshAllocTensors.has(name)) return true;
    const d = tensorMaxDim.get(name) ?? 0;
    return d >= 2;
  };

  // Build the epilogue: write outputs to out-pointers, free scratch buffers.
  const epilogueLines: string[] = [];
  for (const od of outputDescs) {
    if (od.kind === "tensor") {
      if (dynamicOutputs.has(od.name)) {
        // Dynamic output: transfer ownership of the C-malloc'd buffer
        // to the caller (`*_buf_out`) and report the runtime shape.
        // The JS wrapper reads data/d0/d1 and frees the pointer after
        // copying into a JS-owned Float64Array.
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_buf_out = ${tensorData(od.name)};`
        );
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_out_len = ${tensorLen(od.name)};`
        );
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_d0_out = ${tensorD0(od.name)};`
        );
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_d1_out = ${tensorD1(od.name)};`
        );
      } else {
        // Fixed output: data already lives in the pre-allocated _buf; just
        // report the runtime length. (No memcpy: data and _buf share the
        // same pointer in the fixed path.)
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_out_len = ${tensorLen(od.name)};`
        );
      }
    } else {
      // Scalar output: write to out-pointer.
      epilogueLines.push(
        `${indent}*${mangle(od.name)}_out = ${mangle(od.name)};`
      );
    }
  }

  // Free scratch buffers.
  for (const sIdx of ctx.usedScratch) {
    epilogueLines.push(
      `${indent}if (${scratchData(sIdx)}) free(${scratchData(sIdx)});`
    );
  }

  // Free local tensor buffers (C-owned). Dynamic outputs have already
  // transferred ownership above, so exclude them.
  for (const name of ctx.localTensorNames) {
    if (dynamicOutputs.has(name)) continue;
    epilogueLines.push(
      `${indent}if (${tensorData(name)}) free(${tensorData(name)});`
    );
  }

  // Free unshare buffers for pure-input tensor params that were written.
  // If the unshared param is also a dynamic output, ownership was already
  // transferred — skip the free here to avoid a double free.
  for (const p of unshareTensorParams) {
    if (dynamicOutputs.has(p)) continue;
    epilogueLines.push(
      `${indent}if (${tensorData(p)}) free(${tensorData(p)});`
    );
  }

  const paramSet = new Set(params);
  const allLocals = [...localVars].filter(v => !paramSet.has(v)).sort();
  const preludeLines: string[] = [];

  // Emit `v_<p>_d0` / `v_<p>_d1` seeding for a tensor param. Uses the
  // `_in` suffix when the param signature is shadowed (param-output /
  // unshared) and the bare name otherwise.
  const emitParamShapeLocals = (p: string, useInSuffix: boolean): void => {
    const suf = useInSuffix ? "_in" : "";
    const d = tensorMaxDim.get(p) ?? 0;
    if (needsShapeLocals(p)) {
      // Track mutable d0/d1 locals. Seed from the ABI where available;
      // for 1-D-only tensor params we default to (len, 1) so first reads
      // on the "row vector default" shape work for fresh-alloc that
      // later overwrites.
      if (d >= 2) {
        preludeLines.push(
          `${indent}int64_t ${tensorD0(p)} = ${tensorD0(p)}${suf};`
        );
      } else {
        preludeLines.push(
          `${indent}int64_t ${tensorD0(p)} = ${tensorLen(p)}${suf};`
        );
      }
      if (d >= 3) {
        preludeLines.push(
          `${indent}int64_t ${tensorD1(p)} = ${tensorD1(p)}${suf};`
        );
      } else {
        preludeLines.push(`${indent}int64_t ${tensorD1(p)} = 1;`);
      }
    } else {
      if (d >= 2) {
        preludeLines.push(
          `${indent}int64_t ${tensorD0(p)} = ${tensorD0(p)}${suf};`
        );
      }
      if (d >= 3) {
        preludeLines.push(
          `${indent}int64_t ${tensorD1(p)} = ${tensorD1(p)}${suf};`
        );
      }
    }
  };

  // Shadow tensor input-output params with writable locals. The shape
  // dims (`_d0`, `_d1`) are also shadowed so multi-index writes on a
  // param-output use the same local names the body expects.
  for (const p of paramOutputTensors) {
    if (dynamicOutputs.has(p)) {
      // Dynamic param-output: the seeded buffer is passed as const input
      // via `_in`; we unshare-copy it into a C-owned buffer so
      // reassignment / free-on-realloc is safe, and the epilogue
      // transfers the final pointer back.
      preludeLines.push(
        `${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`
      );
      emitParamShapeLocals(p, /*useInSuffix*/ true);
      preludeLines.push(`${indent}double *${tensorData(p)} = NULL;`);
      preludeLines.push(`${indent}if (${tensorLen(p)} > 0) {`);
      preludeLines.push(
        `${indent}  ${tensorData(p)} = (double *)malloc((size_t)${tensorLen(p)} * sizeof(double));`
      );
      preludeLines.push(
        `${indent}  memcpy(${tensorData(p)}, ${tensorData(p)}_in, (size_t)${tensorLen(p)} * sizeof(double));`
      );
      preludeLines.push(`${indent}}`);
    } else {
      // Fixed param-output: data points at the seeded _buf (no copy needed).
      preludeLines.push(
        `${indent}double *${tensorData(p)} = ${mangle(p)}_buf;`
      );
      preludeLines.push(
        `${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`
      );
      emitParamShapeLocals(p, /*useInSuffix*/ true);
    }
  }

  // Unshare-at-entry: pure-input tensor params that are written (either
  // via AssignIndex family or reassigned via a fresh-alloc RHS) get a
  // malloc'd copy of the caller's data. The buffer is freed in the
  // epilogue unless it was also transferred as a dynamic output.
  for (const p of unshareTensorParams) {
    if (paramOutputTensors.has(p)) continue; // already shadowed above
    preludeLines.push(`${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`);
    emitParamShapeLocals(p, /*useInSuffix*/ true);
    preludeLines.push(`${indent}double *${tensorData(p)} = NULL;`);
    preludeLines.push(`${indent}if (${tensorLen(p)} > 0) {`);
    preludeLines.push(
      `${indent}  ${tensorData(p)} = (double *)malloc((size_t)${tensorLen(p)} * sizeof(double));`
    );
    preludeLines.push(
      `${indent}  memcpy(${tensorData(p)}, ${tensorData(p)}_in, (size_t)${tensorLen(p)} * sizeof(double));`
    );
    preludeLines.push(`${indent}}`);
  }

  // Read-only tensor params (no writes, no fresh-alloc, not an output):
  // the function-signature params `v_<p>_d0` / `v_<p>_d1` are already in
  // scope, so no prelude shape-local redeclaration is needed. `needsShapeLocals`
  // is true only for names that also need writable tracking, and those
  // are handled by the param-output / unshare branches above.

  for (const local of allLocals) {
    if (tensorVars.has(local)) {
      // Tensor locals: data pointer + length. For fixed output tensors,
      // the data pointer starts as the pre-allocated output buffer (passed
      // as param). Dynamic outputs and pure locals start as NULL — the
      // C code allocates on first fresh-assign.
      const isOutput = outputTensorNames.has(local);
      if (isOutput && !dynamicOutputs.has(local)) {
        preludeLines.push(
          `${indent}double *${tensorData(local)} = ${mangle(local)}_buf;`
        );
      } else {
        preludeLines.push(`${indent}double *${tensorData(local)} = NULL;`);
      }
      preludeLines.push(`${indent}int64_t ${tensorLen(local)} = 0;`);
      if (needsShapeLocals(local)) {
        preludeLines.push(`${indent}int64_t ${tensorD0(local)} = 0;`);
        preludeLines.push(`${indent}int64_t ${tensorD1(local)} = 0;`);
      }
    } else {
      preludeLines.push(`${indent}double ${mangle(local)} = 0.0;`);
    }
  }

  // Scratch buffer declarations.
  for (const sIdx of ctx.usedScratch) {
    preludeLines.push(`${indent}double *${scratchData(sIdx)} = NULL;`);
    preludeLines.push(`${indent}int64_t ${scratchLen(sIdx)} = 0;`);
  }

  // Build signature. The C function is always void; outputs go through
  // out-pointers. Parameters:
  //   - scalar params: double v_name
  //   - tensor params: const double *v_name_data, int64_t v_name_len
  //   - fixed tensor output: double *v_name_buf, int64_t *v_name_out_len
  //   - dynamic tensor output: double **v_name_buf_out,
  //                            int64_t *v_name_out_len,
  //                            int64_t *v_name_d0_out,
  //                            int64_t *v_name_d1_out
  //   - scalar output out-pointers: double *v_name_out
  const cFnName = `jit_${fnName}`;
  const sigParts: string[] = [];
  const koffiParts: string[] = [];

  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    if (argTypes[i].kind === "tensor") {
      const suffix =
        paramOutputTensors.has(p) || unshareTensorParams.has(p) ? "_in" : "";
      sigParts.push(`const double *${tensorData(p)}${suffix}`);
      sigParts.push(`int64_t ${tensorLen(p)}${suffix}`);
      koffiParts.push("double *");
      koffiParts.push("int64_t");
      const d = tensorMaxDim.get(p) ?? 0;
      if (d >= 2) {
        sigParts.push(`int64_t ${tensorD0(p)}${suffix}`);
        koffiParts.push("int64_t");
      }
      if (d >= 3) {
        sigParts.push(`int64_t ${tensorD1(p)}${suffix}`);
        koffiParts.push("int64_t");
      }
    } else {
      sigParts.push(`double ${mangle(p)}`);
      koffiParts.push("double");
    }
  }

  for (const od of outputDescs) {
    if (od.kind === "tensor") {
      if (dynamicOutputs.has(od.name)) {
        // Dynamic output: C allocates the buffer, transfers ownership
        // via `*_buf_out`. The JS wrapper decodes the pointer + d0/d1,
        // copies data into a fresh Float64Array, then frees the C ptr.
        sigParts.push(`double **${mangle(od.name)}_buf_out`);
        sigParts.push(`int64_t *${mangle(od.name)}_out_len`);
        sigParts.push(`int64_t *${mangle(od.name)}_d0_out`);
        sigParts.push(`int64_t *${mangle(od.name)}_d1_out`);
        koffiParts.push("_Out_ double **");
        koffiParts.push("_Out_ int64_t *");
        koffiParts.push("_Out_ int64_t *");
        koffiParts.push("_Out_ int64_t *");
      } else {
        sigParts.push(`double *${mangle(od.name)}_buf`);
        sigParts.push(`int64_t *${mangle(od.name)}_out_len`);
        koffiParts.push("double *");
        koffiParts.push("int64_t *");
      }
    } else {
      sigParts.push(`double *${mangle(od.name)}_out`);
      koffiParts.push("double *");
    }
  }

  // Append __tic_state parameter when tic/toc are used.
  if (ctx.needsTicState) {
    sigParts.push(`double *__tic_state`);
    koffiParts.push("double *");
  }

  // Append __err_flag when any Index read is emitted. Bounds-violation
  // sets *__err_flag = 1.0; the JS wrapper checks after the call and
  // throws "Index exceeds array bounds" to match JS-JIT semantics.
  if (ctx.needsErrorFlag) {
    sigParts.push(`double *__err_flag`);
    koffiParts.push("double *");
  }

  const paramList = sigParts.length > 0 ? sigParts.join(", ") : "void";
  const signature = `void ${cFnName}(${paramList})`;
  const koffiSignature = `void ${cFnName}(${koffiParts.join(", ")})`;

  // Assemble the full C source.
  const usesTensors = tensorVars.size > 0;

  const parts: string[] = [];
  parts.push(`/* JIT C (koffi): ${fnName}(${params.join(", ")}) */`);
  parts.push(`#include <math.h>`);
  // Always include jit_runtime — the emitter may call any of its helpers
  // (mod, sign, reduce, tic/toc, idx1r) from a non-tensor scalar context.
  parts.push(`#include <stdint.h>`);
  parts.push(`#include "jit_runtime.h"`);
  // Catch a stale jit_runtime.a at compile time so users get a clear
  // "rebuild the addon" message instead of a cryptic linker "undefined
  // reference" error when the emitter starts calling newly-added helpers.
  parts.push(
    `#if !defined(NUMBL_JIT_RT_VERSION) || NUMBL_JIT_RT_VERSION < ${NUMBL_JIT_RT_REQUIRED_VERSION}`
  );
  parts.push(
    `#error "numbl_jit_runtime too old (need version >= ${NUMBL_JIT_RT_REQUIRED_VERSION}); run \`npm run build:addon\` to rebuild"`
  );
  parts.push(`#endif`);
  if (usesTensors) {
    parts.push(`#include <stdlib.h>`);
    parts.push(`#include <string.h>`);
    parts.push(`#include "numbl_ops.h"`);
  }

  parts.push("");
  parts.push(`${signature} {`);
  parts.push(preludeLines.join("\n"));
  parts.push(bodyLines.join("\n"));
  if (epilogueLines.length > 0) {
    parts.push(epilogueLines.join("\n"));
  }
  parts.push(`}`);

  const paramDescs: CParamDesc[] = params.map((p, i) => {
    const desc: CParamDesc = {
      name: p,
      kind: argTypes[i].kind === "tensor" ? "tensor" : "scalar",
    };
    if (desc.kind === "tensor") {
      const d = tensorMaxDim.get(p) ?? 0;
      if (d >= 2) desc.ndim = d;
    }
    return desc;
  });

  return {
    cSource: parts.join("\n"),
    cFnName,
    paramDescs,
    outputDescs,
    usesTensors,
    koffiSignature,
    needsTicState: ctx.needsTicState,
    needsErrorFlag: ctx.needsErrorFlag,
  };
}
