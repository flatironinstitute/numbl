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

import { BinaryOperation, UnaryOperation } from "../../../parser/types.js";
import type { JitExpr, JitStmt, JitType } from "../jitTypes.js";
import { C_TENSOR_REDUCTION_OPS, C_TENSOR_UNARY_OPS } from "./cFeasibility.js";
import { findFusibleChains } from "../fusion.js";
import { emitFusedChain } from "./cFusedCodegen.js";

const MANGLE_PREFIX = "v_";

function mangle(name: string): string {
  return `${MANGLE_PREFIX}${name}`;
}

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

const BUILTINS_NEEDING_HELPERS: Record<string, string> = {
  mod: `static double __numbl_mod(double a, double b) {
  if (b == 0.0) return a;
  double r = fmod(a, b);
  if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
  return r;
}`,
  sign: `static double __numbl_sign(double x) {
  if (x > 0.0) return 1.0;
  if (x < 0.0) return -1.0;
  return 0.0;
}`,
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

function formatNumberLiteral(v: number): string {
  if (!Number.isFinite(v)) {
    if (Number.isNaN(v)) return "(0.0/0.0)";
    return v > 0 ? "(1.0/0.0)" : "(-1.0/0.0)";
  }
  if (Number.isInteger(v)) return `${v}.0`;
  return `${v}`;
}

// ── Tensor data/len accessors ─────────────────────────────────────────

function tensorData(name: string): string {
  return `${mangle(name)}_data`;
}
function tensorLen(name: string): string {
  return `${mangle(name)}_len`;
}

// ── Per-function emit context ─────────────────────────────────────────

interface EmitCtx {
  helpersNeeded: Set<string>;
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

    default:
      throw new Error(`C-JIT codegen: unsupported expr ${expr.tag}`);
  }
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
      return `(((double)((${l}) == (${r}))))`;
    case BinaryOperation.NotEqual:
      return `(((double)((${l}) != (${r}))))`;
    case BinaryOperation.Less:
      return `(((double)((${l}) < (${r}))))`;
    case BinaryOperation.LessEqual:
      return `(((double)((${l}) <= (${r}))))`;
    case BinaryOperation.Greater:
      return `(((double)((${l}) > (${r}))))`;
    case BinaryOperation.GreaterEqual:
      return `(((double)((${l}) >= (${r}))))`;
    case BinaryOperation.AndAnd:
      return `((double)(((${l}) != 0.0) && ((${r}) != 0.0)))`;
    case BinaryOperation.OrOr:
      return `((double)(((${l}) != 0.0) || ((${r}) != 0.0)))`;
    default:
      throw new Error(`C-JIT codegen: unsupported binary op ${expr.op}`);
  }
}

function emitUnary(expr: JitExpr & { tag: "Unary" }, ctx: EmitCtx): string {
  if (isTensorExpr(expr)) {
    throw new Error(
      "C-JIT codegen: tensor unary must be emitted via statement context"
    );
  }

  const operand = emitExpr(expr.operand, ctx);
  switch (expr.op) {
    case UnaryOperation.Plus:
      return `(+${operand})`;
    case UnaryOperation.Minus:
      return `(-${operand})`;
    case UnaryOperation.Not:
      return `((double)((${operand}) == 0.0))`;
    default:
      throw new Error(`C-JIT codegen: unsupported unary op ${expr.op}`);
  }
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
      return `__numbl_reduce(${opEnum}, ${tensorData(arg.name)}, ${tensorLen(arg.name)})`;
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
      return `__numbl_reduce(${opEnum}, ${tensorResult.data}, ${tensorResult.len})`;
    }
    throw new Error(
      `C-JIT codegen: reduction of complex tensor expr outside statement context`
    );
  }
  // Scalar math builtin.
  const cName = BUILTIN_TO_C[expr.name];
  if (!cName) {
    throw new Error(`C-JIT codegen: unmapped builtin ${expr.name}`);
  }
  if (expr.name in BUILTINS_NEEDING_HELPERS) {
    ctx.helpersNeeded.add(expr.name);
  }
  const args = expr.args.map(a => emitExpr(a, ctx));
  return `${cName}(${args.join(", ")})`;
}

function emitTruthiness(expr: JitExpr, ctx: EmitCtx): string {
  if (expr.tag === "Binary" && !isTensorExpr(expr)) {
    switch (expr.op) {
      case BinaryOperation.Equal:
        return `((${emitExpr(expr.left, ctx)}) == (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.NotEqual:
        return `((${emitExpr(expr.left, ctx)}) != (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.Less:
        return `((${emitExpr(expr.left, ctx)}) < (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.LessEqual:
        return `((${emitExpr(expr.left, ctx)}) <= (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.Greater:
        return `((${emitExpr(expr.left, ctx)}) > (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.GreaterEqual:
        return `((${emitExpr(expr.left, ctx)}) >= (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.AndAnd:
        return `((${emitTruthiness(expr.left, ctx)}) && (${emitTruthiness(expr.right, ctx)}))`;
      case BinaryOperation.OrOr:
        return `((${emitTruthiness(expr.left, ctx)}) || (${emitTruthiness(expr.right, ctx)}))`;
      default:
        break;
    }
  }
  if (
    expr.tag === "Unary" &&
    expr.op === UnaryOperation.Not &&
    !isTensorExpr(expr.operand)
  ) {
    return `(!(${emitTruthiness(expr.operand, ctx)}))`;
  }
  return `((${emitExpr(expr, ctx)}) != 0.0)`;
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
      const helpers = emitFusedChain(
        lines,
        indent,
        entry.chain,
        ctx.tensorVars,
        ctx.paramTensorNames,
        ctx.outputTensorNames,
        ctx.localTensorNames
      );
      for (const h of helpers) ctx.helpersNeeded.add(h);
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

  if (expr.tag === "Var" && ctx.tensorVars.has(expr.name)) {
    lines.push(`${indent}${dData} = ${tensorData(expr.name)};`);
    lines.push(`${indent}${dLen} = ${tensorLen(expr.name)};`);
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
    `${indent}${mangle(destName)} = __numbl_reduce(${opEnum}, ${tensorResult.data}, ${tensorResult.len});`
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

export interface CParamDesc {
  name: string;
  kind: "scalar" | "tensor";
}

/** Per-output descriptor. Tells the JS wrapper how to marshal outputs. */
export interface COutputDesc {
  name: string;
  kind: "scalar" | "boolean" | "tensor";
}

export interface GenerateCResult {
  cSource: string;
  cFnName: string;
  helpersUsed: string[];
  paramDescs: CParamDesc[];
  outputDescs: COutputDesc[];
  /** True when any tensor is involved (params, locals, or outputs). */
  usesTensors: boolean;
  /** koffi function signature string for declaring the C function. */
  koffiSignature: string;
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
  fuse?: boolean
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

  const ctx: EmitCtx = {
    helpersNeeded: new Set(),
    tensorVars,
    outputTensorNames,
    scratchCount: 0,
    tmp: { n: 0 },
    usedScratch: new Set(),
    freeStmts: [],
    localTensorNames,
    fuse: fuse ?? false,
    paramTensorNames,
  };

  const indent = "  ";
  const bodyLines: string[] = [];
  emitStmts(bodyLines, body, indent, ctx);

  // Build the epilogue: write outputs to out-pointers, free scratch buffers.
  const epilogueLines: string[] = [];
  for (const od of outputDescs) {
    if (od.kind === "tensor") {
      // Tensor output: copy result data into pre-allocated output buffer.
      // The JS side allocated the output buffer based on the first tensor
      // input's length. The C function must have written into the dest.
      // NOTE: the dest buffer IS the output buffer (passed by the JS wrapper),
      // so no copy is needed — the data is already there. But we need to
      // report the actual length via the out-pointer.
      epilogueLines.push(
        `${indent}*${mangle(od.name)}_out_len = ${tensorLen(od.name)};`
      );
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

  // Free local tensor buffers.
  for (const name of ctx.localTensorNames) {
    epilogueLines.push(
      `${indent}if (${tensorData(name)}) free(${tensorData(name)});`
    );
  }

  // Build the prelude.
  // Tensor params that are also outputs need a writable local that
  // shadows the const input. The signature uses `_in` suffix for the
  // input; the prelude declares the normal data/len locals pointing at
  // the output buffer.
  const paramOutputTensors = new Set<string>();
  for (let i = 0; i < params.length; i++) {
    if (
      argTypes[i].kind === "tensor" &&
      outputDescs.some(od => od.name === params[i] && od.kind === "tensor")
    ) {
      paramOutputTensors.add(params[i]);
    }
  }

  const paramSet = new Set(params);
  const allLocals = [...localVars].filter(v => !paramSet.has(v)).sort();
  const preludeLines: string[] = [];

  // Shadow tensor input-output params with writable locals.
  for (const p of paramOutputTensors) {
    preludeLines.push(`${indent}double *${tensorData(p)} = ${mangle(p)}_buf;`);
    preludeLines.push(`${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`);
  }

  for (const local of allLocals) {
    if (tensorVars.has(local)) {
      // Tensor locals: data pointer + length. For output tensors, the data
      // pointer starts as the pre-allocated output buffer (passed as param).
      const isOutput = outputDescs.some(
        od => od.name === local && od.kind === "tensor"
      );
      if (isOutput) {
        preludeLines.push(
          `${indent}double *${tensorData(local)} = ${mangle(local)}_buf;`
        );
      } else {
        preludeLines.push(`${indent}double *${tensorData(local)} = NULL;`);
      }
      preludeLines.push(`${indent}int64_t ${tensorLen(local)} = 0;`);
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
  //   - tensor output buffers: double *v_name_buf, int64_t *v_name_out_len
  //   - scalar output out-pointers: double *v_name_out
  const cFnName = `jit_${fnName}`;
  const sigParts: string[] = [];
  const koffiParts: string[] = [];

  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    if (argTypes[i].kind === "tensor") {
      const suffix = paramOutputTensors.has(p) ? "_in" : "";
      sigParts.push(`const double *${tensorData(p)}${suffix}`);
      sigParts.push(`int64_t ${tensorLen(p)}${suffix}`);
      koffiParts.push("double *");
      koffiParts.push("int64_t");
    } else {
      sigParts.push(`double ${mangle(p)}`);
      koffiParts.push("double");
    }
  }

  for (const od of outputDescs) {
    if (od.kind === "tensor") {
      sigParts.push(`double *${mangle(od.name)}_buf`);
      sigParts.push(`int64_t *${mangle(od.name)}_out_len`);
      koffiParts.push("double *");
      koffiParts.push("int64_t *");
    } else {
      sigParts.push(`double *${mangle(od.name)}_out`);
      koffiParts.push("double *");
    }
  }

  const paramList = sigParts.length > 0 ? sigParts.join(", ") : "void";
  const signature = `void ${cFnName}(${paramList})`;
  const koffiSignature = `void ${cFnName}(${koffiParts.join(", ")})`;

  // Assemble the full C source.
  const helpersUsed: string[] = [];
  const helperBlocks: string[] = [];
  for (const h of ctx.helpersNeeded) {
    helperBlocks.push(BUILTINS_NEEDING_HELPERS[h]);
    helpersUsed.push(h);
  }

  const usesTensors = tensorVars.size > 0;

  const parts: string[] = [];
  parts.push(`/* JIT C (koffi): ${fnName}(${params.join(", ")}) */`);
  if (usesTensors) {
    parts.push(`#include <math.h>`);
    parts.push(`#include <stdint.h>`);
    parts.push(`#include <stdlib.h>`);
    parts.push(`#include "numbl_ops.h"`);
  } else {
    parts.push(`#include <math.h>`);
  }

  // Reduction helper: always needed if tensors are involved.
  if (usesTensors) {
    parts.push("");
    parts.push(
      `static double __numbl_reduce(int op, const double *data, int64_t len) {`
    );
    parts.push(`  double out = 0.0;`);
    parts.push(`  numbl_real_flat_reduce(op, (size_t)len, data, &out);`);
    parts.push(`  return out;`);
    parts.push(`}`);
  }

  parts.push("");
  if (helperBlocks.length > 0) {
    parts.push(helperBlocks.join("\n\n"));
    parts.push("");
  }
  parts.push(`${signature} {`);
  parts.push(preludeLines.join("\n"));
  parts.push(bodyLines.join("\n"));
  if (epilogueLines.length > 0) {
    parts.push(epilogueLines.join("\n"));
  }
  parts.push(`}`);

  const paramDescs: CParamDesc[] = params.map((p, i) => ({
    name: p,
    kind: argTypes[i].kind === "tensor" ? "tensor" : "scalar",
  }));

  return {
    cSource: parts.join("\n"),
    cFnName,
    helpersUsed,
    paramDescs,
    outputDescs,
    usesTensors,
    koffiSignature,
  };
}
