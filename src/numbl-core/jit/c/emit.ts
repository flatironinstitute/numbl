/**
 * Per-function IR → C emission.
 *
 * Exposes `emitStmts(lines, stmts, indent, ctx)` — walks the JIT IR body
 * and pushes C lines into `lines`. All helpers below are mutually
 * recursive (expr emitters call stmt emitters for fresh-tensor sub-
 * expressions; stmt emitters call expr emitters for value positions) so
 * they all live in this one file.
 *
 * Callers own the classification (`ctx.cls`) and the mutable emitter
 * state (scratch counters, needsTicState/needsErrorFlag flags, etc.).
 * See `codegenCtx.ts` for the shared types.
 */
import { UnaryOperation } from "../../parser/types.js";
import { isKnownInteger, type JitExpr, type JitStmt } from "../jitTypes.js";
import {
  C_TENSOR_BINARY_BUILTINS,
  C_TENSOR_REDUCTION_OPS,
  C_TENSOR_UNARY_OPS,
} from "./cFeasibility.js";
import { findFusibleChains } from "../fusion.js";
import { emitFusedChain } from "./cFusedCodegen.js";
import {
  emitScalarBinaryOp,
  emitScalarUnaryOp,
  emitScalarTruthiness,
} from "../scalarEmit.js";
import { getIBuiltin } from "../../interpreter/builtins/types.js";
import type { TensorMeta } from "./classify.js";
import {
  C_SCALAR_TARGET,
  TENSOR_BIN_OP,
  TENSOR_CMP_OP,
  TENSOR_REDUCE_OP,
  TENSOR_UNARY_OP,
  allocScratch,
  formatNumberLiteral,
  hasFreshAlloc,
  isDynamicOutput,
  isLocalTensor,
  isOutputTensor,
  isTensorVar,
  mangle,
  scratchData,
  scratchLen,
  tensorD0,
  tensorD1,
  tensorData,
  tensorLen,
  tensorMaxDim,
  type EmitCtx,
} from "./codegenCtx.js";

/** Resolve a tensor name's meta or throw — the tensor-creation emit
 *  helpers below depend on the name being classified (with
 *  `hasFreshAlloc`) for the d0/d1 locals they write to actually exist
 *  at runtime. Failing loudly here beats emitting C that references
 *  undeclared identifiers. */
function requireFreshAllocMeta(
  ctx: EmitCtx,
  destName: string,
  site: string
): TensorMeta {
  const m = ctx.cls.meta.get(destName);
  if (!m) {
    throw new Error(
      `C-JIT codegen: ${site}: dest '${destName}' has no TensorMeta (not classified as a tensor)`
    );
  }
  if (!m.hasFreshAlloc) {
    throw new Error(
      `C-JIT codegen: ${site}: dest '${destName}' is not hasFreshAlloc — shape locals (_d0/_d1) wouldn't exist`
    );
  }
  return m;
}

// ── Expression emission ───────────────────────────────────────────────

function isTensorExpr(expr: JitExpr): boolean {
  return expr.jitType.kind === "tensor";
}

/** Emit a value-expression. For scalars, returns a C `double` expression.
 *  For tensors, returns the data-variable name (the caller knows to also
 *  access the corresponding _len variable). */
function emitExpr(expr: JitExpr, ctx: EmitCtx): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return formatNumberLiteral(expr.value);

    case "Var":
      if (isTensorVar(ctx, expr.name)) return tensorData(expr.name);
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
  if (expr.base.tag !== "Var" || !isTensorVar(ctx, expr.base.name)) {
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
    lines.push(`${indent}${destLenVar} = ${leftLen};`);
    const fn = isCmp ? "numbl_real_comparison" : "numbl_real_binary_elemwise";
    lines.push(
      `${indent}${fn}(${opEnum}, (size_t)${destLenVar}, ${leftData}, ${rightData}, ${destDataVar});`
    );
  } else if (leftIsTensor) {
    lines.push(`${indent}${destLenVar} = ${leftLen};`);
    const fn = isCmp
      ? "numbl_real_scalar_comparison"
      : "numbl_real_scalar_binary_elemwise";
    lines.push(
      `${indent}${fn}(${opEnum}, (size_t)${destLenVar}, ${rightData}, ${leftData}, 0, ${destDataVar});`
    );
  } else if (rightIsTensor) {
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
  if (expr.tag === "Var" && isTensorVar(ctx, expr.name)) {
    return { data: tensorData(expr.name), len: tensorLen(expr.name) };
  }

  if (expr.tag === "RangeSliceRead") {
    return emitRangeSliceReadToStmts(lines, indent, expr, ctx);
  }

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
 *  len vars. */
function emitRangeSliceReadToBuf(
  lines: string[],
  indent: string,
  expr: JitExpr & { tag: "RangeSliceRead" },
  ctx: EmitCtx,
  destData: string,
  destLen: string
): void {
  if (!isTensorVar(ctx, expr.baseName)) {
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
  if (expr.tag === "Var" && isTensorVar(ctx, expr.name)) {
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
    if (arg.tag === "Var" && isTensorVar(ctx, arg.name)) {
      return `numbl_reduce_flat(${opEnum}, ${tensorData(arg.name)}, ${tensorLen(arg.name)})`;
    }
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
  if (expr.name === "tic" && expr.args.length === 0) {
    ctx.needsTicState = true;
    return `numbl_tic(__tic_state)`;
  }
  if (expr.name === "toc" && expr.args.length === 0) {
    ctx.needsTicState = true;
    return `numbl_toc(__tic_state)`;
  }
  // length / isempty on a tensor Var.
  if (
    (expr.name === "length" || expr.name === "isempty") &&
    expr.args.length === 1 &&
    expr.args[0].tag === "Var" &&
    isTensorVar(ctx, (expr.args[0] as JitExpr & { tag: "Var" }).name)
  ) {
    const name = (expr.args[0] as JitExpr & { tag: "Var" }).name;
    const lenCode = tensorLen(name);
    if (expr.name === "length") {
      if (hasFreshAlloc(ctx, name)) {
        return `((double)((${tensorD0(name)} > ${tensorD1(name)}) ? ${tensorD0(name)} : ${tensorD1(name)}))`;
      }
      return `((double)${lenCode})`;
    }
    return `((double)(${lenCode} == 0))`;
  }
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
  // Tensor-valued Binary/Unary in condition context: route to value-form
  // so emitExpr raises the "must be emitted via statement context" error.
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

export function emitStmts(
  lines: string[],
  stmts: JitStmt[],
  indent: string,
  ctx: EmitCtx
): void {
  if (!ctx.fuse) {
    for (const s of stmts) emitStmt(lines, s, indent, ctx);
    return;
  }

  const chains = findFusibleChains(
    stmts,
    ctx.cls.paramTensorNames,
    ctx.cls.tensorVars
  );

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
        ctx.cls.tensorVars,
        ctx.cls.paramTensorNames,
        ctx.cls.outputTensorNames,
        ctx.cls.localTensorNames,
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
  if (!isLocalTensor(ctx, destName)) return;
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  lines.push(`${indent}${dLen} = ${lenExpr};`);
  lines.push(
    `${indent}if (!${dData}) ${dData} = (double *)malloc((size_t)${dLen} * sizeof(double));`
  );
}

/** Emit a fresh-alloc pattern: free old buffer, malloc new, fill with
 *  the cells of a TensorLiteral. The caller MUST pass the classified
 *  `TensorMeta` for `destName` (a `hasFreshAlloc=true` meta) so the
 *  `_d0` / `_d1` locals we write to are guaranteed to exist. */
function emitTensorLiteralAssign(
  lines: string[],
  indent: string,
  destName: string,
  destMeta: TensorMeta,
  expr: JitExpr & { tag: "TensorLiteral" },
  ctx: EmitCtx
): void {
  if (!destMeta.hasFreshAlloc) {
    throw new Error(
      `C-JIT codegen: emitTensorLiteralAssign('${destName}'): destMeta.hasFreshAlloc must be true`
    );
  }
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
  lines.push(
    `${inner}double *__tl = (double *)malloc(${nLen} * sizeof(double));`
  );
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
 *  Caller must pass a `hasFreshAlloc=true` TensorMeta for `destName`. */
function emitZerosOnesAssign(
  lines: string[],
  indent: string,
  destName: string,
  destMeta: TensorMeta,
  expr: JitExpr & { tag: "Call" },
  ctx: EmitCtx
): void {
  if (!destMeta.hasFreshAlloc) {
    throw new Error(
      `C-JIT codegen: emitZerosOnesAssign('${destName}'): destMeta.hasFreshAlloc must be true`
    );
  }
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  const dD0 = tensorD0(destName);
  const dD1 = tensorD1(destName);
  const fill = expr.name === "ones" ? "1.0" : "0.0";
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

/** Emit VConcatGrow `dest = [base; value]`. Handles the self-grow case
 *  (base == dest): memcpy completes before the old buffer is freed.
 *  Caller must pass a `hasFreshAlloc=true` TensorMeta for `destName`. */
function emitVConcatGrowAssign(
  lines: string[],
  indent: string,
  destName: string,
  destMeta: TensorMeta,
  expr: JitExpr & { tag: "VConcatGrow" },
  ctx: EmitCtx
): void {
  if (!destMeta.hasFreshAlloc) {
    throw new Error(
      `C-JIT codegen: emitVConcatGrowAssign('${destName}'): destMeta.hasFreshAlloc must be true`
    );
  }
  if (expr.base.tag !== "Var") {
    throw new Error("C-JIT codegen: VConcatGrow base must be a Var");
  }
  const baseName = expr.base.name;
  if (!isTensorVar(ctx, baseName)) {
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
  const needsAlloc = isLocalTensor(ctx, destName);

  if (expr.tag === "TensorLiteral") {
    const destMeta = requireFreshAllocMeta(ctx, destName, "TensorLiteral");
    emitTensorLiteralAssign(lines, indent, destName, destMeta, expr, ctx);
    return;
  }

  if (expr.tag === "VConcatGrow") {
    const destMeta = requireFreshAllocMeta(ctx, destName, "VConcatGrow");
    emitVConcatGrowAssign(lines, indent, destName, destMeta, expr, ctx);
    return;
  }

  if (
    expr.tag === "Call" &&
    (expr.name === "zeros" || expr.name === "ones") &&
    isTensorExpr(expr)
  ) {
    const destMeta = requireFreshAllocMeta(ctx, destName, expr.name);
    emitZerosOnesAssign(lines, indent, destName, destMeta, expr, ctx);
    return;
  }

  if (expr.tag === "RangeSliceRead") {
    if (isLocalTensor(ctx, destName)) {
      emitRangeSliceReadToBuf(lines, indent, expr, ctx, dData, dLen);
      return;
    }
    if (isOutputTensor(ctx, destName)) {
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

  if (expr.tag === "Var" && isTensorVar(ctx, expr.name)) {
    const srcName = expr.name;
    const srcData = tensorData(srcName);
    const srcLen = tensorLen(srcName);
    const destIsDynamic = hasFreshAlloc(ctx, destName);
    const destIsFixedOutput =
      isOutputTensor(ctx, destName) && !isDynamicOutput(ctx, destName);
    if (destIsDynamic) {
      // Deep-copy: the source's buffer may be freed later in the body,
      // so pointer-alias would dangle. Free old dest, malloc fresh, copy,
      // update shape locals. Guarded by self-alias check.
      const dD0 = tensorD0(destName);
      const dD1 = tensorD1(destName);
      const srcD0Expr = hasFreshAlloc(ctx, srcName)
        ? tensorD0(srcName)
        : tensorMaxDim(ctx, srcName) >= 2
          ? tensorD0(srcName)
          : srcLen;
      const srcD1Expr = hasFreshAlloc(ctx, srcName)
        ? tensorD1(srcName)
        : tensorMaxDim(ctx, srcName) >= 3
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

/** Emit an Assign where the RHS is a reduction on a complex tensor
 *  expression (not just a Var). */
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
      if (stmt.expr.jitType.kind === "tensor" && isTensorVar(ctx, stmt.name)) {
        emitTensorAssign(lines, indent, stmt.name, stmt.expr, ctx);
        return;
      }
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
      const n = stmt.indices.length;
      if (n < 1 || n > 3) {
        throw new Error(
          `C-JIT codegen: AssignIndex arity ${n} unsupported (only 1D/2D/3D)`
        );
      }
      if (!isTensorVar(ctx, stmt.baseName)) {
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
      if (!isTensorVar(ctx, stmt.baseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexRange base '${stmt.baseName}' is not a tensor var`
        );
      }
      if (!isTensorVar(ctx, stmt.srcBaseName)) {
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
      if (!isTensorVar(ctx, stmt.baseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexCol base '${stmt.baseName}' is not a tensor var`
        );
      }
      if (!isTensorVar(ctx, stmt.srcBaseName)) {
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
      throw new Error(
        `C-JIT codegen: unsupported stmt ${(stmt as JitStmt).tag}`
      );
  }
}
