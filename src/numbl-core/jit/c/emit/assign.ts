/**
 * Tensor assignment forms.
 *
 * All the `dest = RHS` where `dest` is a tensor Var and the LHS buffer
 * needs sizing / filling. Dispatches on the RHS's tag and complexness.
 *
 * Public entry points:
 *   - `emitTensorAssign` — real tensor dest. Routes complex dests to
 *     `emitComplexTensorAssign`; internal RHS cases handle TensorLiteral,
 *     zeros/ones, VConcatGrow, RangeSliceRead, Var alias, Binary,
 *     Unary (Plus/Minus), Call (tensor unary, complex abs, real/imag),
 *     and two-arg tensor binary builtins (max/min/atan2/...).
 *   - `emitComplexTensorAssign` — complex tensor dest; paired re+im
 *     buffer management via the tensor module's complex helpers.
 *   - `emitReductionOfTensorExpr` — special path for `dest = reduce(EXPR)`
 *     where EXPR is a tensor sub-expression (not a Var).
 *
 * Fresh-alloc assigns (`TensorLiteral`, `zeros`/`ones`, `VConcatGrow`)
 * are delegated to this file's internal helpers; they all malloc a
 * fresh buffer and write shape locals.
 */
import { UnaryOperation } from "../../../parser/types.js";
import { type JitExpr, isKnownInteger } from "../../jitTypes.js";
import type { TensorMeta } from "../classify.js";
import {
  getTensorBinaryFn,
  getTensorReductionOp,
  getTensorUnaryOp,
  hasFreshAlloc,
  isComplexTensorVar,
  isDynamicOutput,
  isLocalTensor,
  isOutputTensor,
  isTensorVar,
  mangle,
  tensorD0,
  tensorD1,
  tensorData,
  tensorDataIm,
  tensorLen,
  tensorMaxDim,
  type EmitCtx,
} from "../context.js";
import { isTensorExpr, requireFreshAllocMeta } from "./helpers.js";
import { emitExpr } from "./scalar.js";
import {
  emitComplexTensorExprToStmts,
  emitElemwiseTensorAssign,
  emitEnsureComplexTensorBuf,
  emitRangeSliceReadToBuf,
  emitTensorBinaryStmts,
  emitTensorExprToStmts,
  findTensorLenExpr,
  shapeExprsFor,
} from "./tensor.js";
import { emitUserCallTensorAssign } from "./userCall.js";
import { withPendingStmts } from "./stmt.js";

// ── Fresh-alloc assigns ──────────────────────────────────────────────

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
  ctx.needsErrorFlag = true;
  lines.push(`${indent}{`);
  const inner = indent + "  ";
  lines.push(`${inner}int64_t __zr = ${arg0};`);
  lines.push(`${inner}int64_t __zc = ${arg1};`);
  lines.push(`${inner}if (__zr < 0) __zr = 0;`);
  lines.push(`${inner}if (__zc < 0) __zc = 0;`);
  // Overflow guard: int64 wrap would silently produce size=[big×big]
  // with numel=0. Flag the error (JS wrapper throws) and size to zero
  // so the rest of the function doesn't allocate a garbage buffer.
  lines.push(`${inner}int64_t __zn;`);
  lines.push(
    `${inner}if (__zr > 0 && __zc > ((int64_t)9223372036854775807LL / 8) / __zr) {`
  );
  lines.push(`${inner}  *__err_flag = 1.0;`);
  lines.push(`${inner}  __zr = 0; __zc = 0; __zn = 0;`);
  lines.push(`${inner}} else {`);
  lines.push(`${inner}  __zn = __zr * __zc;`);
  lines.push(`${inner}}`);
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

// ── Complex tensor assign ─────────────────────────────────────────────

/** Complex-tensor Assign: parallels emitTensorAssign but writes paired
 *  re+im buffers. RHS sub-exprs route through emitComplexTensorExprToStmts
 *  / emitComplexTensorBinaryStmts. For a real RHS (e.g. a Var pointing
 *  to a real tensor), imag is widened via NULL pointer or a zero buffer.
 *
 *  Runs inside a pendingStmts frame so nested complex scalar sub-expressions
 *  (`1i`, `3+4i`, `re(z) + 1i*im(z)`, ...) can materialize their pair
 *  locals into the same `lines` stream ahead of the kernel call. */
export function emitComplexTensorAssign(
  lines: string[],
  indent: string,
  destName: string,
  expr: JitExpr,
  ctx: EmitCtx
): void {
  withPendingStmts(ctx, lines, indent, () =>
    emitComplexTensorAssignInner(lines, indent, destName, expr, ctx)
  );
}

function emitComplexTensorAssignInner(
  lines: string[],
  indent: string,
  destName: string,
  expr: JitExpr,
  ctx: EmitCtx
): void {
  const dData = tensorData(destName);
  const dDataIm = tensorDataIm(destName);
  const dLen = tensorLen(destName);

  // Tensor-return UserCall whose dest is complex: callee transfers
  // ownership of both real + imag buffers via the paired dynOutBuf /
  // dynOutBufIm slots. Reuse the unified real/complex assigner.
  if (expr.tag === "UserCall") {
    const destMeta = requireFreshAllocMeta(ctx, destName, "UserCall");
    emitUserCallTensorAssign(lines, indent, destName, destMeta, expr, ctx);
    return;
  }

  // All compound RHS cases (Binary / Unary Minus / conj) go through
  // the scratch-transfer pattern: eval the whole expression into a
  // fresh complex scratch, then free+malloc dst and memcpy both
  // buffers across. This defuses self-aliasing (`z = z .* conj(z)`,
  // `z = -z`, etc.) — the scratch reads v_z_data before v_z_data is
  // freed.
  if (
    (expr.tag === "Binary" && isTensorExpr(expr)) ||
    (expr.tag === "Unary" &&
      isTensorExpr(expr) &&
      expr.op === UnaryOperation.Minus) ||
    (expr.tag === "Call" && isTensorExpr(expr) && expr.name === "conj")
  ) {
    const scratch = emitComplexTensorExprToStmts(lines, indent, expr, ctx);
    // Shape source: for Binary, the tensor-typed operand; for Unary
    // and conj, the single operand.
    const shapeSrc: JitExpr =
      expr.tag === "Binary"
        ? isTensorExpr(expr.left)
          ? expr.left
          : expr.right
        : expr.tag === "Unary"
          ? expr.operand
          : expr.args[0];
    emitEnsureComplexTensorBuf(
      lines,
      indent,
      destName,
      scratch.len,
      ctx,
      shapeSrc
    );
    lines.push(
      `${indent}if (${dLen} > 0) memcpy(${dData}, ${scratch.data}, (size_t)${dLen} * sizeof(double));`
    );
    lines.push(`${indent}if (${scratch.dataIm}) {`);
    lines.push(
      `${indent}  if (${dLen} > 0) memcpy(${dDataIm}, ${scratch.dataIm}, (size_t)${dLen} * sizeof(double));`
    );
    lines.push(`${indent}} else {`);
    lines.push(
      `${indent}  if (${dLen} > 0) memset(${dDataIm}, 0, (size_t)${dLen} * sizeof(double));`
    );
    lines.push(`${indent}}`);
    return;
  }

  if (expr.tag === "Unary" && isTensorExpr(expr)) {
    if (expr.op === UnaryOperation.Plus) {
      emitComplexTensorAssign(lines, indent, destName, expr.operand, ctx);
      return;
    }
  }

  if (expr.tag === "Var" && isTensorVar(ctx, expr.name)) {
    // Complex-dest Var assign: deep-copy re, and either deep-copy im
    // (source is complex) or zero im (source is real, widened).
    const operand = emitComplexTensorExprToStmts(lines, indent, expr, ctx);
    emitEnsureComplexTensorBuf(lines, indent, destName, operand.len, ctx, expr);
    lines.push(
      `${indent}if (${dLen} > 0) memcpy(${dData}, ${operand.data}, (size_t)${dLen} * sizeof(double));`
    );
    lines.push(`${indent}if (${operand.dataIm}) {`);
    lines.push(
      `${indent}  if (${dLen} > 0) memcpy(${dDataIm}, ${operand.dataIm}, (size_t)${dLen} * sizeof(double));`
    );
    lines.push(`${indent}} else {`);
    lines.push(
      `${indent}  if (${dLen} > 0) memset(${dDataIm}, 0, (size_t)${dLen} * sizeof(double));`
    );
    lines.push(`${indent}}`);
    return;
  }

  throw new Error(
    `C-JIT codegen: unhandled complex tensor assign RHS: ${expr.tag}`
  );
}

// ── Real tensor assign ────────────────────────────────────────────────

/** Emit a tensor-result Assign: handles Binary, Unary, Call on tensors. */
export function emitTensorAssign(
  lines: string[],
  indent: string,
  destName: string,
  expr: JitExpr,
  ctx: EmitCtx
): void {
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);

  // Complex destination tensor: route through the paired-buffer
  // emitter. Real RHS expressions are transparently widened via the
  // complex tensor helpers (NULL imag pointer or 0.0 imag scalar).
  if (isComplexTensorVar(ctx, destName)) {
    emitComplexTensorAssign(lines, indent, destName, expr, ctx);
    return;
  }

  if (expr.tag === "UserCall") {
    const destMeta = requireFreshAllocMeta(ctx, destName, "UserCall");
    emitUserCallTensorAssign(lines, indent, destName, destMeta, expr, ctx);
    return;
  }

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
    // RangeSliceRead always propagates hasFreshAlloc (classify.ts), so
    // an output destination is always isDynamicOutput. Free+malloc so
    // the slice size doesn't have to match the current buffer size.
    if (isLocalTensor(ctx, destName) || isDynamicOutput(ctx, destName)) {
      emitRangeSliceReadToBuf(lines, indent, expr, ctx, dData, dLen);
      if (isDynamicOutput(ctx, destName)) {
        // Preserve source orientation: row source → row slice, else column.
        if (expr.isRow) {
          lines.push(`${indent}${tensorD0(destName)} = 1;`);
          lines.push(`${indent}${tensorD1(destName)} = ${dLen};`);
        } else {
          lines.push(`${indent}${tensorD0(destName)} = ${dLen};`);
          lines.push(`${indent}${tensorD1(destName)} = 1;`);
        }
      }
      return;
    }
    throw new Error(
      `C-JIT codegen: RangeSliceRead assign to '${destName}' unsupported`
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
      // Shape propagation: prefer runtime locals (fresh-alloc / d-indexed
      // params) when present; else read from the RHS's static jitType
      // shape. Falling back to [len, 1] flipped a row `[1, n]` param to
      // a column `[n, 1]` — the source's static shape gives the correct
      // orientation.
      const [d0FromShape, d1FromShape] = shapeExprsFor(expr, srcLen);
      const srcD0Expr = hasFreshAlloc(ctx, srcName)
        ? tensorD0(srcName)
        : tensorMaxDim(ctx, srcName) >= 2
          ? tensorD0(srcName)
          : d0FromShape;
      const srcD1Expr = hasFreshAlloc(ctx, srcName)
        ? tensorD1(srcName)
        : tensorMaxDim(ctx, srcName) >= 3
          ? tensorD1(srcName)
          : d1FromShape;
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
    // Emit the binary kernel with aliasing safety + steady-state fast
    // path: if dst's buffer is already the right size (no realloc
    // needed), the kernel writes directly into dst; otherwise it
    // writes into scratch first, then dst is realloc'd and the
    // scratch copied over. Fast path avoids a full-tensor memcpy.
    const tensorOperand = isTensorExpr(expr.left) ? expr.left : expr.right;
    const lenExpr = findTensorLenExpr(expr, ctx);
    emitElemwiseTensorAssign(
      lines,
      indent,
      destName,
      lenExpr,
      tensorOperand,
      ctx,
      (l, ind, buf, len) => emitTensorBinaryStmts(l, ind, expr, ctx, buf, len)
    );
    return;
  }

  if (expr.tag === "Unary" && isTensorExpr(expr)) {
    if (expr.op === UnaryOperation.Plus) {
      emitTensorAssign(lines, indent, destName, expr.operand, ctx);
      return;
    }
    if (expr.op === UnaryOperation.Minus) {
      const lenExpr = findTensorLenExpr(expr.operand, ctx);
      emitElemwiseTensorAssign(
        lines,
        indent,
        destName,
        lenExpr,
        expr.operand,
        ctx,
        (l, ind, buf, len) => {
          const operand = emitTensorExprToStmts(l, ind, expr.operand, ctx);
          l.push(
            `${ind}numbl_real_scalar_binary_elemwise(NUMBL_REAL_BIN_MUL, (size_t)${len}, -1.0, ${operand.data}, 1, ${buf});`
          );
        }
      );
      return;
    }
  }

  if (expr.tag === "Call" && isTensorExpr(expr)) {
    const opEnum = getTensorUnaryOp(expr.name);
    if (opEnum) {
      const lenExpr = findTensorLenExpr(expr.args[0], ctx);
      emitElemwiseTensorAssign(
        lines,
        indent,
        destName,
        lenExpr,
        expr.args[0],
        ctx,
        (l, ind, buf, len) => {
          const arg = emitTensorExprToStmts(l, ind, expr.args[0], ctx);
          l.push(
            `${ind}numbl_real_unary_elemwise(${opEnum}, (size_t)${len}, ${arg.data}, ${buf});`
          );
        }
      );
      return;
    }
    // abs(complex_tensor) → real tensor via numbl_complex_abs.
    if (
      expr.name === "abs" &&
      expr.args.length === 1 &&
      expr.args[0].jitType.kind === "tensor" &&
      expr.args[0].jitType.isComplex === true
    ) {
      const operand = emitComplexTensorExprToStmts(
        lines,
        indent,
        expr.args[0],
        ctx
      );
      emitElemwiseTensorAssign(
        lines,
        indent,
        destName,
        operand.len,
        expr.args[0],
        ctx,
        (l, ind, buf, len) => {
          l.push(
            `${ind}numbl_complex_abs((size_t)${len}, ${operand.data}, ${operand.dataIm}, ${buf});`
          );
        }
      );
      return;
    }
    // real(complex_tensor) / imag(complex_tensor): produce a real tensor
    // by copying the matching buffer (or zeroing imag when source has
    // NULL imag).
    if (
      (expr.name === "real" || expr.name === "imag") &&
      expr.args.length === 1 &&
      expr.args[0].jitType.kind === "tensor" &&
      expr.args[0].jitType.isComplex === true
    ) {
      const operand = emitComplexTensorExprToStmts(
        lines,
        indent,
        expr.args[0],
        ctx
      );
      emitElemwiseTensorAssign(
        lines,
        indent,
        destName,
        operand.len,
        expr.args[0],
        ctx,
        (l, ind, buf, len) => {
          if (expr.name === "real") {
            l.push(
              `${ind}if (${len} > 0) memcpy(${buf}, ${operand.data}, (size_t)${len} * sizeof(double));`
            );
          } else {
            l.push(`${ind}if (${operand.dataIm}) {`);
            l.push(
              `${ind}  if (${len} > 0) memcpy(${buf}, ${operand.dataIm}, (size_t)${len} * sizeof(double));`
            );
            l.push(`${ind}} else {`);
            l.push(
              `${ind}  if (${len} > 0) memset(${buf}, 0, (size_t)${len} * sizeof(double));`
            );
            l.push(`${ind}}`);
          }
        }
      );
      return;
    }
  }

  // Two-arg tensor binary builtin (max, min, atan2, hypot, mod, rem):
  // emit a per-element loop calling the C math function registered on
  // the builtin's jitCapabilities.tensorBinaryFn. Same steady-state
  // fast path + aliasing safety as the other elemwise assigns.
  if (
    expr.tag === "Call" &&
    isTensorExpr(expr) &&
    expr.args.length === 2 &&
    getTensorBinaryFn(expr.name) !== undefined
  ) {
    const cFn = getTensorBinaryFn(expr.name)!;
    const left = expr.args[0];
    const right = expr.args[1];
    const leftIsTensor = left.jitType.kind === "tensor";
    const rightIsTensor = right.jitType.kind === "tensor";
    const tensorOperand = leftIsTensor ? left : right;
    const lenExpr = findTensorLenExpr(tensorOperand, ctx);

    emitElemwiseTensorAssign(
      lines,
      indent,
      destName,
      lenExpr,
      tensorOperand,
      ctx,
      (l, ind, buf, len) => {
        if (leftIsTensor && rightIsTensor) {
          const lArg = emitTensorExprToStmts(l, ind, left, ctx);
          const rArg = emitTensorExprToStmts(l, ind, right, ctx);
          l.push(`${ind}for (int64_t __i = 0; __i < ${len}; __i++)`);
          l.push(
            `${ind}  ${buf}[__i] = ${cFn}(${lArg.data}[__i], ${rArg.data}[__i]);`
          );
        } else if (leftIsTensor) {
          const lArg = emitTensorExprToStmts(l, ind, left, ctx);
          const rScalar = emitExpr(right, ctx);
          l.push(`${ind}for (int64_t __i = 0; __i < ${len}; __i++)`);
          l.push(
            `${ind}  ${buf}[__i] = ${cFn}(${lArg.data}[__i], ${rScalar});`
          );
        } else {
          const lScalar = emitExpr(left, ctx);
          const rArg = emitTensorExprToStmts(l, ind, right, ctx);
          l.push(`${ind}for (int64_t __i = 0; __i < ${len}; __i++)`);
          l.push(
            `${ind}  ${buf}[__i] = ${cFn}(${lScalar}, ${rArg.data}[__i]);`
          );
        }
      }
    );
    return;
  }

  throw new Error(`C-JIT codegen: unhandled tensor assign RHS: ${expr.tag}`);
}

/** Emit an Assign where the RHS is a reduction on a tensor sub-
 *  expression (not just a Var) — e.g. `y = sum(x .* z)`. The tensor
 *  expression is materialised into a scratch buffer first, then the
 *  scalar reduction reads that buffer. */
export function emitReductionOfTensorExpr(
  lines: string[],
  indent: string,
  destName: string,
  callExpr: JitExpr & { tag: "Call" },
  ctx: EmitCtx
): void {
  const opEnum = getTensorReductionOp(callExpr.name);
  if (!opEnum) {
    throw new Error(
      `C-JIT codegen: tensor reduction ${callExpr.name} has no opcode`
    );
  }
  const arg = callExpr.args[0];
  const tensorResult = emitTensorExprToStmts(lines, indent, arg, ctx);
  lines.push(
    `${indent}${mangle(destName)} = numbl_reduce_flat(${opEnum}, ${tensorResult.data}, ${tensorResult.len});`
  );
}
