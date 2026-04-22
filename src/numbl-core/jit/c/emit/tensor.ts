/**
 * Tensor materialization for the C-JIT emitter.
 *
 * This module owns the tensor side of expression → C emission:
 *   - `emitTensorExprToStmts` / `emitTensorBinaryStmts` — real tensor
 *     sub-expressions into a scratch buffer, via libnumbl_ops kernels.
 *   - `emitComplexTensorExprToStmts` / `emitComplexTensorBinaryStmts` —
 *     parallel complex-tensor paths with paired re+im buffers.
 *   - `emitRangeSliceReadToBuf` / `emitRangeSliceReadToStmts` —
 *     `src(a:b)` materialization into a caller-chosen or scratch buffer.
 *   - Scratch buffer size-check helpers (`emitScratchBufAlloc`,
 *     `emitScratchBufAllocComplex`).
 *   - Dest tensor buffer sizing (`emitEnsureTensorBuf`,
 *     `emitEnsureComplexTensorBuf`).
 *   - `shapeExprsFor` — derive (d0, d1) C expressions for a dynamic
 *     tensor output inheriting an operand's shape.
 *   - `findTensorLenExpr` — scan a tensor-expr tree for a concrete
 *     length-variable to pre-size a scratch buffer.
 *
 * Real and complex paths live in the same file so adding a new tensor
 * op is a single edit; the paths remain structurally parallel (not
 * unified) since the kernel set differs.
 */
import { UnaryOperation } from "../../../parser/types.js";
import { isKnownInteger, type JitExpr } from "../../jitTypes.js";
import {
  allocComplexScratch,
  allocScratch,
  isComplexTensorVar,
  isDynamicOutput,
  isLocalTensor,
  isTensorVar,
  scratchData,
  scratchDataIm,
  scratchLen,
  tensorD0,
  tensorD1,
  tensorData,
  tensorDataIm,
  tensorLen,
  TENSOR_BIN_OP,
  TENSOR_COMPLEX_BIN_OP,
  TENSOR_CMP_OP,
  getTensorUnaryOp,
  type EmitCtx,
} from "../context.js";
import { isTensorExpr, type ComplexTensorResult } from "./helpers.js";
import { emitExpr } from "./scalar.js";
import { emitComplexScalarPair } from "./complexScalar.js";

// ── Scratch buffer size-check helpers ────────────────────────────────

/** Size a scratch tensor buffer to `lenExpr` doubles, freeing any stale
 *  buffer from a prior loop iteration first. A plain `if (!sData) malloc`
 *  would overflow the buffer if a later iteration demanded more bytes. */
export function emitScratchBufAlloc(
  lines: string[],
  indent: string,
  sData: string,
  sLen: string,
  lenExpr: string
): void {
  const inner = indent + "  ";
  lines.push(`${indent}{`);
  lines.push(`${inner}int64_t __need = ${lenExpr};`);
  lines.push(`${inner}if (__need != ${sLen}) {`);
  lines.push(`${inner}  if (${sData}) free(${sData});`);
  lines.push(
    `${inner}  ${sData} = (__need > 0) ? (double *)malloc((size_t)__need * sizeof(double)) : NULL;`
  );
  lines.push(`${inner}  ${sLen} = __need;`);
  lines.push(`${inner}}`);
  lines.push(`${indent}}`);
}

/** Complex scratch alloc: re + im buffers, both sized to `lenExpr`. The
 *  complex kernels in numbl_ops write both buffers unconditionally, so
 *  both must be valid pointers (len > 0).
 *
 *  Same length-match fast-path as emitScratchBufAlloc: both buffers are
 *  kept in lockstep, so a single size check guards both. */
export function emitScratchBufAllocComplex(
  lines: string[],
  indent: string,
  sData: string,
  sDataIm: string,
  sLen: string,
  lenExpr: string
): void {
  const inner = indent + "  ";
  lines.push(`${indent}{`);
  lines.push(`${inner}int64_t __need = ${lenExpr};`);
  lines.push(`${inner}if (__need != ${sLen}) {`);
  lines.push(`${inner}  if (${sData}) free(${sData});`);
  lines.push(`${inner}  if (${sDataIm}) free(${sDataIm});`);
  lines.push(
    `${inner}  ${sData} = (__need > 0) ? (double *)malloc((size_t)__need * sizeof(double)) : NULL;`
  );
  lines.push(
    `${inner}  ${sDataIm} = (__need > 0) ? (double *)malloc((size_t)__need * sizeof(double)) : NULL;`
  );
  // If the second malloc failed (OOM), free the first so we don't leak,
  // and zero sLen so downstream sees "no scratch" rather than a stale size.
  lines.push(`${inner}  if (__need > 0 && ${sDataIm} == NULL && ${sData}) {`);
  lines.push(`${inner}    free(${sData});`);
  lines.push(`${inner}    ${sData} = NULL;`);
  lines.push(`${inner}    ${sLen} = 0;`);
  lines.push(`${inner}  } else {`);
  lines.push(`${inner}    ${sLen} = __need;`);
  lines.push(`${inner}  }`);
  lines.push(`${inner}}`);
  lines.push(`${indent}}`);
}

// ── Shape inference ───────────────────────────────────────────────────

/** Derive (d0, d1) C expressions for a dynamic tensor output whose
 *  shape is inherited from an elemwise operand. Uses the operand's
 *  static shape when known; else recovers the missing dim from lenExpr
 *  and the other dim (e.g. row `[1, -1]` → `[1, len]`). Falls back to
 *  `[len, 1]` (column convention) when both dims are unknown. */
export function shapeExprsFor(
  shapeSrc: JitExpr | undefined,
  lenExpr: string
): [string, string] {
  const shape =
    shapeSrc?.jitType.kind === "tensor" ? shapeSrc.jitType.shape : undefined;
  if (shape && shape.length === 2) {
    // Treat 0 the same as unknown (-1): dividing lenExpr by 0 would be UB
    // in the emitted C. Feasibility normally excludes zero-dim tensors;
    // guard here defensively.
    const d0Known = shape[0] !== -1 && shape[0] !== 0;
    const d1Known = shape[1] !== -1 && shape[1] !== 0;
    if (d0Known && d1Known) return [`${shape[0]}`, `${shape[1]}`];
    if (d0Known) return [`${shape[0]}`, `(${lenExpr} / ${shape[0]})`];
    if (d1Known) return [`(${lenExpr} / ${shape[1]})`, `${shape[1]}`];
  }
  return [lenExpr, "1"];
}

/** Find a tensor-length expression from a tensor expr tree (for pre-
 *  allocating scratch buffers). Returns a C expression for the length. */
export function findTensorLenExpr(expr: JitExpr, ctx: EmitCtx): string {
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

// ── Dest tensor buffer sizing ────────────────────────────────────────

/** Ensure `destName`'s buffer is sized to exactly `lenExpr` doubles
 *  before an elemwise op writes into it. For locals and dynamic
 *  outputs, always free+malloc — loop iterations can produce different
 *  sizes, so a first-time-only malloc would overflow the buffer in
 *  later iterations. For fixed-size outputs (caller-aliased buffer),
 *  just records the length; freeing would corrupt the caller.
 *
 *  When the destination is a dynamic output, also writes its _d0/_d1
 *  shape locals. Pass the tensor operand whose shape the result
 *  inherits (elemwise preserves operand shape); we use its static
 *  jitType.shape when available, else fall back to `[lenExpr, 1]`. */
export function emitEnsureTensorBuf(
  lines: string[],
  indent: string,
  destName: string,
  lenExpr: string,
  ctx: EmitCtx,
  shapeSrc?: JitExpr
): void {
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  const isDyn = isDynamicOutput(ctx, destName);
  if (isLocalTensor(ctx, destName) || isDyn) {
    const inner = indent + "  ";
    lines.push(`${indent}{`);
    lines.push(`${inner}int64_t __need = ${lenExpr};`);
    lines.push(`${inner}if (__need != ${dLen}) {`);
    lines.push(`${inner}  if (${dData}) free(${dData});`);
    lines.push(
      `${inner}  ${dData} = (__need > 0) ? (double *)malloc((size_t)__need * sizeof(double)) : NULL;`
    );
    lines.push(`${inner}  ${dLen} = __need;`);
    lines.push(`${inner}}`);
    lines.push(`${indent}}`);
    if (isDyn) {
      const [d0Expr, d1Expr] = shapeExprsFor(shapeSrc, lenExpr);
      lines.push(`${indent}${tensorD0(destName)} = ${d0Expr};`);
      lines.push(`${indent}${tensorD1(destName)} = ${d1Expr};`);
    }
    return;
  }
  lines.push(`${indent}${dLen} = ${lenExpr};`);
}

/** Complex-tensor version of emitEnsureTensorBuf. Reallocates both re
 *  and im buffers in lockstep, so kernel writes into the paired
 *  destination don't mismatch in size. Fixed outputs keep their
 *  caller-aliased re buffer and (for complex) the caller-aliased im
 *  buffer — we only record the length. */
export function emitEnsureComplexTensorBuf(
  lines: string[],
  indent: string,
  destName: string,
  lenExpr: string,
  ctx: EmitCtx,
  shapeSrc?: JitExpr
): void {
  const dData = tensorData(destName);
  const dDataIm = tensorDataIm(destName);
  const dLen = tensorLen(destName);
  const isDyn = isDynamicOutput(ctx, destName);
  if (isLocalTensor(ctx, destName) || isDyn) {
    const inner = indent + "  ";
    lines.push(`${indent}{`);
    lines.push(`${inner}int64_t __need = ${lenExpr};`);
    lines.push(`${inner}if (__need != ${dLen}) {`);
    lines.push(`${inner}  if (${dData}) free(${dData});`);
    lines.push(`${inner}  if (${dDataIm}) free(${dDataIm});`);
    lines.push(
      `${inner}  ${dData} = (__need > 0) ? (double *)malloc((size_t)__need * sizeof(double)) : NULL;`
    );
    lines.push(
      `${inner}  ${dDataIm} = (__need > 0) ? (double *)malloc((size_t)__need * sizeof(double)) : NULL;`
    );
    lines.push(`${inner}  ${dLen} = __need;`);
    lines.push(`${inner}}`);
    lines.push(`${indent}}`);
    if (isDyn) {
      const [d0Expr, d1Expr] = shapeExprsFor(shapeSrc, lenExpr);
      lines.push(`${indent}${tensorD0(destName)} = ${d0Expr};`);
      lines.push(`${indent}${tensorD1(destName)} = ${d1Expr};`);
    }
    return;
  }
  lines.push(`${indent}${dLen} = ${lenExpr};`);
}

// ── Real tensor expression emission ──────────────────────────────────

/** Emit a tensor expression as statements, returning the data/len vars. */
export function emitTensorExprToStmts(
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
    emitScratchBufAlloc(lines, indent, sData, sLen, tensorLen0);
    emitTensorBinaryStmts(lines, indent, expr, ctx, sData, sLen);
    return { data: sData, len: sLen };
  }

  if (expr.tag === "Unary" && isTensorExpr(expr)) {
    if (expr.op === UnaryOperation.Plus) {
      return emitTensorExprToStmts(lines, indent, expr.operand, ctx);
    }
    if (expr.op === UnaryOperation.Minus) {
      const operand = emitTensorExprToStmts(lines, indent, expr.operand, ctx);
      emitScratchBufAlloc(lines, indent, sData, sLen, operand.len);
      lines.push(
        `${indent}numbl_real_scalar_binary_elemwise(NUMBL_REAL_BIN_MUL, (size_t)${sLen}, -1.0, ${operand.data}, 1, ${sData});`
      );
      return { data: sData, len: sLen };
    }
  }

  if (expr.tag === "Call" && isTensorExpr(expr)) {
    // abs(complex_tensor) as a scratch sub-expression → real scratch.
    // Checked before the tensorUnaryOp path so real-abs doesn't grab
    // a complex arg (the real kernel would silently look at re only).
    if (
      expr.name === "abs" &&
      expr.args.length === 1 &&
      expr.args[0].jitType.kind === "tensor" &&
      expr.args[0].jitType.isComplex === true
    ) {
      const arg = emitComplexTensorExprToStmts(
        lines,
        indent,
        expr.args[0],
        ctx
      );
      emitScratchBufAlloc(lines, indent, sData, sLen, arg.len);
      lines.push(
        `${indent}numbl_complex_abs((size_t)${sLen}, ${arg.data}, ${arg.dataIm}, ${sData});`
      );
      return { data: sData, len: sLen };
    }
    const opEnum = getTensorUnaryOp(expr.name);
    if (opEnum) {
      const arg = emitTensorExprToStmts(lines, indent, expr.args[0], ctx);
      emitScratchBufAlloc(lines, indent, sData, sLen, arg.len);
      lines.push(
        `${indent}numbl_real_unary_elemwise(${opEnum}, (size_t)${sLen}, ${arg.data}, ${sData});`
      );
      return { data: sData, len: sLen };
    }
    // real(complex_tensor) / imag(complex_tensor) → real tensor. The
    // re or im buffer of the complex operand is simply memcpy'd into a
    // fresh real scratch. If the operand's imag is NULL (real tensor
    // widened), imag result is all zero.
    if (
      (expr.name === "real" || expr.name === "imag") &&
      expr.args.length === 1 &&
      expr.args[0].jitType.kind === "tensor" &&
      expr.args[0].jitType.isComplex === true
    ) {
      const arg = emitComplexTensorExprToStmts(
        lines,
        indent,
        expr.args[0],
        ctx
      );
      emitScratchBufAlloc(lines, indent, sData, sLen, arg.len);
      const src = expr.name === "real" ? arg.data : arg.dataIm;
      if (expr.name === "real") {
        lines.push(
          `${indent}if (${sLen} > 0) memcpy(${sData}, ${src}, (size_t)${sLen} * sizeof(double));`
        );
      } else {
        // imag: handle NULL source (real-widened) → zero output.
        lines.push(`${indent}if (${src}) {`);
        lines.push(
          `${indent}  if (${sLen} > 0) memcpy(${sData}, ${src}, (size_t)${sLen} * sizeof(double));`
        );
        lines.push(`${indent}} else {`);
        lines.push(
          `${indent}  if (${sLen} > 0) memset(${sData}, 0, (size_t)${sLen} * sizeof(double));`
        );
        lines.push(`${indent}}`);
      }
      return { data: sData, len: sLen };
    }
  }

  throw new Error(
    `C-JIT codegen: cannot emit tensor expr ${expr.tag} to scratch`
  );
}

/** For tensor binary/unary, we need multi-statement emission. This helper
 *  emits the operation as statements into `lines` and returns the result
 *  data pointer variable name. Callers that route the result into a
 *  named dst (vs. a scratch) must use the scratch-transfer pattern —
 *  let this helper write the full expression into a fresh scratch via
 *  `emitTensorExprToStmts`, then `emitEnsureTensorBuf` + memcpy into
 *  the dst. That ordering avoids clobbering an operand that aliases
 *  the dst (e.g. `r = r .* y + 3.0`). */
export function emitTensorBinaryStmts(
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

  let leftData: string;
  let rightData: string;

  if (leftIsTensor) {
    const lResult = emitTensorExprToStmts(lines, indent, expr.left, ctx);
    leftData = lResult.data;
  } else {
    leftData = emitExpr(expr.left, ctx);
  }

  if (rightIsTensor) {
    const rResult = emitTensorExprToStmts(lines, indent, expr.right, ctx);
    rightData = rResult.data;
  } else {
    rightData = emitExpr(expr.right, ctx);
  }

  // destLenVar was already set by emitScratchBufAlloc (the only caller):
  // the guard block either kept it matching __need, or updated it. No
  // need to re-assign here.

  if (leftIsTensor && rightIsTensor) {
    const fn = isCmp ? "numbl_real_comparison" : "numbl_real_binary_elemwise";
    lines.push(
      `${indent}${fn}(${opEnum}, (size_t)${destLenVar}, ${leftData}, ${rightData}, ${destDataVar});`
    );
  } else if (leftIsTensor) {
    const fn = isCmp
      ? "numbl_real_scalar_comparison"
      : "numbl_real_scalar_binary_elemwise";
    lines.push(
      `${indent}${fn}(${opEnum}, (size_t)${destLenVar}, ${rightData}, ${leftData}, 0, ${destDataVar});`
    );
  } else if (rightIsTensor) {
    const fn = isCmp
      ? "numbl_real_scalar_comparison"
      : "numbl_real_scalar_binary_elemwise";
    lines.push(
      `${indent}${fn}(${opEnum}, (size_t)${destLenVar}, ${leftData}, ${rightData}, 1, ${destDataVar});`
    );
  }
}

// ── Complex tensor expression emission ───────────────────────────────

/** Emit a tensor expression in complex (paired-buffer) form, producing
 *  scratch locals for sub-expressions. Handles:
 *   - Var on a complex tensor: returns the name's `_data`/`_data_im`/`_len`.
 *   - Var on a real tensor: widens via `dataIm = NULL`.
 *   - Binary on a complex-typed tensor expr (the result is complex).
 *   - Unary Plus/Minus on a complex-typed tensor expr.
 *   - conj / real / imag on a complex tensor — real/imag are actually
 *     handled via the real tensor path (result is real), but conj stays
 *     complex and is lowered here.
 */
export function emitComplexTensorExprToStmts(
  lines: string[],
  indent: string,
  expr: JitExpr,
  ctx: EmitCtx
): ComplexTensorResult {
  if (expr.tag === "Var" && isTensorVar(ctx, expr.name)) {
    if (isComplexTensorVar(ctx, expr.name)) {
      return {
        data: tensorData(expr.name),
        dataIm: tensorDataIm(expr.name),
        len: tensorLen(expr.name),
      };
    }
    // Real tensor widened to complex at the kernel boundary. NULL imag
    // = treat as all-zero per numbl_complex_* kernel convention.
    return {
      data: tensorData(expr.name),
      dataIm: "NULL",
      len: tensorLen(expr.name),
    };
  }

  const sIdx = allocComplexScratch(ctx);
  const sData = scratchData(sIdx);
  const sDataIm = scratchDataIm(sIdx);
  const sLen = scratchLen(sIdx);

  if (expr.tag === "Binary" && isTensorExpr(expr)) {
    const tensorLen0 = findTensorLenExpr(expr, ctx);
    emitScratchBufAllocComplex(lines, indent, sData, sDataIm, sLen, tensorLen0);
    emitComplexTensorBinaryStmts(
      lines,
      indent,
      expr,
      ctx,
      sData,
      sDataIm,
      sLen
    );
    return { data: sData, dataIm: sDataIm, len: sLen };
  }

  if (expr.tag === "Unary" && isTensorExpr(expr)) {
    if (expr.op === UnaryOperation.Plus) {
      return emitComplexTensorExprToStmts(lines, indent, expr.operand, ctx);
    }
    if (expr.op === UnaryOperation.Minus) {
      const operand = emitComplexTensorExprToStmts(
        lines,
        indent,
        expr.operand,
        ctx
      );
      emitScratchBufAllocComplex(
        lines,
        indent,
        sData,
        sDataIm,
        sLen,
        operand.len
      );
      // -z element-wise: re=-re, im=-im. Use a scalar binary MUL by -1.
      // numbl_complex_scalar_binary_elemwise accepts NULL imag on arr.
      lines.push(
        `${indent}numbl_complex_scalar_binary_elemwise(NUMBL_COMPLEX_BIN_MUL, (size_t)${sLen}, -1.0, 0.0, ${operand.data}, ${operand.dataIm}, 1, ${sData}, ${sDataIm});`
      );
      return { data: sData, dataIm: sDataIm, len: sLen };
    }
  }

  if (expr.tag === "Call" && isTensorExpr(expr) && expr.name === "conj") {
    const operand = emitComplexTensorExprToStmts(
      lines,
      indent,
      expr.args[0],
      ctx
    );
    emitScratchBufAllocComplex(
      lines,
      indent,
      sData,
      sDataIm,
      sLen,
      operand.len
    );
    // conj: copy re; negate im. If operand.dataIm is NULL (real operand
    // widened), the output imag is all-zero, so just zero it.
    lines.push(
      `${indent}if (${sLen} > 0) memcpy(${sData}, ${operand.data}, (size_t)${sLen} * sizeof(double));`
    );
    lines.push(`${indent}if (${operand.dataIm}) {`);
    lines.push(`${indent}  for (int64_t __i = 0; __i < ${sLen}; __i++)`);
    lines.push(`${indent}    ${sDataIm}[__i] = -((${operand.dataIm})[__i]);`);
    lines.push(`${indent}} else {`);
    lines.push(
      `${indent}  if (${sLen} > 0) memset(${sDataIm}, 0, (size_t)${sLen} * sizeof(double));`
    );
    lines.push(`${indent}}`);
    return { data: sData, dataIm: sDataIm, len: sLen };
  }

  throw new Error(
    `C-JIT codegen: cannot emit complex tensor expr ${expr.tag} to scratch`
  );
}

/** Emit a complex tensor binary op into caller-provided dest buffers
 *  (both re and im). Handles all combos of real-tensor/complex-tensor/
 *  real-scalar/complex-scalar operands. Operand widening is kernel-side
 *  (NULL imag pointer or imag=0.0). */
export function emitComplexTensorBinaryStmts(
  lines: string[],
  indent: string,
  expr: JitExpr & { tag: "Binary" },
  ctx: EmitCtx,
  destDataVar: string,
  destDataImVar: string,
  destLenVar: string
): void {
  const complexOpEnum = TENSOR_COMPLEX_BIN_OP[expr.op];
  if (!complexOpEnum) {
    throw new Error(
      `C-JIT codegen: complex tensor binary op ${expr.op} has no opcode`
    );
  }

  const leftIsTensor = isTensorExpr(expr.left);
  const rightIsTensor = isTensorExpr(expr.right);

  // destLenVar was already set to lenSrc inside emitScratchBufAllocComplex's
  // guarded alloc (or, for non-scratch dests, by the caller). No need to
  // re-assign here.

  if (leftIsTensor && rightIsTensor) {
    const l = emitComplexTensorExprToStmts(lines, indent, expr.left, ctx);
    const r = emitComplexTensorExprToStmts(lines, indent, expr.right, ctx);
    lines.push(
      `${indent}numbl_complex_binary_elemwise(${complexOpEnum}, (size_t)${destLenVar}, ${l.data}, ${l.dataIm}, ${r.data}, ${r.dataIm}, ${destDataVar}, ${destDataImVar});`
    );
    return;
  }
  if (leftIsTensor) {
    const l = emitComplexTensorExprToStmts(lines, indent, expr.left, ctx);
    const sPair = emitComplexScalarPair(expr.right, ctx);
    // scalar on right: out = arr OP scalar
    lines.push(
      `${indent}numbl_complex_scalar_binary_elemwise(${complexOpEnum}, (size_t)${destLenVar}, ${sPair.re}, ${sPair.im}, ${l.data}, ${l.dataIm}, 0, ${destDataVar}, ${destDataImVar});`
    );
    return;
  }
  if (rightIsTensor) {
    const r = emitComplexTensorExprToStmts(lines, indent, expr.right, ctx);
    const sPair = emitComplexScalarPair(expr.left, ctx);
    // scalar on left: out = scalar OP arr
    lines.push(
      `${indent}numbl_complex_scalar_binary_elemwise(${complexOpEnum}, (size_t)${destLenVar}, ${sPair.re}, ${sPair.im}, ${r.data}, ${r.dataIm}, 1, ${destDataVar}, ${destDataImVar});`
    );
    return;
  }
  throw new Error(
    `C-JIT codegen: emitComplexTensorBinaryStmts called with no tensor operand`
  );
}

// ── Range-slice reads ────────────────────────────────────────────────

/** Emit `dstData = src(a:b) copy` pattern into caller-provided dst data /
 *  len vars. */
export function emitRangeSliceReadToBuf(
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
export function emitRangeSliceReadToStmts(
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
