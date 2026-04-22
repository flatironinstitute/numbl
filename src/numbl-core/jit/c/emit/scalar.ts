/**
 * Real scalar expression emission.
 *
 * Exports:
 *   - `emitExpr(expr, ctx)` — the primary value-form emitter for real
 *     scalar expressions. Returns a C `double` expression string. For
 *     tensor Vars, returns the `_data` variable name; callers know to
 *     pick up the matching `_len`.
 *   - `emitBinary`, `emitUnary`, `emitCall`, `emitIndex`,
 *     `emitTruthiness` — specialised entry points used by the
 *     statement-level dispatch.
 *
 * Complex-typed scalar expressions are rejected here — they produce a
 * `(re, im)` pair and must route through `emitComplex` in
 * [complexScalar.ts](./complexScalar.ts).
 *
 * Tensor-valued Binary / Unary / Call expressions are also rejected in
 * value position; their callers route through the tensor module.
 */
import { UnaryOperation } from "../../../parser/types.js";
import { type JitExpr, isKnownInteger } from "../../jitTypes.js";
import { getIBuiltin } from "../../../interpreter/builtins/types.js";
import {
  emitScalarBinaryOp,
  emitScalarUnaryOp,
  emitScalarTruthiness,
} from "../../scalarEmit.js";
import {
  C_SCALAR_TARGET,
  formatNumberLiteral,
  getTensorReductionOp,
  getTensorUnaryOp,
  hasFreshAlloc,
  isTensorVar,
  mangle,
  tensorD0,
  tensorD1,
  tensorData,
  tensorLen,
  type EmitCtx,
} from "../context.js";
import { cStringLiteral, isComplexExpr, isTensorExpr } from "./helpers.js";
import { emitComplex } from "./complexScalar.js";
import {
  emitComplexTensorExprToStmts,
  emitRangeSliceReadToStmts,
  emitTensorExprToStmts,
} from "./tensor.js";
import { emitUserCall } from "./userCall.js";

/** Emit a value-expression. For scalars, returns a C `double` expression.
 *  For tensors, returns the data-variable name (the caller knows to also
 *  access the corresponding _len variable).
 *
 *  Complex-typed expressions are *not* valid here — they produce a pair
 *  of doubles and must go through `emitComplex`. Reaching this function
 *  with a complex expression indicates a missed routing at the caller. */
export function emitExpr(expr: JitExpr, ctx: EmitCtx): string {
  if (isComplexExpr(expr)) {
    throw new Error(
      `C-JIT codegen: emitExpr on complex expr ${expr.tag} — route through emitComplex`
    );
  }
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

    case "UserCall":
      return emitUserCall(expr, ctx);

    default:
      throw new Error(`C-JIT codegen: unsupported expr ${expr.tag}`);
  }
}

export function emitBinary(
  expr: JitExpr & { tag: "Binary" },
  ctx: EmitCtx
): string {
  if (isTensorExpr(expr)) {
    throw new Error(
      "C-JIT codegen: tensor binary must be emitted via statement context"
    );
  }
  const l = emitExpr(expr.left, ctx);
  const r = emitExpr(expr.right, ctx);
  return emitScalarBinaryOp(expr.op, l, r, C_SCALAR_TARGET);
}

export function emitUnary(
  expr: JitExpr & { tag: "Unary" },
  ctx: EmitCtx
): string {
  if (isTensorExpr(expr)) {
    throw new Error(
      "C-JIT codegen: tensor unary must be emitted via statement context"
    );
  }
  const operand = emitExpr(expr.operand, ctx);
  return emitScalarUnaryOp(expr.op, operand, C_SCALAR_TARGET);
}

export function emitIndex(
  expr: JitExpr & { tag: "Index" },
  ctx: EmitCtx
): string {
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

export function emitCall(
  expr: JitExpr & { tag: "Call" },
  ctx: EmitCtx
): string {
  if (isTensorExpr(expr) && getTensorUnaryOp(expr.name)) {
    throw new Error(
      "C-JIT codegen: tensor unary call must be emitted via statement context"
    );
  }
  // Tensor reduction: result is scalar. Emit reduction inline.
  if (!isTensorExpr(expr)) {
    const opEnum = getTensorReductionOp(expr.name);
    if (opEnum) {
      const arg = expr.args[0];
      // Complex-tensor arg with a real scalar result (any/all). Feasibility
      // already restricts to opcodes numbl_complex_flat_reduce accepts.
      // sum/prod on complex go through emitComplex — only any/all arrive
      // here with a real scalar return type.
      if (
        arg.jitType.kind === "tensor" &&
        arg.jitType.isComplex === true &&
        ctx.pendingStmts
      ) {
        const { lines, indent } = ctx.pendingStmts;
        const operand = emitComplexTensorExprToStmts(lines, indent, arg, ctx);
        const n = ++ctx.tmp.n;
        const reVar = `__cr${n}_re`;
        const imVar = `__cr${n}_im`;
        lines.push(`${indent}double ${reVar} = 0.0;`);
        lines.push(`${indent}double ${imVar} = 0.0;`);
        lines.push(
          `${indent}numbl_complex_flat_reduce(${opEnum}, (size_t)${operand.len}, ${operand.data}, ${operand.dataIm}, &${reVar}, &${imVar});`
        );
        // any/all write the flag into out_re; out_im is untouched (per
        // numbl_ops.h). Return re as the scalar result.
        return reVar;
      }
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
        `C-JIT codegen: reduction of tensor expr outside statement context`
      );
    }
  }
  if (expr.name === "tic" && expr.args.length === 0) {
    ctx.needsTicState = true;
    return `numbl_tic(__tic_state)`;
  }
  if (expr.name === "toc" && expr.args.length === 0) {
    ctx.needsTicState = true;
    return `numbl_toc(__tic_state)`;
  }
  // disp(str_literal | numeric_scalar) — route through the
  // JS-registered NumblDispCb callback. The call is a void expression
  // wrapped by ExprStmt's `(void)(...)`. Strings are passed via `s` with
  // kind=0; numbers via `num` with kind=1. Cast result to `double` so
  // the containing expression context (always ExprStmt today) still
  // type-checks; the cast is a no-op at runtime.
  if (expr.name === "disp" && expr.args.length === 1) {
    const a = expr.args[0];
    ctx.needsDispCb = true;
    if (a.tag === "StringLiteral") {
      const s = cStringLiteral(a.value);
      return `(__disp_cb(${s}, 0.0, 0), 0.0)`;
    }
    const numCode = emitExpr(a, ctx);
    return `(__disp_cb((const char *)0, (double)(${numCode}), 1), 0.0)`;
  }
  // real / imag / conj on a complex scalar arg. conj(z) is complex and
  // handled in emitComplex; here we handle the real-returning cases
  // (real(z), imag(z)) where the result goes into a real scalar context.
  if (
    (expr.name === "real" || expr.name === "imag") &&
    expr.args.length === 1 &&
    isComplexExpr(expr.args[0])
  ) {
    const pair = emitComplex(expr.args[0], ctx);
    return expr.name === "real" ? pair.re : pair.im;
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
      // MATLAB: length(A) is 0 if any dim is 0, else max(size(A)). The
      // 2D max(d0, d1) form must guard on len==0 — a 0x5 / 5x0 / 0x1
      // tensor has total length 0 but one non-zero dim.
      if (hasFreshAlloc(ctx, name)) {
        return `((double)((${lenCode} == 0) ? 0 : ((${tensorD0(name)} > ${tensorD1(name)}) ? ${tensorD0(name)} : ${tensorD1(name)})))`;
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

export function emitTruthiness(expr: JitExpr, ctx: EmitCtx): string {
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
