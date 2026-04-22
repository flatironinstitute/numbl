/**
 * Complex-scalar expression emission.
 *
 * Complex scalar values in the JIT IR (`complex_or_number`) are paired
 * `(re, im)` doubles on the C side — `mangle(name)` holds the real
 * part, `mangleIm(name)` holds the imaginary part. This module lowers
 * complex-valued expressions into the matching pair of C expressions.
 *
 * Exports:
 *   - `emitComplex(expr, ctx)` — lower an expression whose JitType is
 *     `complex_or_number` (or a real scalar that needs to be widened
 *     to a complex pair) into a `{re, im}` pair of C expressions.
 *   - `emitComplexScalarPair(expr, ctx)` — thin wrapper used at the
 *     tensor-op boundary; real scalars get im="0.0", complex scalars
 *     go through `emitComplex`.
 *   - `materializeComplexPair` — stage a pair into fresh locals so the
 *     formula can reference each sub-expression more than once without
 *     re-evaluating it.
 *
 * The complex-reduction case (sum/prod on a complex tensor) also lands
 * here since it returns a complex scalar — it delegates tensor-side
 * materialization to the tensor module.
 */
import { BinaryOperation, UnaryOperation } from "../../../parser/types.js";
import type { JitExpr } from "../../jitTypes.js";
import {
  getTensorReductionOp,
  isComplexScalarVar,
  mangle,
  mangleIm,
  type EmitCtx,
} from "../codegenCtx.js";
import {
  isComplexExpr,
  widenRealToComplex,
  type ComplexPair,
} from "./helpers.js";
import { emitExpr } from "./scalar.js";
import { emitComplexTensorExprToStmts } from "./tensor.js";

/** Materialize a ComplexPair into two fresh `__cN_re` / `__cN_im` locals
 *  prepended to `ctx.pendingStmts.lines`. Used when the emitted formula
 *  would otherwise evaluate either sub-expression more than once (e.g.
 *  both Mul and Div reference each operand twice in the per-component
 *  formula). Returns references to the new locals. */
export function materializeComplexPair(
  pair: ComplexPair,
  ctx: EmitCtx
): ComplexPair {
  if (!ctx.pendingStmts) {
    throw new Error(
      "C-JIT codegen: materializeComplexPair outside statement context"
    );
  }
  const n = ++ctx.tmp.n;
  const reVar = `__c${n}_re`;
  const imVar = `__c${n}_im`;
  const { lines, indent } = ctx.pendingStmts;
  lines.push(`${indent}double ${reVar} = ${pair.re};`);
  lines.push(`${indent}double ${imVar} = ${pair.im};`);
  return { re: reVar, im: imVar };
}

/** Emit a complex-valued scalar expression, returning (re, im) C
 *  expression strings. Call sites that need a pair (complex RHS, complex
 *  operand of a complex op, arg of real/imag/conj on complex) route
 *  through here. Real sub-expressions widen implicitly (im = 0). */
export function emitComplex(expr: JitExpr, ctx: EmitCtx): ComplexPair {
  // Real scalar value in a complex position: widen with im = 0.
  if (!isComplexExpr(expr)) {
    if (expr.jitType.kind !== "number" && expr.jitType.kind !== "boolean") {
      throw new Error(
        `C-JIT codegen: emitComplex on non-scalar expr ${expr.tag}:${expr.jitType.kind}`
      );
    }
    return widenRealToComplex(emitExpr(expr, ctx));
  }

  switch (expr.tag) {
    case "ImagLiteral":
      return { re: "0.0", im: "1.0" };

    case "Var": {
      if (!isComplexScalarVar(ctx, expr.name)) {
        throw new Error(
          `C-JIT codegen: complex Var '${expr.name}' not in complexScalarVars`
        );
      }
      return { re: mangle(expr.name), im: mangleIm(expr.name) };
    }

    case "Unary": {
      if (expr.op === UnaryOperation.Plus) {
        return emitComplex(expr.operand, ctx);
      }
      if (expr.op === UnaryOperation.Minus) {
        const o = emitComplex(expr.operand, ctx);
        return { re: `(-(${o.re}))`, im: `(-(${o.im}))` };
      }
      throw new Error(`C-JIT codegen: unsupported complex unary op ${expr.op}`);
    }

    case "Binary": {
      // Both operands lowered to pairs + materialized so the per-
      // component formulas below don't duplicate evaluation.
      const l = materializeComplexPair(emitComplex(expr.left, ctx), ctx);
      const r = materializeComplexPair(emitComplex(expr.right, ctx), ctx);
      switch (expr.op) {
        case BinaryOperation.Add:
          return { re: `(${l.re} + ${r.re})`, im: `(${l.im} + ${r.im})` };
        case BinaryOperation.Sub:
          return { re: `(${l.re} - ${r.re})`, im: `(${l.im} - ${r.im})` };
        case BinaryOperation.Mul:
        case BinaryOperation.ElemMul:
          return {
            re: `(${l.re} * ${r.re} - ${l.im} * ${r.im})`,
            im: `(${l.re} * ${r.im} + ${l.im} * ${r.re})`,
          };
        case BinaryOperation.Div:
        case BinaryOperation.ElemDiv:
          return emitComplexDiv(l, r, ctx);
        default:
          throw new Error(
            `C-JIT codegen: unsupported complex binary op ${expr.op}`
          );
      }
    }

    case "Call": {
      if (expr.name === "conj" && expr.args.length === 1) {
        const o = emitComplex(expr.args[0], ctx);
        return { re: o.re, im: `(-(${o.im}))` };
      }
      if (expr.name === "real" && expr.args.length === 1) {
        const o = emitComplex(expr.args[0], ctx);
        return { re: o.re, im: "0.0" };
      }
      if (expr.name === "imag" && expr.args.length === 1) {
        const o = emitComplex(expr.args[0], ctx);
        return { re: o.im, im: "0.0" };
      }
      // Complex tensor reduction returning a complex scalar (sum / prod
      // / any / all on a complex tensor). numbl_complex_flat_reduce
      // writes into two scratch doubles; return those as the pair.
      const opEnum = getTensorReductionOp(expr.name);
      if (
        opEnum &&
        expr.args.length === 1 &&
        expr.args[0].jitType.kind === "tensor" &&
        expr.args[0].jitType.isComplex === true
      ) {
        if (!ctx.pendingStmts) {
          throw new Error(
            `C-JIT codegen: complex reduction '${expr.name}' outside statement context`
          );
        }
        const { lines, indent } = ctx.pendingStmts;
        const operand = emitComplexTensorExprToStmts(
          lines,
          indent,
          expr.args[0],
          ctx
        );
        const n = ++ctx.tmp.n;
        const reVar = `__cr${n}_re`;
        const imVar = `__cr${n}_im`;
        lines.push(`${indent}double ${reVar} = 0.0;`);
        lines.push(`${indent}double ${imVar} = 0.0;`);
        lines.push(
          `${indent}numbl_complex_flat_reduce(${opEnum}, (size_t)${operand.len}, ${operand.data}, ${operand.dataIm}, &${reVar}, &${imVar});`
        );
        return { re: reVar, im: imVar };
      }
      throw new Error(`C-JIT codegen: unsupported complex Call '${expr.name}'`);
    }

    default:
      throw new Error(`C-JIT codegen: unsupported complex expr ${expr.tag}`);
  }
}

/** Complex division using Smith's method with MATLAB-compatible
 *  division-by-zero semantics (produces signed Inf, not NaN, when the
 *  denominator is exactly zero). Mirrors cDiv in jitHelpersComplex.ts
 *  so C-JIT and JS-JIT agree bit-for-bit on divide-by-zero cases.
 *
 *  Emits scratch locals because the formulas branch on |br| vs |bi| and
 *  reuse the operand components five or six times each. */
function emitComplexDiv(
  a: ComplexPair,
  b: ComplexPair,
  ctx: EmitCtx
): ComplexPair {
  if (!ctx.pendingStmts) {
    throw new Error("C-JIT codegen: emitComplexDiv outside statement context");
  }
  const { lines, indent } = ctx.pendingStmts;
  const n = ++ctx.tmp.n;
  const reVar = `__cdiv${n}_re`;
  const imVar = `__cdiv${n}_im`;
  lines.push(`${indent}double ${reVar};`);
  lines.push(`${indent}double ${imVar};`);
  lines.push(`${indent}if (fabs(${b.re}) >= fabs(${b.im})) {`);
  lines.push(`${indent}  if (${b.re} == 0.0 && ${b.im} == 0.0) {`);
  lines.push(`${indent}    ${reVar} = ${a.re} / 0.0;`);
  lines.push(`${indent}    ${imVar} = ${a.im} / 0.0;`);
  lines.push(`${indent}  } else {`);
  lines.push(`${indent}    double __r = ${b.im} / ${b.re};`);
  lines.push(`${indent}    double __d = ${b.re} + ${b.im} * __r;`);
  lines.push(`${indent}    ${reVar} = (${a.re} + ${a.im} * __r) / __d;`);
  lines.push(`${indent}    ${imVar} = (${a.im} - ${a.re} * __r) / __d;`);
  lines.push(`${indent}  }`);
  lines.push(`${indent}} else {`);
  lines.push(`${indent}  double __r = ${b.re} / ${b.im};`);
  lines.push(`${indent}  double __d = ${b.im} + ${b.re} * __r;`);
  lines.push(`${indent}  ${reVar} = (${a.re} * __r + ${a.im}) / __d;`);
  lines.push(`${indent}  ${imVar} = (${a.im} * __r - ${a.re}) / __d;`);
  lines.push(`${indent}}`);
  return { re: reVar, im: imVar };
}

/** Emit a scalar sub-expression at a complex tensor op boundary. Returns
 *  a (re, im) pair of C expressions. Real scalars become (expr, "0.0");
 *  complex scalars go through emitComplex for their pair form. */
export function emitComplexScalarPair(
  expr: JitExpr,
  ctx: EmitCtx
): ComplexPair {
  if (expr.jitType.kind === "complex_or_number") {
    return emitComplex(expr, ctx);
  }
  return { re: emitExpr(expr, ctx), im: "0.0" };
}
