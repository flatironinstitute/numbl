/**
 * Expression lowering for the JIT. Paired with jitLowerStmt.ts; the two
 * sides are mutually recursive (stmt bodies contain exprs; function-call
 * exprs may trigger nested lowering of a callee's body). jitLower.ts is
 * the orchestrator that ties them together via LowerCtx + lowerFunction.
 */

import type { Expr, Stmt } from "../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../parser/types.js";
import type { FunctionDef } from "../interpreter/types.js";
import type { Interpreter } from "../interpreter/interpreter.js";
import type { CallSite } from "../runtime/runtimeHelpers.js";
import { resolveFunction } from "../functionResolve.js";
import type { ItemType } from "../lowering/itemTypes.js";
import {
  type JitType,
  type JitExpr,
  isNumericScalarType,
  isTensorType,
  jitTypeKey,
  computeJitFnName,
  signFromNumber,
} from "./jitTypes.js";
import {
  KNOWN_CONSTANTS,
  binaryResultType,
  unaryResultType,
} from "./jitLowerTypes.js";
import { generateJS } from "./js/jitCodegen.js";
import { tryEmitScalarFnKernel } from "./e1/scalarFnKernel.js";
import { getIBuiltin, inferJitType } from "../interpreter/builtins/index.js";
import { isRuntimeFunction } from "../runtime/types.js";
import type { RuntimeValue } from "../runtime/types.js";
import { offsetToLineFast } from "../runtime/error.js";
import type { LowerCtx, SliceAlias } from "./jitLower.js";
import { lowerFunction, setBailReason } from "./jitLower.js";

const LOG_CJIT_MISSES =
  typeof process !== "undefined" && !!process.env.NUMBL_LOG_CJIT_MISSES;

/** Pull a literal number out of a JitExpr, if any. Covers NumberLiteral
 *  directly and Var/etc. whose `exact` field was preserved during inference. */
function literalNumber(e: JitExpr): number | null {
  if (e.tag === "NumberLiteral")
    return typeof e.value === "number" ? e.value : null;
  if (e.jitType.kind === "number" && e.jitType.exact !== undefined) {
    return e.jitType.exact;
  }
  if (e.jitType.kind === "boolean" && e.jitType.value !== undefined) {
    return e.jitType.value ? 1 : 0;
  }
  return null;
}

/** Fold a Binary expression whose operands are both literal. Only covers
 *  the ops we need for dead-branch elimination on `if nargout > K`. */
function tryConstantFoldBinary(
  op: BinaryOperation,
  left: JitExpr,
  right: JitExpr
): JitExpr | null {
  const a = literalNumber(left);
  const b = literalNumber(right);
  if (a === null || b === null) return null;
  let v: number | null = null;
  switch (op) {
    case BinaryOperation.Equal:
      v = a === b ? 1 : 0;
      break;
    case BinaryOperation.NotEqual:
      v = a !== b ? 1 : 0;
      break;
    case BinaryOperation.Less:
      v = a < b ? 1 : 0;
      break;
    case BinaryOperation.LessEqual:
      v = a <= b ? 1 : 0;
      break;
    case BinaryOperation.Greater:
      v = a > b ? 1 : 0;
      break;
    case BinaryOperation.GreaterEqual:
      v = a >= b ? 1 : 0;
      break;
    default:
      return null;
  }
  return {
    tag: "NumberLiteral",
    value: v,
    jitType: { kind: "boolean", value: v === 1 },
  };
}

function bailExpr(ctx: LowerCtx, expr: Expr, reason: string): null {
  const line = expr.span
    ? ctx.lineTable
      ? offsetToLineFast(ctx.lineTable, expr.span.start)
      : undefined
    : undefined;
  setBailReason(ctx, `${expr.type}: ${reason}`, line);
  return null;
}

// ── Expression lowering ─────────────────────────────────────────────────

export function lowerExpr(ctx: LowerCtx, expr: Expr): JitExpr | null {
  if (LOG_CJIT_MISSES) {
    ctx.lastExprType = expr.type;
    if (expr.span && ctx.lineTable) {
      ctx.lastExprLine = offsetToLineFast(ctx.lineTable, expr.span.start);
    }
  }
  switch (expr.type) {
    case "Number": {
      const value = parseFloat(expr.value);
      const sign = signFromNumber(value);
      return {
        tag: "NumberLiteral",
        value,
        jitType: {
          kind: "number",
          exact: value,
          ...(sign ? { sign } : {}),
          ...(Number.isInteger(value) ? { isInteger: true } : {}),
        },
      };
    }

    case "ImagUnit":
      return {
        tag: "ImagLiteral",
        jitType: { kind: "complex_or_number", pureImaginary: true },
      };

    case "Ident": {
      // When a slice alias is read as a whole value (not indexed), we
      // materialize it into a real tensor via a helper call. This lets
      // patterns like `rx = A(1,:); r2 = rx .^ 2;` work in the JIT.
      if (ctx.sliceAliases.has(expr.name)) {
        const alias = ctx.sliceAliases.get(expr.name)!;
        return materializeSliceAlias(ctx, alias);
      }

      // Known numeric constants
      const constVal = KNOWN_CONSTANTS[expr.name];
      if (constVal !== undefined) {
        const isBool = expr.name === "true" || expr.name === "false";
        return {
          tag: "NumberLiteral",
          value: constVal,
          jitType: isBool
            ? { kind: "boolean", value: expr.name === "true" }
            : {
                kind: "number",
                exact: constVal,
                ...(signFromNumber(constVal)
                  ? { sign: signFromNumber(constVal) }
                  : {}),
              },
        };
      }
      const type = ctx.env.get(expr.name);
      if (type) return { tag: "Var", name: expr.name, jitType: type };
      // Not a variable — try resolving as a zero-arg builtin call.
      // MATLAB lets bare identifiers like `toc` parse as Ident but
      // execute as zero-arg calls; synthesize the equivalent FuncCall
      // and route through lowerIBuiltinCall so things like `t = toc`
      // can JIT. Skip frame-sensitive builtins (nargin/nargout/etc.)
      // whose IBuiltin apply doesn't see the active call frame —
      // the interpreter handles these via special env lookups that
      // the JIT doesn't yet model.
      // Exception: `nargout` is a constant within a given JIT specialization
      // (the JIT specializes per nargout). Inline it as a number literal.
      // `nargin` isn't quite as simple — callers can omit trailing args so
      // nargin != ctx.params.size — leave it to the bail path for now.
      if (expr.name === "nargout" && ctx.nargout !== undefined) {
        const v = ctx.nargout;
        return {
          tag: "NumberLiteral",
          value: v,
          jitType: {
            kind: "number",
            exact: v,
            isInteger: true,
            sign: "nonneg",
          },
        };
      }
      if (FRAME_SENSITIVE_NO_ARG_BUILTINS.has(expr.name)) return null;
      const zeroArgCall: Expr & { type: "FuncCall" } = {
        type: "FuncCall",
        name: expr.name,
        args: [],
        span: expr.span,
      };
      return lowerIBuiltinCall(ctx, zeroArgCall);
    }

    case "Binary": {
      const left = lowerExpr(ctx, expr.left);
      if (!left) return null;
      const right = lowerExpr(ctx, expr.right);
      if (!right) return null;
      // Constant fold comparisons and a handful of arithmetic ops when both
      // operands are literal-valued — mainly to make `if nargout > K` dead
      // branches vanish before lowering tries (and fails) to lower their
      // bodies. The result is a 0/1 NumberLiteral for comparisons.
      const folded = tryConstantFoldBinary(expr.op, left, right);
      if (folded) return folded;
      const resultType = binaryResultType(
        expr.op,
        left.jitType,
        right.jitType,
        left,
        right
      );

      // Matrix multiply: tensor * tensor goes through the mtimes IBuiltin
      // rather than the element-wise Binary path (which only handles scalar
      // or broadcast element-wise ops).
      if (
        !resultType &&
        expr.op === BinaryOperation.Mul &&
        isTensorType(left.jitType) &&
        isTensorType(right.jitType)
      ) {
        const lt = left.jitType as Extract<JitType, { kind: "tensor" }>;
        const rt = right.jitType as Extract<JitType, { kind: "tensor" }>;
        // Infer output shape from input shapes when known: (M×K) * (K×N) → (M×N)
        const outShape =
          lt.shape && rt.shape && lt.shape.length === 2 && rt.shape.length === 2
            ? [lt.shape[0], rt.shape[1]]
            : undefined;
        const isComplex = (lt.isComplex || rt.isComplex) ?? false;
        ctx._hasTensorOps = true;
        return {
          tag: "Call",
          name: "__mtimes",
          args: [left, right],
          jitType: {
            kind: "tensor",
            isComplex,
            ...(outShape ? { shape: outShape } : {}),
            ndim: 2,
          },
        };
      }

      if (!resultType || resultType.kind === "unknown") return null;

      // Track ops that need $h helpers (tensor or complex)
      if (
        isTensorType(left.jitType) ||
        isTensorType(right.jitType) ||
        left.jitType.kind === "complex_or_number" ||
        right.jitType.kind === "complex_or_number"
      ) {
        ctx._hasTensorOps = true;
      }

      return { tag: "Binary", op: expr.op, left, right, jitType: resultType };
    }

    case "Unary": {
      const operand = lowerExpr(ctx, expr.operand);
      if (!operand) return null;
      const resultType = unaryResultType(expr.op, operand.jitType);
      if (!resultType || resultType.kind === "unknown") return null;

      if (
        isTensorType(operand.jitType) ||
        operand.jitType.kind === "complex_or_number"
      )
        ctx._hasTensorOps = true;

      return { tag: "Unary", op: expr.op, operand, jitType: resultType };
    }

    case "Tensor": {
      // Stage 11 vertical-concat-growth fast path: `[base; value]` where
      // `base` is a real tensor (empty or column vector) and `value` is
      // a numeric scalar. Mirrors the chunkie `it = [it; i]` grow-a-list
      // pattern. Must run before the generic TensorLiteral path because
      // the latter rejects any non-scalar row element.
      if (
        expr.rows.length === 2 &&
        expr.rows[0].length === 1 &&
        expr.rows[1].length === 1
      ) {
        const base = lowerExpr(ctx, expr.rows[0][0]);
        if (
          base &&
          base.jitType.kind === "tensor" &&
          base.jitType.isComplex === false
        ) {
          const value = lowerExpr(ctx, expr.rows[1][0]);
          if (
            value &&
            (value.jitType.kind === "number" ||
              value.jitType.kind === "boolean")
          ) {
            ctx._hasTensorOps = true;
            return {
              tag: "VConcatGrow",
              base,
              value,
              jitType: {
                kind: "tensor",
                isComplex: false,
                shape: [-1, 1],
              },
            };
          }
        }
      }

      // Horizontal concat fast path: [a, b] (single row with 2+ elements)
      // where at least one element is a tensor/unknown (NOT string/char).
      // Emits a runtime helper call. Handles the flagself pattern
      // [c{idx}, scalar] → row vector growth.
      if (expr.rows.length === 1 && expr.rows[0].length >= 2) {
        const loweredElems = expr.rows[0].map(e => lowerExpr(ctx, e));
        if (loweredElems.every(e => e !== null)) {
          const elems = loweredElems as JitExpr[];
          // Only match numeric-ish elements (number, boolean, tensor, unknown)
          // — NOT strings or chars (those use char-concat semantics).
          const isNumericIsh = (k: string) =>
            k === "number" ||
            k === "boolean" ||
            k === "complex_or_number" ||
            k === "tensor" ||
            k === "unknown";
          if (
            elems.every(e => isNumericIsh(e.jitType.kind)) &&
            elems.some(
              e => e.jitType.kind === "tensor" || e.jitType.kind === "unknown"
            )
          ) {
            ctx._hasTensorOps = true;
            return {
              tag: "Call",
              name: "__horzcat",
              args: elems,
              jitType: { kind: "tensor", isComplex: false },
            };
          }
        }
      }

      const rows: JitExpr[][] = [];
      let hasComplex = false;
      for (const row of expr.rows) {
        const loweredRow: JitExpr[] = [];
        for (const elem of row) {
          const lowered = lowerExpr(ctx, elem);
          if (!lowered) return null;
          // Only scalar elements supported in tensor literals
          if (
            lowered.jitType.kind !== "number" &&
            lowered.jitType.kind !== "boolean" &&
            lowered.jitType.kind !== "complex_or_number"
          )
            return null;
          if (lowered.jitType.kind === "complex_or_number") hasComplex = true;
          loweredRow.push(lowered);
        }
        rows.push(loweredRow);
      }
      const nRows = rows.length;
      const nCols = rows[0]?.length ?? 0;
      ctx._hasTensorOps = true;
      return {
        tag: "TensorLiteral",
        rows,
        nRows,
        nCols,
        jitType: {
          kind: "tensor",
          isComplex: hasComplex,
          shape: [nRows, nCols],
        },
      };
    }

    case "FuncCall": {
      // If the name is a function handle variable, emit an indirect call
      // instead of treating it as indexing. This enables JIT compilation of
      // loops that call function handles (e.g. kern(srcinfo, targinfo) in
      // chunkie's adapgausskerneval).
      const varType = ctx.env.get(expr.name);
      if (varType && varType.kind === "function_handle") {
        const args = expr.args.map(a => lowerExpr(ctx, a));
        if (args.some(a => a === null)) return null;
        const loweredArgs = args as JitExpr[];

        // Determine return type by probing the function handle at JIT
        // compile time: call it once with the actual argument values and
        // inspect the result type. If the probe can't determine a type,
        // bail — we won't guess.
        if (!ctx.interp) return null;
        const returnType = probeFuncHandleReturnType(
          ctx.interp,
          expr.name,
          loweredArgs
        );
        if (!returnType) return null;

        ctx._hasTensorOps = true;
        return {
          tag: "FuncHandleCall",
          name: expr.name,
          args: loweredArgs,
          jitType: returnType,
        };
      }

      // Stage 21: range slice read `src(a:b)` — parser may emit
      // FuncCall when `src` is a variable. Match BEFORE lowerIndexExpr
      // so the Range isn't rejected by the all-scalar check.
      if (varType && expr.args.length === 1 && expr.args[0].type === "Range") {
        const result = tryLowerRangeSliceRead(ctx, expr.name, expr.args[0]);
        if (result) return result;
      }

      // If the name is a known variable, treat as indexing (MATLAB ambiguity)
      if (varType) {
        return lowerIndexExpr(ctx, {
          base: { tag: "Var", name: expr.name, jitType: varType },
          indices: expr.args,
        });
      }

      // Slice alias `pt(k)` — parsed as FuncCall because the parser
      // doesn't know `pt` is a variable. Handle it like Index.
      const alias = ctx.sliceAliases.get(expr.name);
      if (alias) return lowerSliceAliasRead(ctx, alias, expr.args);

      // Fold the function-call form `and(a, b)` / `or(a, b)` / `not(a)` to
      // the operator form when both args are simple numeric scalars. This
      // mirrors what `&&` / `||` / `~` already lower to (a JS Binary/Unary
      // node), avoiding the per-iter `$h.ib_and(...)` helper hop. The fold
      // is only safe for plain numeric/boolean operands — complex truthiness
      // doesn't match JS truthiness, so for `complex_or_number` we fall
      // through to the IBuiltin path. Variable shadowing is already handled
      // above; we trust no JIT-able workspace function shadows these
      // builtins.
      if (
        (expr.name === "and" || expr.name === "or") &&
        expr.args.length === 2
      ) {
        const left = lowerExpr(ctx, expr.args[0]);
        if (
          left &&
          (left.jitType.kind === "number" || left.jitType.kind === "boolean")
        ) {
          const right = lowerExpr(ctx, expr.args[1]);
          if (
            right &&
            (right.jitType.kind === "number" ||
              right.jitType.kind === "boolean")
          ) {
            return {
              tag: "Binary",
              op:
                expr.name === "and"
                  ? BinaryOperation.AndAnd
                  : BinaryOperation.OrOr,
              left,
              right,
              jitType: { kind: "boolean" },
            };
          }
        }
      }
      if (expr.name === "not" && expr.args.length === 1) {
        const operand = lowerExpr(ctx, expr.args[0]);
        if (
          operand &&
          (operand.jitType.kind === "number" ||
            operand.jitType.kind === "boolean")
        ) {
          return {
            tag: "Unary",
            op: UnaryOperation.Not,
            operand,
            jitType: { kind: "boolean" },
          };
        }
      }

      // Fold `bsxfun(@op, a, b)` where @op is a known arithmetic operator
      // into a direct call to the runtime's broadcasting-aware arithmetic
      // helpers (mSub, mElemDiv, etc.). These handle shape broadcasting
      // correctly, unlike the JIT's element-wise tSub/tDiv which require
      // same-shape operands.
      if (
        expr.name === "bsxfun" &&
        expr.args.length === 3 &&
        expr.args[0].type === "FuncHandle"
      ) {
        const bsxfunHelperMap: Record<string, string> = {
          minus: "__mSub",
          plus: "__mAdd",
          rdivide: "__mElemDiv",
          // ldivide not yet supported
          times: "__mElemMul",
          power: "__mElemPow",
        };
        const helperName = bsxfunHelperMap[expr.args[0].name];
        if (helperName) {
          const left = lowerExpr(ctx, expr.args[1]);
          if (!left) return null;
          const right = lowerExpr(ctx, expr.args[2]);
          if (!right) return null;
          // Compute broadcast result type
          const binOp =
            expr.args[0].name === "minus"
              ? BinaryOperation.Sub
              : expr.args[0].name === "plus"
                ? BinaryOperation.Add
                : expr.args[0].name === "rdivide"
                  ? BinaryOperation.ElemDiv
                  : expr.args[0].name === "ldivide"
                    ? BinaryOperation.ElemLeftDiv
                    : expr.args[0].name === "times"
                      ? BinaryOperation.ElemMul
                      : BinaryOperation.ElemPow;
          const resultType = binaryResultType(
            binOp,
            left.jitType,
            right.jitType,
            left,
            right
          );
          if (!resultType || resultType.kind === "unknown") return null;
          ctx._hasTensorOps = true;
          return {
            tag: "Call",
            name: helperName,
            args: [left, right],
            jitType: resultType,
          };
        }
      }

      // Try user function resolution (nested → local → workspace → class method)
      const userResult = lowerUserFuncCall(ctx, expr);
      if (userResult !== undefined) return userResult;

      // Try IBuiltin resolution (same priority as builtins — last)
      return lowerIBuiltinCall(ctx, expr);
    }

    case "Char": {
      // Strip enclosing quotes and unescape doubled single-quotes
      const charVal = expr.value.slice(1, -1).replaceAll("''", "'");
      return {
        tag: "StringLiteral",
        value: charVal,
        isChar: true,
        jitType: { kind: "char", value: charVal },
      };
    }

    case "String": {
      // Strip surrounding quotes and unescape doubled quotes (same as interpreter)
      let strVal = expr.value.slice(1, expr.value.length - 1);
      strVal = strVal.replaceAll('""', '"');
      return {
        tag: "StringLiteral",
        value: strVal,
        isChar: false,
        jitType: { kind: "string", value: strVal },
      };
    }

    case "Index": {
      // Slice alias intercept: `pt(k)` where `pt = pts(:, i)` → `pts(k, i)`.
      if (expr.base.type === "Ident") {
        const alias = ctx.sliceAliases.get(expr.base.name);
        if (alias) return lowerSliceAliasRead(ctx, alias, expr.indices);
      }
      // Stage 21: range slice read `src(a:b)` on a real-tensor base
      // returns a fresh column-vector tensor. Must match BEFORE
      // `lowerExpr(expr.base)` so the Range expression isn't lowered
      // as a standalone tensor.
      if (
        expr.base.type === "Ident" &&
        expr.indices.length === 1 &&
        expr.indices[0].type === "Range"
      ) {
        const result = tryLowerRangeSliceRead(
          ctx,
          expr.base.name,
          expr.indices[0]
        );
        if (result) return result;
      }
      const base = lowerExpr(ctx, expr.base);
      if (!base) return null;
      return lowerIndexExpr(ctx, { base, indices: expr.indices });
    }

    case "IndexCell": {
      // Cell array scalar read: c{i}
      if (expr.base.type !== "Ident") return null;
      const cellType = ctx.env.get(expr.base.name);
      if (!cellType || cellType.kind !== "cell") return null;
      if (expr.indices.length !== 1) return null;
      const cellIdx = lowerExpr(ctx, expr.indices[0]);
      if (!cellIdx) return null;
      if (
        cellIdx.jitType.kind !== "number" &&
        cellIdx.jitType.kind !== "boolean"
      )
        return null;
      ctx._hasTensorOps = true;
      // Result type is unknown — the cell element could be any type.
      // Downstream operations that need a specific type (e.g. horzcat)
      // handle this via runtime dispatch in the helper.
      return {
        tag: "Call",
        name: "__cellRead",
        args: [
          { tag: "Var", name: expr.base.name, jitType: cellType },
          cellIdx,
        ],
        jitType: { kind: "unknown" },
      };
    }

    case "Member": {
      // Stage 13: chained struct array member read `T.nodes(i).leaf`.
      // The parser produces this shape in read position as
      //   Member(MethodCall(Ident(T), "nodes", [i]), "leaf")
      // — the middle node is MethodCall, not Index, because the `.` +
      // ident + `(` sequence is parsed as a method-call postfix.
      if (
        expr.base.type === "MethodCall" &&
        expr.base.base.type === "Ident" &&
        expr.base.args.length === 1
      ) {
        const structVarName = expr.base.base.name;
        const structArrayFieldName = expr.base.name;
        const leafFieldName = expr.name;
        const structType = ctx.env.get(structVarName);
        if (structType && structType.kind === "struct" && structType.fields) {
          const arrayFieldType = structType.fields[structArrayFieldName];
          if (
            arrayFieldType &&
            arrayFieldType.kind === "struct_array" &&
            arrayFieldType.elemFields
          ) {
            const leafType = arrayFieldType.elemFields[leafFieldName];
            // Accept scalar numeric fields (read to a scalar local or
            // used inline) or real-tensor fields (assigned to a local
            // which the existing hoist-refresh path picks up).
            const leafOk =
              leafType &&
              (isNumericScalarType(leafType) ||
                (leafType.kind === "tensor" && leafType.isComplex !== true));
            if (leafOk) {
              const idx = lowerExpr(ctx, expr.base.args[0]);
              if (
                idx &&
                (idx.jitType.kind === "number" ||
                  idx.jitType.kind === "boolean")
              ) {
                if (leafType.kind === "tensor") {
                  ctx._hasTensorOps = true;
                }
                return {
                  tag: "StructArrayMemberRead",
                  structVarName,
                  structArrayFieldName,
                  indexExpr: idx,
                  leafFieldName,
                  jitType: leafType,
                };
              }
            }
          }
        }
      }

      // Stage 12: scalar struct field read `s.f` where `s` is an Ident
      // whose type in the env is a struct with a statically-known scalar
      // field. Lowered to a `MemberRead` IR node; codegen hoists each
      // unique `(baseName, fieldName)` pair as a local alias at function
      // entry.
      //
      // The base is restricted to a plain Ident (no chained `a.b.c` yet)
      // and the field type must be a scalar numeric type. Class instances
      // aren't handled because field access may dispatch to a user-defined
      // getter method.
      if (expr.base.type !== "Ident") return null;
      const baseName = expr.base.name;
      const baseType = ctx.env.get(baseName);
      if (!baseType) return null;
      if (baseType.kind !== "struct") return null;
      if (!baseType.fields) return null;
      const fieldType = baseType.fields[expr.name];
      if (!fieldType) return null;
      if (!isNumericScalarType(fieldType)) return null;
      return {
        tag: "MemberRead",
        baseName,
        fieldName: expr.name,
        jitType: fieldType,
      };
    }

    default:
      return bailExpr(ctx, expr, "unsupported expression");
  }
}

// ── Index expression lowering ───────────────────────────────────────────

function lowerIndexExpr(
  ctx: LowerCtx,
  input: { base: JitExpr; indices: Expr[] }
): JitExpr | null {
  const { base } = input;

  // 2D colon-slice read: A(i, :) or A(:, j) on a real-tensor base produces
  // a row/column vector. Catch this before the per-index lowering loop so
  // the Colon indices don't fall through to lowerExpr's default bail.
  if (
    input.indices.length === 2 &&
    base.jitType.kind === "tensor" &&
    base.jitType.isComplex === false &&
    base.jitType.shape &&
    base.jitType.shape.length === 2
  ) {
    const colonLeft = input.indices[0].type === "Colon";
    const colonRight = input.indices[1].type === "Colon";
    if (colonLeft !== colonRight) {
      const fixedExpr = colonLeft ? input.indices[1] : input.indices[0];
      const fixedLowered = lowerExpr(ctx, fixedExpr);
      if (
        fixedLowered &&
        (fixedLowered.jitType.kind === "number" ||
          fixedLowered.jitType.kind === "boolean")
      ) {
        const colonPos = colonLeft ? 0 : 1;
        const sliceLen = base.jitType.shape[colonPos];
        const outShape = colonPos === 0 ? [sliceLen, 1] : [1, sliceLen];
        ctx._hasTensorOps = true;
        return {
          tag: "Call",
          name: "__extractSlice2d",
          args: [
            base,
            fixedLowered,
            {
              tag: "NumberLiteral",
              value: colonPos,
              jitType: { kind: "number", exact: colonPos },
            },
          ],
          jitType: { kind: "tensor", isComplex: false, shape: outShape },
        };
      }
    }
  }

  const indices: JitExpr[] = [];
  for (const idx of input.indices) {
    const lowered = lowerExpr(ctx, idx);
    if (!lowered) return null;
    indices.push(lowered);
  }
  if (indices.length === 0) return null;

  // Check if all indices are scalar
  const allScalar = indices.every(
    i => i.jitType.kind === "number" || i.jitType.kind === "boolean"
  );

  if (allScalar) {
    // Scalar indexing — result is a scalar
    let resultType: JitType;
    switch (base.jitType.kind) {
      case "tensor":
        resultType = base.jitType.isComplex
          ? { kind: "complex_or_number" }
          : { kind: "number" };
        break;
      case "number":
      case "boolean":
        resultType = { kind: "number" };
        break;
      case "complex_or_number":
        resultType = { kind: "complex_or_number" };
        break;
      default:
        return null;
    }
    ctx._hasTensorOps = true;
    return { tag: "Index", base, indices, jitType: resultType };
  }

  // Tensor indexing: base(tensorIdx) — single tensor index into a tensor base
  // Result is a tensor of the same complexity as the base.
  if (
    indices.length === 1 &&
    indices[0].jitType.kind === "tensor" &&
    base.jitType.kind === "tensor"
  ) {
    const isComplex = base.jitType.isComplex === true;
    ctx._hasTensorOps = true;
    return {
      tag: "Call",
      name: "__tensorIndex",
      args: [base, indices[0]],
      jitType: { kind: "tensor", isComplex },
    };
  }

  return null;
}

// ── Slice alias read/materialize ────────────────────────────────────────

/**
 * Lower a read `alias(...)` where `alias` is a slice-aliased name. Supports
 * two shapes:
 *   - linear indexing with a single index into a 1-colon slice:
 *     `pt(k)` where `pt = pts(:, i)` → `pts(k, i)`;
 *   - multi-indexing matching the number of colons:
 *     `pt(r, c)` where `pt = pts(:, :)` → `pts(r, c)`.
 * Anything else (wrong arity, slice-of-slice, etc.) bails.
 */
function lowerSliceAliasRead(
  ctx: LowerCtx,
  alias: SliceAlias,
  readIndices: Expr[]
): JitExpr | null {
  const ncolon = alias.colonPositions.length;

  const lowered: JitExpr[] = [];
  for (const idx of readIndices) {
    const lo = lowerExpr(ctx, idx);
    if (!lo) return null;
    if (lo.jitType.kind !== "number" && lo.jitType.kind !== "boolean")
      return null;
    lowered.push(lo);
  }

  let readForColon: JitExpr[];
  if (lowered.length === ncolon) {
    readForColon = lowered;
  } else if (lowered.length === 1 && ncolon === 1) {
    readForColon = [lowered[0]];
  } else {
    return null;
  }

  const fullIndices: JitExpr[] = [];
  let colonIdx = 0;
  for (const slot of alias.template) {
    if (slot.kind === "colon") {
      fullIndices.push(readForColon[colonIdx++]);
    } else {
      fullIndices.push(slot.expr);
    }
  }

  const baseExpr: JitExpr = {
    tag: "Var",
    name: alias.baseName,
    jitType: alias.baseType,
  };
  const resultType: JitType = { kind: "number" };
  ctx._hasTensorOps = true;
  return {
    tag: "Index",
    base: baseExpr,
    indices: fullIndices,
    jitType: resultType,
  };
}

/**
 * Materialize a slice alias as a real tensor. Called when a slice-aliased
 * name is used as a whole value (e.g. `rx .^ 2` where `rx = A(1,:)`).
 * Only supports 2D base tensors with exactly one colon dimension.
 */
function materializeSliceAlias(
  ctx: LowerCtx,
  alias: SliceAlias
): JitExpr | null {
  const bt = alias.baseType;
  if (bt.kind !== "tensor") return null;
  if (!bt.shape || bt.shape.length !== 2) return null;
  if (alias.colonPositions.length !== 1) return null;
  if (alias.template.length !== 2) return null;

  const colonPos = alias.colonPositions[0];
  const fixedSlot = alias.template[colonPos === 0 ? 1 : 0];
  if (fixedSlot.kind !== "expr") return null;

  const sliceLen = alias.sliceShape[0];
  if (sliceLen <= 0) return null;

  // Emit: $h.__extractSlice2d(base, fixedIdx, colonPos)
  ctx._hasTensorOps = true;
  const baseVar: JitExpr = {
    tag: "Var",
    name: alias.baseName,
    jitType: alias.baseType,
  };
  const shape =
    colonPos === 0
      ? [sliceLen, 1] // column slice → Mx1
      : [1, sliceLen]; // row slice → 1xN
  return {
    tag: "Call",
    name: "__extractSlice2d",
    args: [
      baseVar,
      fixedSlot.expr,
      {
        tag: "NumberLiteral",
        value: colonPos,
        jitType: { kind: "number", exact: colonPos },
      },
    ],
    jitType: { kind: "tensor", isComplex: false, shape, ndim: 2 },
  };
}

/**
 * Stage 21: lower `src(a:b)` on a real-tensor base into a
 * `RangeSliceRead` IR node producing a fresh column-vector tensor.
 *
 * Accepts `Range` with no step (default step 1). `start` and `end`
 * must lower to numeric/boolean scalar exprs. The result is a real
 * tensor with shape `[?, 1]` — the exact length is runtime-dependent.
 *
 * Caller responsibility: match the parent expression shape
 * `Index(Ident(src), [Range])` or `FuncCall(src, [Range])` before
 * calling. Returns null if the source isn't a real tensor or the
 * range isn't the expected shape.
 */
export function tryLowerRangeSliceRead(
  ctx: LowerCtx,
  baseName: string,
  rangeExpr: Expr
): JitExpr | null {
  if (rangeExpr.type !== "Range") return null;
  if (rangeExpr.step !== null) return null;

  const srcType = ctx.env.get(baseName);
  if (!srcType || srcType.kind !== "tensor" || srcType.isComplex === true)
    return null;
  // Slice-alias names don't correspond to a real tensor at runtime.
  if (ctx.sliceAliases.has(baseName)) return null;

  const start = lowerExpr(ctx, rangeExpr.start);
  if (!start) return null;
  if (start.jitType.kind !== "number" && start.jitType.kind !== "boolean")
    return null;

  // `end` keyword inside the indexing context refers to the base's
  // linear length. Codegen substitutes the hoisted `.data.length`
  // alias. Any other expression lowers normally.
  let end: JitExpr | null;
  if (rangeExpr.end.type === "EndKeyword") {
    end = null;
  } else {
    end = lowerExpr(ctx, rangeExpr.end);
    if (!end) return null;
    if (end.jitType.kind !== "number" && end.jitType.kind !== "boolean")
      return null;
  }

  // MATLAB 1-D indexing preserves row/column orientation of a vector.
  // For a statically-known row vector (shape [1, n]) the slice is a row;
  // everything else (column vector or matrix under linear indexing) is a
  // column. We can only claim row-ness when we're sure the first dim is
  // exactly 1 (not -1/unknown).
  const isRow =
    !!srcType.shape &&
    srcType.shape.length === 2 &&
    srcType.shape[0] === 1 &&
    srcType.shape[1] !== 1;
  const resultShape: [number, number] = isRow ? [1, -1] : [-1, 1];

  ctx._hasTensorOps = true;
  return {
    tag: "RangeSliceRead",
    baseName,
    start,
    end,
    isRow,
    jitType: { kind: "tensor", isComplex: false, shape: resultShape },
  };
}

// ── User function call resolution ───────────────────────────────────────

/**
 * Try to resolve and compile a user function call.
 * Returns:
 *   JitExpr - successfully compiled
 *   null    - function found but can't compile (bail out of containing function)
 *   undefined - no user function found (fall through to builtins)
 */
function lowerUserFuncCall(
  ctx: LowerCtx,
  expr: Expr & { type: "FuncCall" }
): JitExpr | null | undefined {
  const interp = ctx.interp;
  if (!interp) return undefined;

  // Lower arguments first to determine types
  const args = expr.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // Resolve the function using the same mechanism as the interpreter
  const calleeFn = resolveUserFunction(interp, expr.name, argJitTypes);
  if (!calleeFn) return undefined; // no user function found — fall through to builtins

  // Build identity string for unique naming
  const calleeNargout = 1; // nested calls always expect 1 output
  const typeKey = argJitTypes.map(jitTypeKey).join(":");
  const identity = `${interp.currentFile}:${calleeFn.name}:${calleeNargout}:${typeKey}`;
  const jitName = computeJitFnName(identity, calleeFn.name);

  // Already generated? Reuse.
  if (ctx.generatedFns.has(jitName)) {
    // Need to determine return type - re-lower to get it (or cache it)
    const returnType = getGeneratedFnReturnType(
      calleeFn,
      argJitTypes,
      calleeNargout,
      ctx
    );
    if (!returnType) return null;
    return {
      tag: "UserCall",
      jitName,
      name: calleeFn.name,
      args: loweredArgs,
      jitType: returnType,
    };
  }

  // Recursion guard
  if (ctx.loweringInProgress.has(jitName)) return null;
  ctx.loweringInProgress.add(jitName);

  try {
    // Recursively lower the callee
    const calleeResult = lowerFunction(
      calleeFn,
      argJitTypes,
      calleeNargout,
      interp,
      ctx.generatedFns,
      ctx.loweringInProgress,
      ctx.generatedIRBodies
    );
    if (!calleeResult) {
      // Stage 24 soft-bail: the callee's body has constructs the JIT
      // can't lower (tensor arithmetic, matrix multiply, bsxfun with a
      // function-handle arg, etc.). Rather than bail the enclosing
      // loop, probe the return type by actually invoking the function
      // once with representative args, then emit a UserDispatchCall
      // that goes through `rt.dispatch` at runtime. The outer loop
      // still JITs — only the callee runs interpreted.
      //
      // Guard: skip the probe+dispatch path for callees whose bodies
      // use caller-aware or frame-sensitive builtins (evalin,
      // assignin, inputname, dbstack, nargin/nargout without arg,
      // etc.). These resolve relative to the MATLAB call stack; a
      // probe call at compile time runs with a different stack than
      // the real runtime call, and going through $h.callUserFunc at
      // runtime may not reproduce the semantics the user expects.
      if (callerAwareBuiltinInBody(calleeFn.body)) {
        return null;
      }
      const returnType = probeUserFuncReturnType(
        interp,
        calleeFn.name,
        loweredArgs
      );
      if (!returnType) return null;
      ctx._hasTensorOps = true;
      return {
        tag: "UserDispatchCall",
        name: calleeFn.name,
        args: loweredArgs,
        jitType: returnType,
      };
    }

    const returnType = calleeResult.outputType ?? { kind: "number" as const };

    // Under --opt e1, try the whole-function scalar C kernel first.
    // When the callee is pure-scalar, we emit a JS wrapper that shells
    // out to `$h.compileKernel` with the C source inlined — visible in
    // --dump-js and compiled on first call.
    let wrappedJS: string | null = null;
    if (interp.experimental === "e1") {
      const scalarKernel = tryEmitScalarFnKernel(
        interp,
        calleeFn,
        calleeResult.body,
        calleeResult.outputNames,
        calleeResult.localVars,
        calleeResult.outputType,
        calleeResult.outputTypes,
        argJitTypes,
        calleeNargout,
        ctx.generatedIRBodies
      );
      if (scalarKernel) {
        const paramComments = calleeFn.params
          .map((p, i) => `${p}: ${jitTypeKey(argJitTypes[i])}`)
          .join(", ");
        const outputComments = calleeResult.outputNames
          .map(
            o =>
              `${o}: ${jitTypeKey(calleeResult.outputType ?? { kind: "number" })}`
          )
          .join(", ");
        const comment = [
          `// JIT (e1 scalar kernel): ${calleeFn.name}(${paramComments}) -> (${outputComments})`,
          `// from: ${interp.currentFile}`,
        ].join("\n");
        // scalarKernel.jsSource defines a function named after the
        // user function; we need the JIT's internal `jitName`. Re-emit
        // with the jitName by string-replacing the signature line.
        const renamed = scalarKernel.jsSource.replace(
          `function ${calleeFn.name}(`,
          `function ${jitName}(`
        );
        wrappedJS = `${comment}\n${renamed}`;
      }
    }

    // Generate JS for the callee and wrap in a named function (fallback
    // path when the e1 scalar-kernel attempt didn't fire).
    if (!wrappedJS) {
      const calleeJS = generateJS(
        calleeResult.body,
        calleeFn.params,
        calleeResult.outputNames,
        calleeNargout,
        calleeResult.localVars,
        interp.currentFile,
        interp.fuse,
        interp.experimental,
        interp.par
      );
      const paramComments = calleeFn.params
        .map((p, i) => `${p}: ${jitTypeKey(argJitTypes[i])}`)
        .join(", ");
      const outputComments = calleeResult.outputNames
        .map(
          o =>
            `${o}: ${jitTypeKey(calleeResult.outputType ?? { kind: "number" })}`
        )
        .join(", ");
      const comment = [
        `// JIT: ${calleeFn.name}(${paramComments}) -> (${outputComments})`,
        `// from: ${interp.currentFile}`,
      ].join("\n");
      wrappedJS = `${comment}\nfunction ${jitName}(${calleeFn.params.join(", ")}) {\n${calleeJS}\n}`;
    }
    ctx.generatedFns.set(jitName, wrappedJS);
    // Cache the lowered IR alongside the JS source. The C-JIT reads this
    // in feasibility / generateC; JS-JIT ignores it.
    ctx.generatedIRBodies.set(jitName, {
      fn: calleeFn,
      argTypes: argJitTypes,
      outputNames: calleeResult.outputNames,
      outputTypes: calleeResult.outputTypes,
      body: calleeResult.body,
      localVars: calleeResult.localVars,
      nargout: calleeNargout,
    });

    // Propagate tensor ops flag
    if (calleeResult.hasTensorOps) ctx._hasTensorOps = true;

    return {
      tag: "UserCall",
      jitName,
      name: calleeFn.name,
      args: loweredArgs,
      jitType: returnType,
    };
  } finally {
    ctx.loweringInProgress.delete(jitName);
  }
}

/** Convert JitType to ItemType for function resolution (only ClassInstance matters). */
function jitTypeToItemType(t: JitType): ItemType {
  if (t.kind === "class_instance") {
    return { kind: "ClassInstance", className: t.className };
  }
  return { kind: "Unknown" };
}

/** Resolve a function name to a FunctionDef using the interpreter's resolution. */
function resolveUserFunction(
  interp: Interpreter,
  name: string,
  argJitTypes: JitType[]
): FunctionDef | null {
  // 1. Check nested functions (mirrors interpreter's callFunction priority)
  const nested = interp.env.getNestedFunction(name);
  if (nested) return nested.fn;

  // 2. Check main local functions
  const localFn = interp.mainLocalFunctions.get(name);
  if (localFn) return localFn;

  // 3. Resolve via function index (for workspace functions, class methods, etc.)
  const callSite: CallSite = {
    file: interp.currentFile,
    ...(interp.currentClassName ? { className: interp.currentClassName } : {}),
    ...(interp.currentMethodName
      ? { methodName: interp.currentMethodName }
      : {}),
  };
  const argItemTypes = argJitTypes.map(jitTypeToItemType);
  const target = resolveFunction(
    name,
    argItemTypes,
    callSite,
    interp.functionIndex
  );
  if (!target) return null;

  if (target.kind === "localFunction" && target.source.from === "main") {
    return interp.mainLocalFunctions.get(target.name) ?? null;
  }

  if (target.kind === "classMethod") {
    const definingClass = interp.ctx.findDefiningClass(
      target.className,
      target.methodName
    );
    const classInfo = interp.ctx.getClassInfo(definingClass);
    if (!classInfo) return null;
    return (
      interp.findMethodInClass(classInfo, target.methodName) ??
      interp.findExternalMethod(classInfo, target.methodName)
    );
  }

  if (target.kind === "workspaceFunction") {
    const dotIdx = target.name.lastIndexOf(".");
    const primaryName =
      dotIdx >= 0 ? target.name.slice(dotIdx + 1) : target.name;
    return interp.findFunctionInWorkspaceFile(target.name, primaryName);
  }

  // jsUserFunction targets are handled by lowerIBuiltinCall (they implement
  // the IBuiltin interface), so this resolver returns null and the caller
  // falls through to the IBuiltin path.

  // Other target kinds (privateFunction, workspaceClassConstructor, etc.) not supported yet
  return null;
}

// ── IBuiltin call resolution ────────────────────────────────────────────

export function lowerIBuiltinCall(
  ctx: LowerCtx,
  expr: Expr & { type: "FuncCall" }
): JitExpr | null {
  // Per-context JS user functions (.numbl.js) take priority over native
  // builtins so a .numbl.js file can shadow a builtin of the same name.
  const jsEntry = ctx.interp?.ctx.registry.jsUserFunctionsByName.get(expr.name);
  const ib = jsEntry?.builtin ?? getIBuiltin(expr.name);
  if (!ib) return null;

  const args = expr.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // If any argument is unknown, it could be a class instance at runtime,
  // and class methods take priority over builtins in MATLAB. Allow the
  // call only if `resolve` succeeds with a non-unknown output type —
  // this lets builtins like `ismember` work with cell-read results while
  // still bailing for builtins that can't determine the output type.

  const resolution = ib.resolve(argJitTypes, 1);
  if (!resolution || resolution.outputTypes.length === 0) return null;
  const outputTypes = resolution.outputTypes;

  // If the output type is unknown, bail — the builtin likely depends on
  // runtime state (e.g. evalin, assignin) and must go through dispatch.
  if (outputTypes[0].kind === "unknown") return null;

  // IBuiltin calls always go through $h helpers
  ctx._hasTensorOps = true;

  return {
    tag: "Call",
    name: expr.name,
    args: loweredArgs,
    jitType: outputTypes[0],
  };
}

/** Get the return type of an already-generated function. */
function getGeneratedFnReturnType(
  fn: FunctionDef,
  argTypes: JitType[],
  nargout: number,
  ctx: LowerCtx
): JitType | null {
  // Re-lower to determine the return type (this is cheap since the code is already generated)
  const result = lowerFunction(
    fn,
    argTypes,
    nargout,
    ctx.interp,
    ctx.generatedFns,
    ctx.loweringInProgress,
    ctx.generatedIRBodies
  );
  return result?.outputType ?? null;
}

// ── Return-type probing for function handles / dispatched user calls ────

/**
 * Probe a function handle's return type at JIT compile time.
 *
 * Calls the function handle once with representative argument values and
 * inspects the result type via `inferJitType`. This is safe for pure
 * numerical functions (the vast majority of function handles in numerical
 * MATLAB code). At runtime, every call verifies the actual return type
 * matches — a mismatch triggers a bail to the interpreter.
 *
 * If the probe fails for any reason, returns null and the caller bails
 * (falls back to interpretation for the whole loop).
 */
function probeFuncHandleReturnType(
  interp: Interpreter,
  fnName: string,
  loweredArgs: JitExpr[]
): JitType | null {
  try {
    const fnVal = interp.env.get(fnName);
    if (!fnVal || !isRuntimeFunction(fnVal as RuntimeValue)) return null;
    const fn = fnVal as import("../runtime/types.js").RuntimeFunction;

    // Only probe function handles that have a direct JS closure — these
    // are anonymous functions and named function references. Builtins
    // that require full interpreter dispatch are too expensive to probe.
    if (!fn.jsFn) return null;

    // Collect argument values for the probe. For each lowered arg:
    // - Var: use the actual value from the env, or synthesize a
    //   representative value from its JIT type (handles loop variables
    //   that don't exist in the env yet at JIT compile time)
    // - NumberLiteral: use the literal value
    // - Other: can't cheaply evaluate, bail
    const argVals: unknown[] = [];
    for (const arg of loweredArgs) {
      if (arg.tag === "NumberLiteral") {
        argVals.push(arg.value);
      } else if (arg.tag === "Var") {
        const val = interp.env.get(arg.name);
        if (val !== undefined) {
          argVals.push(val);
        } else {
          // Variable not in env (e.g. loop iterator before loop starts).
          // Synthesize a representative value from its JIT type.
          const rep = representativeValue(arg.jitType);
          if (rep === undefined) return null;
          argVals.push(rep);
        }
      } else {
        return null;
      }
    }

    // Call the function handle once to determine its return type
    const result = fn.jsFnExpectsNargout
      ? fn.jsFn(1, ...argVals)
      : fn.jsFn(...argVals);
    const resultType = inferJitType(result);
    // Don't accept unknown — that would make downstream lowering bail anyway
    if (resultType.kind === "unknown") return null;
    return resultType;
  } catch {
    // Probe failed (function errored) — bail
    return null;
  }
}

/**
 * Stage 24 safety guard: returns true if the function body references
 * any builtin that reads from or writes to the caller's workspace or
 * the MATLAB call stack. Those builtins (evalin, assignin, inputname,
 * dbstack, …) can't be probed safely at JIT compile time because the
 * probe's call stack differs from the real runtime stack, and they
 * may not survive a round-trip through `$h.callUserFunc` depending on
 * how the runtime resolves the caller frame.
 */
const CALLER_AWARE_BUILTINS = new Set<string>([
  "evalin",
  "assignin",
  "inputname",
  "dbstack",
  "dbstop",
  "keyboard",
  "input",
]);

/**
 * Zero-arg builtins whose correct evaluation depends on the interpreter's
 * call frame (e.g. reading $nargin/$nargout from the enclosing function's
 * env). Their IBuiltin `resolve`/`apply` don't see that frame and return
 * a wrong value, so bail out rather than routing a bare-Ident reference
 * through lowerIBuiltinCall.
 */
const FRAME_SENSITIVE_NO_ARG_BUILTINS = new Set<string>(["nargin", "nargout"]);

function callerAwareBuiltinInBody(body: Stmt[]): boolean {
  const visitExpr = (e: Expr): boolean => {
    if (!e) return false;
    if (e.type === "FuncCall" && CALLER_AWARE_BUILTINS.has(e.name)) return true;
    switch (e.type) {
      case "Binary":
        return visitExpr(e.left) || visitExpr(e.right);
      case "Unary":
        return visitExpr(e.operand);
      case "FuncCall":
        return e.args.some(visitExpr);
      case "Index":
      case "IndexCell":
        return visitExpr(e.base) || e.indices.some(visitExpr);
      case "Member":
        return visitExpr(e.base);
      case "MethodCall":
        return visitExpr(e.base) || e.args.some(visitExpr);
      case "Range":
        return (
          visitExpr(e.start) ||
          (e.step ? visitExpr(e.step) : false) ||
          visitExpr(e.end)
        );
      case "Tensor":
      case "Cell":
        return e.rows.some(row => row.some(visitExpr));
      case "AnonFunc":
        return visitExpr(e.body);
      default:
        return false;
    }
  };
  const visitStmts = (stmts: Stmt[]): boolean => {
    for (const s of stmts) {
      switch (s.type) {
        case "Assign":
        case "AssignLValue":
        case "ExprStmt":
        case "MultiAssign":
          if (visitExpr(s.expr)) return true;
          break;
        case "If":
          if (visitExpr(s.cond)) return true;
          if (visitStmts(s.thenBody)) return true;
          for (const eib of s.elseifBlocks) {
            if (visitExpr(eib.cond)) return true;
            if (visitStmts(eib.body)) return true;
          }
          if (s.elseBody && visitStmts(s.elseBody)) return true;
          break;
        case "For":
          if (visitExpr(s.expr)) return true;
          if (visitStmts(s.body)) return true;
          break;
        case "While":
          if (visitExpr(s.cond)) return true;
          if (visitStmts(s.body)) return true;
          break;
        case "TryCatch":
          if (visitStmts(s.tryBody)) return true;
          if (visitStmts(s.catchBody)) return true;
          break;
        case "Switch":
          if (visitExpr(s.expr)) return true;
          for (const c of s.cases) {
            if (visitExpr(c.value)) return true;
            if (visitStmts(c.body)) return true;
          }
          if (s.otherwise && visitStmts(s.otherwise)) return true;
          break;
        default:
          break;
      }
    }
    return false;
  };
  return visitStmts(body);
}

/**
 * Stage 24: probe a user function's return type by invoking it once
 * through `rt.dispatch` with representative argument values. Mirrors
 * `probeFuncHandleReturnType` but for named user functions. Called
 * when `lowerFunction` fails on the callee's body — we still want
 * the outer loop to JIT via a UserDispatchCall.
 *
 * Args are mapped to runtime values:
 *   - Var with a value in the current env: use that value.
 *   - Var with no env value (e.g. loop iterator before loop starts):
 *     synthesize a representative from its JIT type.
 *   - NumberLiteral: use the literal value.
 *   - Other exprs: bail — we can't cheaply evaluate.
 *
 * The probe call may have side effects (persistent-var init, etc.),
 * so it's wrapped in try/catch; any failure bails the probe.
 */
function probeUserFuncReturnType(
  interp: Interpreter,
  fnName: string,
  loweredArgs: JitExpr[]
): JitType | null {
  try {
    const argVals: unknown[] = [];
    for (const arg of loweredArgs) {
      if (arg.tag === "NumberLiteral") {
        argVals.push(arg.value);
      } else if (arg.tag === "Var") {
        const val = interp.env.get(arg.name);
        if (val !== undefined) {
          argVals.push(val);
        } else {
          const rep = representativeValue(arg.jitType);
          if (rep === undefined) return null;
          argVals.push(rep);
        }
      } else {
        return null;
      }
    }
    const result = interp.rt.dispatch(fnName, 1, argVals);
    const resultType = inferJitType(result);
    if (resultType.kind === "unknown") return null;
    return resultType;
  } catch {
    return null;
  }
}

/** Create a representative runtime value for a JIT type, for probing. */
function representativeValue(t: JitType): unknown | undefined {
  switch (t.kind) {
    case "number":
      return t.exact ?? 1;
    case "boolean":
      return true;
    case "complex_or_number":
      return 1;
    default:
      // For tensors, structs, etc. we can't cheaply synthesize a value
      // that would be meaningful to an arbitrary function handle.
      return undefined;
  }
}
