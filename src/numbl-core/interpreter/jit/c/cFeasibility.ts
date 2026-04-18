/**
 * Feasibility prepass for the C-JIT path.
 *
 * Given the lowered JIT IR for a function and the argument types, decide
 * whether the C codegen can handle it. On any construct that isn't in
 * the whitelist, return `{ok: false, reason}` so the caller falls
 * through to the JS-JIT path.
 *
 * The whitelist intentionally mirrors what [jitCodegenC.ts](./jitCodegenC.ts)
 * can emit, which in turn mirrors what JS-JIT does. Widen all three
 * together.
 *
 * Phase 1 (scalar-only): numbers/booleans, scalar math, control flow.
 * Phase 2 (tensor-real, mirrors JS-JIT): real (non-complex) tensor
 *   parameters / locals / sub-expressions, the same tensor binary +
 *   unary + comparison + call ops the JS-JIT helpers cover, and
 *   reductions to scalar. Tensor reads / writes (Index, AssignIndex)
 *   stay out of scope for now.
 */

import { BinaryOperation, UnaryOperation } from "../../../parser/types.js";
import type { JitExpr, JitStmt, JitType } from "../jitTypes.js";

export type FeasibilityResult =
  | { ok: true }
  | { ok: false; reason: string; line?: number };

/** Mutable traversal state so failure reasons can carry the nearest line. */
interface Ctx {
  /** Updated from `SetLoc` markers during statement traversal. */
  line: number;
  /** Tensor names that are both input params AND outputs — the
   *  `function x = foo(x, ...)` pattern. Only these are safe AssignIndex
   *  targets today: the JS wrapper seeds the output buffer with a copy
   *  of the caller's data so writes don't scribble on caller memory. */
  paramOutputTensors: Set<string>;
}

function fail(ctx: Ctx, reason: string): FeasibilityResult {
  return { ok: false, reason, line: ctx.line || undefined };
}

/**
 * Scalar math builtins that map 1:1 to `<math.h>` functions in the C emitter.
 *
 * **Deliberately excluded** (domain-restricted in MATLAB, where out-of-domain
 * inputs promote to complex rather than returning NaN):
 *   asin, acos, sqrt, log, log2, log10, acosh, atanh, log1p
 * The JS-JIT gates these with `requireNonneg`; we don't track the same
 * type refinement at feasibility time, so the conservative choice is to
 * bail for all call sites, letting JS-JIT handle them.
 */
export const C_SCALAR_MATH_BUILTINS = new Set<string>([
  "sin",
  "cos",
  "tan",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "exp",
  "abs",
  "floor",
  "ceil",
  "fix",
  "round",
  "sign",
  "atan2",
  "hypot",
  "mod",
  "rem",
  "expm1",
]);

/**
 * MATLAB unary builtin → libnumbl_ops `numbl_unary_op_t` code, restricted
 * to the ones we route to a tensor unary helper. Mirrors JS-side
 * UNARY_OP_CODE in jitHelpersTensor.ts (minus the domain-restricted ones).
 */
export const C_TENSOR_UNARY_OPS: Record<string, number> = {
  exp: 0,
  // log/log2/log10/sqrt deliberately excluded (domain-restricted)
  abs: 5,
  floor: 6,
  ceil: 7,
  round: 8,
  fix: 9, // libnumbl_ops calls this TRUNC
  sin: 10,
  cos: 11,
  tan: 12,
  // asin/acos deliberately excluded
  atan: 15,
  sinh: 16,
  cosh: 17,
  tanh: 18,
  sign: 19,
};

/**
 * Two-argument element-wise tensor builtins that the C-JIT can handle.
 * These map to <math.h> functions in both per-op and fused paths.
 */
export const C_TENSOR_BINARY_BUILTINS = new Set<string>([
  "max",
  "min",
  "atan2",
  "hypot",
  "mod",
  "rem",
]);

/**
 * MATLAB reduction builtin → libnumbl_ops `numbl_reduce_op_t` code.
 * Mirrors what JS-JIT's tSum / tMax / tMin / tMean / tProd / tAny / tAll
 * helpers route to.
 */
export const C_TENSOR_REDUCTION_OPS: Record<string, number> = {
  sum: 0,
  prod: 1,
  max: 2,
  min: 3,
  any: 4,
  all: 5,
  mean: 6,
};

function isScalarKind(k: JitType["kind"]): boolean {
  return k === "number" || k === "boolean";
}

/** Real tensor (1-3 D) the C-JIT codegen can handle as a value. */
function isAcceptableTensor(t: JitType): boolean {
  if (t.kind !== "tensor") return false;
  if (t.isComplex !== false) return false;
  if (!t.shape) return true; // shape unknown — runtime helpers will check
  const ndim = t.shape.length;
  return ndim >= 1 && ndim <= 3;
}

/** Type allowed for a value (RHS expression, local, return). */
function checkValueType(t: JitType): FeasibilityResult {
  if (isScalarKind(t.kind)) return { ok: true };
  if (isAcceptableTensor(t)) return { ok: true };
  if (t.kind === "tensor") {
    return { ok: false, reason: "complex tensor / unsupported ndim" };
  }
  return { ok: false, reason: `unsupported type: ${t.kind}` };
}

function checkExpr(expr: JitExpr, ctx: Ctx): FeasibilityResult {
  const typeCheck = checkValueType(expr.jitType);
  if (!typeCheck.ok) return fail(ctx, typeCheck.reason);

  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
      return { ok: true };

    case "Index": {
      // Phase 1 of Index support: single-index (linear) reads into a
      // real-tensor Var. Multi-index reads need shape plumbed through
      // the C ABI; non-Var bases need scratch-buffer evaluation first.
      // Both deferred.
      if (expr.indices.length !== 1) {
        return fail(ctx, "multi-index Index read not supported");
      }
      if (expr.base.tag !== "Var") {
        return fail(ctx, "Index read requires a Var base");
      }
      if (
        expr.base.jitType.kind !== "tensor" ||
        expr.base.jitType.isComplex !== false
      ) {
        return fail(ctx, "Index read base must be a real tensor");
      }
      return checkExpr(expr.indices[0], ctx);
    }

    case "Binary": {
      switch (expr.op) {
        case BinaryOperation.Add:
        case BinaryOperation.Sub:
        case BinaryOperation.Mul:
        case BinaryOperation.ElemMul:
        case BinaryOperation.Div:
        case BinaryOperation.ElemDiv:
        case BinaryOperation.Pow:
        case BinaryOperation.ElemPow:
        case BinaryOperation.Equal:
        case BinaryOperation.NotEqual:
        case BinaryOperation.Less:
        case BinaryOperation.LessEqual:
        case BinaryOperation.Greater:
        case BinaryOperation.GreaterEqual:
        case BinaryOperation.AndAnd:
        case BinaryOperation.OrOr:
          break;
        default:
          return fail(ctx, `unsupported binary op ${expr.op}`);
      }
      // Tensor-result Pow: JS-JIT routes through tensorBinaryOp slow path
      // (tPow), no fast helper. Keep parity by bailing.
      if (
        expr.jitType.kind === "tensor" &&
        (expr.op === BinaryOperation.Pow ||
          expr.op === BinaryOperation.ElemPow ||
          expr.op === BinaryOperation.AndAnd ||
          expr.op === BinaryOperation.OrOr)
      ) {
        return fail(
          ctx,
          `tensor-result binary op ${expr.op} has no C-JIT fast path`
        );
      }
      const l = checkExpr(expr.left, ctx);
      if (!l.ok) return l;
      const r = checkExpr(expr.right, ctx);
      if (!r.ok) return r;
      return { ok: true };
    }

    case "Unary": {
      switch (expr.op) {
        case UnaryOperation.Plus:
        case UnaryOperation.Minus:
        case UnaryOperation.Not:
          break;
        default:
          return fail(ctx, `unsupported unary op ${expr.op}`);
      }
      // Tensor-result Not is unsupported (no `tNot` helper in JS-JIT either).
      if (expr.jitType.kind === "tensor" && expr.op === UnaryOperation.Not) {
        return fail(ctx, "tensor-result Not not supported");
      }
      return checkExpr(expr.operand, ctx);
    }

    case "Call":
      return checkCall(expr, ctx);

    // Everything else is out of scope — bail to JS-JIT.
    case "ImagLiteral":
    case "StringLiteral":
    case "TensorLiteral":
    case "VConcatGrow":
    case "RangeSliceRead":
    case "MemberRead":
    case "StructArrayMemberRead":
    case "UserCall":
    case "FuncHandleCall":
    case "UserDispatchCall":
      return fail(ctx, `unsupported expr: ${expr.tag}`);
  }
}

function checkCall(
  expr: JitExpr & { tag: "Call" },
  ctx: Ctx
): FeasibilityResult {
  // Scalar math builtin (Phase 1 path) — must produce a scalar result.
  if (expr.jitType.kind !== "tensor" && C_SCALAR_MATH_BUILTINS.has(expr.name)) {
    for (const a of expr.args) {
      const r = checkExpr(a, ctx);
      if (!r.ok) return r;
    }
    return { ok: true };
  }
  // Tensor unary builtin: result is tensor, single tensor arg, name in
  // the libnumbl_ops mapping.
  if (expr.jitType.kind === "tensor" && expr.name in C_TENSOR_UNARY_OPS) {
    if (expr.args.length !== 1) {
      return fail(ctx, `${expr.name}: expected 1 tensor arg`);
    }
    return checkExpr(expr.args[0], ctx);
  }
  // Tensor binary builtin (max, min, atan2, hypot, mod, rem):
  // result is tensor, two args (tensor/scalar).
  if (
    expr.jitType.kind === "tensor" &&
    C_TENSOR_BINARY_BUILTINS.has(expr.name)
  ) {
    if (expr.args.length !== 2) {
      return fail(ctx, `${expr.name}: expected 2 args`);
    }
    for (const a of expr.args) {
      const r = checkExpr(a, ctx);
      if (!r.ok) return r;
    }
    return { ok: true };
  }
  // Reduction (tensor → scalar): name in mapping, single tensor arg.
  if (expr.jitType.kind !== "tensor" && expr.name in C_TENSOR_REDUCTION_OPS) {
    if (expr.args.length !== 1) {
      return fail(ctx, `${expr.name}: only single-arg reduction supported`);
    }
    const a = expr.args[0];
    if (a.jitType.kind !== "tensor") {
      return fail(ctx, `${expr.name}: only tensor-arg reduction supported`);
    }
    return checkExpr(a, ctx);
  }
  // tic/toc: scalar timer builtins, no args.
  if (
    (expr.name === "tic" || expr.name === "toc") &&
    expr.args.length === 0 &&
    expr.jitType.kind === "number"
  ) {
    return { ok: true };
  }
  return fail(ctx, `non-C-mappable builtin: ${expr.name}`);
}

function checkStmts(stmts: JitStmt[], ctx: Ctx): FeasibilityResult {
  for (const s of stmts) {
    const r = checkStmt(s, ctx);
    if (!r.ok) return r;
  }
  return { ok: true };
}

function checkStmt(stmt: JitStmt, ctx: Ctx): FeasibilityResult {
  switch (stmt.tag) {
    case "Assign":
      return checkExpr(stmt.expr, ctx);

    case "ExprStmt":
      return checkExpr(stmt.expr, ctx);

    case "If": {
      const c = checkExpr(stmt.cond, ctx);
      if (!c.ok) return c;
      const t = checkStmts(stmt.thenBody, ctx);
      if (!t.ok) return t;
      for (const eib of stmt.elseifBlocks) {
        const ec = checkExpr(eib.cond, ctx);
        if (!ec.ok) return ec;
        const eb = checkStmts(eib.body, ctx);
        if (!eb.ok) return eb;
      }
      if (stmt.elseBody) return checkStmts(stmt.elseBody, ctx);
      return { ok: true };
    }

    case "For": {
      const s = checkExpr(stmt.start, ctx);
      if (!s.ok) return s;
      const e = checkExpr(stmt.end, ctx);
      if (!e.ok) return e;
      if (stmt.step) {
        const stepR = checkExpr(stmt.step, ctx);
        if (!stepR.ok) return stepR;
      }
      return checkStmts(stmt.body, ctx);
    }

    case "While": {
      const c = checkExpr(stmt.cond, ctx);
      if (!c.ok) return c;
      return checkStmts(stmt.body, ctx);
    }

    case "Break":
    case "Continue":
    case "Return":
      return { ok: true };

    case "SetLoc":
      ctx.line = stmt.line;
      return { ok: true };

    case "AssignIndex": {
      // Phase 1 of AssignIndex support: single-index (linear) writes on a
      // real-tensor Var with a scalar RHS, where the target is both an
      // input param AND an output (so the JS wrapper has seeded the
      // output buffer with a copy of the caller's data). Other targets
      // — pure input params, fresh locals, tensors aliased to an input
      // via `y = x` — defer, because correct MATLAB call-by-value
      // semantics on those would require unshare/copy machinery the
      // C-JIT doesn't have yet.
      if (stmt.indices.length !== 1) {
        return fail(ctx, "multi-index AssignIndex not supported");
      }
      if (
        stmt.baseType.kind !== "tensor" ||
        stmt.baseType.isComplex !== false
      ) {
        return fail(ctx, "AssignIndex base must be a real tensor");
      }
      if (!isScalarKind(stmt.value.jitType.kind)) {
        return fail(ctx, "AssignIndex value must be scalar");
      }
      if (!ctx.paramOutputTensors.has(stmt.baseName)) {
        return fail(
          ctx,
          `AssignIndex base '${stmt.baseName}' must be both param and output`
        );
      }
      const vr = checkExpr(stmt.value, ctx);
      if (!vr.ok) return vr;
      return checkExpr(stmt.indices[0], ctx);
    }

    // Out of scope: range/col writes, struct writes, multi-assign.
    case "AssignIndexRange":
    case "AssignIndexCol":
    case "AssignMember":
    case "MultiAssign":
      return fail(ctx, `unsupported stmt: ${stmt.tag}`);
    default:
      return fail(ctx, `unknown stmt: ${(stmt as { tag: string }).tag}`);
  }
}

/**
 * Check if the lowered function can be handled by the C-JIT.
 *
 * `outputTypes` holds the types of every output variable in order;
 * `outputType` is kept for backwards-compatibility and equals
 * `outputTypes[0]` when present.
 *
 * Multi-output mirrors the JS-JIT's `return [out0, out1, ...]` shape:
 * the generated C builds a `napi_value` array of length `nargout`,
 * each entry boxed according to its type (scalar doubles, booleans, or
 * tensors).
 */
export function checkCFeasibility(
  body: JitStmt[],
  paramNames: string[],
  argTypes: JitType[],
  outputNames: string[],
  outputType: JitType | null,
  outputTypes: JitType[],
  nargout: number
): FeasibilityResult {
  for (const t of argTypes) {
    const r = checkValueType(t);
    if (!r.ok) return { ok: false, reason: `arg: ${r.reason}` };
  }
  if (outputType) {
    const r = checkValueType(outputType);
    if (!r.ok) return { ok: false, reason: `return: ${r.reason}` };
  }
  for (let i = 0; i < outputTypes.length; i++) {
    const r = checkValueType(outputTypes[i]);
    if (!r.ok) return { ok: false, reason: `return[${i}]: ${r.reason}` };
  }
  // `nargout` > outputTypes.length means the caller asked for more
  // outputs than the function produces; let the interpreter raise the
  // usual MATLAB "too many output arguments" error.
  if (nargout > outputTypes.length) {
    return {
      ok: false,
      reason: `nargout ${nargout} exceeds available outputs (${outputTypes.length})`,
    };
  }
  // Tensor names that are simultaneously input params AND outputs. The
  // JS wrapper seeds these output buffers with a copy of the caller's
  // data, so writes via AssignIndex land on a buffer independent of the
  // caller's tensor — safe MATLAB call-by-value + local-mutation.
  const outputNameSet = new Set(outputNames);
  const paramOutputTensors = new Set<string>();
  for (let i = 0; i < paramNames.length; i++) {
    if (argTypes[i].kind === "tensor" && outputNameSet.has(paramNames[i])) {
      paramOutputTensors.add(paramNames[i]);
    }
  }
  const ctx: Ctx = { line: 0, paramOutputTensors };
  return checkStmts(body, ctx);
}
