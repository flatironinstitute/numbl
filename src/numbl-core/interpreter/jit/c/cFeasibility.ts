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

export type FeasibilityResult = { ok: true } | { ok: false; reason: string };

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

function checkExpr(expr: JitExpr): FeasibilityResult {
  // Index reads still bail (Phase 2 scope intentionally excludes them).
  if (expr.tag === "Index") {
    return { ok: false, reason: "Index reads not supported (defer to JS-JIT)" };
  }

  const typeCheck = checkValueType(expr.jitType);
  if (!typeCheck.ok) return typeCheck;

  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
      return { ok: true };

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
          return { ok: false, reason: `unsupported binary op ${expr.op}` };
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
        return {
          ok: false,
          reason: `tensor-result binary op ${expr.op} has no C-JIT fast path`,
        };
      }
      const l = checkExpr(expr.left);
      if (!l.ok) return l;
      const r = checkExpr(expr.right);
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
          return { ok: false, reason: `unsupported unary op ${expr.op}` };
      }
      // Tensor-result Not is unsupported (no `tNot` helper in JS-JIT either).
      if (expr.jitType.kind === "tensor" && expr.op === UnaryOperation.Not) {
        return { ok: false, reason: "tensor-result Not not supported" };
      }
      return checkExpr(expr.operand);
    }

    case "Call":
      return checkCall(expr);

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
      return { ok: false, reason: `unsupported expr: ${expr.tag}` };
  }
}

function checkCall(expr: JitExpr & { tag: "Call" }): FeasibilityResult {
  // Scalar math builtin (Phase 1 path) — must produce a scalar result.
  if (expr.jitType.kind !== "tensor" && C_SCALAR_MATH_BUILTINS.has(expr.name)) {
    for (const a of expr.args) {
      const r = checkExpr(a);
      if (!r.ok) return r;
    }
    return { ok: true };
  }
  // Tensor unary builtin: result is tensor, single tensor arg, name in
  // the libnumbl_ops mapping.
  if (expr.jitType.kind === "tensor" && expr.name in C_TENSOR_UNARY_OPS) {
    if (expr.args.length !== 1) {
      return { ok: false, reason: `${expr.name}: expected 1 tensor arg` };
    }
    return checkExpr(expr.args[0]);
  }
  // Reduction (tensor → scalar): name in mapping, single tensor arg.
  if (expr.jitType.kind !== "tensor" && expr.name in C_TENSOR_REDUCTION_OPS) {
    if (expr.args.length !== 1) {
      return {
        ok: false,
        reason: `${expr.name}: only single-arg reduction supported`,
      };
    }
    const a = expr.args[0];
    if (a.jitType.kind !== "tensor") {
      return {
        ok: false,
        reason: `${expr.name}: only tensor-arg reduction supported`,
      };
    }
    return checkExpr(a);
  }
  return { ok: false, reason: `non-C-mappable builtin: ${expr.name}` };
}

function checkStmts(stmts: JitStmt[]): FeasibilityResult {
  for (const s of stmts) {
    const r = checkStmt(s);
    if (!r.ok) return r;
  }
  return { ok: true };
}

function checkStmt(stmt: JitStmt): FeasibilityResult {
  switch (stmt.tag) {
    case "Assign":
      return checkExpr(stmt.expr);

    case "ExprStmt":
      return checkExpr(stmt.expr);

    case "If": {
      const c = checkExpr(stmt.cond);
      if (!c.ok) return c;
      const t = checkStmts(stmt.thenBody);
      if (!t.ok) return t;
      for (const eib of stmt.elseifBlocks) {
        const ec = checkExpr(eib.cond);
        if (!ec.ok) return ec;
        const eb = checkStmts(eib.body);
        if (!eb.ok) return eb;
      }
      if (stmt.elseBody) return checkStmts(stmt.elseBody);
      return { ok: true };
    }

    case "For": {
      const s = checkExpr(stmt.start);
      if (!s.ok) return s;
      const e = checkExpr(stmt.end);
      if (!e.ok) return e;
      if (stmt.step) {
        const stepR = checkExpr(stmt.step);
        if (!stepR.ok) return stepR;
      }
      return checkStmts(stmt.body);
    }

    case "While": {
      const c = checkExpr(stmt.cond);
      if (!c.ok) return c;
      return checkStmts(stmt.body);
    }

    case "Break":
    case "Continue":
    case "Return":
    case "SetLoc":
      return { ok: true };

    // Out of scope: tensor/struct writes, multi-assign, member writes.
    case "AssignIndex":
    case "AssignIndexRange":
    case "AssignIndexCol":
    case "AssignMember":
    case "MultiAssign":
      return { ok: false, reason: `unsupported stmt: ${stmt.tag}` };
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
  argTypes: JitType[],
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
  return checkStmts(body);
}
