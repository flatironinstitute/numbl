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

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { JitExpr, JitStmt, JitType } from "../jitTypes.js";
import { getIBuiltin } from "../../interpreter/builtins/types.js";
import type { GeneratedFn } from "../jitLower.js";
import { analyzeTensorUsage, type ClassificationResult } from "./classify.js";
import {
  getTensorUnaryOp,
  getTensorBinaryFn,
  getTensorReductionOp,
} from "./codegenCtx.js";

export type FeasibilityResult =
  | { ok: true }
  | { ok: false; reason: string; line?: number };

/** Shared across the feasibility check and any recursive UserCall descents.
 *  Caches per-jitName results (ok + reason) so a callee reached from
 *  multiple sites is only analyzed once. `inProgress` blocks reentry for
 *  direct or mutual recursion — we bail on recursive UserCall for now.
 *
 *  `classifications` memoizes analyzeTensorUsage per jitName so the
 *  tensor-return feasibility check doesn't reanalyze the callee body
 *  every time it's called from a different site. */
interface SharedFeasCtx {
  generatedIRBodies: Map<string, GeneratedFn>;
  results: Map<string, FeasibilityResult>;
  inProgress: Set<string>;
  classifications: Map<string, ClassificationResult>;
}

/** Mutable traversal state so failure reasons can carry the nearest line. */
interface Ctx {
  /** Updated from `SetLoc` markers during statement traversal. */
  line: number;
  /** Every tensor-typed input param. AssignIndex is allowed on any of
   *  these: param-outputs land on the seeded output buffer; pure-input
   *  params land on a C-side unshare buffer (malloc'd in the prelude,
   *  memcpy'd from the caller's data). Tensor locals and names aliased
   *  to a param via `y = x` still defer — supporting those requires
   *  flow analysis the emitter doesn't do yet. */
  tensorParams: Set<string>;
  /** Tensor-typed names whose data/len plumbing is reachable in the
   *  emitter — union of `tensorParams`, tensor locals, and tensor
   *  outputs. Used for read-side checks (Index base / RangeSliceRead
   *  base / AssignIndexRange src / AssignIndexCol src). Writes still
   *  require `tensorParams` OR a locally-allocated tensor (covered by
   *  the dynamic-output path in the emitter). */
  tensorVars: Set<string>;
  /** Shared across the whole descent (top-level + every recursive
   *  UserCall). null at the top when no callee map is available (tests). */
  shared: SharedFeasCtx | null;
}

function fail(ctx: Ctx, reason: string): FeasibilityResult {
  return { ok: false, reason, line: ctx.line || undefined };
}

// C-JIT tensor-op membership comes from `IBuiltin.jitCapabilities`; see
// `getTensorUnaryOp` / `getTensorBinaryFn` / `getTensorReductionOp` in
// codegenCtx.ts. Domain-restricted ops (log / sqrt / asin / acos) are
// excluded by simply not setting `tensorUnaryOp` on their IBuiltin.

function isScalarKind(k: JitType["kind"]): boolean {
  return k === "number" || k === "boolean" || k === "complex_or_number";
}

function isRealScalarKind(k: JitType["kind"]): boolean {
  return k === "number" || k === "boolean";
}

function isComplexScalarKind(k: JitType["kind"]): boolean {
  return k === "complex_or_number";
}

/** Real tensor (1-3 D) the C-JIT codegen can handle as a value. */
function isAcceptableRealTensor(t: JitType): boolean {
  if (t.kind !== "tensor") return false;
  if (t.isComplex !== false) return false;
  if (!t.shape) return true; // shape unknown — runtime helpers will check
  const ndim = t.shape.length;
  return ndim >= 1 && ndim <= 3;
}

/** Complex tensor (1-3 D) accepted for the Phase 2 binary/unary/reduce
 *  ops. Reads (Index / RangeSliceRead / AssignIndex / AssignIndexRange /
 *  AssignIndexCol) stay real-only — those require per-site checks below. */
function isAcceptableComplexTensor(t: JitType): boolean {
  if (t.kind !== "tensor") return false;
  if (t.isComplex !== true) return false;
  if (!t.shape) return true;
  const ndim = t.shape.length;
  return ndim >= 1 && ndim <= 3;
}

function isAcceptableTensor(t: JitType): boolean {
  return isAcceptableRealTensor(t) || isAcceptableComplexTensor(t);
}

/** Type allowed for a value (RHS expression, local, return). */
function checkValueType(t: JitType): FeasibilityResult {
  if (isScalarKind(t.kind)) return { ok: true };
  if (isAcceptableTensor(t)) return { ok: true };
  if (t.kind === "tensor") {
    return { ok: false, reason: "unsupported tensor ndim" };
  }
  return { ok: false, reason: `unsupported type: ${t.kind}` };
}

function checkExpr(
  expr: JitExpr,
  ctx: Ctx,
  allowTensorUserCall: boolean = false
): FeasibilityResult {
  // Void-returning I/O calls (disp) carry jitType:unknown by design —
  // they only appear in ExprStmt position where the result is
  // discarded. Skip the type check and let `checkCall` decide.
  const isVoidIoCall = expr.tag === "Call" && expr.name === "disp";
  if (!isVoidIoCall) {
    const typeCheck = checkValueType(expr.jitType);
    if (!typeCheck.ok) return fail(ctx, typeCheck.reason);
  }

  switch (expr.tag) {
    case "NumberLiteral":
    case "Var":
    case "ImagLiteral":
      return { ok: true };

    case "Index": {
      // 1-3 D scalar Index reads into a real-tensor Var. 1D goes through
      // numbl_idx1r (len-only); 2D/3D go through numbl_idx2r / numbl_idx3r
      // with shape dims (`_d0` for 2D, `_d0`/`_d1` for 3D) threaded in
      // via the ABI. Non-Var bases would need scratch-buffer evaluation
      // first — deferred. 4D+ has no helper.
      const n = expr.indices.length;
      if (n < 1 || n > 3) {
        return fail(ctx, `Index arity ${n} unsupported (only 1D/2D/3D)`);
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
      // Multi-index reads require the base's shape dims — tensor
      // params and dynamic tensor locals both carry `_d0`/`_d1`. The
      // tensorVars check covers both.
      if (n >= 2 && !ctx.tensorVars.has(expr.base.name)) {
        return fail(
          ctx,
          `multi-index Index read base '${expr.base.name}' must be a tensor var`
        );
      }
      for (const idx of expr.indices) {
        if (!isRealScalarKind(idx.jitType.kind)) {
          return fail(ctx, "Index index must be a real scalar");
        }
        const r = checkExpr(idx, ctx);
        if (!r.ok) return r;
      }
      return { ok: true };
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
      // Complex tensor binary ops: only Add/Sub/Mul/ElemMul/Div/ElemDiv
      // have numbl_complex_binary_elemwise opcodes. Comparisons on
      // complex are defined by the kernels but not wired through the
      // emitter yet (would need a real-output tensor target); logical
      // ops (AndAnd/OrOr) on complex tensors aren't defined.
      if (expr.jitType.kind === "tensor" && expr.jitType.isComplex === true) {
        switch (expr.op) {
          case BinaryOperation.Add:
          case BinaryOperation.Sub:
          case BinaryOperation.Mul:
          case BinaryOperation.ElemMul:
          case BinaryOperation.Div:
          case BinaryOperation.ElemDiv:
            break;
          default:
            return fail(
              ctx,
              `complex tensor binary op ${expr.op} not supported in C-JIT`
            );
        }
      }
      // Complex scalar binary ops: only Add/Sub/Mul/ElemMul/Div/ElemDiv
      // are supported in Phase 1. Comparisons, logicals, and pow all bail.
      const anyComplex =
        isComplexScalarKind(expr.left.jitType.kind) ||
        isComplexScalarKind(expr.right.jitType.kind) ||
        isComplexScalarKind(expr.jitType.kind);
      if (anyComplex) {
        switch (expr.op) {
          case BinaryOperation.Add:
          case BinaryOperation.Sub:
          case BinaryOperation.Mul:
          case BinaryOperation.ElemMul:
          case BinaryOperation.Div:
          case BinaryOperation.ElemDiv:
            break;
          default:
            return fail(
              ctx,
              `complex scalar binary op ${expr.op} not supported in C-JIT`
            );
        }
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
      // Complex scalar Not is not supported — requires cTruthy semantics.
      if (
        isComplexScalarKind(expr.operand.jitType.kind) &&
        expr.op === UnaryOperation.Not
      ) {
        return fail(ctx, "complex scalar Not not supported in C-JIT");
      }
      return checkExpr(expr.operand, ctx);
    }

    case "Call":
      return checkCall(expr, ctx);

    case "RangeSliceRead": {
      // `src(a:b)` producing a fresh column-vector tensor. Requires the
      // src to have data/len plumbed — tensor params qualify directly,
      // and tensor locals/outputs plumb through the same `v_<n>_data` /
      // `_len` locals. Phase 2 keeps this real-only; complex range
      // slices need paired-imag plumbing that's deferred to Phase 3.
      if (!ctx.tensorVars.has(expr.baseName)) {
        return fail(
          ctx,
          `RangeSliceRead base '${expr.baseName}' must be a tensor var`
        );
      }
      if (expr.jitType.kind === "tensor" && expr.jitType.isComplex === true) {
        return fail(ctx, "RangeSliceRead on complex tensor not yet supported");
      }
      const sr = checkExpr(expr.start, ctx);
      if (!sr.ok) return sr;
      if (expr.end) {
        const er = checkExpr(expr.end, ctx);
        if (!er.ok) return er;
      }
      return { ok: true };
    }

    case "TensorLiteral": {
      // Real-tensor literal: `[1 2 3]`, `[a b; c d]`, `[]`. Empty matrix
      // (nRows == 0 && nCols == 0) is allowed — emits NULL / len 0.
      // Non-empty cells must all be real scalar-typed (the dest is a
      // real tensor — complex cells would imply a complex tensor,
      // unsupported in Phase 1).
      if (expr.jitType.kind !== "tensor" || expr.jitType.isComplex !== false) {
        return fail(ctx, "TensorLiteral must be a real tensor");
      }
      if (expr.nRows === 0 && expr.nCols === 0) return { ok: true };
      for (let r = 0; r < expr.nRows; r++) {
        for (let c = 0; c < expr.nCols; c++) {
          const cell = expr.rows[r][c];
          if (!isRealScalarKind(cell.jitType.kind)) {
            return fail(ctx, "TensorLiteral cell must be a real scalar");
          }
          const cr = checkExpr(cell, ctx);
          if (!cr.ok) return cr;
        }
      }
      return { ok: true };
    }

    case "VConcatGrow": {
      // `[base; value]` grow-by-one: base is a real column-vector (or
      // empty) tensor Var, value is a scalar. Codegen supports any
      // tensor Var as base (common case: self-grow, `it = [it; i]`).
      if (expr.jitType.kind !== "tensor" || expr.jitType.isComplex !== false) {
        return fail(ctx, "VConcatGrow must be a real tensor");
      }
      if (expr.base.tag !== "Var") {
        return fail(ctx, "VConcatGrow base must be a Var");
      }
      if (
        expr.base.jitType.kind !== "tensor" ||
        expr.base.jitType.isComplex !== false
      ) {
        return fail(ctx, "VConcatGrow base must be a real tensor");
      }
      if (!isScalarKind(expr.value.jitType.kind)) {
        return fail(ctx, "VConcatGrow value must be scalar");
      }
      return checkExpr(expr.value, ctx);
    }

    case "UserCall":
      return checkUserCall(expr, ctx, allowTensorUserCall);

    // Everything else is out of scope — bail to JS-JIT.
    case "StringLiteral":
    case "MemberRead":
    case "StructArrayMemberRead":
    case "FuncHandleCall":
    case "UserDispatchCall":
      return fail(ctx, `unsupported expr: ${expr.tag}`);
  }
}

/** UserCall: the callee must itself be C-feasible.
 *  - Scalar args: always allowed.
 *  - Tensor args: real (1-3D), must appear as a Var at the caller (so
 *    the existing tensor locals plumb straight through to the callee's
 *    signature). Inline tensor literals / expressions would need scratch
 *    materialization — deferred.
 *  - Scalar return: always allowed.
 *  - Tensor return: accepted only when (a) the callee's output[0] is
 *    classified `isDynamicOutput` (fresh-alloc + output kind — the
 *    callee mallocs and transfers ownership via the `double **` ABI)
 *    AND (b) the caller is in Assign-RHS position (`allowTensorUserCall`).
 *    Aliased param-output tensor returns violate COW and are rejected.
 *
 *  Args are checked in the caller's ctx; the callee body is checked in
 *  a fresh ctx built from the cached `GeneratedFn` meta. Results are
 *  memoized across the whole feasibility descent. */
function checkUserCall(
  expr: JitExpr & { tag: "UserCall" },
  ctx: Ctx,
  allowTensorUserCall: boolean
): FeasibilityResult {
  if (!ctx.shared) {
    return fail(ctx, "UserCall: no generatedIRBodies in feasibility ctx");
  }
  for (let i = 0; i < expr.args.length; i++) {
    const a = expr.args[i];
    if (isScalarKind(a.jitType.kind)) {
      const r = checkExpr(a, ctx);
      if (!r.ok) return r;
      continue;
    }
    if (isAcceptableTensor(a.jitType)) {
      if (a.tag !== "Var") {
        return fail(
          ctx,
          `UserCall '${expr.name}': tensor arg must be a Var (inline tensor expressions not yet supported)`
        );
      }
      if (!ctx.tensorVars.has(a.name)) {
        return fail(
          ctx,
          `UserCall '${expr.name}': tensor arg '${a.name}' is not a tensor var`
        );
      }
      continue;
    }
    return fail(ctx, `UserCall '${expr.name}': unsupported arg type`);
  }

  if (!isScalarKind(expr.jitType.kind)) {
    if (!isAcceptableTensor(expr.jitType)) {
      return fail(ctx, `UserCall '${expr.name}': unsupported return type`);
    }
    if (!allowTensorUserCall) {
      return fail(
        ctx,
        `UserCall '${expr.name}': tensor-return UserCall only allowed as Assign RHS`
      );
    }
  }

  // Descend into the callee body. Memoize + guard against recursion.
  const { shared } = ctx;
  const cached = shared.results.get(expr.jitName);
  if (cached) {
    if (!cached.ok) {
      // Re-wrap the cached failure with the caller's line so the parity
      // reporter blames the call site, not the callee body.
      return fail(ctx, `UserCall '${expr.name}': ${cached.reason}`);
    }
    // Callee body is feasible — still need to verify the tensor-return
    // invariant using the (cached) classification below.
  } else if (shared.inProgress.has(expr.jitName)) {
    return fail(ctx, `UserCall '${expr.name}': recursion not supported`);
  }

  const callee = shared.generatedIRBodies.get(expr.jitName);
  if (!callee) {
    return fail(
      ctx,
      `UserCall '${expr.name}': missing IR body for ${expr.jitName}`
    );
  }

  if (!cached) {
    shared.inProgress.add(expr.jitName);
    let calleeRes: FeasibilityResult;
    try {
      calleeRes = checkCFeasibilityInternal(
        callee.body,
        callee.fn.params,
        callee.argTypes,
        callee.outputTypes[0] ?? null,
        callee.outputTypes,
        callee.nargout,
        shared
      );
    } finally {
      shared.inProgress.delete(expr.jitName);
    }
    shared.results.set(expr.jitName, calleeRes);
    if (!calleeRes.ok) {
      return fail(ctx, `UserCall '${expr.name}': ${calleeRes.reason}`);
    }
  }

  // Tensor-return case: require the callee's output[0] to be a
  // fresh-alloc dynamic output. Classification is memoized per jitName.
  if (expr.jitType.kind === "tensor") {
    let cls = shared.classifications.get(expr.jitName);
    if (!cls) {
      const effectiveOutputs = callee.outputNames.slice(0, callee.nargout || 1);
      const effectiveOutputTypes = callee.outputTypes.slice(
        0,
        effectiveOutputs.length
      );
      cls = analyzeTensorUsage(
        callee.body,
        callee.fn.params,
        callee.argTypes,
        effectiveOutputs,
        effectiveOutputTypes
      );
      shared.classifications.set(expr.jitName, cls);
    }
    const outName = callee.outputNames[0];
    const outMeta = outName ? cls.meta.get(outName) : undefined;
    if (!outMeta || !outMeta.isDynamicOutput) {
      return fail(
        ctx,
        `UserCall '${expr.name}': tensor return requires fresh-alloc output (aliased param-output not supported)`
      );
    }
  }
  return { ok: true };
}

function checkCall(
  expr: JitExpr & { tag: "Call" },
  ctx: Ctx
): FeasibilityResult {
  // Scalar math builtin — accept any builtin that provides a C
  // scalar-emission hook and returns a non-null emission for the
  // arg types at hand. Dummy argCode is passed since jitEmitC should
  // only inspect argTypes to decide feasibility.
  if (expr.jitType.kind !== "tensor") {
    const ib = getIBuiltin(expr.name);
    if (ib?.jitEmitC) {
      const argTypes = expr.args.map(a => a.jitType);
      const dummyArgs = expr.args.map(() => "_");
      if (ib.jitEmitC(dummyArgs, argTypes) !== null) {
        for (const a of expr.args) {
          const r = checkExpr(a, ctx);
          if (!r.ok) return r;
        }
        return { ok: true };
      }
    }
    // Complex-scalar special cases: real / imag / conj on a
    // complex_or_number arg. The emitter decomposes the arg via
    // emitComplex() and returns the appropriate component — no jitEmitC
    // hook is needed since jitEmitC returns a single string and we need
    // different per-component handling.
    if (
      (expr.name === "real" || expr.name === "imag" || expr.name === "conj") &&
      expr.args.length === 1 &&
      isComplexScalarKind(expr.args[0].jitType.kind)
    ) {
      return checkExpr(expr.args[0], ctx);
    }
  }
  // abs(complex_tensor): complex → real tensor via numbl_complex_abs.
  // Checked BEFORE the generic tensor-unary kernel dispatch below so
  // the real-kernel guard doesn't reject a complex arg whose result
  // lives on the complex-abs path. Emit handles it at the real-tensor
  // Assign site / scratch path.
  if (
    expr.name === "abs" &&
    expr.args.length === 1 &&
    expr.jitType.kind === "tensor" &&
    expr.jitType.isComplex === false &&
    expr.args[0].jitType.kind === "tensor" &&
    expr.args[0].jitType.isComplex === true
  ) {
    return checkExpr(expr.args[0], ctx);
  }
  // Tensor unary builtin: result is tensor, single tensor arg, name has
  // a `tensorUnaryOp` capability on its IBuiltin. These kernels are
  // real-only in libnumbl_ops; the complex variant requires separate
  // dispatch (Phase 3+) or is not defined. Keep this path real-only.
  if (
    expr.jitType.kind === "tensor" &&
    expr.jitType.isComplex === false &&
    getTensorUnaryOp(expr.name)
  ) {
    if (expr.args.length !== 1) {
      return fail(ctx, `${expr.name}: expected 1 tensor arg`);
    }
    if (
      expr.args[0].jitType.kind === "tensor" &&
      expr.args[0].jitType.isComplex === true
    ) {
      return fail(
        ctx,
        `${expr.name}: complex tensor arg not supported by real unary kernel`
      );
    }
    return checkExpr(expr.args[0], ctx);
  }
  // Complex-tensor special cases: conj / real / imag on a complex tensor
  // arg. conj preserves shape+complexness; real/imag produce a real
  // tensor of the same shape. Handled in the emitter directly since
  // they don't match the tensorUnaryOp kernel dispatch.
  if (
    (expr.name === "conj" || expr.name === "real" || expr.name === "imag") &&
    expr.args.length === 1 &&
    expr.args[0].jitType.kind === "tensor" &&
    expr.args[0].jitType.isComplex === true
  ) {
    return checkExpr(expr.args[0], ctx);
  }
  // Tensor binary builtin (max, min, atan2, hypot, mod, rem):
  // result is tensor, two args (tensor/scalar). Real-only kernels.
  if (
    expr.jitType.kind === "tensor" &&
    expr.jitType.isComplex === false &&
    getTensorBinaryFn(expr.name)
  ) {
    if (expr.args.length !== 2) {
      return fail(ctx, `${expr.name}: expected 2 args`);
    }
    for (const a of expr.args) {
      if (a.jitType.kind === "tensor" && a.jitType.isComplex === true) {
        return fail(
          ctx,
          `${expr.name}: complex tensor arg not supported by real binary kernel`
        );
      }
      const r = checkExpr(a, ctx);
      if (!r.ok) return r;
    }
    return { ok: true };
  }
  // Reduction (tensor → scalar): name has a `tensorReductionOp`
  // capability, single tensor arg. Real reductions always OK; complex
  // reductions limited to SUM/PROD/ANY/ALL (the opcodes the
  // `numbl_complex_flat_reduce` kernel supports — MAX/MIN/MEAN aren't
  // defined for complex values).
  if (expr.jitType.kind !== "tensor" && getTensorReductionOp(expr.name)) {
    if (expr.args.length !== 1) {
      return fail(ctx, `${expr.name}: only single-arg reduction supported`);
    }
    const a = expr.args[0];
    if (a.jitType.kind !== "tensor") {
      return fail(ctx, `${expr.name}: only tensor-arg reduction supported`);
    }
    if (a.jitType.isComplex === true) {
      const opEnum = getTensorReductionOp(expr.name);
      const allowed =
        opEnum === "NUMBL_REDUCE_SUM" ||
        opEnum === "NUMBL_REDUCE_PROD" ||
        opEnum === "NUMBL_REDUCE_ANY" ||
        opEnum === "NUMBL_REDUCE_ALL";
      if (!allowed) {
        return fail(
          ctx,
          `${expr.name}: complex reduction not defined (only sum/prod/any/all)`
        );
      }
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
  // zeros / ones: fresh-tensor builtin (1 or 2 numeric args). Feasible
  // only at the statement level where a malloc can be emitted; the
  // codegen handles both tensor locals and dynamic tensor outputs.
  // Refuse 3D+ and tensor-valued shape args — no test needs those today.
  if (
    (expr.name === "zeros" || expr.name === "ones") &&
    expr.jitType.kind === "tensor" &&
    (expr.args.length === 1 || expr.args.length === 2)
  ) {
    for (const a of expr.args) {
      if (!isScalarKind(a.jitType.kind)) {
        return fail(ctx, `${expr.name}: shape args must be scalar`);
      }
      const r = checkExpr(a, ctx);
      if (!r.ok) return r;
    }
    return { ok: true };
  }
  // length / isempty on a tensor Var: fast scalar-result accessors
  // that only need the tensor's `_len` (and optionally `_d0`/`_d1`)
  // locals. Keep the arg-Var restriction — anything more general would
  // need a scratch evaluation that these cheap accessors don't want.
  if (
    (expr.name === "length" || expr.name === "isempty") &&
    expr.args.length === 1 &&
    expr.args[0].tag === "Var" &&
    expr.args[0].jitType.kind === "tensor"
  ) {
    // length / isempty only need `_len` (and optionally `_d0`/`_d1`) —
    // they don't care whether the tensor is real or complex.
    return { ok: true };
  }
  // disp: supported for string/char literals and real numeric scalars.
  // The emitter routes these through a JS-registered callback
  // (NumblDispCb) that calls back into `rt.output`. Tensor / complex /
  // char-var args are still out of scope — fall through to the generic
  // "non-C-mappable" failure.
  if (expr.name === "disp" && expr.args.length === 1) {
    const a = expr.args[0];
    if (a.tag === "StringLiteral") return { ok: true };
    if (a.jitType.kind === "number" || a.jitType.kind === "boolean") {
      return checkExpr(a, ctx);
    }
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
      // Assign-RHS is the one position where a tensor-return UserCall
      // is allowed — the existing local-tensor + epilogue-free()
      // machinery hooks it automatically if the RHS is the top expr.
      return checkExpr(stmt.expr, ctx, true);

    case "ExprStmt":
      return checkExpr(stmt.expr, ctx);

    case "If": {
      if (isComplexScalarKind(stmt.cond.jitType.kind)) {
        return fail(ctx, "If condition cannot be complex scalar");
      }
      const c = checkExpr(stmt.cond, ctx);
      if (!c.ok) return c;
      const t = checkStmts(stmt.thenBody, ctx);
      if (!t.ok) return t;
      for (const eib of stmt.elseifBlocks) {
        if (isComplexScalarKind(eib.cond.jitType.kind)) {
          return fail(ctx, "If condition cannot be complex scalar");
        }
        const ec = checkExpr(eib.cond, ctx);
        if (!ec.ok) return ec;
        const eb = checkStmts(eib.body, ctx);
        if (!eb.ok) return eb;
      }
      if (stmt.elseBody) return checkStmts(stmt.elseBody, ctx);
      return { ok: true };
    }

    case "For": {
      if (isComplexScalarKind(stmt.start.jitType.kind)) {
        return fail(ctx, "For loop bound cannot be complex scalar");
      }
      if (isComplexScalarKind(stmt.end.jitType.kind)) {
        return fail(ctx, "For loop bound cannot be complex scalar");
      }
      if (stmt.step && isComplexScalarKind(stmt.step.jitType.kind)) {
        return fail(ctx, "For loop step cannot be complex scalar");
      }
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
      if (isComplexScalarKind(stmt.cond.jitType.kind)) {
        return fail(ctx, "While condition cannot be complex scalar");
      }
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
      // 1-3 D scalar writes to a real-tensor param (so the caller's
      // buffer is either the seeded output or a prelude unshare copy).
      // Shape dims for 2D/3D are threaded through the ABI. Fresh tensor
      // locals and `y = x` aliases still defer — those would need shape
      // tracking on the callee side that the emitter doesn't do yet.
      const n = stmt.indices.length;
      if (n < 1 || n > 3) {
        return fail(ctx, `AssignIndex arity ${n} unsupported (only 1D/2D/3D)`);
      }
      if (
        stmt.baseType.kind !== "tensor" ||
        stmt.baseType.isComplex !== false
      ) {
        return fail(ctx, "AssignIndex base must be a real tensor");
      }
      if (!isRealScalarKind(stmt.value.jitType.kind)) {
        return fail(ctx, "AssignIndex value must be a real scalar");
      }
      // Writes land on (a) a tensor param (unshare-at-entry for pure
      // inputs, seeded output for param-outputs), or (b) a local /
      // dynamic-output tensor whose buffer the emitter has already
      // malloc'd. Either way the backing data is writable.
      if (!ctx.tensorVars.has(stmt.baseName)) {
        return fail(
          ctx,
          `AssignIndex base '${stmt.baseName}' must be a tensor var`
        );
      }
      const vr = checkExpr(stmt.value, ctx);
      if (!vr.ok) return vr;
      for (const idx of stmt.indices) {
        if (!isRealScalarKind(idx.jitType.kind)) {
          return fail(ctx, "AssignIndex index must be a real scalar");
        }
        const r = checkExpr(idx, ctx);
        if (!r.ok) return r;
      }
      return { ok: true };
    }

    case "AssignIndexRange": {
      // `dst(a:b) = src(c:d)` (or `dst(a:b) = src` — whole-tensor RHS,
      // srcStart/srcEnd are both null). Both tensors must be real. Dst
      // is written like AssignIndex so it must be a tensor param (for
      // either the unshare-at-entry copy or the seeded output buffer).
      // Src just needs data/len — also a tensor param for now, which
      // keeps the rule symmetric and avoids having to track local
      // tensor buffer state here.
      if (
        stmt.baseType.kind !== "tensor" ||
        stmt.baseType.isComplex !== false
      ) {
        return fail(ctx, "AssignIndexRange base must be a real tensor");
      }
      if (stmt.srcType.kind !== "tensor" || stmt.srcType.isComplex !== false) {
        return fail(ctx, "AssignIndexRange src must be a real tensor");
      }
      if (!ctx.tensorVars.has(stmt.baseName)) {
        return fail(
          ctx,
          `AssignIndexRange base '${stmt.baseName}' must be a tensor var`
        );
      }
      if (!ctx.tensorVars.has(stmt.srcBaseName)) {
        return fail(
          ctx,
          `AssignIndexRange src '${stmt.srcBaseName}' must be a tensor var`
        );
      }
      const ds = checkExpr(stmt.dstStart, ctx);
      if (!ds.ok) return ds;
      const de = checkExpr(stmt.dstEnd, ctx);
      if (!de.ok) return de;
      if (stmt.srcStart) {
        const ss = checkExpr(stmt.srcStart, ctx);
        if (!ss.ok) return ss;
      }
      if (stmt.srcEnd) {
        const se = checkExpr(stmt.srcEnd, ctx);
        if (!se.ok) return se;
      }
      return { ok: true };
    }

    case "AssignIndexCol": {
      // `dst(:, j) = src` — dst must have 2-D shape plumbing (numbl_setCol2r_h
      // needs dstRows), src needs data/len. Both tensor params for now.
      if (
        stmt.baseType.kind !== "tensor" ||
        stmt.baseType.isComplex !== false
      ) {
        return fail(ctx, "AssignIndexCol base must be a real tensor");
      }
      if (stmt.srcType.kind !== "tensor" || stmt.srcType.isComplex !== false) {
        return fail(ctx, "AssignIndexCol src must be a real tensor");
      }
      if (!ctx.tensorVars.has(stmt.baseName)) {
        return fail(
          ctx,
          `AssignIndexCol base '${stmt.baseName}' must be a tensor var`
        );
      }
      if (!ctx.tensorVars.has(stmt.srcBaseName)) {
        return fail(
          ctx,
          `AssignIndexCol src '${stmt.srcBaseName}' must be a tensor var`
        );
      }
      return checkExpr(stmt.colIndex, ctx);
    }

    // Out of scope: struct writes, multi-assign, 3-D page-slice writes.
    case "AssignMember":
    case "MultiAssign":
    case "AssignIndexPage3d":
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
  outputType: JitType | null,
  outputTypes: JitType[],
  nargout: number,
  generatedIRBodies?: Map<string, GeneratedFn>
): FeasibilityResult {
  const shared: SharedFeasCtx | null = generatedIRBodies
    ? {
        generatedIRBodies,
        results: new Map(),
        inProgress: new Set(),
        classifications: new Map(),
      }
    : null;
  return checkCFeasibilityInternal(
    body,
    paramNames,
    argTypes,
    outputType,
    outputTypes,
    nargout,
    shared
  );
}

function checkCFeasibilityInternal(
  body: JitStmt[],
  paramNames: string[],
  argTypes: JitType[],
  outputType: JitType | null,
  outputTypes: JitType[],
  nargout: number,
  shared: SharedFeasCtx | null
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
  // All tensor params. AssignIndex targets must be one of these — param-
  // outputs are safe because the JS wrapper seeds the output buffer
  // from the caller's data, and pure-input tensor params are made safe
  // by the emitter's unshare-at-entry copy (see jitCodegenC.ts).
  const tensorParams = new Set<string>();
  for (let i = 0; i < paramNames.length; i++) {
    if (argTypes[i].kind === "tensor") tensorParams.add(paramNames[i]);
  }
  // Tensor-typed names reachable in the emitter: params + any local
  // that receives at least one tensor-valued Assign RHS. Used for
  // read-side feasibility (src of AssignIndexRange / AssignIndexCol /
  // RangeSliceRead / multi-index Index). Writes still go through the
  // stricter `tensorParams` check.
  const tensorVars = new Set<string>(tensorParams);
  collectTensorLocals(body, tensorVars);
  const ctx: Ctx = { line: 0, tensorParams, tensorVars, shared };
  return checkStmts(body, ctx);
}

function collectTensorLocals(body: JitStmt[], out: Set<string>): void {
  const visit = (s: JitStmt): void => {
    switch (s.tag) {
      case "Assign":
        if (s.expr.jitType.kind === "tensor") out.add(s.name);
        break;
      case "If":
        s.thenBody.forEach(visit);
        s.elseifBlocks.forEach(eb => eb.body.forEach(visit));
        if (s.elseBody) s.elseBody.forEach(visit);
        break;
      case "For":
      case "While":
        s.body.forEach(visit);
        break;
      default:
        break;
    }
  };
  body.forEach(visit);
}
