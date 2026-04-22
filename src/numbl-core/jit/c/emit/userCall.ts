/**
 * UserCall emission — scalar returns in value position, and tensor
 * returns from an Assign RHS.
 *
 * A UserCall lowers to a static C function (`jit_<jitName>`) generated
 * alongside the outer function by [../assemble.ts](../assemble.ts).
 * This module marshals arg slots per the callee's ABI and emits the
 * call itself.
 *
 * Exports:
 *   - `emitUserCall` — scalar-return UserCall in value position (Assign
 *     RHS / nested expr). Stashes the result into a fresh `__ucN_out`
 *     local and returns the name as the expression text.
 *   - `emitUserCallArgSlots` — helper used by `emitUserCall` *and* the
 *     tensor-return variant in [./assign.ts](./assign.ts) to convert
 *     one arg into its ABI-slot expression list.
 *   - `emitUserCallTensorAssign` — tensor-return UserCall, only allowed
 *     as the top RHS of an Assign. Uses the dynamic-output ABI: the
 *     callee mallocs + transfers ownership via `double **` out-params.
 */
import type { JitExpr } from "../../jitTypes.js";
import type { TensorMeta } from "../classify.js";
import {
  hasFreshAlloc,
  isComplexTensorVar,
  isTensorVar,
  tensorD0,
  tensorD1,
  tensorData,
  tensorDataIm,
  tensorLen,
  tensorMaxDim,
  type EmitCtx,
} from "../context.js";
import { emitComplex } from "./complexScalar.js";
import { emitExpr } from "./scalar.js";
import { withPendingStmts } from "./stmt.js";

/** Emit the C expressions for one arg's ABI slots, consulting the
 *  callee's paramDesc so the slot order matches the callee's signature.
 *  Scalars contribute one slot; tensors contribute data + len + optional
 *  d0 / d1. For the shape slots the caller falls back to
 *  `(int64_t)tensorLen(arg)` / `1` when its own arg-var wasn't classified
 *  with matching shape plumbing. */
export function emitUserCallArgSlots(
  a: JitExpr,
  paramDesc: {
    kind: "scalar" | "complexScalar" | "tensor";
    slots: { kind: string }[];
  },
  ctx: EmitCtx
): string[] {
  if (paramDesc.kind === "scalar") {
    return [emitExpr(a, ctx)];
  }
  if (paramDesc.kind === "complexScalar") {
    // Callee expects (re, im). The caller's arg may be a bare real
    // scalar (widened to im = 0) or a complex_or_number value —
    // emitComplex handles both cases.
    const pair = emitComplex(a, ctx);
    return [pair.re, pair.im];
  }
  if (a.tag !== "Var") {
    throw new Error(
      `C-JIT codegen: UserCall tensor arg must be a Var (got ${a.tag})`
    );
  }
  const argName = a.name;
  if (!isTensorVar(ctx, argName)) {
    throw new Error(
      `C-JIT codegen: UserCall tensor arg '${argName}' is not a tensor var`
    );
  }
  const hasShape =
    hasFreshAlloc(ctx, argName) || tensorMaxDim(ctx, argName) >= 2;
  const slotCodes: string[] = [];
  for (const s of paramDesc.slots) {
    switch (s.kind) {
      case "tensorData":
        slotCodes.push(tensorData(argName));
        break;
      case "tensorDataIm":
        // Callee expects a complex-tensor imag slot. If the caller's arg
        // is a complex tensor, pass its imag pointer; if the arg is real,
        // widen by passing NULL — numbl_ops complex kernels treat NULL
        // imag as all-zero.
        slotCodes.push(
          isComplexTensorVar(ctx, argName) ? tensorDataIm(argName) : "NULL"
        );
        break;
      case "tensorLen":
        slotCodes.push(tensorLen(argName));
        break;
      case "tensorD0":
        slotCodes.push(
          hasShape ? tensorD0(argName) : `(int64_t)${tensorLen(argName)}`
        );
        break;
      case "tensorD1":
        slotCodes.push(hasShape ? tensorD1(argName) : "1");
        break;
      default:
        throw new Error(
          `C-JIT codegen: unexpected callee param slot kind '${s.kind}'`
        );
    }
  }
  return slotCodes;
}

/** Scalar-return UserCall. Tensor args are marshaled via the callee's
 *  paramDescs (data + len + optional d0/d1 slots). The callee is emitted
 *  as `static void jit_<jitName>(...)` in the same .c file by
 *  `generateC`, with a trailing `__err_flag` pointer. We stash the
 *  return value in a fresh local and return its name as the expression
 *  text. Must be invoked from statement context so the decl + call can
 *  be inserted before the surrounding expression.
 *
 *  Tensor-return UserCall is handled upstream by emitTensorAssign (only
 *  allowed as an Assign RHS), not here. */
export function emitUserCall(
  expr: JitExpr & { tag: "UserCall" },
  ctx: EmitCtx
): string {
  if (!ctx.pendingStmts) {
    throw new Error(`C-JIT codegen: UserCall outside statement context`);
  }
  if (expr.jitType.kind === "tensor") {
    throw new Error(
      `C-JIT codegen: tensor-return UserCall '${expr.name}' must appear as the top RHS of an Assign`
    );
  }
  const calleeAbi = ctx.calleeAbi?.get(expr.jitName);
  if (!calleeAbi) {
    throw new Error(
      `C-JIT codegen: UserCall '${expr.name}' missing callee ABI for ${expr.jitName}`
    );
  }
  if (calleeAbi.paramDescs.length !== expr.args.length) {
    throw new Error(
      `C-JIT codegen: UserCall '${expr.name}' arg count (${expr.args.length}) differs from callee paramDescs (${calleeAbi.paramDescs.length})`
    );
  }
  const argCodes: string[] = [];
  for (let i = 0; i < expr.args.length; i++) {
    const slots = emitUserCallArgSlots(
      expr.args[i],
      calleeAbi.paramDescs[i],
      ctx
    );
    argCodes.push(...slots);
  }
  const n = ++ctx.tmp.n;
  const tmpVar = `__uc${n}_out`;
  ctx.needsErrorFlag = true;
  const indent = ctx.pendingStmts.indent;
  ctx.pendingStmts.lines.push(`${indent}double ${tmpVar} = 0.0;`);
  const callArgs = [...argCodes, `&${tmpVar}`, `__err_flag`];
  ctx.pendingStmts.lines.push(
    `${indent}jit_${expr.jitName}(${callArgs.join(", ")});`
  );
  return tmpVar;
}

/** Emit `dest = foo(...)` where foo returns a tensor via the dynamic-
 *  output ABI. Feasibility has already verified the callee's output[0]
 *  is a fresh-alloc dynamic output, so the callee fills
 *  `buf_out / out_len / d0_out / d1_out` and transfers ownership. The
 *  caller frees the old dest buffer (if any), takes the new buffer, and
 *  lets the epilogue free() it at end-of-scope alongside the other
 *  local tensors. */
export function emitUserCallTensorAssign(
  lines: string[],
  indent: string,
  destName: string,
  destMeta: TensorMeta,
  expr: JitExpr & { tag: "UserCall" },
  ctx: EmitCtx
): void {
  if (!destMeta.hasFreshAlloc) {
    throw new Error(
      `C-JIT codegen: emitUserCallTensorAssign('${destName}'): destMeta.hasFreshAlloc must be true`
    );
  }
  const calleeAbi = ctx.calleeAbi?.get(expr.jitName);
  if (!calleeAbi) {
    throw new Error(
      `C-JIT codegen: UserCall '${expr.name}' missing callee ABI for ${expr.jitName}`
    );
  }
  if (calleeAbi.paramDescs.length !== expr.args.length) {
    throw new Error(
      `C-JIT codegen: UserCall '${expr.name}' arg count (${expr.args.length}) differs from callee paramDescs (${calleeAbi.paramDescs.length})`
    );
  }
  const out0 = calleeAbi.outputDescs[0];
  if (!out0 || out0.kind !== "tensor" || !out0.dynamic) {
    throw new Error(
      `C-JIT codegen: UserCall '${expr.name}' tensor-return requires dynamic tensor output[0]`
    );
  }
  const destIsComplex = isComplexTensorVar(ctx, destName);
  const calleeIsComplex = out0.isComplex === true;
  if (destIsComplex !== calleeIsComplex) {
    throw new Error(
      `C-JIT codegen: UserCall '${expr.name}' dest/callee complex mismatch (dest=${destIsComplex}, callee=${calleeIsComplex})`
    );
  }
  ctx.needsErrorFlag = true;
  // Marshal args inside a pendingStmts frame so a complex scalar arg
  // whose expression needs materialized pair locals (e.g. `(1+2i)*z`)
  // can hoist its decls into `lines` ahead of the callee invocation.
  // emitComplexTensorAssign wraps its caller path, but the real-tensor
  // emitTensorAssign path into this function does not — wrapping here
  // makes the function self-sufficient.
  const argCodes: string[] = withPendingStmts(ctx, lines, indent, () => {
    const codes: string[] = [];
    for (let i = 0; i < expr.args.length; i++) {
      const slots = emitUserCallArgSlots(
        expr.args[i],
        calleeAbi.paramDescs[i],
        ctx
      );
      codes.push(...slots);
    }
    return codes;
  });
  const dData = tensorData(destName);
  const dLen = tensorLen(destName);
  const dD0 = tensorD0(destName);
  const dD1 = tensorD1(destName);
  const n = ++ctx.tmp.n;
  const prefix = `__uc${n}`;
  const inner = indent + "  ";
  lines.push(`${indent}{`);
  lines.push(`${inner}double *${prefix}_buf = NULL;`);
  if (calleeIsComplex) {
    lines.push(`${inner}double *${prefix}_buf_im = NULL;`);
  }
  lines.push(`${inner}int64_t ${prefix}_len = 0;`);
  lines.push(`${inner}int64_t ${prefix}_d0 = 0;`);
  lines.push(`${inner}int64_t ${prefix}_d1 = 0;`);
  const callArgs = [...argCodes, `&${prefix}_buf`];
  if (calleeIsComplex) callArgs.push(`&${prefix}_buf_im`);
  callArgs.push(
    `&${prefix}_len`,
    `&${prefix}_d0`,
    `&${prefix}_d1`,
    `__err_flag`
  );
  lines.push(`${inner}jit_${expr.jitName}(${callArgs.join(", ")});`);
  lines.push(`${inner}if (${dData}) free(${dData});`);
  if (calleeIsComplex) {
    const dDataIm = tensorDataIm(destName);
    lines.push(`${inner}if (${dDataIm}) free(${dDataIm});`);
  }
  lines.push(`${inner}${dData} = ${prefix}_buf;`);
  if (calleeIsComplex) {
    const dDataIm = tensorDataIm(destName);
    lines.push(`${inner}${dDataIm} = ${prefix}_buf_im;`);
  }
  lines.push(`${inner}${dLen} = ${prefix}_len;`);
  lines.push(`${inner}${dD0} = ${prefix}_d0;`);
  lines.push(`${inner}${dD1} = ${prefix}_d1;`);
  lines.push(`${indent}}`);
}
