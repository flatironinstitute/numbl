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
import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import { isKnownInteger, type JitExpr, type JitStmt } from "../jitTypes.js";
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
  getTensorUnaryOp,
  getTensorBinaryFn,
  getTensorReductionOp,
  allocComplexScratch,
  allocScratch,
  formatNumberLiteral,
  hasFreshAlloc,
  isComplexScalarVar,
  isComplexTensorVar,
  isDynamicOutput,
  isLocalTensor,
  isOutputTensor,
  isTensorVar,
  mangle,
  mangleIm,
  scratchData,
  scratchDataIm,
  scratchLen,
  tensorD0,
  tensorD1,
  tensorData,
  tensorDataIm,
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

function isComplexExpr(expr: JitExpr): boolean {
  return expr.jitType.kind === "complex_or_number";
}

/** Pair of C expressions holding the real and imaginary parts of a
 *  complex scalar value. Produced by `emitComplex`. */
interface ComplexPair {
  re: string;
  im: string;
}

/** Materialize a ComplexPair into two fresh `__cN_re` / `__cN_im` locals
 *  prepended to `ctx.pendingStmts.lines`. Used when the emitted formula
 *  would otherwise evaluate either sub-expression more than once (e.g.
 *  both Mul and Div reference each operand twice in the per-component
 *  formula). Returns references to the new locals. */
function materializeComplexPair(pair: ComplexPair, ctx: EmitCtx): ComplexPair {
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

/** Widen a real scalar C expression to a complex pair (im = 0). */
function widenRealToComplex(realCode: string): ComplexPair {
  return { re: realCode, im: "0.0" };
}

/** Emit a complex-valued scalar expression, returning (re, im) C
 *  expression strings. Call sites that need a pair (complex RHS, complex
 *  operand of a complex op, arg of real/imag/conj on complex) route
 *  through here. Real sub-expressions widen implicitly (im = 0). */
function emitComplex(expr: JitExpr, ctx: EmitCtx): ComplexPair {
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

/** Emit a value-expression. For scalars, returns a C `double` expression.
 *  For tensors, returns the data-variable name (the caller knows to also
 *  access the corresponding _len variable).
 *
 *  Complex-typed expressions are *not* valid here — they produce a pair
 *  of doubles and must go through `emitComplex`. Reaching this function
 *  with a complex expression indicates a missed routing at the caller. */
function emitExpr(expr: JitExpr, ctx: EmitCtx): string {
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

/** Emit the C expressions for one arg's ABI slots, consulting the
 *  callee's paramDesc so the slot order matches the callee's signature.
 *  Scalars contribute one slot; tensors contribute data + len + optional
 *  d0 / d1. For the shape slots the caller falls back to
 *  `(int64_t)tensorLen(arg)` / `1` when its own arg-var wasn't classified
 *  with matching shape plumbing. */
function emitUserCallArgSlots(
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
function emitUserCall(
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
 *  data pointer variable name. Callers that route the result into a
 *  named dst (vs. a scratch) must use the scratch-transfer pattern —
 *  let this helper write the full expression into a fresh scratch via
 *  `emitTensorExprToStmts`, then `emitEnsureTensorBuf` + memcpy into
 *  the dst. That ordering avoids clobbering an operand that aliases
 *  the dst (e.g. `r = r .* y + 3.0`). */
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

/** Size a scratch tensor buffer to `lenExpr` doubles, freeing any stale
 *  buffer from a prior loop iteration first. A plain `if (!sData) malloc`
 *  would overflow the buffer if a later iteration demanded more bytes. */
function emitScratchBufAlloc(
  lines: string[],
  indent: string,
  sData: string,
  sLen: string,
  lenExpr: string
): void {
  lines.push(`${indent}${sLen} = ${lenExpr};`);
  lines.push(`${indent}if (${sData}) free(${sData});`);
  lines.push(
    `${indent}${sData} = (${sLen} > 0) ? (double *)malloc((size_t)${sLen} * sizeof(double)) : NULL;`
  );
}

/** Complex scratch alloc: re + im buffers, both sized to `lenExpr`. The
 *  complex kernels in numbl_ops write both buffers unconditionally, so
 *  both must be valid pointers (len > 0). */
function emitScratchBufAllocComplex(
  lines: string[],
  indent: string,
  sData: string,
  sDataIm: string,
  sLen: string,
  lenExpr: string
): void {
  lines.push(`${indent}${sLen} = ${lenExpr};`);
  lines.push(`${indent}if (${sData}) free(${sData});`);
  lines.push(`${indent}if (${sDataIm}) free(${sDataIm});`);
  lines.push(
    `${indent}${sData} = (${sLen} > 0) ? (double *)malloc((size_t)${sLen} * sizeof(double)) : NULL;`
  );
  lines.push(
    `${indent}${sDataIm} = (${sLen} > 0) ? (double *)malloc((size_t)${sLen} * sizeof(double)) : NULL;`
  );
}

/** Complex tensor expression result: data + dataIm + len in C. For a
 *  Var whose JitType is a real tensor, `dataIm` is the literal string
 *  `"NULL"` — the numbl_ops complex kernels treat that as "all zero",
 *  so a real tensor flowing into a complex op doesn't need a zero
 *  buffer. */
interface ComplexTensorResult {
  data: string;
  dataIm: string;
  len: string;
}

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
function emitComplexTensorExprToStmts(
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
function emitComplexTensorBinaryStmts(
  lines: string[],
  indent: string,
  expr: JitExpr & { tag: "Binary" },
  ctx: EmitCtx,
  destDataVar: string,
  destDataImVar: string,
  destLenVar: string
): void {
  const opEnum = TENSOR_BIN_OP[expr.op];
  if (!opEnum) {
    throw new Error(
      `C-JIT codegen: complex tensor binary op ${expr.op} has no opcode`
    );
  }
  // TENSOR_BIN_OP enum values are numerically aligned with the
  // NUMBL_COMPLEX_BIN_* enum (ADD=0, SUB=1, MUL=2, DIV=3), so the same
  // string works at the C-side switch. Confirmed by numbl_ops.h.
  const complexOpEnum = opEnum.replace("NUMBL_REAL_BIN_", "NUMBL_COMPLEX_BIN_");

  const leftIsTensor = isTensorExpr(expr.left);
  const rightIsTensor = isTensorExpr(expr.right);

  const lenSrc = findTensorLenExpr(expr, ctx);
  lines.push(`${indent}${destLenVar} = ${lenSrc};`);

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

/** Emit a scalar sub-expression at a complex tensor op boundary. Returns
 *  a (re, im) pair of C expressions. Real scalars become (expr, "0.0");
 *  complex scalars go through emitComplex for their pair form. */
function emitComplexScalarPair(
  expr: JitExpr,
  ctx: EmitCtx
): {
  re: string;
  im: string;
} {
  if (expr.jitType.kind === "complex_or_number") {
    return emitComplex(expr, ctx);
  }
  return { re: emitExpr(expr, ctx), im: "0.0" };
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

/** Evaluate `fn` with `ctx.pendingStmts` set to `{lines, indent}` so any
 *  UserCall / RangeSliceRead nested inside can hoist its decl+call into
 *  `lines` ahead of the calling statement. Restores the prior value on
 *  exit (safe even if nested save/restore frames stack). */
function withPendingStmts<T>(
  ctx: EmitCtx,
  lines: string[],
  indent: string,
  fn: () => T
): T {
  const prev = ctx.pendingStmts;
  ctx.pendingStmts = { lines, indent };
  try {
    return fn();
  } finally {
    ctx.pendingStmts = prev;
  }
}

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
      const dynamicOutputNames = new Set<string>();
      for (const n of ctx.cls.outputTensorNames) {
        if (ctx.cls.meta.get(n)?.isDynamicOutput) dynamicOutputNames.add(n);
      }
      emitFusedChain(
        lines,
        indent,
        entry.chain,
        ctx.cls.tensorVars,
        ctx.cls.paramTensorNames,
        ctx.cls.outputTensorNames,
        ctx.cls.localTensorNames,
        dynamicOutputNames,
        ctx.openmp
      );
      i += entry.chain.length;
    } else {
      emitStmt(lines, stmts[i], indent, ctx);
      i++;
    }
  }
}

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
function emitEnsureTensorBuf(
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
    lines.push(`${indent}${dLen} = ${lenExpr};`);
    lines.push(`${indent}if (${dData}) free(${dData});`);
    lines.push(
      `${indent}${dData} = (${dLen} > 0) ? (double *)malloc((size_t)${dLen} * sizeof(double)) : NULL;`
    );
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
function emitEnsureComplexTensorBuf(
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
    lines.push(`${indent}${dLen} = ${lenExpr};`);
    lines.push(`${indent}if (${dData}) free(${dData});`);
    lines.push(`${indent}if (${dDataIm}) free(${dDataIm});`);
    lines.push(
      `${indent}${dData} = (${dLen} > 0) ? (double *)malloc((size_t)${dLen} * sizeof(double)) : NULL;`
    );
    lines.push(
      `${indent}${dDataIm} = (${dLen} > 0) ? (double *)malloc((size_t)${dLen} * sizeof(double)) : NULL;`
    );
    if (isDyn) {
      const [d0Expr, d1Expr] = shapeExprsFor(shapeSrc, lenExpr);
      lines.push(`${indent}${tensorD0(destName)} = ${d0Expr};`);
      lines.push(`${indent}${tensorD1(destName)} = ${d1Expr};`);
    }
    return;
  }
  lines.push(`${indent}${dLen} = ${lenExpr};`);
}

/** Derive (d0, d1) C expressions for a dynamic tensor output whose
 *  shape is inherited from an elemwise operand. Uses the operand's
 *  static shape when known; else falls back to `[len, 1]` (1D column
 *  convention). */
function shapeExprsFor(
  shapeSrc: JitExpr | undefined,
  lenExpr: string
): [string, string] {
  const shape =
    shapeSrc?.jitType.kind === "tensor" ? shapeSrc.jitType.shape : undefined;
  if (shape && shape.length === 2) {
    return [`${shape[0]}`, `${shape[1]}`];
  }
  return [lenExpr, "1"];
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

/** Emit `dest = foo(...)` where foo returns a tensor via the dynamic-
 *  output ABI. Feasibility has already verified the callee's output[0]
 *  is a fresh-alloc dynamic output, so the callee fills
 *  `buf_out / out_len / d0_out / d1_out` and transfers ownership. The
 *  caller frees the old dest buffer (if any), takes the new buffer, and
 *  lets the epilogue free() it at end-of-scope alongside the other
 *  local tensors. */
function emitUserCallTensorAssign(
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
  const argCodes: string[] = [];
  for (let i = 0; i < expr.args.length; i++) {
    const slots = emitUserCallArgSlots(
      expr.args[i],
      calleeAbi.paramDescs[i],
      ctx
    );
    argCodes.push(...slots);
  }
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

/** Emit a tensor-result Assign: handles Binary, Unary, Call on tensors. */
/** Complex-tensor Assign: parallels emitTensorAssign but writes paired
 *  re+im buffers. RHS sub-exprs route through emitComplexTensorExprToStmts
 *  / emitComplexTensorBinaryStmts. For a real RHS (e.g. a Var pointing
 *  to a real tensor), imag is widened via NULL pointer or a zero buffer.
 *
 *  Runs inside a pendingStmts frame so nested complex scalar sub-expressions
 *  (`1i`, `3+4i`, `re(z) + 1i*im(z)`, ...) can materialize their pair
 *  locals into the same `lines` stream ahead of the kernel call. */
function emitComplexTensorAssign(
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

function emitTensorAssign(
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
        // 1D slice — emit column-vector shape for the dynamic output.
        // Row-vector sources lose orientation here; C-JIT doesn't yet
        // track RangeSliceRead's isRow flag (the JS-JIT does).
        lines.push(`${indent}${tensorD0(destName)} = ${dLen};`);
        lines.push(`${indent}${tensorD1(destName)} = 1;`);
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
    // Route the whole RHS through a scratch so that a self-aliasing
    // sub-expression like `r = r .* y + 3.0` sees the OLD v_r_data
    // while the kernel writes into a fresh scratch. Then transfer
    // the scratch into dst (free+malloc for locals/dyn outputs, or
    // memcpy for fixed outputs). Fixes the heap-overflow regression
    // where emitEnsureTensorBuf clobbered an aliased operand before
    // the kernel read it.
    const scratch = emitTensorExprToStmts(lines, indent, expr, ctx);
    const tensorOperand = isTensorExpr(expr.left) ? expr.left : expr.right;
    emitEnsureTensorBuf(
      lines,
      indent,
      destName,
      scratch.len,
      ctx,
      tensorOperand
    );
    lines.push(
      `${indent}if (${dLen} > 0) memcpy(${dData}, ${scratch.data}, (size_t)${dLen} * sizeof(double));`
    );
    return;
  }

  if (expr.tag === "Unary" && isTensorExpr(expr)) {
    if (expr.op === UnaryOperation.Plus) {
      emitTensorAssign(lines, indent, destName, expr.operand, ctx);
      return;
    }
    if (expr.op === UnaryOperation.Minus) {
      // scratch-transfer pattern — evaluate `-operand` into a scratch
      // first, then free+malloc dst. Guards against `r = -r` aliasing,
      // where operand.data would otherwise dangle through the dst
      // free before the kernel reads it.
      const scratch = emitTensorExprToStmts(lines, indent, expr, ctx);
      emitEnsureTensorBuf(
        lines,
        indent,
        destName,
        scratch.len,
        ctx,
        expr.operand
      );
      lines.push(
        `${indent}if (${dLen} > 0) memcpy(${dData}, ${scratch.data}, (size_t)${dLen} * sizeof(double));`
      );
      return;
    }
  }

  if (expr.tag === "Call" && isTensorExpr(expr)) {
    const opEnum = getTensorUnaryOp(expr.name);
    if (opEnum) {
      // Scratch-transfer to defuse self-aliasing (e.g. `r = exp(r)`).
      const scratch = emitTensorExprToStmts(lines, indent, expr, ctx);
      emitEnsureTensorBuf(
        lines,
        indent,
        destName,
        scratch.len,
        ctx,
        expr.args[0]
      );
      lines.push(
        `${indent}if (${dLen} > 0) memcpy(${dData}, ${scratch.data}, (size_t)${dLen} * sizeof(double));`
      );
      return;
    }
    // abs(complex_tensor) → real tensor, via numbl_complex_abs. Sits
    // here rather than in the tensorUnaryOp path because the output
    // type changes (complex operand → real result). Scratch-transfer
    // pattern keeps this aliasing-safe even if the real-tensor dst
    // somehow shared a name with the complex operand (shouldn't
    // happen given the type flip, but cheap insurance).
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
      const sIdx = allocScratch(ctx);
      const sData = scratchData(sIdx);
      const sLen = scratchLen(sIdx);
      emitScratchBufAlloc(lines, indent, sData, sLen, operand.len);
      lines.push(
        `${indent}numbl_complex_abs((size_t)${sLen}, ${operand.data}, ${operand.dataIm}, ${sData});`
      );
      emitEnsureTensorBuf(lines, indent, destName, sLen, ctx, expr.args[0]);
      lines.push(
        `${indent}if (${dLen} > 0) memcpy(${dData}, ${sData}, (size_t)${dLen} * sizeof(double));`
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
      emitEnsureTensorBuf(
        lines,
        indent,
        destName,
        operand.len,
        ctx,
        expr.args[0]
      );
      if (expr.name === "real") {
        lines.push(
          `${indent}if (${dLen} > 0) memcpy(${dData}, ${operand.data}, (size_t)${dLen} * sizeof(double));`
        );
      } else {
        lines.push(`${indent}if (${operand.dataIm}) {`);
        lines.push(
          `${indent}  if (${dLen} > 0) memcpy(${dData}, ${operand.dataIm}, (size_t)${dLen} * sizeof(double));`
        );
        lines.push(`${indent}} else {`);
        lines.push(
          `${indent}  if (${dLen} > 0) memset(${dData}, 0, (size_t)${dLen} * sizeof(double));`
        );
        lines.push(`${indent}}`);
      }
      return;
    }
  }

  // Two-arg tensor binary builtin (max, min, atan2, hypot, mod, rem):
  // emit a per-element loop calling the C math function registered on
  // the builtin's jitCapabilities.tensorBinaryFn. Scratch-transfer
  // pattern avoids self-aliasing (e.g. `r = max(r, y)`).
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

    const sIdx = allocScratch(ctx);
    const sData = scratchData(sIdx);
    const sLen = scratchLen(sIdx);

    if (leftIsTensor && rightIsTensor) {
      const lArg = emitTensorExprToStmts(lines, indent, left, ctx);
      const rArg = emitTensorExprToStmts(lines, indent, right, ctx);
      emitScratchBufAlloc(lines, indent, sData, sLen, lArg.len);
      lines.push(`${indent}for (int64_t __i = 0; __i < ${sLen}; __i++)`);
      lines.push(
        `${indent}  ${sData}[__i] = ${cFn}(${lArg.data}[__i], ${rArg.data}[__i]);`
      );
    } else if (leftIsTensor) {
      const lArg = emitTensorExprToStmts(lines, indent, left, ctx);
      const rScalar = emitExpr(right, ctx);
      emitScratchBufAlloc(lines, indent, sData, sLen, lArg.len);
      lines.push(`${indent}for (int64_t __i = 0; __i < ${sLen}; __i++)`);
      lines.push(
        `${indent}  ${sData}[__i] = ${cFn}(${lArg.data}[__i], ${rScalar});`
      );
    } else {
      const lScalar = emitExpr(left, ctx);
      const rArg = emitTensorExprToStmts(lines, indent, right, ctx);
      emitScratchBufAlloc(lines, indent, sData, sLen, rArg.len);
      lines.push(`${indent}for (int64_t __i = 0; __i < ${sLen}; __i++)`);
      lines.push(
        `${indent}  ${sData}[__i] = ${cFn}(${lScalar}, ${rArg.data}[__i]);`
      );
    }
    emitEnsureTensorBuf(lines, indent, destName, sLen, ctx, tensorOperand);
    lines.push(
      `${indent}if (${dLen} > 0) memcpy(${dData}, ${sData}, (size_t)${dLen} * sizeof(double));`
    );
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
        getTensorReductionOp(stmt.expr.name) !== undefined &&
        stmt.expr.args[0]?.jitType.kind === "tensor" &&
        stmt.expr.args[0].tag !== "Var"
      ) {
        emitReductionOfTensorExpr(lines, indent, stmt.name, stmt.expr, ctx);
        return;
      }
      // Complex scalar assign: emit pair into pendingStmts then write to
      // the two paired locals.
      if (isComplexScalarVar(ctx, stmt.name)) {
        const pair = withPendingStmts(ctx, lines, indent, () =>
          emitComplex(stmt.expr, ctx)
        );
        lines.push(`${indent}${mangle(stmt.name)} = ${pair.re};`);
        lines.push(`${indent}${mangleIm(stmt.name)} = ${pair.im};`);
        return;
      }
      const rhs = withPendingStmts(ctx, lines, indent, () =>
        emitExpr(stmt.expr, ctx)
      );
      lines.push(`${indent}${mangle(stmt.name)} = ${rhs};`);
      return;
    }

    case "ExprStmt": {
      const code = withPendingStmts(ctx, lines, indent, () =>
        emitExpr(stmt.expr, ctx)
      );
      lines.push(`${indent}(void)(${code});`);
      return;
    }

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
      // Allow UserCall / RangeSliceRead in the index or value by letting
      // them prepend helper statements (decl + call) before the
      // numbl_set*r_h line.
      const { idxCodes, v } = withPendingStmts(ctx, lines, indent, () => ({
        idxCodes: stmt.indices.map(idx => {
          let s = emitExpr(idx, ctx);
          if (!isKnownInteger(idx.jitType)) s = `round(${s})`;
          return s;
        }),
        v: emitExpr(stmt.value, ctx),
      }));
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
      const { dStart, dEnd, srcStart, srcEnd } = withPendingStmts(
        ctx,
        lines,
        indent,
        () => {
          const emitRI = (e: JitExpr): string => {
            let code = emitExpr(e, ctx);
            if (!isKnownInteger(e.jitType)) code = `round(${code})`;
            return code;
          };
          return {
            dStart: emitRI(stmt.dstStart),
            dEnd: emitRI(stmt.dstEnd),
            srcStart: stmt.srcStart !== null ? emitRI(stmt.srcStart) : `1.0`,
            srcEnd:
              stmt.srcEnd !== null ? emitRI(stmt.srcEnd) : `(double)${sLen}`,
          };
        }
      );
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
      const colCode = withPendingStmts(ctx, lines, indent, () => {
        let code = emitExpr(stmt.colIndex, ctx);
        if (!isKnownInteger(stmt.colIndex.jitType)) code = `round(${code})`;
        return code;
      });
      lines.push(
        `${indent}numbl_setCol2r_h(${dData}, (size_t)${dRows}, (size_t)${dLen}, ${colCode}, ${sData}, (size_t)${sLen}, __err_flag);`
      );
      return;
    }

    case "If": {
      const condCode = withPendingStmts(ctx, lines, indent, () =>
        emitTruthiness(stmt.cond, ctx)
      );
      lines.push(`${indent}if (${condCode}) {`);
      emitStmts(lines, stmt.thenBody, indent + "  ", ctx);
      for (const eib of stmt.elseifBlocks) {
        const eibCondCode = withPendingStmts(ctx, lines, indent, () =>
          emitTruthiness(eib.cond, ctx)
        );
        lines.push(`${indent}} else if (${eibCondCode}) {`);
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
      // MATLAB evaluates start / end / step once at loop entry. Hoisting
      // once matches that; C's for header re-reads end/step each iter,
      // which is semantically neutral against a local already assigned.
      const { start, end, step } = withPendingStmts(ctx, lines, indent, () => ({
        start: emitExpr(stmt.start, ctx),
        end: emitExpr(stmt.end, ctx),
        step: stmt.step ? emitExpr(stmt.step, ctx) : "1.0",
      }));
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
      // NO pendingStmts here: While's cond is re-evaluated every iter,
      // so a UserCall / RangeSliceRead in it can't be hoisted once —
      // those cases throw "outside statement context" and bail.
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
