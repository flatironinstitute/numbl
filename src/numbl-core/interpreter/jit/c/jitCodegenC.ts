/**
 * JIT IR -> C code generation.
 *
 * Sibling of [jitCodegen.ts](../jitCodegen.ts). Covers the IR subset gated
 * by [cFeasibility.ts](./cFeasibility.ts):
 *
 *   - Phase 1: numbers/booleans, scalar math, control flow.
 *   - Phase 2: real (non-complex) tensors via the per-statement helpers in
 *     [cJitHelpers.ts](./cJitHelpers.ts), structured 1:1 with the JS-JIT
 *     helpers in [jitHelpersTensor.ts](../jitHelpersTensor.ts). Same
 *     dest-hint propagation, same buffer-reuse rules, same fallback to
 *     interpreter on unsupported inputs.
 *
 * Anything outside the whitelist should be blocked by the feasibility
 * prepass before this function is called.
 *
 * Every scalar (number or boolean) is represented as a C `double`. Booleans
 * flow as 0.0 / 1.0 so they compose with numeric arithmetic cleanly, which
 * mirrors how the JS path lets `true`/`false` coerce via `+x`. Truthiness
 * in `if`/`while`/&&/|| is `(x != 0.0)`.
 *
 * Tensor params, locals, and sub-expression results are `napi_value`s
 * (RuntimeTensor objects). Any function that touches tensors takes
 * `napi_env env` as its first parameter; the shim threads it through
 * unchanged.
 */

import { BinaryOperation, UnaryOperation } from "../../../parser/types.js";
import type { JitExpr, JitStmt, JitType } from "../jitTypes.js";
import { cJitTensorHelpersSource } from "./cJitHelpers.js";
import { C_TENSOR_REDUCTION_OPS, C_TENSOR_UNARY_OPS } from "./cFeasibility.js";

// All user-visible identifiers get a prefix so they can never collide with
// C keywords, libc names (`sin`, `cos`, ...) or our internal locals.
const MANGLE_PREFIX = "v_";

function mangle(name: string): string {
  return `${MANGLE_PREFIX}${name}`;
}

// Scalar MATLAB builtin name → C expression form. Same map as MVP.
const BUILTIN_TO_C: Record<string, string> = {
  sin: "sin",
  cos: "cos",
  tan: "tan",
  asin: "asin",
  acos: "acos",
  atan: "atan",
  sinh: "sinh",
  cosh: "cosh",
  tanh: "tanh",
  asinh: "asinh",
  acosh: "acosh",
  atanh: "atanh",
  exp: "exp",
  log: "log",
  log2: "log2",
  log10: "log10",
  sqrt: "sqrt",
  abs: "fabs",
  floor: "floor",
  ceil: "ceil",
  fix: "trunc",
  round: "round",
  atan2: "atan2",
  hypot: "hypot",
  rem: "fmod",
  expm1: "expm1",
  log1p: "log1p",
  pow: "pow",
  // Special-cased in emitCall (use our inline helpers):
  mod: "__numbl_mod",
  sign: "__numbl_sign",
};

// Builtins whose call emits a call to a local helper we prepend to the TU.
const BUILTINS_NEEDING_HELPERS: Record<string, string> = {
  mod: `static double __numbl_mod(double a, double b) {
  if (b == 0.0) return a;
  double r = fmod(a, b);
  if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
  return r;
}`,
  sign: `static double __numbl_sign(double x) {
  if (x > 0.0) return 1.0;
  if (x < 0.0) return -1.0;
  return 0.0;
}`,
};

// MATLAB binary op → C-JIT tensor helper name. Mirrors the $h.t* names
// in jitCodegen.ts:emitTensorBinary.
const TENSOR_BIN_HELPER: Partial<Record<BinaryOperation, string>> = {
  [BinaryOperation.Add]: "numbl_jit_tAdd",
  [BinaryOperation.Sub]: "numbl_jit_tSub",
  [BinaryOperation.Mul]: "numbl_jit_tMul",
  [BinaryOperation.ElemMul]: "numbl_jit_tMul",
  [BinaryOperation.Div]: "numbl_jit_tDiv",
  [BinaryOperation.ElemDiv]: "numbl_jit_tDiv",
  [BinaryOperation.Equal]: "numbl_jit_tEq",
  [BinaryOperation.NotEqual]: "numbl_jit_tNeq",
  [BinaryOperation.Less]: "numbl_jit_tLt",
  [BinaryOperation.LessEqual]: "numbl_jit_tLe",
  [BinaryOperation.Greater]: "numbl_jit_tGt",
  [BinaryOperation.GreaterEqual]: "numbl_jit_tGe",
};

// MATLAB unary builtin → C-JIT helper name. Mirrors UNARY_OP_CODE in
// jitHelpersTensor.ts (subset; domain-restricted ones excluded).
const TENSOR_UNARY_HELPER: Record<string, string> = {
  exp: "numbl_jit_tExp",
  abs: "numbl_jit_tAbs",
  floor: "numbl_jit_tFloor",
  ceil: "numbl_jit_tCeil",
  round: "numbl_jit_tRound",
  fix: "numbl_jit_tFix",
  sin: "numbl_jit_tSin",
  cos: "numbl_jit_tCos",
  tan: "numbl_jit_tTan",
  atan: "numbl_jit_tAtan",
  sinh: "numbl_jit_tSinh",
  cosh: "numbl_jit_tCosh",
  tanh: "numbl_jit_tTanh",
  sign: "numbl_jit_tSign",
};

const TENSOR_REDUCTION_HELPER: Record<string, string> = {
  sum: "numbl_jit_tSum",
  prod: "numbl_jit_tProd",
  max: "numbl_jit_tMax",
  min: "numbl_jit_tMin",
  any: "numbl_jit_tAny",
  all: "numbl_jit_tAll",
  mean: "numbl_jit_tMean",
};

function formatNumberLiteral(v: number): string {
  if (!Number.isFinite(v)) {
    if (Number.isNaN(v)) return "(0.0/0.0)";
    return v > 0 ? "(1.0/0.0)" : "(-1.0/0.0)";
  }
  if (Number.isInteger(v)) return `${v}.0`;
  return `${v}`;
}

// ── Per-function emit context ──────────────────────────────────────────

interface EmitCtx {
  returnExpr: string;
  helpersNeeded: Set<string>;
  /** Names (un-mangled) of variables typed as tensor. Includes params,
   *  locals, and outputs. Used by the codegen to pick `napi_value` over
   *  `double` for declarations and to emit the right helper for ops. */
  tensorVars: Set<string>;
  /** Becomes true if the function ever needs `napi_env env`. Set when the
   *  body emits any `numbl_jit_*` call or any tensor-typed value flows
   *  through it. */
  needsEnv: boolean;
  /** Counter for inner-sub-expression scratch napi_value slots (__s1, __s2,
   *  ...). Each tensor sub-expression that doesn't have a top-level dest
   *  gets one. They're declared at the top of the function and reused
   *  across loop iterations — same as JS-JIT's allocScratch(). */
  scratchCount: number;
  /** Counter for `__t` for-loop step temps (legacy from MVP). */
  tmp: { n: number };
}

// ── Expression emission ────────────────────────────────────────────────

function isTensorExpr(expr: JitExpr): boolean {
  return expr.jitType.kind === "tensor";
}

function allocScratch(ctx: EmitCtx): string {
  ctx.scratchCount += 1;
  return `__s${ctx.scratchCount}`;
}

/** Emit `expr`, ensuring the result is a `napi_value`. Tensor expressions
 *  already produce napi_value; scalar expressions get wrapped in
 *  `numbl_jit_box_double(env, ...)`. Used whenever a scalar sub-expression
 *  feeds into a tensor helper (the numbl_jit_t* functions all take
 *  napi_value arguments, mirroring JS-JIT which passes raw JS numbers
 *  that its fastBinaryOp helper dispatches on). */
function emitAsNapiValue(expr: JitExpr, ctx: EmitCtx): string {
  const code = emitExpr(expr, ctx);
  if (isTensorExpr(expr)) return code;
  ctx.needsEnv = true;
  return `numbl_jit_box_double(env, ${code})`;
}

/** Emit a value-expression. For scalars, returns a C `double` expression.
 *  For tensors, returns a `napi_value` expression. */
function emitExpr(expr: JitExpr, ctx: EmitCtx): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return formatNumberLiteral(expr.value);

    case "Var":
      return mangle(expr.name);

    case "Binary":
      return emitBinary(expr, ctx);

    case "Unary":
      return emitUnary(expr, ctx);

    case "Call":
      return emitCall(expr, ctx);

    // All other expression tags are blocked by cFeasibility.
    default:
      throw new Error(`C-JIT codegen: unsupported expr ${expr.tag}`);
  }
}

function emitBinary(
  expr: JitExpr & { tag: "Binary" },
  ctx: EmitCtx,
  destName?: string
): string {
  // Tensor-result binary: route through the helper. Mirrors JS-JIT
  // emitTensorBinary exactly: top-level Assign passes destName; inner
  // sub-expressions get a fresh scratch slot. Scalar operands get boxed
  // into a napi_value (the JS side passes raw JS numbers — in C we have
  // to call napi_create_double explicitly).
  if (isTensorExpr(expr)) {
    const helper = TENSOR_BIN_HELPER[expr.op];
    if (!helper) {
      throw new Error(
        `C-JIT codegen: tensor-result binary op ${expr.op} has no helper`
      );
    }
    ctx.needsEnv = true;
    const left = emitAsNapiValue(expr.left, ctx);
    const right = emitAsNapiValue(expr.right, ctx);
    if (destName !== undefined) {
      const dest = mangle(destName);
      return `${helper}(env, ${dest}, ${left}, ${right})`;
    }
    const scratch = allocScratch(ctx);
    return `(${scratch} = ${helper}(env, ${scratch}, ${left}, ${right}))`;
  }

  // Scalar-result binary: same as MVP, but operands may themselves be
  // tensor sub-expressions if comparisons coerce — feasibility blocks
  // those. So this reduces to MVP's pure-scalar path.
  const l = emitExpr(expr.left, ctx);
  const r = emitExpr(expr.right, ctx);
  switch (expr.op) {
    case BinaryOperation.Add:
      return `(${l} + ${r})`;
    case BinaryOperation.Sub:
      return `(${l} - ${r})`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `(${l} * ${r})`;
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return `(${l} / ${r})`;
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return `pow(${l}, ${r})`;
    case BinaryOperation.Equal:
      return `(((double)((${l}) == (${r}))))`;
    case BinaryOperation.NotEqual:
      return `(((double)((${l}) != (${r}))))`;
    case BinaryOperation.Less:
      return `(((double)((${l}) < (${r}))))`;
    case BinaryOperation.LessEqual:
      return `(((double)((${l}) <= (${r}))))`;
    case BinaryOperation.Greater:
      return `(((double)((${l}) > (${r}))))`;
    case BinaryOperation.GreaterEqual:
      return `(((double)((${l}) >= (${r}))))`;
    case BinaryOperation.AndAnd:
      return `((double)(((${l}) != 0.0) && ((${r}) != 0.0)))`;
    case BinaryOperation.OrOr:
      return `((double)(((${l}) != 0.0) || ((${r}) != 0.0)))`;
    default:
      throw new Error(`C-JIT codegen: unsupported binary op ${expr.op}`);
  }
}

function emitUnary(
  expr: JitExpr & { tag: "Unary" },
  ctx: EmitCtx,
  destName?: string
): string {
  // Tensor-result unary: only Minus and Plus are reachable (cFeasibility
  // blocks Not on tensors).
  if (isTensorExpr(expr)) {
    const operand = emitExpr(expr.operand, ctx);
    if (expr.op === UnaryOperation.Plus) {
      // Plus is a no-op (mirrors JS-JIT, which returns operand directly).
      return operand;
    }
    if (expr.op === UnaryOperation.Minus) {
      ctx.needsEnv = true;
      if (destName !== undefined) {
        return `numbl_jit_tNeg(env, ${mangle(destName)}, ${operand})`;
      }
      const scratch = allocScratch(ctx);
      return `(${scratch} = numbl_jit_tNeg(env, ${scratch}, ${operand}))`;
    }
    throw new Error(`C-JIT codegen: tensor unary op ${expr.op} not handled`);
  }

  const operand = emitExpr(expr.operand, ctx);
  switch (expr.op) {
    case UnaryOperation.Plus:
      return `(+${operand})`;
    case UnaryOperation.Minus:
      return `(-${operand})`;
    case UnaryOperation.Not:
      return `((double)((${operand}) == 0.0))`;
    default:
      throw new Error(`C-JIT codegen: unsupported unary op ${expr.op}`);
  }
}

function emitCall(
  expr: JitExpr & { tag: "Call" },
  ctx: EmitCtx,
  destName?: string
): string {
  // Tensor unary builtin (exp, sin, cos, ..., abs).
  if (isTensorExpr(expr) && expr.name in C_TENSOR_UNARY_OPS) {
    const helper = TENSOR_UNARY_HELPER[expr.name];
    if (!helper) {
      throw new Error(
        `C-JIT codegen: tensor unary call ${expr.name} has no helper mapping`
      );
    }
    ctx.needsEnv = true;
    const arg = emitExpr(expr.args[0], ctx);
    if (destName !== undefined) {
      return `${helper}(env, ${mangle(destName)}, ${arg})`;
    }
    const scratch = allocScratch(ctx);
    return `(${scratch} = ${helper}(env, ${scratch}, ${arg}))`;
  }
  // Tensor reduction (sum, max, ...). Result is scalar (number/boolean).
  if (!isTensorExpr(expr) && expr.name in C_TENSOR_REDUCTION_OPS) {
    const helper = TENSOR_REDUCTION_HELPER[expr.name];
    if (!helper) {
      throw new Error(
        `C-JIT codegen: tensor reduction ${expr.name} has no helper mapping`
      );
    }
    ctx.needsEnv = true;
    const arg = emitExpr(expr.args[0], ctx);
    // Reduction returns a JS number (napi_value); we extract a double for
    // use in scalar arithmetic. The helper itself does shape/complex
    // checks and bails on unsupported inputs.
    return `numbl_jit_napi_to_double(env, ${helper}(env, ${arg}))`;
  }
  // Scalar math builtin: same as MVP.
  const cName = BUILTIN_TO_C[expr.name];
  if (!cName) {
    throw new Error(`C-JIT codegen: unmapped builtin ${expr.name}`);
  }
  if (expr.name in BUILTINS_NEEDING_HELPERS) {
    ctx.helpersNeeded.add(expr.name);
  }
  const args = expr.args.map(a => emitExpr(a, ctx));
  return `${cName}(${args.join(", ")})`;
}

function emitTruthiness(expr: JitExpr, ctx: EmitCtx): string {
  if (expr.tag === "Binary" && !isTensorExpr(expr)) {
    switch (expr.op) {
      case BinaryOperation.Equal:
        return `((${emitExpr(expr.left, ctx)}) == (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.NotEqual:
        return `((${emitExpr(expr.left, ctx)}) != (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.Less:
        return `((${emitExpr(expr.left, ctx)}) < (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.LessEqual:
        return `((${emitExpr(expr.left, ctx)}) <= (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.Greater:
        return `((${emitExpr(expr.left, ctx)}) > (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.GreaterEqual:
        return `((${emitExpr(expr.left, ctx)}) >= (${emitExpr(expr.right, ctx)}))`;
      case BinaryOperation.AndAnd:
        return `((${emitTruthiness(expr.left, ctx)}) && (${emitTruthiness(expr.right, ctx)}))`;
      case BinaryOperation.OrOr:
        return `((${emitTruthiness(expr.left, ctx)}) || (${emitTruthiness(expr.right, ctx)}))`;
      default:
        break;
    }
  }
  if (
    expr.tag === "Unary" &&
    expr.op === UnaryOperation.Not &&
    !isTensorExpr(expr.operand)
  ) {
    return `(!(${emitTruthiness(expr.operand, ctx)}))`;
  }
  return `((${emitExpr(expr, ctx)}) != 0.0)`;
}

// ── Statement emission ─────────────────────────────────────────────────

function emitStmts(
  lines: string[],
  stmts: JitStmt[],
  indent: string,
  ctx: EmitCtx
): void {
  for (const s of stmts) emitStmt(lines, s, indent, ctx);
}

/** Emit the RHS of an Assign with a destName hint propagated to the
 *  tensor-helper layer. The hint lets the top-level helper reuse the LHS
 *  variable's existing buffer when uniquely owned (mirrors JS-JIT). */
function emitAssignRhs(expr: JitExpr, ctx: EmitCtx, destName: string): string {
  if (expr.tag === "Binary") return emitBinary(expr, ctx, destName);
  if (expr.tag === "Unary") return emitUnary(expr, ctx, destName);
  if (expr.tag === "Call") return emitCall(expr, ctx, destName);
  return emitExpr(expr, ctx);
}

function emitStmt(
  lines: string[],
  stmt: JitStmt,
  indent: string,
  ctx: EmitCtx
): void {
  switch (stmt.tag) {
    case "Assign": {
      const rhs = emitAssignRhs(stmt.expr, ctx, stmt.name);
      lines.push(`${indent}${mangle(stmt.name)} = ${rhs};`);
      return;
    }

    case "ExprStmt":
      lines.push(`${indent}(void)(${emitExpr(stmt.expr, ctx)});`);
      return;

    case "If": {
      lines.push(`${indent}if (${emitTruthiness(stmt.cond, ctx)}) {`);
      emitStmts(lines, stmt.thenBody, indent + "  ", ctx);
      for (const eib of stmt.elseifBlocks) {
        lines.push(`${indent}} else if (${emitTruthiness(eib.cond, ctx)}) {`);
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
      const start = emitExpr(stmt.start, ctx);
      const end = emitExpr(stmt.end, ctx);
      const step = stmt.step ? emitExpr(stmt.step, ctx) : "1.0";
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
      lines.push(`${indent}return ${ctx.returnExpr};`);
      return;

    case "SetLoc":
      // Line tracking is not plumbed into the C codegen — runtime $rt.$line
      // is simply not updated while we're inside C-JITed code. Acceptable
      // because errors from pure C scalar code are rare; tensor-helper
      // bails come back through the JitBailToInterpreter sentinel which
      // the interpreter's slow path then attributes correctly.
      return;

    // All other stmt tags are blocked by cFeasibility.
    default:
      throw new Error(`C-JIT codegen: unsupported stmt ${stmt.tag}`);
  }
}

// ── Top-level: tensor-var discovery + signature shaping ────────────────

/** Walk the body to find every variable that is ever assigned a tensor-
 *  typed value. Mirrors what JS-JIT discovers via lowering's type
 *  propagation. */
function findTensorLocals(body: JitStmt[], out: Set<string>): void {
  const visitStmt = (s: JitStmt): void => {
    switch (s.tag) {
      case "Assign":
        if (s.expr.jitType.kind === "tensor") out.add(s.name);
        break;
      case "If":
        s.thenBody.forEach(visitStmt);
        s.elseifBlocks.forEach(eb => eb.body.forEach(visitStmt));
        if (s.elseBody) s.elseBody.forEach(visitStmt);
        break;
      case "For":
        s.body.forEach(visitStmt);
        break;
      case "While":
        s.body.forEach(visitStmt);
        break;
      default:
        break;
    }
  };
  body.forEach(visitStmt);
}

/** Per-param descriptor for the generated C function signature. The shim
 *  reads this to know how to marshal each JS-side argument. */
export interface CParamDesc {
  name: string;
  kind: "scalar" | "tensor";
}

/** Result of generating C code. */
export interface GenerateCResult {
  /** Full C source: includes, helpers, user function. (No N-API shim.) */
  cSource: string;
  /** Name of the generated C function. */
  cFnName: string;
  /** Names of helper functions actually emitted (for debug/testing). */
  helpersUsed: string[];
  /** Per-param marshalling info for the shim generator. */
  paramDescs: CParamDesc[];
  /** True when the function returns a tensor (napi_value) vs a scalar (double). */
  returnIsTensor: boolean;
  /** True when the function takes `napi_env env` as its first parameter
   *  and includes the tensor helper block. */
  usesTensors: boolean;
}

/**
 * Generate a standalone C function (no N-API shim) for the given lowered IR.
 *
 * `argTypes` is needed to detect tensor-typed params; the signature
 * differs by param kind (`double` vs `napi_value`).
 */
export function generateC(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  argTypes: JitType[],
  outputType: JitType | null,
  fnName: string
): GenerateCResult {
  if (params.length !== argTypes.length) {
    throw new Error("C-JIT codegen: params/argTypes length mismatch");
  }

  // Tensor-typed variables: params + locals (locals discovered by walking
  // the body for tensor-typed Assign LHSs).
  const tensorVars = new Set<string>();
  for (let i = 0; i < params.length; i++) {
    if (argTypes[i].kind === "tensor") tensorVars.add(params[i]);
  }
  findTensorLocals(body, tensorVars);

  const returnIsTensor = !!(outputType && outputType.kind === "tensor");
  const effectiveOutputs = outputs.slice(0, nargout || 1);
  const returnVar = effectiveOutputs.length > 0 ? effectiveOutputs[0] : "";
  const returnExpr = returnVar
    ? mangle(returnVar)
    : returnIsTensor
      ? "NULL"
      : "0.0";

  const ctx: EmitCtx = {
    returnExpr,
    helpersNeeded: new Set(),
    tensorVars,
    needsEnv: returnIsTensor || tensorVars.size > 0,
    scratchCount: 0,
    tmp: { n: 0 },
  };

  const indent = "  ";
  const bodyLines: string[] = [];
  emitStmts(bodyLines, body, indent, ctx);
  bodyLines.push(`${indent}return ${ctx.returnExpr};`);

  // Build the prelude: scalar-double + napi_value local declarations,
  // then scratch slots.
  const paramSet = new Set(params);
  const allLocals = [...localVars].filter(v => !paramSet.has(v)).sort();

  const preludeLines: string[] = [];
  for (const local of allLocals) {
    if (tensorVars.has(local)) {
      // Outputs declared as tensor must live in napi_value; init to NULL
      // (the helper treats NULL dest as "alloc fresh", same as JS-JIT
      // treats undefined).
      preludeLines.push(`${indent}napi_value ${mangle(local)} = NULL;`);
    } else {
      preludeLines.push(`${indent}double ${mangle(local)} = 0.0;`);
    }
  }
  for (let i = 1; i <= ctx.scratchCount; i++) {
    // Persistent scratch: declared at function top so its previous value
    // (a RuntimeTensor with _rc==1) is available next time the helper is
    // called, enabling buffer reuse across loop iterations. Same shape as
    // JS-JIT's `var $s1, $s2, ...` declarations.
    preludeLines.push(`${indent}static __thread napi_value __s${i} = NULL;`);
  }

  // Build signature.
  const cFnName = `jit_${fnName}`;
  const sigParts: string[] = [];
  if (ctx.needsEnv) sigParts.push("napi_env env");
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    if (argTypes[i].kind === "tensor") {
      sigParts.push(`napi_value ${mangle(p)}`);
    } else {
      sigParts.push(`double ${mangle(p)}`);
    }
  }
  const paramList = sigParts.length > 0 ? sigParts.join(", ") : "void";
  const returnCType = returnIsTensor ? "napi_value" : "double";
  const signature = `${returnCType} ${cFnName}(${paramList})`;

  // Helper for scalar reductions: napi_value → double extraction. Emitted
  // unconditionally when needsEnv, since the codegen may have used it.
  const napiToDoubleHelper = ctx.needsEnv
    ? `static double numbl_jit_napi_to_double(napi_env env, napi_value v) {
  if (!v) return 0.0;
  double d = 0.0;
  napi_value n;
  if (napi_coerce_to_number(env, v, &n) != napi_ok) return 0.0;
  napi_get_value_double(env, n, &d);
  return d;
}
`
    : "";

  const helpersUsed: string[] = [];
  const helperBlocks: string[] = [];
  for (const h of ctx.helpersNeeded) {
    helperBlocks.push(BUILTINS_NEEDING_HELPERS[h]);
    helpersUsed.push(h);
  }

  const parts: string[] = [];
  parts.push(`/* JIT C: ${fnName}(${params.join(", ")}) */`);
  if (ctx.needsEnv) {
    // The tensor helper block already pulls in <node_api.h>, <math.h>,
    // <stdint.h>, <stdlib.h>, <string.h>, "numbl_ops.h", so we don't
    // need the bare <math.h> include the scalar path uses.
    parts.push(cJitTensorHelpersSource());
    parts.push("");
    parts.push(napiToDoubleHelper);
  } else {
    parts.push(`#include <math.h>`);
    parts.push("");
  }
  if (helperBlocks.length > 0) {
    parts.push(helperBlocks.join("\n\n"));
    parts.push("");
  }
  parts.push(`${signature} {`);
  parts.push(preludeLines.join("\n"));
  parts.push(bodyLines.join("\n"));
  parts.push(`}`);

  const paramDescs: CParamDesc[] = params.map((p, i) => ({
    name: p,
    kind: argTypes[i].kind === "tensor" ? "tensor" : "scalar",
  }));

  return {
    cSource: parts.join("\n"),
    cFnName,
    helpersUsed,
    paramDescs,
    returnIsTensor,
    usesTensors: ctx.needsEnv,
  };
}
