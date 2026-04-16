/**
 * JIT IR -> C code generation (scalar-only MVP).
 *
 * Sibling of [jitCodegen.ts](../jitCodegen.ts). Covers only the IR subset
 * gated by [cFeasibility.ts](./cFeasibility.ts): numbers/booleans, scalar
 * math, and control flow. Anything outside the whitelist should be blocked
 * by the feasibility prepass before this function is called.
 *
 * Every scalar (number or boolean) is represented as a C `double`. Booleans
 * flow as 0.0 / 1.0 so they compose with numeric arithmetic cleanly, which
 * mirrors how the JS path lets `true`/`false` coerce via `+x`. Truthiness
 * in `if`/`while`/&&/|| is `(x != 0.0)`.
 */

import { BinaryOperation, UnaryOperation } from "../../../parser/types.js";
import type { JitExpr, JitStmt } from "../jitTypes.js";

// All user-visible identifiers get a prefix so they can never collide with
// C keywords, libc names (`sin`, `cos`, ...) or our internal locals.
const MANGLE_PREFIX = "v_";

function mangle(name: string): string {
  return `${MANGLE_PREFIX}${name}`;
}

// Scalar MATLAB builtin name → C expression form. For single-arg forms the
// C name is applied to the emitted operand(s); for multi-arg forms the
// caller stringifies the arg list and wraps with the function name.
//
// MATLAB semantics diverge from libc for:
//   - `abs` on real scalars (libc `abs` is int-only; use `fabs`)
//   - `fix` truncates toward zero (libc `trunc`)
//   - `mod(a,b)` is MATLAB-specific (differs from libc `fmod` for signed b,
//     handles b==0 specially) — emit via a local static helper
//   - `rem(a,b)` follows libc `fmod` semantics, so map to `fmod`
//   - `sign` is defined inline via a static helper (avoid copysign subtlety)
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

function formatNumberLiteral(v: number): string {
  if (!Number.isFinite(v)) {
    if (Number.isNaN(v)) return "(0.0/0.0)";
    return v > 0 ? "(1.0/0.0)" : "(-1.0/0.0)";
  }
  if (Number.isInteger(v)) return `${v}.0`;
  return `${v}`;
}

interface EmitCtx {
  returnExpr: string;
  helpersNeeded: Set<string>;
}

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

function emitBinary(expr: JitExpr & { tag: "Binary" }, ctx: EmitCtx): string {
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

function emitUnary(expr: JitExpr & { tag: "Unary" }, ctx: EmitCtx): string {
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

function emitCall(expr: JitExpr & { tag: "Call" }, ctx: EmitCtx): string {
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
  if (expr.tag === "Binary") {
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
  if (expr.tag === "Unary" && expr.op === UnaryOperation.Not) {
    return `(!(${emitTruthiness(expr.operand, ctx)}))`;
  }
  return `((${emitExpr(expr, ctx)}) != 0.0)`;
}

function emitStmts(
  lines: string[],
  stmts: JitStmt[],
  indent: string,
  ctx: EmitCtx,
  tmpCounter: { n: number }
): void {
  for (const s of stmts) emitStmt(lines, s, indent, ctx, tmpCounter);
}

function emitStmt(
  lines: string[],
  stmt: JitStmt,
  indent: string,
  ctx: EmitCtx,
  tmpCounter: { n: number }
): void {
  switch (stmt.tag) {
    case "Assign":
      lines.push(
        `${indent}${mangle(stmt.name)} = ${emitExpr(stmt.expr, ctx)};`
      );
      return;

    case "ExprStmt":
      lines.push(`${indent}(void)(${emitExpr(stmt.expr, ctx)});`);
      return;

    case "If": {
      lines.push(`${indent}if (${emitTruthiness(stmt.cond, ctx)}) {`);
      emitStmts(lines, stmt.thenBody, indent + "  ", ctx, tmpCounter);
      for (const eib of stmt.elseifBlocks) {
        lines.push(`${indent}} else if (${emitTruthiness(eib.cond, ctx)}) {`);
        emitStmts(lines, eib.body, indent + "  ", ctx, tmpCounter);
      }
      if (stmt.elseBody) {
        lines.push(`${indent}} else {`);
        emitStmts(lines, stmt.elseBody, indent + "  ", ctx, tmpCounter);
      }
      lines.push(`${indent}}`);
      return;
    }

    case "For": {
      const v = mangle(stmt.varName);
      const t = `__t${++tmpCounter.n}`;
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
      emitStmts(lines, stmt.body, indent + "  ", ctx, tmpCounter);
      lines.push(`${indent}}`);
      return;
    }

    case "While":
      lines.push(`${indent}while (${emitTruthiness(stmt.cond, ctx)}) {`);
      emitStmts(lines, stmt.body, indent + "  ", ctx, tmpCounter);
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
      // Line tracking is not plumbed into the C MVP — runtime $rt.$line is
      // simply not updated while we're inside C-JITed code. Acceptable
      // because errors from pure C scalar code are rare (divide-by-zero
      // yields Inf, not a trap). Re-enable if/when callbacks to JS are
      // added for richer builtins.
      return;

    // All other stmt tags are blocked by cFeasibility.
    default:
      throw new Error(`C-JIT codegen: unsupported stmt ${stmt.tag}`);
  }
}

/** Result of generating C code. */
export interface GenerateCResult {
  /** Full C source: includes, helpers, user function. (No N-API shim — that's added separately.) */
  cSource: string;
  /** Name of the generated C function. */
  cFnName: string;
  /** Names of helper functions actually emitted (for debug/testing). */
  helpersUsed: string[];
}

/**
 * Generate a standalone C function (no N-API shim) for the given lowered IR.
 */
export function generateC(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  fnName: string
): GenerateCResult {
  const effectiveOutputs = outputs.slice(0, nargout || 1);
  const returnExpr =
    effectiveOutputs.length > 0 ? mangle(effectiveOutputs[0]) : "0.0";

  const ctx: EmitCtx = { returnExpr, helpersNeeded: new Set() };

  const indent = "  ";
  const lines: string[] = [];

  // Declare locals (excluding params — params are in the function signature).
  const paramSet = new Set(params);
  const locals = [...localVars].filter(v => !paramSet.has(v));
  // Initialize locals to 0.0 for determinism (MATLAB-style undefined-var
  // reads already get caught by lowering, but an uninit C local is UB).
  for (const local of locals.sort()) {
    lines.push(`${indent}double ${mangle(local)} = 0.0;`);
  }

  const tmpCounter = { n: 0 };
  emitStmts(lines, body, indent, ctx, tmpCounter);

  lines.push(`${indent}return ${returnExpr};`);

  const cFnName = `jit_${fnName}`;
  const paramList =
    params.length > 0
      ? params.map(p => `double ${mangle(p)}`).join(", ")
      : "void";
  const signature = `double ${cFnName}(${paramList})`;

  const helpersUsed: string[] = [];
  const helperBlocks: string[] = [];
  for (const h of ctx.helpersNeeded) {
    helperBlocks.push(BUILTINS_NEEDING_HELPERS[h]);
    helpersUsed.push(h);
  }

  const parts: string[] = [];
  parts.push(`/* JIT C: ${fnName}(${params.join(", ")}) */`);
  parts.push(`#include <math.h>`);
  parts.push("");
  if (helperBlocks.length > 0) {
    parts.push(helperBlocks.join("\n\n"));
    parts.push("");
  }
  parts.push(`${signature} {`);
  parts.push(lines.join("\n"));
  parts.push(`}`);

  return {
    cSource: parts.join("\n"),
    cFnName,
    helpersUsed,
  };
}
