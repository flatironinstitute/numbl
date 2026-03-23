/**
 * AST -> JIT IR lowering with type propagation.
 *
 * Returns null if any unsupported construct is encountered,
 * causing the entire function to fall back to interpretation.
 */

import type { Expr, Stmt } from "../../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { FunctionDef } from "../types.js";
import {
  type JitType,
  type JitExpr,
  type JitStmt,
  unifyJitTypes,
  isScalarType,
  isTensorType,
  SCALAR_MATH,
} from "./jitTypes.js";

// ── Type Environment ────────────────────────────────────────────────────

type TypeEnv = Map<string, JitType>;

function cloneEnv(env: TypeEnv): TypeEnv {
  return new Map(env);
}

/** Merge two type environments at a join point. Returns null if any type becomes unknown. */
function mergeEnvs(a: TypeEnv, b: TypeEnv): TypeEnv | null {
  const result = cloneEnv(a);
  for (const [name, typeB] of b) {
    const typeA = result.get(name);
    if (typeA) {
      const unified = unifyJitTypes(typeA, typeB);
      if (unified.kind === "unknown") return null;
      result.set(name, unified);
    } else {
      result.set(name, typeB);
    }
  }
  return result;
}

// ── Type propagation for binary operations ──────────────────────────────

function binaryResultType(
  op: BinaryOperation,
  left: JitType,
  right: JitType
): JitType | null {
  // Comparisons always produce number(nonneg) for scalars
  if (
    op === BinaryOperation.Equal ||
    op === BinaryOperation.NotEqual ||
    op === BinaryOperation.Less ||
    op === BinaryOperation.LessEqual ||
    op === BinaryOperation.Greater ||
    op === BinaryOperation.GreaterEqual
  ) {
    if (isScalarType(left) && isScalarType(right))
      return { kind: "number", nonneg: true };
    // Tensor comparison: not supported yet
    return null;
  }

  // Logical: scalar only
  if (op === BinaryOperation.AndAnd || op === BinaryOperation.OrOr) {
    if (isScalarType(left) && isScalarType(right))
      return { kind: "number", nonneg: true };
    return null;
  }

  // Arithmetic
  if (left.kind === "number" && right.kind === "number") {
    return scalarArithResultType(op, left, right);
  }

  // Tensor + scalar or tensor + tensor (element-wise)
  if (isTensorType(left) || isTensorType(right)) {
    return tensorArithResultType(op, left, right);
  }

  return null;
}

function scalarArithResultType(
  op: BinaryOperation,
  left: JitType & { kind: "number" },
  right: JitType & { kind: "number" }
): JitType | null {
  const bothNonneg = !!left.nonneg && !!right.nonneg;
  switch (op) {
    case BinaryOperation.Add:
      return { kind: "number", nonneg: bothNonneg };
    case BinaryOperation.Sub:
      return { kind: "number" }; // subtraction can go negative
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return { kind: "number", nonneg: bothNonneg };
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return { kind: "number" };
    default:
      return null;
  }
}

function tensorArithResultType(
  op: BinaryOperation,
  left: JitType,
  right: JitType
): JitType | null {
  // Matrix multiply: not supported
  if (op === BinaryOperation.Mul) {
    if (isTensorType(left) && isTensorType(right)) return null;
    // scalar * tensor or tensor * scalar is OK (element-wise scale)
  }

  // Only element-wise ops on tensors
  switch (op) {
    case BinaryOperation.Add:
    case BinaryOperation.Sub:
    case BinaryOperation.Mul: // scalar * tensor case
    case BinaryOperation.ElemMul:
    case BinaryOperation.Div: // scalar / tensor or tensor / scalar
    case BinaryOperation.ElemDiv:
      break;
    default:
      return null; // Pow, LeftDiv, etc. not supported for tensors yet
  }

  // Determine if result is complex
  if (left.kind === "complexTensor" || right.kind === "complexTensor")
    return null; // complex tensor ops not supported yet
  if (left.kind === "complex" || right.kind === "complex") return null;

  return { kind: "realTensor" };
}

function unaryResultType(op: UnaryOperation, operand: JitType): JitType | null {
  switch (op) {
    case UnaryOperation.Plus:
      return operand;
    case UnaryOperation.Minus:
      if (operand.kind === "number") return { kind: "number" };
      if (operand.kind === "realTensor") return { kind: "realTensor" };
      return null;
    case UnaryOperation.Not:
      if (isScalarType(operand)) return { kind: "number", nonneg: true };
      return null;
    default:
      return null; // Transpose not supported
  }
}

// ── Lowering entry point ────────────────────────────────────────────────

export interface LoweringResult {
  body: JitStmt[];
  outputNames: string[];
  localVars: Set<string>;
  hasTensorOps: boolean;
}

export function lowerFunction(
  fn: FunctionDef,
  argTypes: JitType[],
  nargout: number
): LoweringResult | null {
  if (argTypes.length !== fn.params.length) return null;

  const env: TypeEnv = new Map();
  const localVars = new Set<string>();
  let hasTensorOps = false;

  // Initialize parameters
  for (let i = 0; i < fn.params.length; i++) {
    env.set(fn.params[i], argTypes[i]);
  }

  // Initialize output variables (default to number 0)
  const outputNames = fn.outputs.slice(0, nargout || 1);
  for (const name of outputNames) {
    if (!env.has(name)) {
      env.set(name, { kind: "number", nonneg: true }); // default 0
      localVars.add(name);
    }
  }

  const ctx: LowerCtx = { env, localVars, params: new Set(fn.params) };
  const body = lowerStmts(ctx, fn.body);
  if (!body) return null;

  hasTensorOps = ctx._hasTensorOps ?? false;

  return { body, outputNames, localVars: ctx.localVars, hasTensorOps };
}

// ── Internal lowering context ───────────────────────────────────────────

interface LowerCtx {
  env: TypeEnv;
  localVars: Set<string>;
  params: Set<string>;
  _hasTensorOps?: boolean;
}

// ── Statement lowering ──────────────────────────────────────────────────

function lowerStmts(ctx: LowerCtx, stmts: Stmt[]): JitStmt[] | null {
  const result: JitStmt[] = [];
  for (const stmt of stmts) {
    const lowered = lowerStmt(ctx, stmt);
    if (!lowered) return null;
    result.push(...lowered);
  }
  return result;
}

function lowerStmt(ctx: LowerCtx, stmt: Stmt): JitStmt[] | null {
  switch (stmt.type) {
    case "Assign":
      return lowerAssign(ctx, stmt);
    case "ExprStmt":
      if (!stmt.suppressed) return null; // unsuppressed display not supported
      return lowerExprStmt(ctx, stmt);
    case "If":
      return lowerIf(ctx, stmt);
    case "For":
      return lowerFor(ctx, stmt);
    case "While":
      return lowerWhile(ctx, stmt);
    case "Break":
      return [{ tag: "Break" }];
    case "Continue":
      return [{ tag: "Continue" }];
    case "Return":
      return [{ tag: "Return" }];
    default:
      return null; // unsupported statement
  }
}

function lowerAssign(
  ctx: LowerCtx,
  stmt: Stmt & { type: "Assign" }
): JitStmt[] | null {
  const expr = lowerExpr(ctx, stmt.expr);
  if (!expr) return null;

  ctx.env.set(stmt.name, expr.jitType);
  if (!ctx.params.has(stmt.name)) ctx.localVars.add(stmt.name);

  return [{ tag: "Assign", name: stmt.name, expr }];
}

function lowerExprStmt(
  ctx: LowerCtx,
  stmt: Stmt & { type: "ExprStmt" }
): JitStmt[] | null {
  const expr = lowerExpr(ctx, stmt.expr);
  if (!expr) return null;
  return [{ tag: "ExprStmt", expr }];
}

function lowerIf(ctx: LowerCtx, stmt: Stmt & { type: "If" }): JitStmt[] | null {
  const cond = lowerExpr(ctx, stmt.cond);
  if (!cond) return null;

  const envBefore = cloneEnv(ctx.env);

  // Then branch
  ctx.env = cloneEnv(envBefore);
  const thenBody = lowerStmts(ctx, stmt.thenBody);
  if (!thenBody) return null;
  let mergedEnv = cloneEnv(ctx.env);

  // Elseif branches
  const elseifBlocks: { cond: JitExpr; body: JitStmt[] }[] = [];
  for (const eib of stmt.elseifBlocks) {
    ctx.env = cloneEnv(envBefore);
    const eibCond = lowerExpr(ctx, eib.cond);
    if (!eibCond) return null;
    const eibBody = lowerStmts(ctx, eib.body);
    if (!eibBody) return null;
    elseifBlocks.push({ cond: eibCond, body: eibBody });

    const merged = mergeEnvs(mergedEnv, ctx.env);
    if (!merged) return null;
    mergedEnv = merged;
  }

  // Else branch
  let elseBody: JitStmt[] | null = null;
  if (stmt.elseBody) {
    ctx.env = cloneEnv(envBefore);
    elseBody = lowerStmts(ctx, stmt.elseBody);
    if (!elseBody) return null;

    const merged = mergeEnvs(mergedEnv, ctx.env);
    if (!merged) return null;
    mergedEnv = merged;
  } else {
    // No else: merge with pre-if env (variable might not be assigned)
    const merged = mergeEnvs(mergedEnv, envBefore);
    if (!merged) return null;
    mergedEnv = merged;
  }

  ctx.env = mergedEnv;

  return [{ tag: "If", cond, thenBody, elseifBlocks, elseBody }];
}

function lowerFor(
  ctx: LowerCtx,
  stmt: Stmt & { type: "For" }
): JitStmt[] | null {
  // Only Range-based for loops
  if (stmt.expr.type !== "Range") return null;

  const start = lowerExpr(ctx, stmt.expr.start);
  if (!start || !isScalarType(start.jitType)) return null;
  const step = stmt.expr.step ? lowerExpr(ctx, stmt.expr.step) : null;
  if (stmt.expr.step && (!step || !isScalarType(step!.jitType))) return null;
  const end = lowerExpr(ctx, stmt.expr.end);
  if (!end || !isScalarType(end.jitType)) return null;

  // Loop variable is always number
  ctx.env.set(stmt.varName, { kind: "number" });
  if (!ctx.params.has(stmt.varName)) ctx.localVars.add(stmt.varName);

  const envBefore = cloneEnv(ctx.env);

  // Lower body (first pass)
  const body = lowerStmts(ctx, stmt.body);
  if (!body) return null;

  // Merge pre-loop and post-body envs (loop might not execute)
  const merged = mergeEnvs(envBefore, ctx.env);
  if (!merged) return null;

  // Check if types changed - if so, re-lower with merged types
  let needRepass = false;
  for (const [name, type] of merged) {
    const before = envBefore.get(name);
    if (
      before &&
      (before.kind !== type.kind ||
        (before as { nonneg?: boolean }).nonneg !==
          (type as { nonneg?: boolean }).nonneg)
    ) {
      needRepass = true;
      break;
    }
  }

  let finalBody = body;
  if (needRepass) {
    ctx.env = cloneEnv(merged);
    ctx.env.set(stmt.varName, { kind: "number" });
    finalBody = lowerStmts(ctx, stmt.body)!;
    if (!finalBody) return null;

    const merged2 = mergeEnvs(merged, ctx.env);
    if (!merged2) return null;
    ctx.env = merged2;
  } else {
    ctx.env = merged;
  }

  return [
    {
      tag: "For",
      varName: stmt.varName,
      start,
      step,
      end,
      body: finalBody,
    },
  ];
}

function lowerWhile(
  ctx: LowerCtx,
  stmt: Stmt & { type: "While" }
): JitStmt[] | null {
  const envBefore = cloneEnv(ctx.env);

  const cond = lowerExpr(ctx, stmt.cond);
  if (!cond) return null;

  const body = lowerStmts(ctx, stmt.body);
  if (!body) return null;

  // Merge pre-loop and post-body
  const merged = mergeEnvs(envBefore, ctx.env);
  if (!merged) return null;

  // Re-lower if types changed
  let needRepass = false;
  for (const [name, type] of merged) {
    const before = envBefore.get(name);
    if (
      before &&
      (before.kind !== type.kind ||
        (before as { nonneg?: boolean }).nonneg !==
          (type as { nonneg?: boolean }).nonneg)
    ) {
      needRepass = true;
      break;
    }
  }

  let finalCond = cond;
  let finalBody = body;
  if (needRepass) {
    ctx.env = cloneEnv(merged);
    finalCond = lowerExpr(ctx, stmt.cond)!;
    if (!finalCond) return null;
    finalBody = lowerStmts(ctx, stmt.body)!;
    if (!finalBody) return null;

    const merged2 = mergeEnvs(merged, ctx.env);
    if (!merged2) return null;
    ctx.env = merged2;
  } else {
    ctx.env = merged;
  }

  return [{ tag: "While", cond: finalCond, body: finalBody }];
}

// ── Expression lowering ─────────────────────────────────────────────────

function lowerExpr(ctx: LowerCtx, expr: Expr): JitExpr | null {
  switch (expr.type) {
    case "Number": {
      const value = parseFloat(expr.value);
      return {
        tag: "NumberLiteral",
        value,
        jitType: { kind: "number", nonneg: value >= 0 },
      };
    }

    case "Ident": {
      const type = ctx.env.get(expr.name);
      if (!type) return null; // undefined variable
      return { tag: "Var", name: expr.name, jitType: type };
    }

    case "Binary": {
      const left = lowerExpr(ctx, expr.left);
      if (!left) return null;
      const right = lowerExpr(ctx, expr.right);
      if (!right) return null;
      const resultType = binaryResultType(expr.op, left.jitType, right.jitType);
      if (!resultType || resultType.kind === "unknown") return null;

      // Track tensor ops
      if (isTensorType(left.jitType) || isTensorType(right.jitType)) {
        ctx._hasTensorOps = true;
      }

      return { tag: "Binary", op: expr.op, left, right, jitType: resultType };
    }

    case "Unary": {
      const operand = lowerExpr(ctx, expr.operand);
      if (!operand) return null;
      const resultType = unaryResultType(expr.op, operand.jitType);
      if (!resultType || resultType.kind === "unknown") return null;

      if (isTensorType(operand.jitType)) ctx._hasTensorOps = true;

      return { tag: "Unary", op: expr.op, operand, jitType: resultType };
    }

    case "FuncCall": {
      const entry = SCALAR_MATH[expr.name];
      if (!entry) return null; // not a known math function
      if (expr.args.length !== entry.arity) return null;

      const args = expr.args.map(a => lowerExpr(ctx, a));
      if (args.some(a => a === null)) return null;
      const loweredArgs = args as JitExpr[];

      const resultType = entry.resultType(loweredArgs.map(a => a.jitType));
      if (!resultType || resultType.kind === "unknown") return null;

      // Track tensor ops
      if (loweredArgs.some(a => isTensorType(a.jitType)))
        ctx._hasTensorOps = true;

      return {
        tag: "Call",
        name: expr.name,
        args: loweredArgs,
        jitType: resultType,
      };
    }

    default:
      return null; // unsupported expression
  }
}
