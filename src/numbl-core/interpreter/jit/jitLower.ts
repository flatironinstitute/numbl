/**
 * AST -> JIT IR lowering with type propagation.
 *
 * Returns null if any unsupported construct is encountered,
 * causing the entire function to fall back to interpretation.
 */

import type { Expr, Stmt } from "../../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { FunctionDef } from "../types.js";
import type { Interpreter } from "../interpreter.js";
import type { CallSite } from "../../runtime/runtimeHelpers.js";
import { resolveFunction } from "../../functionResolve.js";
import type { ItemType } from "../../lowering/itemTypes.js";
import {
  type JitType,
  type JitExpr,
  type JitStmt,
  type SignCategory,
  unifyJitTypes,
  isScalarType,
  isNumericScalarType,
  isTensorType,
  isComplexType,
  isArithmeticType,
  jitTypeKey,
  computeJitFnName,
  signFromNumber,
  flipSign,
} from "./jitTypes.js";
import { generateJS } from "./jitCodegen.js";
import { getIBuiltin } from "../builtins/index.js";
import { offsetToLineFast, buildLineTable } from "../../runtime/error.js";

// ── Known constants ─────────────────────────────────────────────────────

const KNOWN_CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  inf: Infinity,
  Inf: Infinity,
  nan: NaN,
  NaN: NaN,
  eps: 2.220446049250313e-16,
  true: 1,
  false: 0,
};

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

/** Check if two type environments are identical (by JSON comparison). */
function envsEqual(a: TypeEnv, b: TypeEnv): boolean {
  if (a.size !== b.size) return false;
  for (const [name, type] of a) {
    const other = b.get(name);
    if (!other || JSON.stringify(type) !== JSON.stringify(other)) return false;
  }
  return true;
}

// ── Sign algebra ────────────────────────────────────────────────────────

function addSigns(a: SignCategory, b: SignCategory): SignCategory | undefined {
  if (a === "positive" && b === "positive") return "positive";
  if (a === "negative" && b === "negative") return "negative";
  if (
    (a === "nonneg" || a === "positive") &&
    (b === "nonneg" || b === "positive")
  )
    return "nonneg";
  if (
    (a === "nonpositive" || a === "negative") &&
    (b === "nonpositive" || b === "negative")
  )
    return "nonpositive";
  return undefined;
}

function mulSigns(a: SignCategory, b: SignCategory): SignCategory | undefined {
  // positive * positive -> positive, negative * negative -> positive
  const aPos = a === "positive" || a === "nonneg";
  const aNeg = a === "negative" || a === "nonpositive";
  const bPos = b === "positive" || b === "nonneg";
  const bNeg = b === "negative" || b === "nonpositive";
  const aStrict = a === "positive" || a === "negative";
  const bStrict = b === "positive" || b === "negative";

  if ((aPos && bPos) || (aNeg && bNeg)) {
    return aStrict && bStrict ? "positive" : "nonneg";
  }
  if ((aPos && bNeg) || (aNeg && bPos)) {
    return aStrict && bStrict ? "negative" : "nonpositive";
  }
  return undefined;
}

function combineSigns(
  a: SignCategory | undefined,
  b: SignCategory | undefined,
  op: BinaryOperation
): SignCategory | undefined {
  if (!a || !b) return undefined;
  switch (op) {
    case BinaryOperation.Add:
      return addSigns(a, b);
    case BinaryOperation.Sub:
      return addSigns(a, flipSign(b)!);
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return mulSigns(a, b);
    default:
      return undefined;
  }
}

// ── Type propagation for binary operations ──────────────────────────────

function binaryResultType(
  op: BinaryOperation,
  left: JitType,
  right: JitType
): JitType | null {
  // Reject non-arithmetic types (class_instance, struct, string, char, etc.)
  if (!isArithmeticType(left) || !isArithmeticType(right)) return null;

  // Comparisons always produce logical for scalars
  if (
    op === BinaryOperation.Equal ||
    op === BinaryOperation.NotEqual ||
    op === BinaryOperation.Less ||
    op === BinaryOperation.LessEqual ||
    op === BinaryOperation.Greater ||
    op === BinaryOperation.GreaterEqual
  ) {
    if (isScalarType(left) && isScalarType(right)) return { kind: "boolean" };
    // Tensor comparison: real tensors only (MATLAB errors on complex comparisons)
    const anyTensor = isTensorType(left) || isTensorType(right);
    const anyComplex = isComplexType(left) || isComplexType(right);
    if (anyTensor && !anyComplex) {
      const lt = isTensorType(left)
        ? (left as Extract<JitType, { kind: "tensor" }>)
        : undefined;
      const rt = isTensorType(right)
        ? (right as Extract<JitType, { kind: "tensor" }>)
        : undefined;
      const shape = lt?.shape ?? rt?.shape;
      const ndim = shape ? undefined : (lt?.ndim ?? rt?.ndim);
      return {
        kind: "tensor",
        isComplex: false,
        isLogical: true,
        ...(shape ? { shape } : {}),
        ...(ndim !== undefined ? { ndim } : {}),
      };
    }
    return null;
  }

  // Logical operators: scalar only
  if (op === BinaryOperation.AndAnd || op === BinaryOperation.OrOr) {
    if (isScalarType(left) && isScalarType(right)) return { kind: "boolean" };
    return null;
  }

  // Only element-wise arithmetic ops
  switch (op) {
    case BinaryOperation.Add:
    case BinaryOperation.Sub:
    case BinaryOperation.ElemMul:
    case BinaryOperation.ElemDiv:
      break;
    case BinaryOperation.Mul:
      // Matrix multiply (both tensors) not supported
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    case BinaryOperation.Div:
      // tensor / tensor not supported (use ./ instead)
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    case BinaryOperation.ElemPow:
      break;
    case BinaryOperation.Pow:
      // Matrix power (both tensors) not supported
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    default:
      return null;
  }

  // Coerce logical to number for arithmetic
  const effLeft: JitType =
    left.kind === "boolean" ? { kind: "number", sign: "nonneg" } : left;
  const effRight: JitType =
    right.kind === "boolean" ? { kind: "number", sign: "nonneg" } : right;

  // Determine result type based on operand types
  const anyComplex = isComplexType(effLeft) || isComplexType(effRight);
  const anyTensor = isTensorType(effLeft) || isTensorType(effRight);

  if (anyTensor) {
    const lt = isTensorType(effLeft)
      ? (effLeft as Extract<JitType, { kind: "tensor" }>)
      : undefined;
    const rt = isTensorType(effRight)
      ? (effRight as Extract<JitType, { kind: "tensor" }>)
      : undefined;
    const shape = lt?.shape ?? rt?.shape;
    const ndim = shape ? undefined : (lt?.ndim ?? rt?.ndim);
    const isComplex =
      anyComplex || (lt?.isComplex ?? false) || (rt?.isComplex ?? false);
    return {
      kind: "tensor",
      isComplex,
      ...(shape ? { shape } : {}),
      ...(ndim !== undefined ? { ndim } : {}),
    };
  }

  if (anyComplex) {
    return { kind: "complex_or_number" };
  }

  // Both are numbers (or coerced from logical)
  if (effLeft.kind === "number" && effRight.kind === "number") {
    const sign = combineSigns(effLeft.sign, effRight.sign, op);
    return { kind: "number", ...(sign ? { sign } : {}) };
  }

  return null;
}

function unaryResultType(op: UnaryOperation, operand: JitType): JitType | null {
  switch (op) {
    case UnaryOperation.Plus:
      return operand;
    case UnaryOperation.Minus:
      if (operand.kind === "number") {
        const sign = flipSign(operand.sign);
        return { kind: "number", ...(sign ? { sign } : {}) };
      }
      if (operand.kind === "boolean")
        return { kind: "number", sign: "nonpositive" };
      if (operand.kind === "complex_or_number")
        return { kind: "complex_or_number" };
      if (operand.kind === "tensor")
        return {
          kind: "tensor",
          isComplex: operand.isComplex,
          shape: operand.shape,
          ndim: operand.ndim,
        };
      return null;
    case UnaryOperation.Not:
      if (isNumericScalarType(operand)) return { kind: "boolean" };
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
  /** Generated JS code for called user functions: jitName → code */
  generatedFns: Map<string, string>;
  /** Type of the first output variable after lowering */
  outputType: JitType | null;
}

export function lowerFunction(
  fn: FunctionDef,
  argTypes: JitType[],
  nargout: number,
  interp?: Interpreter,
  generatedFns?: Map<string, string>,
  loweringInProgress?: Set<string>
): LoweringResult | null {
  if (argTypes.length !== fn.params.length) return null;

  const env: TypeEnv = new Map();
  const localVars = new Set<string>();

  // Initialize parameters
  for (let i = 0; i < fn.params.length; i++) {
    env.set(fn.params[i], argTypes[i]);
  }

  // Initialize output variables (default to number 0)
  const outputNames = fn.outputs.slice(0, nargout || 1);
  for (const name of outputNames) {
    if (!env.has(name)) {
      env.set(name, { kind: "number", exact: 0, sign: "nonneg" }); // default 0
      localVars.add(name);
    }
  }

  const sharedGeneratedFns = generatedFns ?? new Map<string, string>();
  const sharedInProgress = loweringInProgress ?? new Set<string>();

  // Build line table for offset→line lookup from file sources
  let lineTable: number[] | undefined;
  if (interp && fn.body.length > 0 && fn.body[0].span) {
    const file = fn.body[0].span.file;
    lineTable = interp.lineTableCache.get(file);
    if (!lineTable) {
      const src = interp.fileSources.get(file) ?? "";
      lineTable = buildLineTable(src);
      interp.lineTableCache.set(file, lineTable);
    }
  }

  const ctx: LowerCtx = {
    env,
    localVars,
    params: new Set(fn.params),
    interp,
    generatedFns: sharedGeneratedFns,
    loweringInProgress: sharedInProgress,
    lineTable,
  };
  const body = lowerStmts(ctx, fn.body);
  if (!body) return null;

  const outputType =
    outputNames.length > 0 ? (ctx.env.get(outputNames[0]) ?? null) : null;

  return {
    body,
    outputNames,
    localVars: ctx.localVars,
    hasTensorOps: ctx._hasTensorOps ?? false,
    generatedFns: sharedGeneratedFns,
    outputType,
  };
}

// ── Internal lowering context ───────────────────────────────────────────

interface LowerCtx {
  env: TypeEnv;
  localVars: Set<string>;
  params: Set<string>;
  _hasTensorOps?: boolean;
  interp?: Interpreter;
  generatedFns: Map<string, string>;
  loweringInProgress: Set<string>;
  /** Pre-built line break table for offset→line lookup. */
  lineTable?: number[];
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
  // Emit SetLoc for line tracking if span info is available
  const prefix: JitStmt[] = [];
  if (ctx.lineTable && stmt.span) {
    const line = offsetToLineFast(ctx.lineTable, stmt.span.start);
    prefix.push({ tag: "SetLoc", line });
  }

  let result: JitStmt[] | null;
  switch (stmt.type) {
    case "Assign":
      result = lowerAssign(ctx, stmt);
      break;
    case "ExprStmt":
      if (!stmt.suppressed) return null; // unsuppressed display not supported
      result = lowerExprStmt(ctx, stmt);
      break;
    case "If":
      result = lowerIf(ctx, stmt);
      break;
    case "For":
      result = lowerFor(ctx, stmt);
      break;
    case "While":
      result = lowerWhile(ctx, stmt);
      break;
    case "Break":
      result = [{ tag: "Break" }];
      break;
    case "Continue":
      result = [{ tag: "Continue" }];
      break;
    case "Return":
      result = [{ tag: "Return" }];
      break;
    case "MultiAssign":
      result = lowerMultiAssign(ctx, stmt);
      break;
    default:
      return null; // unsupported statement
  }
  if (!result) return null;
  return [...prefix, ...result];
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

function lowerMultiAssign(
  ctx: LowerCtx,
  stmt: Stmt & { type: "MultiAssign" }
): JitStmt[] | null {
  const nargout = stmt.lvalues.length;

  // Only support simple variable lvalues and ~ (Ignore)
  const names: (string | null)[] = [];
  for (const lv of stmt.lvalues) {
    if (lv.type === "Var") names.push(lv.name);
    else if (lv.type === "Ignore") names.push(null);
    else return null; // unsupported lvalue (index, member, etc.)
  }

  // RHS must be a FuncCall (either IBuiltin or user function)
  const rhs = stmt.expr;
  if (rhs.type !== "FuncCall") return null;

  // Lower the arguments
  const args = rhs.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // Try IBuiltin resolution with actual nargout
  const ib = getIBuiltin(rhs.name);
  if (!ib) return null; // user function multi-output not yet supported

  const resolution = ib.resolve(argJitTypes, nargout);
  if (!resolution || resolution.outputTypes.length < nargout) return null;
  const outputTypes = resolution.outputTypes;

  // Update type environment for each output variable
  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    if (name !== null) {
      ctx.env.set(name, outputTypes[i]);
      if (!ctx.params.has(name)) ctx.localVars.add(name);
    }
  }

  ctx._hasTensorOps = true;

  return [
    {
      tag: "MultiAssign",
      names,
      callName: rhs.name,
      args: loweredArgs,
      outputTypes,
    },
  ];
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
  // Only numeric scalar conditions supported
  if (!isScalarType(cond.jitType)) return null;
  if (cond.jitType.kind === "string" || cond.jitType.kind === "char")
    return null;
  if (cond.jitType.kind === "complex_or_number") ctx._hasTensorOps = true;

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
    if (!isScalarType(eibCond.jitType)) return null;
    if (eibCond.jitType.kind === "string" || eibCond.jitType.kind === "char")
      return null;
    if (eibCond.jitType.kind === "complex_or_number") ctx._hasTensorOps = true;
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
  if (!start || !isNumericScalarType(start.jitType)) return null;
  const step = stmt.expr.step ? lowerExpr(ctx, stmt.expr.step) : null;
  if (stmt.expr.step && (!step || !isNumericScalarType(step!.jitType)))
    return null;
  const end = lowerExpr(ctx, stmt.expr.end);
  if (!end || !isNumericScalarType(end.jitType)) return null;

  // Loop variable is always number
  ctx.env.set(stmt.varName, { kind: "number" });
  if (!ctx.params.has(stmt.varName)) ctx.localVars.add(stmt.varName);

  const envBefore = cloneEnv(ctx.env);

  // Lower body (first pass)
  const body = lowerStmts(ctx, stmt.body);
  if (!body) return null;

  // Merge pre-loop and post-body envs (loop might not execute)
  let merged = mergeEnvs(envBefore, ctx.env);
  if (!merged) return null;

  // Fixed-point iteration: re-lower until types stabilize (per-loop budget)
  let finalBody = body;
  let prevEnv = envBefore;
  let repassBudget = 20;
  while (!envsEqual(merged, prevEnv)) {
    if (repassBudget <= 0) return null;
    repassBudget--;
    ctx.env = cloneEnv(merged);
    ctx.env.set(stmt.varName, { kind: "number" });
    const newBody = lowerStmts(ctx, stmt.body);
    if (!newBody) return null;

    const newMerged = mergeEnvs(merged, ctx.env);
    if (!newMerged) return null;
    prevEnv = merged;
    merged = newMerged;
    finalBody = newBody;
  }

  ctx.env = merged;

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
  if (!isScalarType(cond.jitType)) return null;
  if (cond.jitType.kind === "string" || cond.jitType.kind === "char")
    return null;
  if (cond.jitType.kind === "complex_or_number") ctx._hasTensorOps = true;

  const body = lowerStmts(ctx, stmt.body);
  if (!body) return null;

  // Merge pre-loop and post-body
  let merged = mergeEnvs(envBefore, ctx.env);
  if (!merged) return null;

  // Fixed-point iteration: re-lower until types stabilize (per-loop budget)
  let finalCond = cond;
  let finalBody = body;
  let prevEnv = envBefore;
  let repassBudget = 20;
  while (!envsEqual(merged, prevEnv)) {
    if (repassBudget <= 0) return null;
    repassBudget--;
    ctx.env = cloneEnv(merged);
    const newCond = lowerExpr(ctx, stmt.cond);
    if (!newCond) return null;
    const newBody = lowerStmts(ctx, stmt.body);
    if (!newBody) return null;

    const newMerged = mergeEnvs(merged, ctx.env);
    if (!newMerged) return null;
    prevEnv = merged;
    merged = newMerged;
    finalCond = newCond;
    finalBody = newBody;
  }

  ctx.env = merged;

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
        jitType: {
          kind: "number",
          exact: value,
          ...(signFromNumber(value) ? { sign: signFromNumber(value) } : {}),
        },
      };
    }

    case "ImagUnit":
      return {
        tag: "ImagLiteral",
        jitType: { kind: "complex_or_number", pureImaginary: true },
      };

    case "Ident": {
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
      // If the name is a known variable, treat as indexing (MATLAB ambiguity)
      const varType = ctx.env.get(expr.name);
      if (varType) {
        return lowerIndexExpr(ctx, {
          base: { tag: "Var", name: expr.name, jitType: varType },
          indices: expr.args,
        });
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

    case "String":
      return {
        tag: "StringLiteral",
        value: expr.value,
        isChar: false,
        jitType: { kind: "string", value: expr.value },
      };

    case "Index": {
      const base = lowerExpr(ctx, expr.base);
      if (!base) return null;
      return lowerIndexExpr(ctx, { base, indices: expr.indices });
    }

    default:
      return null; // unsupported expression
  }
}

// ── Index expression lowering ───────────────────────────────────────────

function lowerIndexExpr(
  ctx: LowerCtx,
  input: { base: JitExpr; indices: Expr[] }
): JitExpr | null {
  const { base } = input;
  const indices: JitExpr[] = [];
  for (const idx of input.indices) {
    const lowered = lowerExpr(ctx, idx);
    if (!lowered) return null;
    // Only scalar numeric indices supported
    if (lowered.jitType.kind !== "number" && lowered.jitType.kind !== "boolean")
      return null;
    indices.push(lowered);
  }
  if (indices.length === 0) return null;

  // Determine result type based on base type
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
      ctx.loweringInProgress
    );
    if (!calleeResult) return null;

    // Generate JS for the callee and wrap in a named function
    const calleeJS = generateJS(
      calleeResult.body,
      calleeFn.params,
      calleeResult.outputNames,
      calleeNargout,
      calleeResult.localVars,
      interp.currentFile
    );
    const returnType = calleeResult.outputType ?? { kind: "number" as const };
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
    const wrappedJS = `${comment}\nfunction ${jitName}(${calleeFn.params.join(", ")}) {\n${calleeJS}\n}`;
    ctx.generatedFns.set(jitName, wrappedJS);

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

  // Other target kinds (privateFunction, workspaceClassConstructor, etc.) not supported yet
  return null;
}

// ── IBuiltin call resolution ────────────────────────────────────────────

function lowerIBuiltinCall(
  ctx: LowerCtx,
  expr: Expr & { type: "FuncCall" }
): JitExpr | null {
  const ib = getIBuiltin(expr.name);
  if (!ib) return null;

  const args = expr.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // If any argument is unknown, bail — it could be a class instance at runtime,
  // and class methods take priority over builtins in MATLAB.
  if (argJitTypes.some(t => t.kind === "unknown")) return null;

  const resolution = ib.resolve(argJitTypes, 1);
  if (!resolution || resolution.outputTypes.length === 0) return null;
  const outputTypes = resolution.outputTypes;

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
    ctx.loweringInProgress
  );
  return result?.outputType ?? null;
}
