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
import {
  type JitType,
  type JitExpr,
  type JitStmt,
  unifyJitTypes,
  isScalarType,
  isTensorType,
  jitTypeKey,
  computeJitFnName,
  SCALAR_MATH,
} from "./jitTypes.js";
import { generateJS } from "./jitCodegen.js";

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
    default:
      return null;
  }

  // Determine result type based on operand types
  const anyComplex =
    left.kind === "complex" ||
    right.kind === "complex" ||
    left.kind === "complexTensor" ||
    right.kind === "complexTensor";
  const anyTensor = isTensorType(left) || isTensorType(right);

  if (anyTensor) {
    return { kind: anyComplex ? "complexTensor" : "realTensor" };
  }

  if (anyComplex) {
    return { kind: "complex" };
  }

  // Both are numbers
  if (left.kind === "number" && right.kind === "number") {
    const bothNonneg = !!left.nonneg && !!right.nonneg;
    switch (op) {
      case BinaryOperation.Add:
        return { kind: "number", nonneg: bothNonneg };
      case BinaryOperation.Sub:
        return { kind: "number" };
      case BinaryOperation.Mul:
      case BinaryOperation.ElemMul:
        return { kind: "number", nonneg: bothNonneg };
      default:
        return { kind: "number" };
    }
  }

  return null;
}

function unaryResultType(op: UnaryOperation, operand: JitType): JitType | null {
  switch (op) {
    case UnaryOperation.Plus:
      return operand;
    case UnaryOperation.Minus:
      if (operand.kind === "number") return { kind: "number" };
      if (operand.kind === "complex") return { kind: "complex" };
      if (operand.kind === "realTensor") return { kind: "realTensor" };
      if (operand.kind === "complexTensor") return { kind: "complexTensor" };
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
      env.set(name, { kind: "number", nonneg: true }); // default 0
      localVars.add(name);
    }
  }

  const sharedGeneratedFns = generatedFns ?? new Map<string, string>();
  const sharedInProgress = loweringInProgress ?? new Set<string>();

  const ctx: LowerCtx = {
    env,
    localVars,
    params: new Set(fn.params),
    interp,
    generatedFns: sharedGeneratedFns,
    loweringInProgress: sharedInProgress,
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

    case "ImagUnit":
      return { tag: "ImagLiteral", jitType: { kind: "complex" } };

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

      // Track ops that need $h helpers (tensor or complex)
      if (
        isTensorType(left.jitType) ||
        isTensorType(right.jitType) ||
        left.jitType.kind === "complex" ||
        right.jitType.kind === "complex"
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

      if (isTensorType(operand.jitType) || operand.jitType.kind === "complex")
        ctx._hasTensorOps = true;

      return { tag: "Unary", op: expr.op, operand, jitType: resultType };
    }

    case "FuncCall": {
      // First try scalar math builtins
      const entry = SCALAR_MATH[expr.name];
      if (entry && expr.args.length === entry.arity) {
        const args = expr.args.map(a => lowerExpr(ctx, a));
        if (args.some(a => a === null)) return null;
        const loweredArgs = args as JitExpr[];

        const resultType = entry.resultType(loweredArgs.map(a => a.jitType));
        if (resultType && resultType.kind !== "unknown") {
          if (loweredArgs.some(a => isTensorType(a.jitType)))
            ctx._hasTensorOps = true;

          return {
            tag: "Call",
            name: expr.name,
            args: loweredArgs,
            jitType: resultType,
          };
        }
      }

      // Try user function resolution
      return lowerUserFuncCall(ctx, expr);
    }

    default:
      return null; // unsupported expression
  }
}

// ── User function call resolution ───────────────────────────────────────

function lowerUserFuncCall(
  ctx: LowerCtx,
  expr: Expr & { type: "FuncCall" }
): JitExpr | null {
  const interp = ctx.interp;
  if (!interp) return null;

  // Lower arguments first to determine types
  const args = expr.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // Resolve the function using the same mechanism as the interpreter
  const calleeFn = resolveUserFunction(interp, expr.name);
  if (!calleeFn) return null;

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
    return { tag: "UserCall", jitName, args: loweredArgs, jitType: returnType };
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
      calleeResult.hasTensorOps
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

    return { tag: "UserCall", jitName, args: loweredArgs, jitType: returnType };
  } finally {
    ctx.loweringInProgress.delete(jitName);
  }
}

/** Resolve a function name to a FunctionDef using the interpreter's resolution. */
function resolveUserFunction(
  interp: Interpreter,
  name: string
): FunctionDef | null {
  // 1. Check nested functions (mirrors interpreter's callFunction priority)
  const nested = interp.env.getNestedFunction(name);
  if (nested) return nested.fn;

  // 2. Check main local functions
  const localFn = interp.mainLocalFunctions.get(name);
  if (localFn) return localFn;

  // 3. Resolve via function index (for workspace functions etc.)
  const callSite: CallSite = {
    file: interp.currentFile,
    ...(interp.currentClassName ? { className: interp.currentClassName } : {}),
    ...(interp.currentMethodName
      ? { methodName: interp.currentMethodName }
      : {}),
  };
  // Use empty argTypes for resolution - we just need the target identity
  const target = resolveFunction(name, [], callSite, interp.functionIndex);
  if (!target) return null;

  if (target.kind === "localFunction" && target.source.from === "main") {
    return interp.mainLocalFunctions.get(target.name) ?? null;
  }

  // Other target kinds (workspace, class, etc.) not supported yet
  return null;
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
