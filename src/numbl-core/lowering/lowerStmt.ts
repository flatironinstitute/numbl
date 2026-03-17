/**
 * Statement and function lowering (AST → IR).
 *
 * Handles all statement types including function definitions.
 */

import { type Stmt as AstStmt } from "../parser/index.js";
import { SemanticError } from "../lowering/errors.js";
import { type ItemType, IType } from "../lowering/itemTypes.js";
import {
  type IRExprKind,
  type IRLValue,
  type IRStmt,
} from "../lowering/nodes.js";
import { itemTypeForExprKind } from "../lowering/nodeUtils.js";
import { lowerArgumentsBlocks } from "../lowering/lowerArguments.js";
import { type LoweringContext } from "./loweringContext.js";
import { lowerExpr } from "./lowerExpr.js";
import { lowerLValue } from "./lowerLValue.js";
import { preDefineBodyVars } from "./loweringHelpers.js";
import { isScalarType } from "./itemTypes.js";
import { type TypeEnv } from "./typeEnv.js";

// ── Public API ──────────────────────────────────────────────────────────

export function lowerStmts(ctx: LoweringContext, stmts: AstStmt[]): IRStmt[] {
  return stmts.map(s => lowerStmt(ctx, s));
}

export function lowerStmt(ctx: LoweringContext, stmt: AstStmt): IRStmt {
  const span = stmt.span;

  switch (stmt.type) {
    case "ExprStmt": {
      // 'clear' and 'clf' are no-ops — drop args
      const exprToLower =
        stmt.expr.type === "FuncCall" &&
        ["clear", "clf"].includes(stmt.expr.name)
          ? { ...stmt.expr, args: [] }
          : stmt.expr;
      return {
        type: "ExprStmt",
        expr: lowerExpr(ctx, exprToLower, stmt.suppressed ? 0 : 1),
        suppressed: stmt.suppressed,
        span,
      };
    }

    case "Assign": {
      const value = lowerExpr(ctx, stmt.expr, 1);
      let variable = ctx.lookup(stmt.name);
      const rhsType = itemTypeForExprKind(value.kind, ctx.typeEnv);
      if (variable) {
        ctx.typeEnv.set(variable.id, rhsType);
      } else {
        variable = ctx.defineVariable(stmt.name, rhsType);
      }
      return {
        type: "Assign",
        variable,
        expr: value,
        assignedType: rhsType,
        suppressed: stmt.suppressed,
        span,
      };
    }

    case "MultiAssign": {
      const value = lowerExpr(ctx, stmt.expr, stmt.lvalues.length);
      const valueItemType = itemTypeForExprKind(value.kind, ctx.typeEnv);
      let outputTypes = stmt.lvalues.map(
        () => ({ kind: "Unknown" }) as ItemType
      );
      if (valueItemType.kind === "MultipleOutputs") {
        if (valueItemType.outputTypes.length !== stmt.lvalues.length) {
          throw new SemanticError(
            `Expected ${stmt.lvalues.length} outputs, got ${valueItemType.outputTypes.length}`,
            span
          );
        }
        outputTypes = valueItemType.outputTypes;
      }

      const irLvalues: (IRLValue | null)[] = stmt.lvalues.map((lv, i) => {
        if (lv.type === "Ignore") return null;
        if (lv.type === "Var") {
          const vv = ctx.lookup(lv.name);
          if (vv) {
            ctx.typeEnv.set(vv.id, outputTypes[i]);
            return { type: "Var" as const, variable: vv };
          } else {
            return {
              type: "Var" as const,
              variable: ctx.defineVariable(lv.name, outputTypes[i]),
            };
          }
        }
        return lowerLValue(ctx, lv);
      });

      // Build assignedTypes: type per lvalue (null for ignored or non-Var)
      const assignedTypes: (ItemType | null)[] = stmt.lvalues.map((lv, i) => {
        if (lv.type === "Ignore") return null;
        if (lv.type === "Var") return outputTypes[i];
        return null;
      });

      return {
        type: "MultiAssign",
        lvalues: irLvalues,
        expr: value,
        assignedTypes,
        suppressed: stmt.suppressed,
        span,
      };
    }

    case "AssignLValue": {
      const irLv = lowerLValue(ctx, stmt.lvalue);
      const value = lowerExpr(ctx, stmt.expr, 1);

      // If target is a plain variable, update its type from RHS
      if (irLv.type === "Var") {
        const rhsType = itemTypeForExprKind(value.kind, ctx.typeEnv);
        ctx.typeEnv.set(irLv.variable.id, rhsType);
        return {
          type: "Assign",
          variable: irLv.variable,
          expr: value,
          assignedType: rhsType,
          suppressed: stmt.suppressed,
          span,
        };
      }
      // Member assignment (e.g. s.x = val, s.a.b = val): update the root
      // variable's Struct type to include the assigned field and its type.
      // For chained access like s.a.b = val, builds nested Struct types.
      // Skip for ClassInstance — property assignments don't change the class type.
      if (irLv.type === "Member") {
        const fieldType = itemTypeForExprKind(value.kind, ctx.typeEnv);
        updateStructTypeForMemberAssign(ctx, irLv, fieldType);
      }
      // Indexed assignment (e.g. X(i,j)=val, X{i}=val) can change the
      // runtime type of the base variable (e.g. scalar → tensor via
      // auto-grow).  Widen scalar static types to Unknown so codegen does
      // not apply scalar-only fast-paths (transpose elision, raw JS
      // arithmetic, native math inlining, skipped $rt.share) that would
      // produce wrong results at runtime.
      if (
        (irLv.type === "Index" || irLv.type === "IndexCell") &&
        irLv.base.kind.type === "Var"
      ) {
        const v = irLv.base.kind.variable;
        const vTy = ctx.typeEnv.get(v.id);
        if (vTy && isScalarType(vTy)) {
          ctx.typeEnv.set(v.id, IType.Unknown);
        }
      }
      return {
        type: "AssignLValue",
        lvalue: irLv,
        expr: value,
        suppressed: stmt.suppressed,
        span,
      };
    }

    case "If": {
      const cond = lowerExpr(ctx, stmt.cond);

      // Collect variables assigned in any branch for type join
      const ifAssigned = new Set<string>();
      preCollectAssignedVarIds(ctx, stmt.thenBody, ifAssigned);
      for (const b of stmt.elseifBlocks)
        preCollectAssignedVarIds(ctx, b.body, ifAssigned);
      if (stmt.elseBody)
        preCollectAssignedVarIds(ctx, stmt.elseBody, ifAssigned);

      // Save pre-branch types
      const preBranchTypes = snapshotTypes(ctx.typeEnv, ifAssigned);

      // Lower each branch, capturing post-branch types
      const thenBody = lowerStmts(ctx, stmt.thenBody);
      const postThen = snapshotTypes(ctx.typeEnv, ifAssigned);
      restoreTypes(ctx.typeEnv, preBranchTypes);

      const elseifBlocks = stmt.elseifBlocks.map(b => {
        const result = {
          cond: lowerExpr(ctx, b.cond),
          body: lowerStmts(ctx, b.body),
        };
        const postElseif = snapshotTypes(ctx.typeEnv, ifAssigned);
        restoreTypes(ctx.typeEnv, preBranchTypes);
        return { ...result, postTypes: postElseif };
      });

      const elseBody = stmt.elseBody ? lowerStmts(ctx, stmt.elseBody) : null;
      const postElse = elseBody ? snapshotTypes(ctx.typeEnv, ifAssigned) : null;
      restoreTypes(ctx.typeEnv, preBranchTypes);

      // Join: unify types across all branches (and pre-branch if no else)
      const allBranchTypes = [postThen, ...elseifBlocks.map(b => b.postTypes)];
      if (postElse) allBranchTypes.push(postElse);
      joinBranchTypes(
        ctx.typeEnv,
        ifAssigned,
        preBranchTypes,
        allBranchTypes,
        !elseBody
      );

      return {
        type: "If",
        cond,
        thenBody,
        elseifBlocks: elseifBlocks.map(b => ({ cond: b.cond, body: b.body })),
        elseBody,
        span,
      };
    }

    case "While": {
      const cond = lowerExpr(ctx, stmt.cond);
      preDefineBodyVars(ctx, stmt.body, new Set());

      const whileAssigned = new Set<string>();
      preCollectAssignedVarIds(ctx, stmt.body, whileAssigned);
      const preLoopTypes = snapshotTypes(ctx.typeEnv, whileAssigned);

      const body = lowerStmts(ctx, stmt.body);
      const postBody = snapshotTypes(ctx.typeEnv, whileAssigned);
      restoreTypes(ctx.typeEnv, preLoopTypes);
      // Loop may not execute, so include pre-loop types
      joinBranchTypes(
        ctx.typeEnv,
        whileAssigned,
        preLoopTypes,
        [postBody],
        true
      );

      return { type: "While", cond, body, span };
    }

    case "For": {
      const expr = lowerExpr(ctx, stmt.expr);
      const exprItemType = itemTypeForExprKind(expr.kind, ctx.typeEnv);
      let varType: ItemType = { kind: "Unknown" };
      if (expr.kind.type === "Range") {
        varType = IType.Num;
      } else if (exprItemType.kind === "Tensor") {
        if (exprItemType.isComplex) {
          varType = IType.Complex;
        } else if (exprItemType.isLogical) {
          varType = IType.Bool;
        } else {
          varType = IType.Num;
        }
      } else if (exprItemType.kind === "Number") {
        varType = IType.Num;
      }
      const vv =
        ctx.lookup(stmt.varName) ?? ctx.defineVariable(stmt.varName, undefined);
      ctx.typeEnv.set(vv.id, varType);
      preDefineBodyVars(ctx, stmt.body, new Set([stmt.varName]));

      const forAssigned = new Set<string>();
      preCollectAssignedVarIds(ctx, stmt.body, forAssigned);
      const preLoopTypes = snapshotTypes(ctx.typeEnv, forAssigned);

      const body = lowerStmts(ctx, stmt.body);
      const postBody = snapshotTypes(ctx.typeEnv, forAssigned);
      restoreTypes(ctx.typeEnv, preLoopTypes);
      // Loop may not execute, so include pre-loop types
      joinBranchTypes(ctx.typeEnv, forAssigned, preLoopTypes, [postBody], true);

      return {
        type: "For",
        variable: vv,
        expr,
        body,
        iterVarType: varType,
        span,
      };
    }

    case "Switch": {
      const control = lowerExpr(ctx, stmt.expr);

      // Collect variables assigned in any case
      const switchAssigned = new Set<string>();
      for (const c of stmt.cases)
        preCollectAssignedVarIds(ctx, c.body, switchAssigned);
      if (stmt.otherwise)
        preCollectAssignedVarIds(ctx, stmt.otherwise, switchAssigned);

      const preBranchTypes = snapshotTypes(ctx.typeEnv, switchAssigned);

      const cases = stmt.cases.map(c => {
        const result = {
          value: lowerExpr(ctx, c.value),
          body: lowerStmts(ctx, c.body),
        };
        const postCase = snapshotTypes(ctx.typeEnv, switchAssigned);
        restoreTypes(ctx.typeEnv, preBranchTypes);
        return { ...result, postTypes: postCase };
      });

      const otherwise = stmt.otherwise ? lowerStmts(ctx, stmt.otherwise) : null;
      const postOtherwise = otherwise
        ? snapshotTypes(ctx.typeEnv, switchAssigned)
        : null;
      restoreTypes(ctx.typeEnv, preBranchTypes);

      const allBranchTypes = cases.map(c => c.postTypes);
      if (postOtherwise) allBranchTypes.push(postOtherwise);
      joinBranchTypes(
        ctx.typeEnv,
        switchAssigned,
        preBranchTypes,
        allBranchTypes,
        !otherwise
      );

      return {
        type: "Switch",
        expr: control,
        cases: cases.map(c => ({ value: c.value, body: c.body })),
        otherwise,
        span,
      };
    }

    case "TryCatch": {
      const tcAssigned = new Set<string>();
      preCollectAssignedVarIds(ctx, stmt.tryBody, tcAssigned);
      preCollectAssignedVarIds(ctx, stmt.catchBody, tcAssigned);
      const preBranchTypes = snapshotTypes(ctx.typeEnv, tcAssigned);

      const tryBody = lowerStmts(ctx, stmt.tryBody);
      const postTry = snapshotTypes(ctx.typeEnv, tcAssigned);
      restoreTypes(ctx.typeEnv, preBranchTypes);

      const catchVar = stmt.catchVar
        ? (ctx.lookup(stmt.catchVar) ??
          ctx.defineVariable(stmt.catchVar, IType.Unknown))
        : null;
      const catchBody = lowerStmts(ctx, stmt.catchBody);
      const postCatch = snapshotTypes(ctx.typeEnv, tcAssigned);
      restoreTypes(ctx.typeEnv, preBranchTypes);

      // Both branches can execute (try may throw at any point),
      // so include pre-branch types in the join.
      joinBranchTypes(
        ctx.typeEnv,
        tcAssigned,
        preBranchTypes,
        [postTry, postCatch],
        true
      );

      return { type: "TryCatch", tryBody, catchVar, catchBody, span };
    }

    case "Global": {
      const vars = stmt.names.map(n => {
        const variable = ctx.lookup(n) ?? ctx.defineVariable(n, undefined);
        return { variable, name: n };
      });
      return { type: "Global", vars, span };
    }

    case "Persistent": {
      const vars = stmt.names.map(n => {
        const variable = ctx.lookup(n) ?? ctx.defineVariable(n, undefined);
        return { variable, name: n };
      });
      return { type: "Persistent", vars, span };
    }

    case "Break":
      return { type: "Break", span };

    case "Continue":
      return { type: "Continue", span };

    case "Return":
      return { type: "Return", span };

    case "Function":
      return lowerFunction(ctx, stmt);

    case "Import":
      // Declarative; imports are collected by buildFunctionIndex()
      return {
        type: "ExprStmt",
        expr: { kind: { type: "Number", value: "0" }, span },
        suppressed: true,
        span,
      };

    case "ClassDef": {
      // Register the class with the lowering context (metadata, not executed code)
      ctx.registerLocalClass(stmt);
      return {
        type: "ExprStmt",
        expr: { kind: { type: "Number", value: "0" }, span },
        suppressed: true,
        span,
      };
    }

    default:
      throw new SemanticError(
        `Unknown statement type: ${(stmt as AstStmt).type}`,
        span
      );
  }
}

// ── Flow-dependent type helpers ─────────────────────────────────────────

/** Collect VarIds of all variables assigned in a list of IR statements (recursive into control flow). */
export function collectAssignedVarIds(stmts: IRStmt[], out: Set<string>): void {
  for (const s of stmts) {
    if (s.type === "Assign") out.add(s.variable.id.id);
    if (s.type === "MultiAssign") {
      for (const lv of s.lvalues) {
        if (lv?.type === "Var") out.add(lv.variable.id.id);
      }
    }
    if (s.type === "If") {
      collectAssignedVarIds(s.thenBody, out);
      for (const b of s.elseifBlocks) collectAssignedVarIds(b.body, out);
      if (s.elseBody) collectAssignedVarIds(s.elseBody, out);
    }
    if (s.type === "While" || s.type === "For")
      collectAssignedVarIds(s.body, out);
    if (s.type === "Switch") {
      for (const c of s.cases) collectAssignedVarIds(c.body, out);
      if (s.otherwise) collectAssignedVarIds(s.otherwise, out);
    }
    if (s.type === "TryCatch") {
      collectAssignedVarIds(s.tryBody, out);
      collectAssignedVarIds(s.catchBody, out);
    }
  }
}

/**
 * Pre-collect variable names assigned in AST statements (before lowering).
 * Uses LoweringContext.lookup to find VarIds for already-known variables.
 * Newly defined variables won't be found yet, but those don't need
 * snapshot/restore since their type starts as undefined.
 */
function preCollectAssignedVarIds(
  ctx: LoweringContext,
  stmts: AstStmt[],
  out: Set<string>
): void {
  for (const s of stmts) {
    if (s.type === "Assign") {
      const v = ctx.lookup(s.name);
      if (v) out.add(v.id.id);
    }
    if (s.type === "MultiAssign") {
      for (const lv of s.lvalues) {
        if (lv.type === "Var") {
          const v = ctx.lookup(lv.name);
          if (v) out.add(v.id.id);
        }
      }
    }
    if (s.type === "If") {
      preCollectAssignedVarIds(ctx, s.thenBody, out);
      for (const b of s.elseifBlocks)
        preCollectAssignedVarIds(ctx, b.body, out);
      if (s.elseBody) preCollectAssignedVarIds(ctx, s.elseBody, out);
    }
    if (s.type === "While" || s.type === "For")
      preCollectAssignedVarIds(ctx, s.body, out);
    if (s.type === "Switch") {
      for (const c of s.cases) preCollectAssignedVarIds(ctx, c.body, out);
      if (s.otherwise) preCollectAssignedVarIds(ctx, s.otherwise, out);
    }
    if (s.type === "TryCatch") {
      preCollectAssignedVarIds(ctx, s.tryBody, out);
      preCollectAssignedVarIds(ctx, s.catchBody, out);
    }
  }
}

/** Save current types for a set of variable IDs. */
function snapshotTypes(
  typeEnv: TypeEnv,
  varIds: Set<string>
): Map<string, ItemType | undefined> {
  const snap = new Map<string, ItemType | undefined>();
  for (const id of varIds) {
    snap.set(id, typeEnv.get({ id }));
  }
  return snap;
}

/** Restore types from a snapshot. */
function restoreTypes(
  typeEnv: TypeEnv,
  snap: Map<string, ItemType | undefined>
): void {
  for (const [id, ty] of snap) {
    if (ty !== undefined) {
      typeEnv.set({ id }, ty);
    }
  }
}

/**
 * Join branch types at a control flow merge point.
 * For each variable, unifies types from all branches.
 * If `includePreBranch` is true, the pre-branch type is also included
 * (for when not all paths assign, e.g. if without else).
 */
function joinBranchTypes(
  typeEnv: TypeEnv,
  varIds: Set<string>,
  preBranchTypes: Map<string, ItemType | undefined>,
  branchTypes: Map<string, ItemType | undefined>[],
  includePreBranch: boolean
): void {
  for (const id of varIds) {
    let joined: ItemType | undefined = includePreBranch
      ? preBranchTypes.get(id)
      : undefined;
    for (const bt of branchTypes) {
      const ty = bt.get(id);
      if (ty !== undefined) {
        joined = joined !== undefined ? IType.unify(joined, ty) : ty;
      }
    }
    if (joined !== undefined) {
      typeEnv.set({ id }, joined);
    }
  }
}

// ── Struct field type helpers ────────────────────────────────────────────

/**
 * Walk a Member lvalue chain to the root Var, building nested Struct types
 * from inside out. For s.a.b = val with fieldType Number:
 *   innermost: Struct<b: Number>
 *   next:      Struct<a: Struct<b: Number>>
 *   root var s gets unified with that.
 */
function updateStructTypeForMemberAssign(
  ctx: LoweringContext,
  lv: IRLValue & { type: "Member" },
  fieldType: ItemType
): void {
  // Collect the chain of field names from outermost to innermost
  const chain: string[] = [lv.name];
  let cursor: IRExprKind = lv.base.kind;
  while (cursor.type === "Member") {
    chain.unshift(cursor.name);
    cursor = cursor.base.kind;
  }
  if (cursor.type !== "Var") return;
  const v = cursor.variable;
  // Skip ClassInstance — property assignments don't change the class type
  const vTy = ctx.typeEnv.get(v.id);
  if (vTy && vTy.kind !== "Struct" && vTy.kind !== "Unknown") return;

  // Build nested Struct type from inside out
  let ty: ItemType = IType.struct({ [chain[chain.length - 1]]: fieldType });
  for (let i = chain.length - 2; i >= 0; i--) {
    ty = IType.struct({ [chain[i]]: ty });
  }
  if (!vTy || vTy.kind === "Struct") {
    ctx.typeEnv.unify(v.id, ty);
  }
}

// ── Function lowering ───────────────────────────────────────────────────

/**
 * Lower a function definition. Called by LoweringContext for on-demand lowering.
 */
export function lowerFunction(
  ctx: LoweringContext,
  stmt: AstStmt & { type: "Function" },
  paramTypes?: ItemType[]
): IRStmt & { type: "Function" } {
  // Determine if this is a true nested function (defined inside another
  // function body) vs a top-level subfunction. Nested functions share the
  // parent workspace via closure; subfunctions use isolated scope.
  // We detect nesting by scope depth: if we're deeper than root (scopes > 1),
  // we're inside a parent function and this is a nested function.
  const isNested = ctx.scopes.length > 1 && !stmt.isFileLocalSubfunction;
  if (isNested) {
    ctx.pushScope();
  } else {
    ctx.pushIsolatedScope();
  }

  const paramVars = stmt.params.map((p, i) => {
    if (p === "~") {
      return ctx.defineVariable(`$ignored${i}`, paramTypes?.[i] ?? undefined);
    } else if (p !== "varargin") {
      return ctx.defineVariable(p, paramTypes?.[i] ?? undefined);
    } else {
      if (i !== stmt.params.length - 1) {
        throw new SemanticError(
          "Only the last parameter can be 'varargin'",
          stmt.span
        );
      }
      return ctx.defineVariable(p, IType.cell("unknown"));
    }
  });

  // Build param name → type map so outputs matching a parameter inherit its type
  const paramTypeByName = new Map<string, ItemType | undefined>();
  for (let i = 0; i < stmt.params.length; i++) {
    paramTypeByName.set(stmt.params[i], paramTypes?.[i] ?? undefined);
  }

  const outputVars = stmt.outputs.map((o, i) => {
    if (o !== "varargout") {
      // If output name matches a parameter, inherit the parameter's type
      const ty = paramTypeByName.get(o);
      return ctx.defineVariable(o, ty);
    } else {
      if (i !== stmt.outputs.length - 1) {
        throw new SemanticError(
          "Only the last output can be 'varargout'",
          stmt.span
        );
      }
      return ctx.defineVariable(o, IType.cell("unknown"));
    }
  });

  const hasVarargin =
    stmt.params.length > 0 &&
    stmt.params[stmt.params.length - 1] === "varargin";
  const hasVarargout =
    stmt.outputs.length > 0 &&
    stmt.outputs[stmt.outputs.length - 1] === "varargout";

  // Lower arguments blocks
  const argumentsBlocks = lowerArgumentsBlocks(stmt.argumentsBlocks, expr =>
    lowerExpr(ctx, expr)
  );

  // Pre-register the function stub so recursive calls work
  const funcStmt: IRStmt & { type: "Function" } = {
    type: "Function",
    originalName: stmt.name,
    functionId: stmt.functionId,
    params: paramVars,
    outputs: outputVars,
    body: [],
    hasVarargin,
    hasVarargout,
    argumentsBlocks,
    span: stmt.span,
  };
  ctx.registerLoweredFunction(funcStmt);

  // Pre-register nested functions and pre-define body variables
  if (stmt.body.some(s => s.type === "Function")) {
    for (const s of stmt.body) {
      if (s.type === "Function") {
        ctx.registerLocalFunctionAST(s);
      }
    }
    preDefineBodyVars(
      ctx,
      stmt.body,
      new Set([...stmt.params, ...stmt.outputs])
    );
  }

  // Lower the body
  funcStmt.body = lowerStmts(ctx, stmt.body);

  // Snapshot output types for cross-context reads (e.g. returnTypeInference)
  funcStmt.outputTypes = outputVars.map(
    v => ctx.typeEnv.get(v.id) ?? IType.Unknown
  );

  ctx.popScope();

  return funcStmt;
}
