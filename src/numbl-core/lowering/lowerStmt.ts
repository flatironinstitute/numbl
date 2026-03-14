/**
 * Statement and function lowering (AST → IR).
 *
 * Handles all statement types including function definitions.
 */

import { type Stmt as AstStmt } from "../parser/index.js";
import { SemanticError } from "../lowering/errors.js";
import { type ItemType, IType } from "../lowering/itemTypes.js";
import { type IRLValue, type IRStmt } from "../lowering/nodes.js";
import { itemTypeForExprKind } from "../lowering/nodeUtils.js";
import { lowerArgumentsBlocks } from "../lowering/lowerArguments.js";
import { type LoweringContext } from "./loweringContext.js";
import { lowerExpr } from "./lowerExpr.js";
import { lowerLValue } from "./lowerLValue.js";
import { preDefineBodyVars } from "./loweringHelpers.js";
import { isScalarType } from "./nodeUtils.js";

// Re-export lowerLValue for backwards compatibility
export { lowerLValue } from "./lowerLValue.js";

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
      if (variable) {
        variable.ty = IType.unify(variable.ty, itemTypeForExprKind(value.kind));
      } else {
        variable = ctx.defineVariable(
          stmt.name,
          itemTypeForExprKind(value.kind)
        );
      }
      return {
        type: "Assign",
        variable,
        expr: value,
        suppressed: stmt.suppressed,
        span,
      };
    }

    case "MultiAssign": {
      const value = lowerExpr(ctx, stmt.expr, stmt.lvalues.length);
      const valueItemType = itemTypeForExprKind(value.kind);
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
            vv.ty = IType.unify(vv.ty, outputTypes[i]);
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

      return {
        type: "MultiAssign",
        lvalues: irLvalues,
        expr: value,
        suppressed: stmt.suppressed,
        span,
      };
    }

    case "AssignLValue": {
      const irLv = lowerLValue(ctx, stmt.lvalue);
      const value = lowerExpr(ctx, stmt.expr, 1);

      // If target is a plain variable, update its type from RHS
      if (irLv.type === "Var") {
        irLv.variable.ty = IType.unify(
          irLv.variable.ty,
          itemTypeForExprKind(value.kind)
        );
        return {
          type: "Assign",
          variable: irLv.variable,
          expr: value,
          suppressed: stmt.suppressed,
          span,
        };
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
        if (v.ty && isScalarType(v.ty)) {
          v.ty = IType.Unknown;
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
      const thenBody = lowerStmts(ctx, stmt.thenBody);
      const elseifBlocks = stmt.elseifBlocks.map(b => ({
        cond: lowerExpr(ctx, b.cond),
        body: lowerStmts(ctx, b.body),
      }));
      const elseBody = stmt.elseBody ? lowerStmts(ctx, stmt.elseBody) : null;
      return { type: "If", cond, thenBody, elseifBlocks, elseBody, span };
    }

    case "While": {
      const cond = lowerExpr(ctx, stmt.cond);
      preDefineBodyVars(ctx, stmt.body, new Set());
      const body = lowerStmts(ctx, stmt.body);
      return { type: "While", cond, body, span };
    }

    case "For": {
      const expr = lowerExpr(ctx, stmt.expr);
      const exprItemType = itemTypeForExprKind(expr.kind);
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
      vv.ty = IType.unify(vv.ty, varType);
      preDefineBodyVars(ctx, stmt.body, new Set([stmt.varName]));
      const body = lowerStmts(ctx, stmt.body);
      return { type: "For", variable: vv, expr, body, span };
    }

    case "Switch": {
      const control = lowerExpr(ctx, stmt.expr);
      const cases = stmt.cases.map(c => ({
        value: lowerExpr(ctx, c.value),
        body: lowerStmts(ctx, c.body),
      }));
      const otherwise = stmt.otherwise ? lowerStmts(ctx, stmt.otherwise) : null;
      return { type: "Switch", expr: control, cases, otherwise, span };
    }

    case "TryCatch": {
      const tryBody = lowerStmts(ctx, stmt.tryBody);
      const catchVar = stmt.catchVar
        ? (ctx.lookup(stmt.catchVar) ??
          ctx.defineVariable(stmt.catchVar, IType.Unknown))
        : null;
      const catchBody = lowerStmts(ctx, stmt.catchBody);
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
  ctx.popScope();

  return funcStmt;
}
