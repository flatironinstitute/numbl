/**
 * LValue lowering (AST → IR).
 *
 * Handles assignment targets like variables, member access, and indexing.
 */

import {
  type Expr as AstExpr,
  type LValue as AstLValue,
} from "../parser/index.js";
import { SemanticError } from "../lowering/errors.js";
import { IType } from "../lowering/itemTypes.js";
import { type IRExpr, type IRLValue } from "../lowering/nodes.js";
import { type LoweringContext } from "./loweringContext.js";
import { lowerExpr } from "./lowerExpr.js";

/**
 * Lower an LValue (assignment target) from AST to IR.
 */
export function lowerLValue(ctx: LoweringContext, lv: AstLValue): IRLValue {
  switch (lv.type) {
    case "Ignore":
      throw new SemanticError("Ignore (~) lvalue should be handled by caller");

    case "Var": {
      const vv = ctx.lookup(lv.name) ?? ctx.defineVariable(lv.name, undefined);
      return { type: "Var", variable: vv };
    }

    case "Member": {
      if (lv.base.type === "Ident") {
        const vv =
          ctx.lookup(lv.base.name) ??
          ctx.defineVariable(lv.base.name, undefined);
        const base: IRExpr = {
          kind: { type: "Var", variable: vv },
          span: lv.base.span,
        };
        return { type: "Member", base, name: lv.name };
      }
      // Chained member — auto-define root
      if (lv.base.type === "Member") {
        let root: AstExpr = lv.base;
        while (root.type === "Member") root = root.base;
        if (root.type === "Ident" && ctx.lookup(root.name) === null) {
          ctx.defineVariable(root.name, undefined);
        }
      }
      if (lv.base.type === "Index" || lv.base.type === "IndexCell") {
        autoDefineRootVar(ctx, lv.base);
      }
      const base = lowerExpr(ctx, lv.base);
      return { type: "Member", base, name: lv.name };
    }

    case "MemberDynamic": {
      let root: AstExpr = lv.base;
      while (root.type === "Member") root = root.base;
      if (root.type === "Ident" && ctx.lookup(root.name) === null) {
        ctx.defineVariable(root.name, IType.Unknown);
      }
      const base = lowerExpr(ctx, lv.base);
      const nameExpr = lowerExpr(ctx, lv.nameExpr);
      return { type: "MemberDynamic", base, nameExpr };
    }

    case "Index": {
      autoDefineRootVar(ctx, lv.base);
      const base = lowerExpr(ctx, lv.base);
      const indices = lv.indices.map(i => lowerExpr(ctx, i));
      return { type: "Index", base, indices };
    }

    case "IndexCell": {
      autoDefineRootVar(ctx, lv.base);
      const base = lowerExpr(ctx, lv.base);
      const indices = lv.indices.map(i => lowerExpr(ctx, i));
      return { type: "IndexCell", base, indices };
    }

    default:
      throw new SemanticError(`Unknown lvalue type: ${(lv as AstLValue).type}`);
  }
}

/**
 * Walk to root Ident and auto-define it if not in scope.
 */
function autoDefineRootVar(ctx: LoweringContext, expr: AstExpr): void {
  let root = expr;
  while (
    root.type === "Member" ||
    root.type === "Index" ||
    root.type === "IndexCell"
  ) {
    root = root.base;
  }
  if (root.type === "Ident" && ctx.lookup(root.name) === null) {
    ctx.defineVariable(root.name, undefined);
  }
}
