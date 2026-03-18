/**
 * Expression lowering (AST → IR).
 */

import { type Expr as AstExpr } from "../parser/index.js";
import { SemanticError } from "../lowering/errors.js";
import { type ItemType, IType } from "./itemTypes.js";
import { type IRExpr } from "./nodes.js";
import { itemTypeForExprKind } from "./nodeUtils.js";
import { type LoweringContext } from "./loweringContext.js";
import {
  determineReturnType,
  determineMethodReturnType,
} from "./returnTypeInference.js";
import { resolveFunction } from "../functionResolve.js";
import type { CallSite } from "../runtime/runtimeHelpers.js";
import type { Span } from "../parser/index.js";

export type FunctionCandidate =
  | { type: "userFunction"; functionId: string }
  | { type: "builtin"; builtin: true };

// ── Shared function resolution ──────────────────────────────────────────

/**
 * Resolve a name to a function call IR node.
 * Delegates to the unified resolveFunction() from functionResolve.ts,
 * then converts the ResolvedTarget into the appropriate IR node.
 * Returns null if the name doesn't match any known function.
 */
/** Build a FuncCall IR node with return type inference. */
function makeFuncCallExpr(
  ctx: LoweringContext,
  name: string,
  args: IRExpr[],
  nargout: number,
  span: Span,
  candidates: FunctionCandidate[]
): IRExpr {
  const returnType = determineReturnType(ctx, name, candidates, args, nargout);
  return {
    kind: { type: "FuncCall", name, args, nargout, returnType, candidates },
    span,
  };
}

function resolveFuncCall(
  ctx: LoweringContext,
  name: string,
  args: IRExpr[],
  nargout: number,
  span: Span
): IRExpr | null {
  const argTypes: ItemType[] = args.map(a =>
    itemTypeForExprKind(a.kind, ctx.typeEnv)
  );
  const callSite: CallSite = {
    file: ctx.mainFileName,
    className: ctx.ownerClassName ?? undefined,
  };
  const target = resolveFunction(name, argTypes, callSite, ctx.functionIndex);
  if (!target) return null;

  switch (target.kind) {
    case "classMethod": {
      const returnType = determineMethodReturnType(
        ctx,
        target.className,
        target.methodName,
        argTypes,
        nargout
      );
      return {
        kind: {
          type: "FuncCall",
          name,
          args,
          nargout,
          returnType,
          candidates: [{ type: "userFunction", functionId: name }],
        },
        span,
      };
    }

    case "localFunction": {
      let functionId = name;
      if (target.source.from === "main") {
        const stub = ctx.getLocalFunctionStub(name);
        if (stub) functionId = stub.functionId;
      }
      return makeFuncCallExpr(ctx, name, args, nargout, span, [
        { type: "userFunction", functionId },
      ]);
    }

    case "privateFunction":
      return makeFuncCallExpr(ctx, name, args, nargout, span, [
        { type: "userFunction", functionId: `private:${name}` },
      ]);

    case "workspaceFunction":
      return makeFuncCallExpr(ctx, target.name, args, nargout, span, [
        { type: "userFunction", functionId: target.name },
      ]);

    case "workspaceClassConstructor":
      return {
        kind: {
          type: "ClassInstantiation",
          className: target.className,
          args,
          nargout,
        },
        span,
      };

    case "builtin":
      return makeFuncCallExpr(ctx, name, args, nargout, span, [
        { type: "builtin", builtin: true },
      ]);
  }
}

// ── Public API ──────────────────────────────────────────────────────────

export function lowerExpr(
  ctx: LoweringContext,
  expr: AstExpr,
  nargout: number = 1
): IRExpr {
  try {
    return lowerExprInner(ctx, expr, nargout);
  } catch (e) {
    if (e instanceof SemanticError) throw e;
    throw new SemanticError(
      e instanceof Error ? e.message : String(e),
      expr.span
    );
  }
}

// ── Expression Lowering ─────────────────────────────────────────────────

function lowerExprInner(
  ctx: LoweringContext,
  expr: AstExpr,
  nargout: number
): IRExpr {
  const span = expr.span;

  switch (expr.type) {
    case "Number":
      return { kind: { type: "Number", value: expr.value }, span };

    case "Char":
      return { kind: { type: "Char", value: expr.value }, span };

    case "String":
      return { kind: { type: "String", value: expr.value }, span };

    case "ImagUnit":
      return { kind: { type: "Constant", name: "i" }, span };

    case "Ident": {
      // Variable in scope?
      const vv = ctx.lookup(expr.name);
      if (vv !== null) {
        return { kind: { type: "Var", variable: vv }, span };
      }
      // Constant?
      if (ctx.isConstant(expr.name)) {
        return { kind: { type: "Constant", name: expr.name }, span };
      }
      // Class / local / workspace / builtin function with no args?
      const resolved = resolveFuncCall(ctx, expr.name, [], nargout, span);
      if (resolved) return resolved;
      // Variable may be defined externally via assignin
      if (ctx.isExternallyDefinable(expr.name)) {
        const variable = ctx.defineVariable(expr.name, IType.Unknown);
        return { kind: { type: "Var", variable }, span };
      }
      // Undefined
      return {
        kind: {
          type: "RuntimeError",
          message: `Undefined variable: ${expr.name}`,
        },
        span,
      };
    }

    case "EndKeyword":
      return { kind: { type: "End" }, span };

    case "Colon":
      return { kind: { type: "Colon" }, span };

    case "Unary": {
      const inner = lowerExpr(ctx, expr.operand);
      return { kind: { type: "Unary", op: expr.op, operand: inner }, span };
    }

    case "Binary": {
      const left = lowerExpr(ctx, expr.left);
      const right = lowerExpr(ctx, expr.right);
      return { kind: { type: "Binary", left, op: expr.op, right }, span };
    }

    case "Tensor": {
      const rows = expr.rows.map(row => row.map(e => lowerExpr(ctx, e)));
      return { kind: { type: "Tensor", rows }, span };
    }

    case "Cell": {
      const rows = expr.rows.map(row => row.map(e => lowerExpr(ctx, e)));
      return { kind: { type: "Cell", rows }, span };
    }

    case "Index": {
      const base = lowerExpr(ctx, expr.base);
      const indices = expr.indices.map(i => lowerExpr(ctx, i));
      return {
        kind: {
          type: "Index",
          base,
          indices,
          ...(nargout > 1 ? { nargout } : {}),
        },
        span,
      };
    }

    case "IndexCell": {
      const base = lowerExpr(ctx, expr.base);
      const indices = expr.indices.map(i => lowerExpr(ctx, i));
      return { kind: { type: "IndexCell", base, indices }, span };
    }

    case "Range": {
      const start = lowerExpr(ctx, expr.start);
      const end = lowerExpr(ctx, expr.end);
      const step = expr.step ? lowerExpr(ctx, expr.step) : null;
      return { kind: { type: "Range", start, step, end }, span };
    }

    case "FuncCall": {
      const args = expr.args.map(a => lowerExpr(ctx, a));

      // Constant called as zero-arg function? e.g. eps(), pi(), inf()
      if (args.length === 0 && ctx.isConstant(expr.name)) {
        return { kind: { type: "Constant", name: expr.name }, span };
      }

      // Variable in scope? → array indexing, not function call
      const vv = ctx.lookup(expr.name);
      if (vv !== null) {
        const varExpr: IRExpr = { kind: { type: "Var", variable: vv }, span };
        return {
          kind: {
            type: "Index",
            base: varExpr,
            indices: args,
            ...(nargout > 1 ? { nargout } : {}),
          },
          span,
        };
      }

      // Class / local / workspace / class-method / builtin
      const resolved = resolveFuncCall(ctx, expr.name, args, nargout, span);
      if (resolved) return resolved;

      // Truly unknown function → dispatch at runtime
      const returnType = determineReturnType(ctx, expr.name, [], args, nargout);
      return {
        kind: {
          type: "FuncCall",
          name: expr.name,
          args,
          nargout,
          returnType,
          candidates: [],
        },
        span,
      };
    }

    case "Member": {
      // Check for ClassName.staticMethod (no parentheses) — call with 0 args
      // ClassName.staticMethod is equivalent to ClassName.staticMethod()
      const memberBaseName = tryExtractDottedName(expr.base);
      if (
        memberBaseName !== null &&
        !ctx.lookup(memberBaseName) &&
        ctx.isClass(memberBaseName)
      ) {
        const classInfo = ctx.getClassInfo(memberBaseName);
        if (classInfo && ctx.classHasStaticMethod(memberBaseName, expr.name)) {
          const argTypes: ItemType[] = [];
          const returnType = determineMethodReturnType(
            ctx,
            memberBaseName,
            expr.name,
            argTypes,
            nargout
          );
          return {
            kind: {
              type: "FuncCall",
              name: expr.name,
              args: [],
              nargout,
              returnType,
              candidates: [],
              targetClassName: memberBaseName,
              methodName: expr.name,
            },
            span,
          };
        }
      }
      const base = lowerExpr(ctx, expr.base);
      return { kind: { type: "Member", base, name: expr.name }, span };
    }

    case "MemberDynamic": {
      const base = lowerExpr(ctx, expr.base);
      const nameExpr = lowerExpr(ctx, expr.nameExpr);
      return { kind: { type: "MemberDynamic", base, nameExpr }, span };
    }

    case "MethodCall": {
      // Check for namespace workspace function/class: pkg.func(args), pkg.sub.func(args), pkg.ClassName(args)
      const namespacePath = tryExtractDottedName(expr.base);
      if (namespacePath !== null) {
        // Ensure the root identifier is not a variable in scope
        const rootName = namespacePath.split(".")[0];
        if (!ctx.lookup(rootName)) {
          const qualifiedName = `${namespacePath}.${expr.name}`;
          // Namespace function or class constructor?
          if (
            ctx.isWorkspaceFunction(qualifiedName) ||
            ctx.isClass(qualifiedName)
          ) {
            const args = expr.args.map(a => lowerExpr(ctx, a));
            const resolved = resolveFuncCall(
              ctx,
              qualifiedName,
              args,
              nargout,
              span
            );
            if (resolved) return resolved;
          }

          // Static method call: ClassName.method(args) where ClassName is a known class
          if (ctx.isClass(namespacePath)) {
            const classInfo = ctx.getClassInfo(namespacePath);
            if (
              classInfo &&
              ctx.classHasStaticMethod(namespacePath, expr.name)
            ) {
              const args = expr.args.map(a => lowerExpr(ctx, a));
              const argTypes: ItemType[] = args.map(a =>
                itemTypeForExprKind(a.kind, ctx.typeEnv)
              );
              const returnType = determineMethodReturnType(
                ctx,
                namespacePath,
                expr.name,
                argTypes,
                nargout
              );
              return {
                kind: {
                  type: "FuncCall",
                  name: expr.name,
                  args,
                  nargout,
                  returnType,
                  candidates: [],
                  targetClassName: namespacePath,
                  methodName: expr.name,
                },
                span,
              };
            }
          }
        }
      }

      const base = lowerExpr(ctx, expr.base);
      const args = expr.args.map(a => lowerExpr(ctx, a));

      // If base type is ClassInstance and method is known, infer return type
      const baseType = itemTypeForExprKind(base.kind, ctx.typeEnv);
      if (baseType.kind === "ClassInstance") {
        const classInfo = ctx.getClassInfo(baseType.className);
        // When a property and method share the same name, dot-syntax
        // obj.name(args) means property access + indexing.
        // Only use method call if the name is NOT also a property.
        const isAlsoProperty =
          classInfo && classInfo.propertyNames.includes(expr.name);
        if (isAlsoProperty) {
          // Property access + indexing: obj.name(args) → index(getMember(obj, name), args)
          const memberExpr: IRExpr = {
            kind: { type: "Member", base, name: expr.name },
            span,
          };
          return {
            kind: {
              type: "Index",
              base: memberExpr,
              indices: args,
              ...(nargout > 1 ? { nargout } : {}),
            },
            span,
          };
        }
        if (classInfo && classInfo.methodNames.has(expr.name)) {
          const methodArgTypes: ItemType[] = [
            baseType,
            ...args.map(a => itemTypeForExprKind(a.kind, ctx.typeEnv)),
          ];
          const returnType = determineMethodReturnType(
            ctx,
            baseType.className,
            expr.name,
            methodArgTypes,
            nargout
          );
          return {
            kind: {
              type: "FuncCall",
              name: expr.name,
              args: [base, ...args],
              nargout,
              returnType,
              candidates: [],
              targetClassName: baseType.className,
              methodName: expr.name,
            },
            span,
          };
        }
        // Static method called on an instance: instance.staticMethod(args)
        // Dispatches to the runtime type's implementation.
        if (
          classInfo &&
          ctx.classHasStaticMethod(baseType.className, expr.name)
        ) {
          const argTypes: ItemType[] = args.map(a =>
            itemTypeForExprKind(a.kind, ctx.typeEnv)
          );
          const returnType = determineMethodReturnType(
            ctx,
            baseType.className,
            expr.name,
            argTypes,
            nargout
          );
          return {
            kind: {
              type: "FuncCall",
              name: expr.name,
              args,
              nargout,
              returnType,
              candidates: [],
              targetClassName: baseType.className,
              methodName: expr.name,
              // Carry the instance so codegen can resolve the class at runtime
              instanceBase: base,
            },
            span,
          };
        }
      }

      const returnType: ItemType = { kind: "Unknown" };
      return {
        kind: {
          type: "MethodCall",
          base,
          name: expr.name,
          args,
          nargout,
          returnType,
        },
        span,
      };
    }

    case "AnonFunc": {
      const savedLen = ctx.scopes.length;
      ctx.pushScope();
      const paramIds = expr.params.map(p =>
        ctx.defineVariable(p, IType.Unknown)
      );
      const body = lowerExpr(ctx, expr.body);
      while (ctx.scopes.length > savedLen) {
        ctx.popScope();
      }
      return { kind: { type: "AnonFunc", params: paramIds, body }, span };
    }

    case "FuncHandle": {
      // Resolve functionId from local functions
      let functionId: string = expr.name; // fallback for builtins
      const localFunc = ctx.getLocalFunctionStub(expr.name);
      if (localFunc) {
        functionId = localFunc.functionId;
      }
      return {
        kind: { type: "FuncHandle", name: expr.name, functionId },
        span,
      };
    }

    case "SuperMethodCall": {
      const args = expr.args.map(a => lowerExpr(ctx, a));
      const objVar = ctx.lookup(expr.methodName);
      if (objVar) {
        return {
          kind: {
            type: "SuperConstructorCall",
            superClassName: expr.superClassName,
            objVar,
            args,
          },
          span,
        };
      }
      return {
        kind: {
          type: "FuncCall",
          name: expr.methodName,
          args,
          nargout,
          returnType: { kind: "Unknown" },
          candidates: [],
          targetClassName: expr.superClassName,
          methodName: expr.methodName,
        },
        span,
      };
    }

    case "MetaClass":
      return { kind: { type: "MetaClass", name: expr.name }, span };

    default:
      throw new SemanticError(
        `Unknown expression type: ${(expr as AstExpr).type}`,
        span
      );
  }
}

// ── Namespace helpers ───────────────────────────────────────────────────

/**
 * Try to extract a dotted name from a chain of Member accesses rooted at an Ident.
 * e.g. Member(Member(Ident("a"), "b"), "c") → "a.b.c"
 * Returns null if the expression is not a pure Ident/Member chain.
 */
function tryExtractDottedName(expr: AstExpr): string | null {
  if (expr.type === "Ident") return expr.name;
  if (expr.type === "Member") {
    const baseName = tryExtractDottedName(expr.base);
    if (baseName !== null) return `${baseName}.${expr.name}`;
  }
  return null;
}
