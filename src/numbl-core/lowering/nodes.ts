/**
 * IR Node Definitions
 */

import {
  Attr,
  BinaryOperation,
  Span,
  UnaryOperation,
  MethodSignature,
  ArgumentsBlockKind,
} from "../parser/index.js";
import { ItemType } from "./itemTypes.js";
import { FunctionCandidate } from "./lowerExpr.js";
import { IRVariable } from "./loweringTypes.js";

// ── IR Expression ──────────────────────────────────────────────────────

export type IRExprKind =
  | { type: "Number"; value: string }
  | { type: "Char"; value: string } // single-quoted char array: 'hello'
  | { type: "String"; value: string } // double-quoted string: "hello"
  | { type: "Var"; variable: IRVariable }
  | { type: "Constant"; name: string }
  | { type: "Unary"; op: UnaryOperation; operand: IRExpr }
  | { type: "Binary"; left: IRExpr; op: BinaryOperation; right: IRExpr }
  | { type: "Tensor"; rows: IRExpr[][] }
  | { type: "Cell"; rows: IRExpr[][] }
  | { type: "Index"; base: IRExpr; indices: IRExpr[]; nargout?: number }
  | { type: "IndexCell"; base: IRExpr; indices: IRExpr[] }
  | { type: "Range"; start: IRExpr; step: IRExpr | null; end: IRExpr }
  | { type: "Colon" }
  | { type: "End" }
  | { type: "Member"; base: IRExpr; name: string }
  | { type: "MemberDynamic"; base: IRExpr; nameExpr: IRExpr }
  | {
      type: "MethodCall";
      base: IRExpr;
      name: string;
      args: IRExpr[];
      nargout: number;
      returnType: ItemType;
    }
  | {
      type: "SuperConstructorCall";
      superClassName: string;
      objVar: IRVariable;
      args: IRExpr[];
    }
  | { type: "AnonFunc"; params: IRVariable[]; body: IRExpr }
  | { type: "FuncHandle"; name: string; functionId: string }
  | {
      type: "FuncCall";
      name: string;
      args: IRExpr[];
      nargout: number;
      returnType: ItemType;
      candidates: FunctionCandidate[];
      /** When resolved to a specific class method at lowering time. */
      targetClassName?: string;
      /** When the method name differs from `name` (e.g. super.method, static calls). */
      methodName?: string;
      /** When a static method is called on an instance (obj.staticMethod(args)),
       *  carries the instance expression for runtime polymorphic dispatch. */
      instanceBase?: IRExpr;
    }
  | { type: "MetaClass"; name: string }
  | {
      type: "ClassInstantiation";
      className: string;
      args: IRExpr[];
      nargout: number;
    }
  | { type: "RuntimeError"; message: string };

export interface IRExpr {
  kind: IRExprKind;
  // ty: Type;
  span: Span;
}

// ── IR LValue ──────────────────────────────────────────────────────────

export type IRLValue =
  | { type: "Var"; variable: IRVariable }
  | { type: "Member"; base: IRExpr; name: string }
  | { type: "MemberDynamic"; base: IRExpr; nameExpr: IRExpr }
  | { type: "Index"; base: IRExpr; indices: IRExpr[] }
  | { type: "IndexCell"; base: IRExpr; indices: IRExpr[] };

// ── IR Arguments Block ─────────────────────────────────────────────────

export interface IRArgumentEntry {
  name: string; // "x" or "options.Name" for name-value args
  dimensions: string[] | null;
  className: string | null;
  validators: string[];
  defaultValue: IRExpr | null;
}

export interface IRArgumentsBlock {
  kind: ArgumentsBlockKind;
  entries: IRArgumentEntry[];
}

// ── IR Class Member ────────────────────────────────────────────────────

export type IRClassMember =
  | {
      type: "Properties";
      attributes: Attr[];
      names: string[];
      defaultValues: (IRExpr | null)[];
    }
  | {
      type: "Methods";
      attributes: Attr[];
      body: IRStmt[];
      signatures?: MethodSignature[];
    }
  | { type: "Events"; attributes: Attr[]; names: string[] }
  | { type: "Enumeration"; attributes: Attr[]; names: string[] }
  | { type: "Arguments"; attributes: Attr[]; names: string[] };

// ── IR Statement ───────────────────────────────────────────────────────

export type IRStmt =
  | { type: "ExprStmt"; expr: IRExpr; suppressed: boolean; span: Span }
  | {
      type: "Assign";
      variable: IRVariable;
      expr: IRExpr;
      /** Flow-dependent type: the inferred type of the RHS at this program point. */
      assignedType?: ItemType;
      suppressed: boolean;
      span: Span;
    }
  | {
      type: "MultiAssign";
      lvalues: (IRLValue | null)[];
      expr: IRExpr;
      /** Flow-dependent types: inferred type per lvalue at this program point. */
      assignedTypes?: (ItemType | null)[];
      suppressed: boolean;
      span: Span;
    }
  | {
      type: "AssignLValue";
      lvalue: IRLValue;
      expr: IRExpr;
      suppressed: boolean;
      span: Span;
    }
  | {
      type: "If";
      cond: IRExpr;
      thenBody: IRStmt[];
      elseifBlocks: Array<{ cond: IRExpr; body: IRStmt[] }>;
      elseBody: IRStmt[] | null;
      span: Span;
    }
  | { type: "While"; cond: IRExpr; body: IRStmt[]; span: Span }
  | {
      type: "For";
      variable: IRVariable;
      expr: IRExpr;
      body: IRStmt[];
      /** Flow-dependent type of the iteration variable. */
      iterVarType?: ItemType;
      span: Span;
    }
  | {
      type: "Switch";
      expr: IRExpr;
      cases: Array<{ value: IRExpr; body: IRStmt[] }>;
      otherwise: IRStmt[] | null;
      span: Span;
    }
  | {
      type: "TryCatch";
      tryBody: IRStmt[];
      catchVar: IRVariable | null;
      catchBody: IRStmt[];
      span: Span;
    }
  | {
      type: "Global";
      vars: Array<{ variable: IRVariable; name: string }>;
      span: Span;
    }
  | {
      type: "Persistent";
      vars: Array<{ variable: IRVariable; name: string }>;
      span: Span;
    }
  | { type: "Break"; span: Span }
  | { type: "Continue"; span: Span }
  | { type: "Return"; span: Span }
  | {
      type: "Function";
      originalName: string;
      functionId: string;
      params: IRVariable[];
      outputs: IRVariable[];
      body: IRStmt[];
      hasVarargin: boolean;
      hasVarargout: boolean;
      argumentsBlocks: IRArgumentsBlock[];
      /** True if this is the primary (public entry-point) function of its .m file. */
      isPrimaryFunction?: boolean;
      /** For file-local subfunctions inside classdef files, the enclosing
       *  class name so that the codegen can skip overloaded subsref for
       *  same-class paren indexing. */
      classContext?: string;
      /** Snapshot of output variable types at end of lowering. */
      outputTypes?: ItemType[];
      span: Span;
    }
  | {
      type: "ClassDef";
      name: string;
      classAttributes: Attr[];
      superClass: string | null;
      members: IRClassMember[];
      span: Span;
    }
  | { type: "Import"; path: string[]; wildcard: boolean; span: Span };

// ── IR Program ─────────────────────────────────────────────────────────

export interface IRProgram {
  body: IRStmt[];
  // varTypes: Map<string, Type>;
}
