import { Token } from "../lexer/index.js";

// ── Source Location ──────────────────────────────────────────────────────

export interface Span {
  file: string;
  start: number;
  end: number;
}

export interface TokenInfo {
  token: Token;
  lexeme: string;
  position: number;
  end: number;
}

// ── Operators ────────────────────────────────────────────────────────────

export enum BinaryOperation {
  Add = "Add",
  Sub = "Sub",
  Mul = "Mul",
  ElemMul = "ElemMul",
  Div = "Div",
  ElemDiv = "ElemDiv",
  LeftDiv = "LeftDiv",
  ElemLeftDiv = "ElemLeftDiv",
  Pow = "Pow",
  ElemPow = "ElemPow",
  Equal = "Equal",
  NotEqual = "NotEqual",
  Less = "Less",
  LessEqual = "LessEqual",
  Greater = "Greater",
  GreaterEqual = "GreaterEqual",
  OrOr = "OrOr",
  AndAnd = "AndAnd",
  BitOr = "BitOr",
  BitAnd = "BitAnd",
}

export enum UnaryOperation {
  Plus = "Plus",
  Minus = "Minus",
  Not = "Not",
  Transpose = "Transpose",
  NonConjugateTranspose = "NonConjugateTranspose",
}

// ── Expressions ──────────────────────────────────────────────────────────

export type Expr =
  | { type: "Number"; value: string; span: Span }
  | { type: "Char"; value: string; span: Span }
  | { type: "String"; value: string; span: Span }
  | { type: "Ident"; name: string; span: Span }
  | { type: "EndKeyword"; span: Span }
  | { type: "ImagUnit"; span: Span }
  | { type: "Colon"; span: Span }
  | { type: "MetaClass"; name: string; span: Span }
  | { type: "Binary"; left: Expr; op: BinaryOperation; right: Expr; span: Span }
  | { type: "Unary"; op: UnaryOperation; operand: Expr; span: Span }
  | { type: "Range"; start: Expr; step: Expr | null; end: Expr; span: Span }
  | { type: "FuncCall"; name: string; args: Expr[]; span: Span }
  | { type: "Index"; base: Expr; indices: Expr[]; span: Span }
  | { type: "IndexCell"; base: Expr; indices: Expr[]; span: Span }
  | { type: "Member"; base: Expr; name: string; span: Span }
  | { type: "MemberDynamic"; base: Expr; nameExpr: Expr; span: Span }
  | {
      type: "MethodCall";
      base: Expr;
      name: string;
      args: Expr[];
      span: Span;
    }
  | {
      type: "SuperMethodCall";
      methodName: string;
      superClassName: string;
      args: Expr[];
      span: Span;
    }
  | { type: "AnonFunc"; params: string[]; body: Expr; span: Span }
  | { type: "FuncHandle"; name: string; span: Span }
  | { type: "Tensor"; rows: Expr[][]; span: Span }
  | { type: "Cell"; rows: Expr[][]; span: Span }
  | { type: "ClassInstantiation"; className: string; args: Expr[]; span: Span };

// ── LValues ──────────────────────────────────────────────────────────────

export type LValue =
  | { type: "Var"; name: string }
  | { type: "Ignore" }
  | { type: "Index"; base: Expr; indices: Expr[] }
  | { type: "IndexCell"; base: Expr; indices: Expr[] }
  | { type: "Member"; base: Expr; name: string }
  | { type: "MemberDynamic"; base: Expr; nameExpr: Expr };

// ── Class-related Types ──────────────────────────────────────────────────

export interface Attr {
  name: string;
  value: string | null;
}

export interface MethodSignature {
  name: string;
  params: string[];
  outputs: string[];
  span: Span;
}

export type ClassMember =
  | {
      type: "Properties";
      attributes: Attr[];
      names: string[];
      defaultValues: (Expr | null)[];
    }
  | {
      type: "Methods";
      attributes: Attr[];
      body: Stmt[];
      signatures?: MethodSignature[];
    }
  | { type: "Events"; attributes: Attr[]; names: string[] }
  | { type: "Enumeration"; attributes: Attr[]; names: string[] }
  | { type: "Arguments"; attributes: Attr[]; names: string[] };

// ── Arguments Blocks ─────────────────────────────────────────────────────

export type ArgumentsBlockKind =
  | "Input"
  | "Output"
  | "Repeating"
  | "OutputRepeating";

export interface ArgumentEntry {
  name: string;
  dimensions: string[] | null;
  className: string | null;
  validators: string[];
  defaultValue: Expr | null;
}

export interface ArgumentsBlock {
  kind: ArgumentsBlockKind;
  entries: ArgumentEntry[];
}

// ── Statements ───────────────────────────────────────────────────────────

export type Stmt =
  | { type: "ExprStmt"; expr: Expr; suppressed: boolean; span: Span }
  | {
      type: "Assign";
      name: string;
      expr: Expr;
      suppressed: boolean;
      span: Span;
    }
  | {
      type: "MultiAssign";
      lvalues: LValue[];
      expr: Expr;
      suppressed: boolean;
      span: Span;
    }
  | {
      type: "AssignLValue";
      lvalue: LValue;
      expr: Expr;
      suppressed: boolean;
      span: Span;
    }
  | {
      type: "If";
      cond: Expr;
      thenBody: Stmt[];
      elseifBlocks: Array<{ cond: Expr; body: Stmt[] }>;
      elseBody: Stmt[] | null;
      span: Span;
    }
  | { type: "While"; cond: Expr; body: Stmt[]; span: Span }
  | { type: "For"; varName: string; expr: Expr; body: Stmt[]; span: Span }
  | {
      type: "Switch";
      expr: Expr;
      cases: Array<{ value: Expr; body: Stmt[] }>;
      otherwise: Stmt[] | null;
      span: Span;
    }
  | {
      type: "TryCatch";
      tryBody: Stmt[];
      catchVar: string | null;
      catchBody: Stmt[];
      span: Span;
    }
  | {
      type: "Function";
      name: string;
      functionId: string;
      params: string[];
      outputs: string[];
      body: Stmt[];
      argumentsBlocks: ArgumentsBlock[];
      isFileLocalSubfunction?: boolean;
      span: Span;
    }
  | { type: "Global"; names: string[]; span: Span }
  | { type: "Persistent"; names: string[]; span: Span }
  | { type: "Break"; span: Span }
  | { type: "Continue"; span: Span }
  | { type: "Return"; span: Span }
  | {
      type: "Import";
      path: string[];
      wildcard: boolean;
      span: Span;
    }
  | {
      type: "ClassDef";
      name: string;
      classAttributes: Attr[];
      superClass: string | null;
      members: ClassMember[];
      span: Span;
    }
  | {
      /** Magic comment directive, e.g. `%!numbl:assert_jit` or `%!numbl:assert_jit c`. */
      type: "Directive";
      directive: string;
      args: string[];
      span: Span;
    };

// ── AST Root ─────────────────────────────────────────────────────────────

export interface AbstractSyntaxTree {
  body: Stmt[];
}
