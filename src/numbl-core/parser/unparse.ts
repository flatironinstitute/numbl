/**
 * MATLAB source text of an expression, in the compact form `func2str`
 * gives an anonymous function (`@(x)x.^2+1`): no spaces around operators,
 * commas between arguments and matrix elements. The AST does not record
 * parentheses, so only those that precedence requires are written.
 */

import { BinaryOperation, UnaryOperation, type Expr } from "./types.js";

const BINARY: Record<BinaryOperation, { text: string; prec: number }> = {
  [BinaryOperation.OrOr]: { text: "||", prec: 1 },
  [BinaryOperation.AndAnd]: { text: "&&", prec: 2 },
  [BinaryOperation.BitOr]: { text: "|", prec: 3 },
  [BinaryOperation.BitAnd]: { text: "&", prec: 4 },
  [BinaryOperation.Equal]: { text: "==", prec: 5 },
  [BinaryOperation.NotEqual]: { text: "~=", prec: 5 },
  [BinaryOperation.Less]: { text: "<", prec: 5 },
  [BinaryOperation.LessEqual]: { text: "<=", prec: 5 },
  [BinaryOperation.Greater]: { text: ">", prec: 5 },
  [BinaryOperation.GreaterEqual]: { text: ">=", prec: 5 },
  [BinaryOperation.Add]: { text: "+", prec: 7 },
  [BinaryOperation.Sub]: { text: "-", prec: 7 },
  [BinaryOperation.Mul]: { text: "*", prec: 8 },
  [BinaryOperation.ElemMul]: { text: ".*", prec: 8 },
  [BinaryOperation.Div]: { text: "/", prec: 8 },
  [BinaryOperation.ElemDiv]: { text: "./", prec: 8 },
  [BinaryOperation.LeftDiv]: { text: "\\", prec: 8 },
  [BinaryOperation.ElemLeftDiv]: { text: ".\\", prec: 8 },
  [BinaryOperation.Pow]: { text: "^", prec: 10 },
  [BinaryOperation.ElemPow]: { text: ".^", prec: 10 },
};

const RANGE_PREC = 6;
const PREFIX_PREC = 9;
const POW_PREC = 10;
const POSTFIX_PREC = 11;
const ATOM_PREC = 12;

/** `3i` parses as 3 * (imaginary unit); write it back as a literal. */
function isImagLiteral(e: Expr): boolean {
  return (
    e.type === "Binary" &&
    e.op === BinaryOperation.Mul &&
    e.left.type === "Number" &&
    e.right.type === "ImagUnit"
  );
}

function prec(e: Expr): number {
  switch (e.type) {
    case "AnonFunc":
      return 0;
    case "Binary":
      return isImagLiteral(e) ? ATOM_PREC : BINARY[e.op].prec;
    case "Range":
      return RANGE_PREC;
    case "Unary":
      return e.op === UnaryOperation.Transpose ||
        e.op === UnaryOperation.NonConjugateTranspose
        ? POSTFIX_PREC
        : PREFIX_PREC;
    default:
      return ATOM_PREC;
  }
}

/** `e`, parenthesized when its precedence is below `min`. */
function sub(e: Expr, min: number): string {
  const s = unparseExpr(e);
  return prec(e) < min ? `(${s})` : s;
}

const list = (args: Expr[]) => args.map(unparseExpr).join(",");

const rows = (rs: Expr[][]) => rs.map(list).join(";");

export function unparseExpr(e: Expr): string {
  switch (e.type) {
    case "Number":
    case "Char":
    case "String":
      return e.value;
    case "Ident":
      return e.name;
    case "EndKeyword":
      return "end";
    case "ImagUnit":
      return "i";
    case "Colon":
      return ":";
    case "MetaClass":
      return `?${e.name}`;
    case "Binary": {
      if (isImagLiteral(e)) return `${unparseExpr(e.left)}i`;
      const { text, prec: p } = BINARY[e.op];
      // Left-associative: an equal-precedence right operand needs parens.
      // A power's exponent may be a signed postfix expression (2^-x).
      const signedPostfix =
        e.right.type === "Unary" &&
        prec(e.right) === PREFIX_PREC &&
        prec(e.right.operand) >= POSTFIX_PREC;
      const right =
        p === POW_PREC && signedPostfix
          ? unparseExpr(e.right)
          : sub(e.right, p + 1);
      return `${sub(e.left, p)}${text}${right}`;
    }
    case "Unary":
      switch (e.op) {
        case UnaryOperation.Transpose:
          return `${sub(e.operand, POSTFIX_PREC)}'`;
        case UnaryOperation.NonConjugateTranspose:
          return `${sub(e.operand, POSTFIX_PREC)}.'`;
        case UnaryOperation.Plus:
          return `+${sub(e.operand, PREFIX_PREC)}`;
        case UnaryOperation.Minus:
          return `-${sub(e.operand, PREFIX_PREC)}`;
        case UnaryOperation.Not:
          return `~${sub(e.operand, PREFIX_PREC)}`;
      }
      break;
    case "Range": {
      const part = (x: Expr) => sub(x, RANGE_PREC + 1);
      return e.step
        ? `${part(e.start)}:${part(e.step)}:${part(e.end)}`
        : `${part(e.start)}:${part(e.end)}`;
    }
    case "FuncCall":
      return `${e.name}(${list(e.args)})`;
    case "Index":
      return `${sub(e.base, POSTFIX_PREC)}(${list(e.indices)})`;
    case "IndexCell":
      return `${sub(e.base, POSTFIX_PREC)}{${list(e.indices)}}`;
    case "Member":
      return `${sub(e.base, POSTFIX_PREC)}.${e.name}`;
    case "MemberDynamic":
      return `${sub(e.base, POSTFIX_PREC)}.(${unparseExpr(e.nameExpr)})`;
    case "MethodCall":
      return `${sub(e.base, POSTFIX_PREC)}.${e.name}(${list(e.args)})`;
    case "SuperMethodCall":
      return `${e.methodName}@${e.superClassName}(${list(e.args)})`;
    case "ClassInstantiation":
      return `${e.className}(${list(e.args)})`;
    case "AnonFunc":
      return `@(${e.params.join(",")})${unparseExpr(e.body)}`;
    case "FuncHandle":
      return `@${e.name}`;
    case "Tensor":
      return `[${rows(e.rows)}]`;
    case "Cell":
      return `{${rows(e.rows)}}`;
  }
  throw new Error(`unparseExpr: unhandled expression`);
}
