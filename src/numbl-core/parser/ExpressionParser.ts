/**
 * ExpressionParser - Expression parsing methods
 */

import { Token } from "../lexer/index.js";
import { Expr, BinaryOperation, UnaryOperation } from "./types.js";
import { ParserBase } from "./ParserBase.js";

export class ExpressionParser extends ParserBase {
  // ── Expression Parsing ───────────────────────────────────────────────

  parseExpr(): Expr {
    return this.parseLogicalOr();
  }

  private parseLogicalOr(): Expr {
    let node = this.parseLogicalAnd();
    while (this.consume(Token.OrOr)) {
      const rhs = this.parseLogicalAnd();
      node = this.makeBinary(node, BinaryOperation.OrOr, rhs);
    }
    return node;
  }

  private parseLogicalAnd(): Expr {
    let node = this.parseBitwiseOr();
    while (this.consume(Token.AndAnd)) {
      const rhs = this.parseBitwiseOr();
      node = this.makeBinary(node, BinaryOperation.AndAnd, rhs);
    }
    return node;
  }

  private parseBitwiseOr(): Expr {
    let node = this.parseBitwiseAnd();
    while (this.consume(Token.Or)) {
      const rhs = this.parseBitwiseAnd();
      node = this.makeBinary(node, BinaryOperation.BitOr, rhs);
    }
    return node;
  }

  private parseBitwiseAnd(): Expr {
    let node = this.parseComparison();
    while (this.consume(Token.And)) {
      const rhs = this.parseComparison();
      node = this.makeBinary(node, BinaryOperation.BitAnd, rhs);
    }
    return node;
  }

  private parseComparison(): Expr {
    let node = this.parseRange();
    while (true) {
      let op: BinaryOperation | undefined;
      const tok = this.peekToken();
      switch (tok) {
        case Token.Equal:
          op = BinaryOperation.Equal;
          break;
        case Token.NotEqual:
          op = BinaryOperation.NotEqual;
          break;
        case Token.Less:
          op = BinaryOperation.Less;
          break;
        case Token.LessEqual:
          op = BinaryOperation.LessEqual;
          break;
        case Token.Greater:
          op = BinaryOperation.Greater;
          break;
        case Token.GreaterEqual:
          op = BinaryOperation.GreaterEqual;
          break;
        default:
          return node;
      }
      this.pos++;
      const rhs = this.parseRange();
      node = this.makeBinary(node, op, rhs);
    }
  }

  private parseRange(): Expr {
    const node = this.parseAddSub();
    if (this.consume(Token.Colon)) {
      const mid = this.parseAddSub();
      if (this.consume(Token.Colon)) {
        const end = this.parseAddSub();
        const span = this.spanBetween(node.span, end.span);
        return { type: "Range", start: node, step: mid, end, span };
      } else {
        const span = this.spanBetween(node.span, mid.span);
        return { type: "Range", start: node, step: null, end: mid, span };
      }
    }
    return node;
  }

  private parseAddSub(): Expr {
    let node = this.parseMulDiv();
    while (true) {
      // Matrix expression whitespace handling
      if (
        this.inMatrixExpr &&
        (this.peekToken() === Token.Plus || this.peekToken() === Token.Minus) &&
        this.pos > 0 &&
        !this.tokensAdjacent(this.pos - 1, this.pos) &&
        this.tokensAdjacent(this.pos, this.pos + 1)
      ) {
        const rhsIndex = this.pos + 1;
        const rhsIsImagLiteral =
          (this.tokens[rhsIndex]?.token === Token.Integer ||
            this.tokens[rhsIndex]?.token === Token.Float) &&
          this.tokens[rhsIndex + 1]?.token === Token.Ident &&
          (this.tokens[rhsIndex + 1]?.lexeme.toLowerCase() === "i" ||
            this.tokens[rhsIndex + 1]?.lexeme.toLowerCase() === "j") &&
          this.tokensAdjacent(rhsIndex, rhsIndex + 1);

        if (!rhsIsImagLiteral) break;
      }

      let op: BinaryOperation | undefined;
      if (this.consume(Token.Plus)) {
        op = BinaryOperation.Add;
      } else if (this.consume(Token.Minus)) {
        op = BinaryOperation.Sub;
      } else if (
        this.peekToken() === Token.Dot &&
        (this.peekTokenAt(1) === Token.Plus ||
          this.peekTokenAt(1) === Token.Minus)
      ) {
        // '.+' or '.-' tokenized as Dot then Plus/Minus
        this.pos += 2;
        op =
          this.tokens[this.pos - 1].token === Token.Plus
            ? BinaryOperation.Add
            : BinaryOperation.Sub;
      } else {
        break;
      }

      const rhs = this.parseMulDiv();
      node = this.makeBinary(node, op, rhs);
    }
    return node;
  }

  private parseMulDiv(): Expr {
    let node = this.parseUnary();
    while (true) {
      let op: BinaryOperation | undefined;
      const tok = this.peekToken();
      switch (tok) {
        case Token.Star:
          op = BinaryOperation.Mul;
          break;
        case Token.DotStar:
          op = BinaryOperation.ElemMul;
          break;
        case Token.Slash:
          op = BinaryOperation.Div;
          break;
        case Token.DotSlash:
          op = BinaryOperation.ElemDiv;
          break;
        case Token.Backslash:
          op = BinaryOperation.LeftDiv;
          break;
        case Token.DotBackslash:
          op = BinaryOperation.ElemLeftDiv;
          break;
        default:
          return node;
      }
      this.pos++;
      const rhs = this.parseUnary();
      node = this.makeBinary(node, op, rhs);
    }
  }

  private parsePow(): Expr {
    const node = this.parsePostfix();
    const tok = this.peekToken();
    let op: BinaryOperation | undefined;
    if (tok === Token.Caret) {
      op = BinaryOperation.Pow;
    } else if (tok === Token.DotCaret) {
      op = BinaryOperation.ElemPow;
    } else {
      return node;
    }
    this.pos++;
    // Call parseUnary (not parsePow) to allow unary operators in the exponent
    // (e.g., 2^-3, 1i.^-nu). Right-associativity is preserved because
    // parseUnary calls parsePow for non-unary cases.
    const rhs = this.parseUnary();
    return this.makeBinary(node, op, rhs);
  }

  private parsePostfixWithBase(expr: Expr): Expr {
    while (true) {
      // In matrix context, a '(' separated by whitespace starts a new
      // element rather than being a postfix call/index (e.g., [(1 -2) (3 +4)]
      // is two elements, not indexing the result of (1 -2)).
      if (
        this.peekToken() === Token.LParen &&
        this.inMatrixExpr &&
        this.pos > 0 &&
        !this.tokensAdjacent(this.pos - 1, this.pos)
      ) {
        break;
      }
      if (this.consume(Token.LParen)) {
        const start = expr.span.start;
        const args: Expr[] = [];
        // Disable matrix whitespace heuristic inside call/index parens
        const priorMatrix = this.inMatrixExpr;
        this.inMatrixExpr = false;
        if (!this.consume(Token.RParen)) {
          args.push(this.parseExpr());
          while (this.consume(Token.Comma)) {
            args.push(this.parseExpr());
          }
          if (!this.consume(Token.RParen)) {
            this.inMatrixExpr = priorMatrix;
            throw this.error("expected ')' after arguments");
          }
        }
        this.inMatrixExpr = priorMatrix;
        const end = this.lastTokenEnd();
        const span = this.spanFrom(start, end);

        // If callee is identifier, parse as function call
        if (expr.type === "Ident") {
          expr = { type: "FuncCall", name: expr.name, args, span };
        } else {
          // For non-ident bases, this is indexing
          expr = { type: "Index", base: expr, indices: args, span };
        }
      } else if (!this.inMatrixExpr && this.consume(Token.LBracket)) {
        const start = expr.span.start;
        const indices: Expr[] = [];
        indices.push(this.parseExpr());
        while (this.consume(Token.Comma)) {
          indices.push(this.parseExpr());
        }
        if (!this.consume(Token.RBracket)) {
          throw this.error("expected ']'");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(start, end);
        expr = { type: "Index", base: expr, indices, span };
      } else if (this.consume(Token.LBrace)) {
        const start = expr.span.start;
        const indices: Expr[] = [];
        indices.push(this.parseExpr());
        while (this.consume(Token.Comma)) {
          indices.push(this.parseExpr());
        }
        if (!this.consume(Token.RBrace)) {
          throw this.error("expected '}'");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(start, end);
        expr = { type: "IndexCell", base: expr, indices, span };
      } else if (this.peekToken() === Token.Dot) {
        // Could be .', .+, .- or member access
        if (this.peekTokenAt(1) === Token.Transpose) {
          this.pos += 2;
          const end = this.lastTokenEnd();
          const span = this.spanFrom(expr.span.start, end);
          expr = {
            type: "Unary",
            op: UnaryOperation.NonConjugateTranspose,
            operand: expr,
            span,
          };
          continue;
        }
        if (
          this.peekTokenAt(1) === Token.Plus ||
          this.peekTokenAt(1) === Token.Minus
        ) {
          // '.+' or '.-' belong to additive level
          break;
        }
        // Member access
        this.pos++; // consume '.'

        // Check for dynamic field access .(expr)
        if (this.peekToken() === Token.LParen) {
          this.pos++; // consume '('
          const nameExpr = this.parseExpr();
          if (!this.consume(Token.RParen)) {
            throw this.error("expected ')' after dynamic field name");
          }
          const end = this.lastTokenEnd();
          const span = this.spanFrom(expr.span.start, end);
          expr = { type: "MemberDynamic", base: expr, nameExpr, span };
        } else {
          // Static member access
          const memberName = this.expectMemberName();

          if (this.consume(Token.LParen)) {
            const args: Expr[] = [];
            if (!this.consume(Token.RParen)) {
              args.push(this.parseExpr());
              while (this.consume(Token.Comma)) {
                args.push(this.parseExpr());
              }
              if (!this.consume(Token.RParen)) {
                throw this.error("expected ')' after method arguments");
              }
            }
            const end = this.lastTokenEnd();
            const span = this.spanFrom(expr.span.start, end);
            expr = {
              type: "MethodCall",
              base: expr,
              name: memberName,
              args,
              span,
            };
          } else {
            const end = this.lastTokenEnd();
            const span = this.spanFrom(expr.span.start, end);
            expr = { type: "Member", base: expr, name: memberName, span };
          }
        }
      } else if (
        expr.type === "Ident" &&
        this.peekToken() === Token.At &&
        this.peekTokenAt(1) === Token.Ident
      ) {
        // superMethod@ClassName(args) or obj@SuperClass(args)
        this.pos++; // consume '@'
        const className = this.expectIdent();
        if (!this.consume(Token.LParen)) {
          throw this.error("expected '(' after superclass name in super call");
        }
        const args: Expr[] = [];
        if (!this.consume(Token.RParen)) {
          args.push(this.parseExpr());
          while (this.consume(Token.Comma)) {
            args.push(this.parseExpr());
          }
          if (!this.consume(Token.RParen)) {
            throw this.error("expected ')' after super call arguments");
          }
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(expr.span.start, end);
        expr = {
          type: "SuperMethodCall",
          methodName: expr.name,
          superClassName: className,
          args,
          span,
        };
      } else if (this.consume(Token.Transpose)) {
        const end = this.lastTokenEnd();
        const span = this.spanFrom(expr.span.start, end);
        expr = {
          type: "Unary",
          op: UnaryOperation.Transpose,
          operand: expr,
          span,
        };
      } else {
        break;
      }
    }
    return expr;
  }

  private parsePostfix(): Expr {
    let expr = this.parsePrimary();

    // Handle imaginary unit suffix (e.g., 3i, 1.5j) at postfix level
    // so it binds tighter than power operators (^, .^)
    if (
      expr.type === "Number" &&
      this.peekToken() === Token.Ident &&
      this.pos > 0
    ) {
      const prev = this.tokens[this.pos - 1];
      const curr = this.tokens[this.pos];
      const isAdjacent = this.tokensAdjacent(this.pos - 1, this.pos);
      const isImag =
        curr.lexeme.toLowerCase() === "i" || curr.lexeme.toLowerCase() === "j";
      if (
        isAdjacent &&
        isImag &&
        (prev.token === Token.Integer || prev.token === Token.Float)
      ) {
        const token = this.next()!;
        const rhs: Expr = {
          type: "ImagUnit",
          span: this.spanFrom(token.position, token.end),
        };
        expr = this.makeBinary(expr, BinaryOperation.Mul, rhs);
      }
    }

    return this.parsePostfixWithBase(expr);
  }

  private parseUnary(): Expr {
    if (this.peekToken() === Token.Plus) {
      const start = this.tokens[this.pos].position;
      this.pos++;
      const expr = this.parseUnary();
      return this.makeUnary(UnaryOperation.Plus, expr, start);
    } else if (this.peekToken() === Token.Minus) {
      const start = this.tokens[this.pos].position;
      this.pos++;
      const expr = this.parseUnary();
      return this.makeUnary(UnaryOperation.Minus, expr, start);
    } else if (this.peekToken() === Token.Tilde) {
      const start = this.tokens[this.pos].position;
      this.pos++;
      const expr = this.parseUnary();
      return this.makeUnary(UnaryOperation.Not, expr, start);
    } else if (this.peekToken() === Token.Question) {
      const start = this.tokens[this.pos].position;
      this.pos++;
      // Meta-class query
      const parts: string[] = [];
      const first = this.expectIdent();
      const classConsumed =
        first.charAt(0) === first.charAt(0).toUpperCase() &&
        /[A-Z]/.test(first.charAt(0));
      parts.push(first);

      while (
        this.peekToken() === Token.Dot &&
        this.peekTokenAt(1) === Token.Ident
      ) {
        const nextLex = this.tokens[this.pos + 1]?.lexeme ?? "";
        const isUpper =
          nextLex.charAt(0) === nextLex.charAt(0).toUpperCase() &&
          /[A-Z]/.test(nextLex.charAt(0));
        if (classConsumed) break;
        this.pos++; // consume '.'
        const seg = this.expectIdent();
        parts.push(seg);
        if (isUpper) break;
      }

      const end = this.lastTokenEnd();
      const span = this.spanFrom(start, end);
      const base: Expr = { type: "MetaClass", name: parts.join("."), span };
      return this.parsePostfixWithBase(base);
    } else {
      return this.parsePow();
    }
  }

  private parsePrimary(): Expr {
    const info = this.next();
    if (!info) {
      throw this.error("unexpected end of input");
    }

    switch (info.token) {
      case Token.Integer:
      case Token.Float: {
        const span = this.spanFrom(info.position, info.end);
        return { type: "Number", value: info.lexeme, span };
      }
      case Token.Char: {
        const span = this.spanFrom(info.position, info.end);
        return { type: "Char", value: info.lexeme, span };
      }
      case Token.Str: {
        const span = this.spanFrom(info.position, info.end);
        return { type: "String", value: info.lexeme, span };
      }
      case Token.True: {
        const span = this.spanFrom(info.position, info.end);
        return { type: "Ident", name: "true", span };
      }
      case Token.False: {
        const span = this.spanFrom(info.position, info.end);
        return { type: "Ident", name: "false", span };
      }
      case Token.Ident: {
        const span = this.spanFrom(info.position, info.end);
        return { type: "Ident", name: info.lexeme, span };
      }
      case Token.End: {
        const span = this.spanFrom(info.position, info.end);
        return { type: "EndKeyword", span };
      }
      case Token.At: {
        const start = info.position;
        // Anonymous function or function handle
        if (this.consume(Token.LParen)) {
          const params: string[] = [];
          if (!this.consume(Token.RParen)) {
            params.push(this.expectIdent());
            while (this.consume(Token.Comma)) {
              params.push(this.expectIdent());
            }
            if (!this.consume(Token.RParen)) {
              throw this.error(
                "expected ')' after anonymous function parameters"
              );
            }
          }
          const body = this.parseExpr();
          const span = this.spanFrom(start, body.span.end);
          return { type: "AnonFunc", params, body, span };
        } else {
          // function handle @name or @ClassName.method
          let name = this.expectIdent();
          // Handle dotted names like @chebfun.ode113 (static method handles)
          while (this.consume(Token.Dot)) {
            name += "." + this.expectIdent();
          }
          const end = this.lastTokenEnd();
          const span = this.spanFrom(start, end);
          return { type: "FuncHandle", name, span };
        }
      }
      case Token.LParen: {
        const start = info.position;
        // Disable matrix whitespace heuristic inside parentheses:
        // [(1 -2)] should parse as subtraction, not two elements
        const priorMatrix = this.inMatrixExpr;
        this.inMatrixExpr = false;
        const expr = this.parseExpr();
        this.inMatrixExpr = priorMatrix;
        if (!this.consume(Token.RParen)) {
          throw this.error("expected ')' to close parentheses");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(start, end);
        return { ...expr, span };
      }
      case Token.LBracket: {
        const start = info.position;
        const matrix = this.parseMatrix();
        if (!this.consume(Token.RBracket)) {
          throw this.error("expected ']' to close matrix literal");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(start, end);
        return { ...matrix, span };
      }
      case Token.LBrace: {
        const start = info.position;
        const cell = this.parseCell();
        if (!this.consume(Token.RBrace)) {
          throw this.error("expected '}' to close cell literal");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(start, end);
        return { ...cell, span };
      }
      case Token.Colon: {
        const span = this.spanFrom(info.position, info.end);
        return { type: "Colon", span };
      }
      default:
        throw this.error(`unexpected token: ${info.token}`);
    }
  }

  // ── Matrix/Cell Parsing ──────────────────────────────────────────────

  private parseMatrix(): Expr {
    this.skipNewlines();
    const rows: Expr[][] = [];
    if (this.peekToken() === Token.RBracket) {
      return {
        type: "Tensor",
        rows,
        span: { file: this.fileName, start: 0, end: 0 },
      };
    }

    while (true) {
      this.skipNewlines();
      if (this.peekToken() === Token.RBracket) break;

      const row: Expr[] = [];
      row.push(this.parseMatrixExpr());

      while (true) {
        if (this.consume(Token.Comma)) {
          // Trailing comma before semicolon, newline, or closing bracket
          if (
            this.peekToken() === Token.Semicolon ||
            this.peekToken() === Token.Newline ||
            this.peekToken() === Token.RBracket
          ) {
            break;
          }
          row.push(this.parseMatrixExpr());
          continue;
        }
        if (
          this.peekToken() === Token.Semicolon ||
          this.peekToken() === Token.Newline ||
          this.peekToken() === Token.RBracket
        ) {
          break;
        }
        // Whitespace-separated
        const canStart = [
          Token.Ident,
          Token.Integer,
          Token.Float,
          Token.Char,
          Token.Str,
          Token.LParen,
          Token.LBracket,
          Token.LBrace,
          Token.At,
          Token.Plus,
          Token.Minus,
          Token.Colon,
          Token.True,
          Token.False,
          Token.End,
        ];
        if (canStart.includes(this.peekToken()!)) {
          row.push(this.parseMatrixExpr());
        } else {
          break;
        }
      }

      rows.push(row);
      if (this.consume(Token.Semicolon) || this.consume(Token.Newline)) {
        this.skipNewlines();
        continue;
      } else {
        break;
      }
    }

    this.skipNewlines();
    return {
      type: "Tensor",
      rows,
      span: { file: this.fileName, start: 0, end: 0 },
    };
  }

  private parseMatrixExpr(): Expr {
    const prior = this.inMatrixExpr;
    this.inMatrixExpr = true;
    const expr = this.parseExpr();
    this.inMatrixExpr = prior;
    return expr;
  }

  private parseCell(): Expr {
    this.skipNewlines();
    const rows: Expr[][] = [];
    if (this.peekToken() === Token.RBrace) {
      return {
        type: "Cell",
        rows,
        span: { file: this.fileName, start: 0, end: 0 },
      };
    }

    while (true) {
      this.skipNewlines();
      if (this.peekToken() === Token.RBrace) break;

      const row: Expr[] = [];
      row.push(this.parseMatrixExpr());

      while (true) {
        if (this.consume(Token.Comma)) {
          // Trailing comma before semicolon, newline, or closing brace
          if (
            this.peekToken() === Token.Semicolon ||
            this.peekToken() === Token.Newline ||
            this.peekToken() === Token.RBrace
          ) {
            break;
          }
          row.push(this.parseMatrixExpr());
          continue;
        }
        if (
          this.peekToken() === Token.Semicolon ||
          this.peekToken() === Token.Newline ||
          this.peekToken() === Token.RBrace
        ) {
          break;
        }
        // Whitespace-separated
        const canStart = [
          Token.Ident,
          Token.Integer,
          Token.Float,
          Token.Char,
          Token.Str,
          Token.LParen,
          Token.LBracket,
          Token.LBrace,
          Token.At,
          Token.Plus,
          Token.Minus,
          Token.Colon,
          Token.True,
          Token.False,
          Token.End,
        ];
        if (canStart.includes(this.peekToken()!)) {
          row.push(this.parseMatrixExpr());
        } else {
          break;
        }
      }

      rows.push(row);
      if (this.consume(Token.Semicolon) || this.consume(Token.Newline)) {
        this.skipNewlines();
        continue;
      } else {
        break;
      }
    }

    this.skipNewlines();
    return {
      type: "Cell",
      rows,
      span: { file: this.fileName, start: 0, end: 0 },
    };
  }
}
