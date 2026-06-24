/**
 * ControlFlowParser - Control flow parsing methods (if/while/for/switch/try-catch)
 */

import { Token } from "../lexer/index.js";
import { Stmt, Expr } from "./types.js";
import { CommandParser } from "./CommandParser.js";

export class ControlFlowParser extends CommandParser {
  // ── Control Flow ─────────────────────────────────────────────────────

  parseIf(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.If);
    const cond = this.parseExpr();
    const thenBody = this.parseBlock(
      t => t === Token.Else || t === Token.ElseIf || t === Token.End
    );

    const elseifBlocks: Array<{ cond: Expr; body: Stmt[] }> = [];
    while (this.consume(Token.ElseIf)) {
      const c = this.parseExpr();
      const body = this.parseBlock(
        t => t === Token.Else || t === Token.ElseIf || t === Token.End
      );
      elseifBlocks.push({ cond: c, body });
    }

    let elseBody: Stmt[] | null = null;
    if (this.consume(Token.Else)) {
      elseBody = this.parseBlock(t => t === Token.End);
    }

    if (!this.consume(Token.End)) {
      throw this.error("expected 'end'");
    }
    const end = this.lastTokenEnd();
    return {
      type: "If",
      cond,
      thenBody,
      elseifBlocks,
      elseBody,
      span: this.spanFrom(start, end),
    };
  }

  parseWhile(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.While);
    const cond = this.parseExpr();
    const body = this.parseBlock(t => t === Token.End);
    if (!this.consume(Token.End)) {
      throw this.error("expected 'end'");
    }
    const end = this.lastTokenEnd();
    return { type: "While", cond, body, span: this.spanFrom(start, end) };
  }

  parseFor(): Stmt {
    const start = this.tokens[this.pos].position;
    // `parfor` parses exactly like `for` and runs serially (results are
    // order-independent). Accept both opener tokens.
    if (!this.consume(Token.For)) {
      this.consume(Token.ParFor);
    }
    const hasParen = this.consume(Token.LParen);
    const varName = this.expectIdent();
    if (!this.consume(Token.Assign)) {
      throw this.error("expected '='");
    }
    const expr = this.parseExpr();
    // parfor's optional worker-count argument, e.g. `parfor (n=1:N, W)` — parse
    // and ignore it (serial execution is correct for order-independent bodies).
    if (hasParen && this.consume(Token.Comma)) {
      this.parseExpr();
    }
    if (hasParen && !this.consume(Token.RParen)) {
      throw this.error("expected ')' to close for loop");
    }
    const body = this.parseBlock(t => t === Token.End);
    if (!this.consume(Token.End)) {
      throw this.error("expected 'end'");
    }
    const end = this.lastTokenEnd();
    return {
      type: "For",
      varName,
      expr,
      body,
      span: this.spanFrom(start, end),
    };
  }

  parseSwitch(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.Switch);
    const control = this.parseExpr();

    const cases: Array<{ value: Expr; body: Stmt[] }> = [];
    let otherwise: Stmt[] | null = null;

    while (true) {
      if (this.consume(Token.Newline) || this.consume(Token.Semicolon))
        continue;
      if (this.consume(Token.Case)) {
        const val = this.parseExpr();
        const body = this.parseBlock(
          t => t === Token.Case || t === Token.Otherwise || t === Token.End
        );
        cases.push({ value: val, body });
      } else if (this.consume(Token.Otherwise)) {
        otherwise = this.parseBlock(t => t === Token.End);
      } else if (this.consume(Token.Comma)) {
        continue;
      } else {
        break;
      }
    }

    if (!this.consume(Token.End)) {
      throw this.error("expected 'end' for switch");
    }
    const end = this.lastTokenEnd();
    return {
      type: "Switch",
      expr: control,
      cases,
      otherwise,
      span: this.spanFrom(start, end),
    };
  }

  parseTryCatch(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.Try);
    const tryBody = this.parseBlock(t => t === Token.Catch || t === Token.End);
    let catchVar: string | null = null;
    let catchBody: Stmt[] = [];
    if (this.consume(Token.Catch)) {
      // `catch e` captures the exception only when the identifier stands
      // alone on the line (followed by a separator or `end`). `catch e = ...`
      // or `catch e(...)` instead begins the catch body with a statement, so
      // the identifier must not be swallowed as the exception variable.
      if (this.peekToken() === Token.Ident) {
        const after = this.peekTokenAt(1);
        if (
          after === undefined ||
          after === Token.Newline ||
          after === Token.Semicolon ||
          after === Token.Comma ||
          after === Token.End
        ) {
          catchVar = this.expectIdent();
        }
      }
      catchBody = this.parseBlock(t => t === Token.End);
    }
    if (!this.consume(Token.End)) {
      throw this.error("expected 'end' after try");
    }
    const end = this.lastTokenEnd();
    return {
      type: "TryCatch",
      tryBody,
      catchVar,
      catchBody,
      span: this.spanFrom(start, end),
    };
  }

  parseBlock(term: (t: Token) => boolean): Stmt[] {
    const body: Stmt[] = [];
    while (this.peekToken() !== undefined) {
      if (term(this.peekToken()!)) break;
      if (
        this.consume(Token.Semicolon) ||
        this.consume(Token.Comma) ||
        this.consume(Token.Newline)
      ) {
        continue;
      }

      let stmt: Stmt;
      if (this.peekToken() === Token.LBracket) {
        stmt = this.tryParseMultiAssign();
      } else {
        stmt = this.parseStmt();
      }

      const isSemicolonTerminated = this.consume(Token.Semicolon);
      switch (stmt.type) {
        case "ExprStmt":
          body.push({ ...stmt, suppressed: isSemicolonTerminated });
          break;
        case "Assign":
          body.push({ ...stmt, suppressed: isSemicolonTerminated });
          break;
        case "MultiAssign":
          body.push({ ...stmt, suppressed: isSemicolonTerminated });
          break;
        case "AssignLValue":
          body.push({ ...stmt, suppressed: isSemicolonTerminated });
          break;
        default:
          body.push(stmt);
      }
    }
    return body;
  }

  // Forward declarations for methods implemented in other parsers
  protected parseStmt(): Stmt {
    throw new Error("parseStmt must be implemented by StatementParser");
  }

  protected tryParseMultiAssign(): Stmt {
    throw new Error(
      "tryParseMultiAssign must be implemented by StatementParser"
    );
  }
}
