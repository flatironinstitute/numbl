/* eslint-disable no-empty */
/**
 * ParserBase - Core parser state and utility methods
 */

import { Token, KEYWORDS } from "../lexer/index.js";
import {
  Span,
  TokenInfo,
  Expr,
  BinaryOperation,
  UnaryOperation,
} from "./types.js";
import { SyntaxError } from "./errors.js";
import { offsetToLine } from "../runtime/index.js";

const KEYWORD_TOKENS: ReadonlySet<Token> = new Set(KEYWORDS.values());

export class ParserBase {
  protected tokens: TokenInfo[];
  protected pos: number;
  protected input: string;
  protected fileName: string;
  protected inMatrixExpr: boolean;

  constructor(
    tokens: TokenInfo[],
    input: string,
    fileName: string = "script.m"
  ) {
    this.tokens = tokens;
    this.pos = 0;
    this.input = input;
    this.fileName = fileName;
    this.inMatrixExpr = false;
  }

  // ── Token Navigation ─────────────────────────────────────────────────

  protected peek(): TokenInfo | undefined {
    return this.tokens[this.pos];
  }

  protected currentPosition(): number {
    const tok = this.peek();
    return tok ? tok.position : this.input.length;
  }

  protected peekToken(): Token | undefined {
    return this.tokens[this.pos]?.token;
  }

  protected peekTokenAt(offset: number): Token | undefined {
    return this.tokens[this.pos + offset]?.token;
  }

  protected next(): TokenInfo | undefined {
    if (this.pos < this.tokens.length) {
      const info = this.tokens[this.pos];
      this.pos++;
      return info;
    }
    return undefined;
  }

  protected consume(t: Token): boolean {
    if (this.peekToken() === t) {
      this.pos++;
      return true;
    }
    return false;
  }

  // ── Helper Methods ───────────────────────────────────────────────────

  protected skipNewlines(): void {
    while (this.consume(Token.Newline)) {}
  }

  protected tokensAdjacent(left: number, right: number): boolean {
    const a = this.tokens[left];
    const b = this.tokens[right];
    if (a && b) {
      return a.end === b.position;
    }
    return false;
  }

  protected spanFrom(start: number, end: number): Span {
    return { file: this.fileName, start, end };
  }

  protected spanBetween(start: Span, end: Span): Span {
    return { file: start.file, start: start.start, end: end.end };
  }

  protected lastTokenEnd(): number {
    const idx = this.pos - 1;
    if (idx >= 0 && idx < this.tokens.length) {
      return this.tokens[idx].end;
    }
    return this.input.length;
  }

  protected makeBinary(left: Expr, op: BinaryOperation, right: Expr): Expr {
    const span = this.spanBetween(left.span, right.span);
    return { type: "Binary", left, op, right, span };
  }

  protected makeUnary(
    op: UnaryOperation,
    operand: Expr,
    opStart: number
  ): Expr {
    const span = this.spanFrom(opStart, operand.span.end);
    return { type: "Unary", op, operand, span };
  }

  protected isSimpleAssignmentAhead(): boolean {
    return (
      this.peekToken() === Token.Ident && this.peekTokenAt(1) === Token.Assign
    );
  }

  protected error(message: string): SyntaxError {
    const tok = this.peek();
    const position = this.currentPosition();
    const line = offsetToLine(this.input, position);
    return new SyntaxError(message, position, tok?.lexeme ?? null, null, line);
  }

  protected errorWithExpected(message: string, expected: string): SyntaxError {
    const tok = this.peek();
    const position = this.currentPosition();
    const line = offsetToLine(this.input, position);
    return new SyntaxError(
      message,
      position,
      tok?.lexeme ?? null,
      expected,
      line
    );
  }

  // ── Utility Methods ──────────────────────────────────────────────────

  protected expectIdent(): string {
    const tok = this.next();
    if (tok?.token === Token.Ident || tok?.token === Token.Import) {
      return tok.lexeme;
    }
    throw this.error("expected identifier");
  }

  /** Accept an identifier or any keyword token as a member name (after '.') */
  protected expectMemberName(): string {
    const tok = this.next();
    if (tok && (tok.token === Token.Ident || KEYWORD_TOKENS.has(tok.token))) {
      return tok.lexeme;
    }
    throw this.error("expected member name after '.'");
  }

  protected expectIdentOrTilde(): string {
    const tok = this.next();
    if (tok?.token === Token.Ident) {
      return tok.lexeme;
    }
    if (tok?.token === Token.Tilde) {
      return "~";
    }
    throw this.error("expected identifier or '~'");
  }

  protected parseQualifiedName(): string {
    const parts: string[] = [];
    parts.push(this.expectIdent());
    while (this.consume(Token.Dot)) {
      parts.push(this.expectIdent());
    }
    return parts.join(".");
  }
}
