import { Token } from "../lexer/index.js";
import {
  ArgumentEntry,
  ArgumentsBlock,
  ArgumentsBlockKind,
  Expr,
  ValidationSpec,
} from "./types.js";
import { ControlFlowParser } from "./ControlFlowParser.js";

export class ArgumentsParser extends ControlFlowParser {
  // ── Arguments blocks ─────────────────────────────────────────────────

  /**
   * Parse all consecutive arguments blocks at the top of a function body.
   * Returns an array of ArgumentsBlock (may be empty if none present).
   */
  parseArgumentsBlocks(): ArgumentsBlock[] {
    const blocks: ArgumentsBlock[] = [];
    // Skip newlines before the first arguments block (and between consecutive blocks)
    this.consumeLineEnds();
    while (this.peekToken() === Token.Arguments) {
      blocks.push(this.parseOneArgumentsBlock());
      this.consumeLineEnds();
    }
    return blocks;
  }

  private parseOneArgumentsBlock(): ArgumentsBlock {
    this.consume(Token.Arguments);

    // Parse optional attribute list: (Input), (Repeating), (Output), (Output,Repeating), etc.
    const kind = this.parseArgumentsBlockKind();

    // Skip trailing newline/semicolon after the block header
    this.consumeLineEnd();

    const entries: ArgumentEntry[] = [];

    while (true) {
      // Skip blank lines
      this.consumeLineEnds();

      // End of block
      if (this.peekToken() === Token.End) {
        this.consume(Token.End);
        break;
      }
      if (this.peekToken() === undefined) break;

      entries.push(this.parseArgumentEntry());
    }

    return { kind, entries };
  }

  private parseArgumentsBlockKind(): ArgumentsBlockKind {
    if (!this.consume(Token.LParen)) {
      return "Input";
    }

    // Read attribute names until RParen
    const attrs: string[] = [];
    while (
      this.peekToken() !== Token.RParen &&
      this.peekToken() !== undefined
    ) {
      if (this.consume(Token.Comma)) continue;
      if (this.peekToken() === Token.Ident) {
        attrs.push(this.next()!.lexeme);
      } else {
        // Unexpected token, bail out
        break;
      }
    }
    this.consume(Token.RParen);

    const lower = attrs.map(a => a.toLowerCase());
    const hasOutput = lower.includes("output");
    const hasRepeating = lower.includes("repeating");

    if (hasOutput && hasRepeating) return "OutputRepeating";
    if (hasOutput) return "Output";
    if (hasRepeating) return "Repeating";
    return "Input";
  }

  private parseArgumentEntry(): ArgumentEntry {
    // Argument name, or "struct.field" for a name-value argument.
    let name = this.expectIdent();
    if (this.consume(Token.Dot)) {
      const field = this.expectIdent();
      name = `${name}.${field}`;
    }
    const spec = this.parseValidationSpec();

    // Optional default value: = expr
    let defaultValue = null;
    if (this.consume(Token.Assign)) {
      defaultValue = this.parseExpr();
    }
    this.consumeLineEnd();
    return { name, ...spec, defaultValue };
  }

  /**
   * Parse the constraints that follow an argument or property name:
   * `(dims) className {validators}`, each part optional. Shared by
   * `arguments` blocks and classdef `properties` blocks.
   */
  protected parseValidationSpec(): ValidationSpec {
    let dimensions: Expr[] | null = null;
    if (this.peekToken() === Token.LParen) {
      dimensions = this.parseArgDimensions();
    }

    // Class name: an identifier on the same line, possibly dotted
    // (`matlab.io.xml.dom.Element`).
    let className: string | null = null;
    if (this.peekToken() === Token.Ident) {
      className = this.next()!.lexeme;
      while (this.peekToken() === Token.Dot) {
        this.consume(Token.Dot);
        className += "." + this.expectIdent();
      }
    }

    // Validators: {mustBeNumeric, mustBeInRange(x,0,10)}, parsed as a cell
    // literal whose elements are the validator expressions.
    const validators: Expr[] = [];
    if (this.peekToken() === Token.LBrace) {
      const cell = this.parseExpr();
      if (cell.type !== "Cell") {
        throw this.error("expected a validator list '{...}'");
      }
      for (const row of cell.rows) validators.push(...row);
    }
    return { dimensions, className, validators };
  }

  /** Parse a size specification: `(1,:)` → [Number 1, Colon]. */
  private parseArgDimensions(): Expr[] {
    this.consume(Token.LParen);
    const dims: Expr[] = [];
    if (this.consume(Token.RParen)) return dims;
    while (true) {
      const tok = this.peek();
      const after = this.tokens[this.pos + 1]?.token;
      if (
        tok?.token === Token.Colon &&
        (after === Token.Comma || after === Token.RParen)
      ) {
        this.next();
        dims.push({
          type: "Colon",
          span: this.spanFrom(tok.position, tok.end),
        });
      } else {
        dims.push(this.parseExpr());
      }
      if (this.consume(Token.Comma)) continue;
      if (this.consume(Token.RParen)) break;
      throw this.error("expected ',' or ')' in size specification");
    }
    return dims;
  }

  /**
   * Consume a single line terminator (newline or semicolon).
   */
  private consumeLineEnd(): void {
    if (
      this.consume(Token.Newline) ||
      this.consume(Token.Semicolon) ||
      this.consume(Token.Comma)
    ) {
      // consumed a line end
    }
  }

  /**
   * Consume all consecutive line terminators.
   */
  private consumeLineEnds(): void {
    while (
      this.consume(Token.Newline) ||
      this.consume(Token.Semicolon) ||
      this.consume(Token.Comma)
    ) {
      // continue
    }
  }
}
