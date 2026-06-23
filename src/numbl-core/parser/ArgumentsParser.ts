import { Token } from "../lexer/index.js";
import { ArgumentEntry, ArgumentsBlock, ArgumentsBlockKind } from "./types.js";
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
    // Parse argument name (possibly "struct.field" for name-value args)
    let name = this.expectIdent();
    if (this.consume(Token.Dot)) {
      const field = this.expectIdent();
      name = `${name}.${field}`;
    }

    // Parse optional dimensions: (1,:) or (m,n,p) etc.
    let dimensions: string[] | null = null;
    if (this.peekToken() === Token.LParen) {
      dimensions = this.parseArgDimensions();
    }

    // Parse optional class name (an identifier on the same line, not { or =).
    // May be namespace-qualified, e.g. `matlab.io.xml.dom.Element`.
    let className: string | null = null;
    if (this.peekToken() === Token.Ident) {
      className = this.next()!.lexeme;
      while (this.peekToken() === Token.Dot) {
        this.consume(Token.Dot);
        className += "." + this.expectIdent();
      }
    }

    // Parse optional validators: {mustBeNumeric, ...}
    let validators: string[] = [];
    if (this.peekToken() === Token.LBrace) {
      validators = this.parseArgValidators();
    }

    // Parse optional default value: = expr
    let defaultValue = null;
    if (this.consume(Token.Assign)) {
      defaultValue = this.parseExpr();
    }

    // Consume end of line
    this.consumeLineEnd();

    return { name, dimensions, className, validators, defaultValue };
  }

  /**
   * Parse dimension list: (1,:) → ["1", ":"]
   */
  private parseArgDimensions(): string[] {
    this.consume(Token.LParen);
    const dims: string[] = [];

    while (
      this.peekToken() !== Token.RParen &&
      this.peekToken() !== undefined
    ) {
      if (this.consume(Token.Comma)) continue;
      if (this.peekToken() === Token.Colon) {
        this.next();
        dims.push(":");
      } else if (
        this.peekToken() === Token.Integer ||
        this.peekToken() === Token.Float
      ) {
        dims.push(this.next()!.lexeme);
      } else if (this.peekToken() === Token.Ident) {
        // Allow named dimension like 'n' (uncommon, treat as string)
        dims.push(this.next()!.lexeme);
      } else {
        break;
      }
    }

    this.consume(Token.RParen);
    return dims;
  }

  /**
   * Parse validator list: {mustBeNumeric, mustBePositive}. Validators may be
   * full function calls with arguments, e.g.
   * `{mustBeMember(x,["a","b"]), mustBeInRange(x,0,7)}`. The validator bodies
   * are not enforced at runtime, so we capture the top-level validator name
   * tokens and skip past any nested `(...)`, `[...]`, `{...}` so parsing
   * resumes correctly at the default value (`= expr`) or end of line.
   */
  private parseArgValidators(): string[] {
    this.consume(Token.LBrace);
    const validators: string[] = [];
    let depth = 0; // nesting depth inside the validator braces

    while (this.peekToken() !== undefined) {
      const tok = this.peekToken();
      if (depth === 0 && tok === Token.RBrace) break;

      if (
        tok === Token.LParen ||
        tok === Token.LBracket ||
        tok === Token.LBrace
      ) {
        depth++;
      } else if (
        tok === Token.RParen ||
        tok === Token.RBracket ||
        tok === Token.RBrace
      ) {
        depth--;
      } else if (depth === 0 && tok === Token.Ident) {
        // Top-level identifier is a validator function name.
        validators.push(this.peek()!.lexeme);
      }
      this.next();
    }

    this.consume(Token.RBrace);
    return validators;
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
