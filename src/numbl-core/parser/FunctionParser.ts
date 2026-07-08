/**
 * FunctionParser - Function definition parsing methods
 */

import { Token } from "../lexer/index.js";
import { Stmt } from "./types.js";
import { ArgumentsParser } from "./ArgumentsParser.js";

export class FunctionParser extends ArgumentsParser {
  // ── Functions ────────────────────────────────────────────────────────

  parseFunction(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.Function);

    const outputs: string[] = [];
    if (this.consume(Token.LBracket)) {
      // Empty output list: `function [] = name(...)` (no outputs).
      // Output names may be separated by commas or just whitespace, e.g.
      // `function [a, b] = f()` and `function [a b] = f()` are both valid.
      // Only continue while the next token can begin another output name, so
      // malformed lists still report the expected ']' below.
      if (this.peekToken() !== Token.RBracket) {
        outputs.push(this.expectIdentOrTilde());
        while (
          this.peekToken() === Token.Comma ||
          this.peekToken() === Token.Ident ||
          this.peekToken() === Token.Tilde
        ) {
          this.consume(Token.Comma); // optional separator
          outputs.push(this.expectIdentOrTilde());
        }
      }
      if (!this.consume(Token.RBracket)) {
        throw this.error("expected ']'");
      }
      if (!this.consume(Token.Assign)) {
        throw this.error("expected '='");
      }
    } else if (
      this.peekToken() === Token.Ident &&
      this.peekTokenAt(1) === Token.Assign
    ) {
      outputs.push(this.next()!.lexeme);
      this.consume(Token.Assign);
    }

    let name = this.expectIdent();
    // Support property accessor methods: get.PropertyName / set.PropertyName
    if ((name === "get" || name === "set") && this.peekToken() === Token.Dot) {
      this.consume(Token.Dot);
      name = `${name}.${this.expectIdent()}`;
    }

    const params: string[] = [];
    if (this.consume(Token.LParen)) {
      if (!this.consume(Token.RParen)) {
        params.push(this.expectIdentOrTilde());
        while (this.consume(Token.Comma)) {
          params.push(this.expectIdentOrTilde());
        }
        if (!this.consume(Token.RParen)) {
          throw this.error("expected ')'");
        }
      }
    }

    // Enforce varargs placement constraints
    const vararginIdx = params.indexOf("varargin");
    if (vararginIdx !== -1 && params.filter(p => p === "varargin").length > 1) {
      throw this.error("'varargin' cannot appear more than once");
    }
    const varargoutIdx = outputs.indexOf("varargout");
    if (varargoutIdx !== -1) {
      if (varargoutIdx !== outputs.length - 1) {
        throw this.error("'varargout' must be the last output parameter");
      }
      if (outputs.filter(o => o === "varargout").length > 1) {
        throw this.error("'varargout' cannot appear more than once");
      }
    }

    // Optional arguments blocks (must appear before any executable statements)
    const argumentsBlocks = this.parseArgumentsBlocks();

    // varargin must be the last input parameter, except that name-value
    // struct parameters (declared via dotted `arguments` entries like
    // `options.field`) may follow it, e.g. `function f(a, varargin, options)`.
    if (vararginIdx !== -1 && vararginIdx !== params.length - 1) {
      const nvBases = new Set<string>();
      for (const block of argumentsBlocks) {
        if (block.kind === "Output" || block.kind === "OutputRepeating")
          continue;
        for (const e of block.entries) {
          const dot = e.name.indexOf(".");
          if (dot >= 0) nvBases.add(e.name.slice(0, dot));
        }
      }
      for (const p of params.slice(vararginIdx + 1)) {
        if (!nvBases.has(p)) {
          throw this.error("'varargin' must be the last input parameter");
        }
      }
    }

    const body = this.parseBlock(t => t === Token.End);
    if (!this.consume(Token.End)) {
      throw this.error("expected 'end'");
    }
    const end = this.lastTokenEnd();
    return {
      type: "Function",
      name,
      functionId: name + "_" + Math.random().toString(36).substring(2, 10),
      params,
      outputs,
      body,
      argumentsBlocks,
      span: this.spanFrom(start, end),
    };
  }

  // ── Global/Persistent ────────────────────────────────────────────────

  parseGlobal(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.Global);
    const names: string[] = [];
    names.push(this.expectIdent());
    while (true) {
      if (this.consume(Token.Comma)) {
        names.push(this.expectIdent());
        continue;
      }
      if (this.peekToken() === Token.Ident) {
        names.push(this.expectIdent());
        continue;
      }
      break;
    }
    const end = this.lastTokenEnd();
    return { type: "Global", names, span: this.spanFrom(start, end) };
  }

  parsePersistent(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.Persistent);
    const names: string[] = [];
    names.push(this.expectIdent());
    while (true) {
      if (this.consume(Token.Comma)) {
        names.push(this.expectIdent());
        continue;
      }
      if (this.peekToken() === Token.Ident) {
        names.push(this.expectIdent());
        continue;
      }
      break;
    }
    const end = this.lastTokenEnd();
    return { type: "Persistent", names, span: this.spanFrom(start, end) };
  }
}
