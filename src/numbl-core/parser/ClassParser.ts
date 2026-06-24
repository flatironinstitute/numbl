/**
 * ClassParser - Class and import definition parsing methods
 */

import { Token } from "../lexer/index.js";
import { Stmt, Expr, Attr, ClassMember, MethodSignature } from "./types.js";
import { FunctionParser } from "./FunctionParser.js";

/** Well-known MATLAB base classes that confer handle (reference) semantics.
 *  A class deriving from any of these is a handle class. */
const HANDLE_BASE_CLASSES = new Set([
  "handle",
  "dynamicprops",
  "matlab.mixin.Copyable",
  "matlab.mixin.SetGet",
  "matlab.mixin.SetGetExactNames",
  "hgsetget",
]);

export class ClassParser extends FunctionParser {
  // ── Imports & ClassDef ───────────────────────────────────────────────

  parseImport(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.Import);

    const path: string[] = [];
    path.push(this.expectIdent());
    let wildcard = false;

    while (true) {
      if (this.consume(Token.DotStar)) {
        wildcard = true;
        break;
      }
      if (this.consume(Token.Dot)) {
        if (this.consume(Token.Star)) {
          wildcard = true;
          break;
        } else {
          path.push(this.expectIdent());
          continue;
        }
      }
      break;
    }

    const end = this.lastTokenEnd();
    return { type: "Import", path, wildcard, span: this.spanFrom(start, end) };
  }

  parseClassDef(): Stmt {
    const start = this.tokens[this.pos].position;
    this.consume(Token.ClassDef);
    const classAttributes = this.parseOptionalAttrList();
    const name = this.parseQualifiedName();

    let superClass: string | null = null;
    if (this.consume(Token.Less)) {
      // A class may list multiple superclasses joined by `&` (multiple
      // inheritance), e.g. `handle & matlab.mixin.CustomDisplay`. numbl models
      // only a single superclass; if any super confers handle (reference)
      // semantics, normalize to `handle` so downstream handle detection works —
      // otherwise keep the first.
      const supers: string[] = [this.parseQualifiedName()];
      while (this.consume(Token.And)) {
        supers.push(this.parseQualifiedName());
      }
      superClass = supers.some(s => HANDLE_BASE_CLASSES.has(s))
        ? "handle"
        : supers[0];
    }

    const members: ClassMember[] = [];
    while (true) {
      if (
        this.consume(Token.Semicolon) ||
        this.consume(Token.Comma) ||
        this.consume(Token.Newline)
      ) {
        continue;
      }

      const tok = this.peekToken();
      if (tok === Token.Properties) {
        this.pos++;
        const attrs = this.parseOptionalAttrList();
        const props = this.parsePropertiesNamesBlock();
        if (!this.consume(Token.End)) {
          throw this.error("expected 'end' after properties");
        }
        members.push({
          type: "Properties",
          attributes: attrs,
          names: props.map(p => p.name),
          defaultValues: props.map(p => p.defaultValue),
        });
      } else if (tok === Token.Methods) {
        this.pos++;
        const attrs = this.parseOptionalAttrList();

        // Check if this is an Abstract methods block
        const isAbstract = attrs.some(
          attr =>
            attr.name.toLowerCase() === "abstract" &&
            (attr.value === null || attr.value === "true")
        );

        if (isAbstract) {
          // Parse method signatures for abstract methods (no function bodies)
          const signatures = this.parseMethodSignatures();
          if (!this.consume(Token.End)) {
            throw this.error("expected 'end' after methods");
          }
          members.push({
            type: "Methods",
            attributes: attrs,
            body: [],
            signatures,
          });
        } else {
          // Parse methods block - can contain both inline functions and signatures
          const body: Stmt[] = [];
          const signatures: MethodSignature[] = [];

          while (
            this.peekToken() !== undefined &&
            this.peekToken() !== Token.End
          ) {
            if (
              this.consume(Token.Semicolon) ||
              this.consume(Token.Comma) ||
              this.consume(Token.Newline)
            ) {
              continue;
            }

            // Check if next item is a function definition
            if (this.peekToken() === Token.Function) {
              const stmt = this.parseStmt();
              body.push(stmt);
            } else if (
              this.peekToken() === Token.Ident ||
              this.peekToken() === Token.Import ||
              this.peekToken() === Token.LBracket
            ) {
              // Parse as a method signature
              const sig = this.parseSingleMethodSignature();
              if (sig) {
                signatures.push(sig);
              }
            } else {
              break;
            }
          }

          if (!this.consume(Token.End)) {
            throw this.error("expected 'end' after methods");
          }

          members.push({
            type: "Methods",
            attributes: attrs,
            body,
            signatures: signatures.length > 0 ? signatures : undefined,
          });
        }
      } else if (tok === Token.Events) {
        this.pos++;
        const attrs = this.parseOptionalAttrList();
        const names = this.parseNameBlock();
        if (!this.consume(Token.End)) {
          throw this.error("expected 'end' after events");
        }
        members.push({ type: "Events", attributes: attrs, names });
      } else if (tok === Token.Enumeration) {
        this.pos++;
        const attrs = this.parseOptionalAttrList();
        const names = this.parseNameBlock();
        if (!this.consume(Token.End)) {
          throw this.error("expected 'end' after enumeration");
        }
        members.push({ type: "Enumeration", attributes: attrs, names });
      } else if (tok === Token.Arguments) {
        this.pos++;
        const attrs = this.parseOptionalAttrList();
        const names = this.parseNameBlock();
        if (!this.consume(Token.End)) {
          throw this.error("expected 'end' after arguments");
        }
        members.push({ type: "Arguments", attributes: attrs, names });
      } else if (tok === Token.End) {
        this.pos++;
        break;
      } else {
        break;
      }
    }

    const end = this.lastTokenEnd();
    return {
      type: "ClassDef",
      name,
      classAttributes,
      superClass,
      members,
      span: this.spanFrom(start, end),
    };
  }

  private parseNameBlock(): string[] {
    const names: string[] = [];
    while (this.peekToken() !== undefined) {
      if (this.peekToken() === Token.End) break;
      if (
        this.consume(Token.Semicolon) ||
        this.consume(Token.Comma) ||
        this.consume(Token.Newline)
      ) {
        continue;
      }
      if (this.peekToken() === Token.Ident) {
        names.push(this.expectIdent());
      } else {
        break;
      }
    }
    return names;
  }

  private parsePropertiesNamesBlock(): {
    name: string;
    defaultValue: Expr | null;
  }[] {
    const props: { name: string; defaultValue: Expr | null }[] = [];
    while (this.peekToken() !== undefined) {
      if (this.peekToken() === Token.End) break;
      if (
        this.consume(Token.Semicolon) ||
        this.consume(Token.Comma) ||
        this.consume(Token.Newline)
      ) {
        continue;
      }
      if (this.peekToken() === Token.Ident) {
        const name = this.expectIdent();

        // Optional MATLAB property validation syntax (R2019b+):
        //   PropName (dims) Type {validators} = default
        // numbl does not enforce size/type/validators, but it must still
        // consume them — otherwise a bare type annotation gets mis-read as a
        // second property (and the real property loses its default) and a
        // dims/validators clause raises a hard syntax error that drops the
        // whole class. Parse and discard them; keep name + default.

        // (dims) — e.g. (1,1) or (:,2)
        if (this.peekToken() === Token.LParen) {
          this.skipBalanced(Token.LParen, Token.RParen);
        }
        // Type — an identifier on the same line, possibly dotted (pkg.Class)
        if (this.peekToken() === Token.Ident) {
          this.next();
          while (this.consume(Token.Dot)) {
            this.expectIdent();
          }
        }
        // {validators} — e.g. {mustBePositive, mustBeFinite}
        if (this.peekToken() === Token.LBrace) {
          this.skipBalanced(Token.LBrace, Token.RBrace);
        }

        let defaultValue: Expr | null = null;
        if (this.consume(Token.Assign)) {
          defaultValue = this.parseExpr();
        }
        props.push({ name, defaultValue });
      } else {
        break;
      }
    }
    return props;
  }

  /** Consume a balanced `open ... close` run (handling nesting). Assumes the
   *  next token is `open`; no-op otherwise. */
  private skipBalanced(open: Token, close: Token): void {
    if (!this.consume(open)) return;
    let depth = 1;
    while (this.peekToken() !== undefined && depth > 0) {
      const tok = this.next()!;
      if (tok.token === open) depth++;
      else if (tok.token === close) depth--;
    }
  }

  private parseOptionalAttrList(): Attr[] {
    const attrs: Attr[] = [];
    if (!this.consume(Token.LParen)) {
      return attrs;
    }

    while (true) {
      if (this.consume(Token.RParen)) break;
      if (this.peekToken() === Token.Ident) {
        const name = this.expectIdent();
        let value: string | null = null;
        if (this.consume(Token.Assign)) {
          // Handle brace-delimited values like {?ClassName1, ?ClassName2}
          if (this.peekToken() === Token.LBrace) {
            let braceDepth = 0;
            const parts: string[] = [];
            while (this.peekToken() !== undefined) {
              const tok = this.next()!;
              parts.push(tok.lexeme);
              if (tok.token === Token.LBrace) braceDepth++;
              else if (tok.token === Token.RBrace) {
                braceDepth--;
                if (braceDepth === 0) break;
              }
            }
            value = parts.join("");
          } else {
            const tok = this.next();
            if (tok) {
              value = tok.lexeme;
            }
          }
        }
        attrs.push({ name, value });
        this.consume(Token.Comma);
      } else if (this.consume(Token.Comma)) {
        continue;
      } else if (this.peekToken() === Token.RParen) {
        this.pos++;
        break;
      } else if (this.peekToken() !== undefined) {
        this.pos++;
      } else {
        break;
      }
    }

    return attrs;
  }

  /**
   * Parse a single method signature.
   * Returns null if parsing fails or current position doesn't look like a signature.
   */
  private parseSingleMethodSignature(): MethodSignature | null {
    const start = this.tokens[this.pos].position;
    const outputs: string[] = [];

    // Check for output arguments
    if (this.consume(Token.LBracket)) {
      // Empty output list: `function [] = name(...)` (no outputs)
      if (this.peekToken() !== Token.RBracket) {
        // Multiple outputs: [out1, out2, ...] =
        outputs.push(this.expectIdent());
        while (this.consume(Token.Comma)) {
          outputs.push(this.expectIdent());
        }
      }
      if (!this.consume(Token.RBracket)) {
        throw this.errorWithExpected(
          "expected ']' after output arguments",
          "]"
        );
      }
      if (!this.consume(Token.Assign)) {
        throw this.errorWithExpected(
          "expected '=' after output arguments",
          "="
        );
      }
    } else if (this.peekToken() === Token.Ident) {
      // Could be: output = name(params) or just name(params)
      // Look ahead to check if there's an '=' after the identifier
      const savedPos = this.pos;
      const maybeOutput = this.expectIdent();
      if (this.consume(Token.Assign)) {
        // Single output: output = name(params)
        outputs.push(maybeOutput);
      } else {
        // No output, restore position: name(params)
        this.pos = savedPos;
      }
    }

    // Parse method name
    if (this.peekToken() !== Token.Ident && this.peekToken() !== Token.Import) {
      return null;
    }
    let name = this.expectIdent();
    // Support property accessor methods: get.PropertyName / set.PropertyName
    if ((name === "get" || name === "set") && this.peekToken() === Token.Dot) {
      this.consume(Token.Dot);
      name = `${name}.${this.expectIdent()}`;
    }

    // Parse input parameters
    const params: string[] = [];
    if (this.consume(Token.LParen)) {
      if (!this.consume(Token.RParen)) {
        params.push(this.expectIdent());
        while (this.consume(Token.Comma)) {
          params.push(this.expectIdent());
        }
        if (!this.consume(Token.RParen)) {
          throw this.errorWithExpected("expected ')' after parameters", ")");
        }
      }
    }

    const end = this.lastTokenEnd();
    return { name, params, outputs, span: this.spanFrom(start, end) };
  }

  private parseMethodSignatures(): MethodSignature[] {
    const signatures: MethodSignature[] = [];

    while (this.peekToken() !== undefined) {
      if (this.peekToken() === Token.End) break;
      if (
        this.consume(Token.Semicolon) ||
        this.consume(Token.Comma) ||
        this.consume(Token.Newline)
      ) {
        continue;
      }

      const sig = this.parseSingleMethodSignature();
      if (sig) {
        signatures.push(sig);
      } else {
        break;
      }
    }

    return signatures;
  }
}
