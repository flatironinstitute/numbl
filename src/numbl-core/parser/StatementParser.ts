/**
 * StatementParser - Statement parsing methods
 */

import { Token } from "../lexer/index.js";
import { Stmt, Expr, AbstractSyntaxTree, LValue } from "./types.js";
import { ClassParser } from "./ClassParser.js";

export class StatementParser extends ClassParser {
  // ── Program ──────────────────────────────────────────────────────────

  parseProgram(): AbstractSyntaxTree {
    const body: Stmt[] = [];
    while (this.pos < this.tokens.length) {
      if (
        this.consume(Token.Semicolon) ||
        this.consume(Token.Comma) ||
        this.consume(Token.Newline)
      ) {
        continue;
      }
      body.push(this.parseStmtWithSemicolon());
    }
    return { body };
  }

  // ── Statements ───────────────────────────────────────────────────────

  private parseStmtWithSemicolon(): Stmt {
    const stmt = this.parseStmt();
    const isSemicolonTerminated = this.consume(Token.Semicolon);

    // Update suppression flag
    switch (stmt.type) {
      case "ExprStmt":
        return { ...stmt, suppressed: isSemicolonTerminated };
      case "Assign":
        return { ...stmt, suppressed: isSemicolonTerminated };
      case "MultiAssign":
        return { ...stmt, suppressed: isSemicolonTerminated };
      case "AssignLValue":
        return { ...stmt, suppressed: isSemicolonTerminated };
      default:
        return stmt;
    }
  }

  parseStmt(): Stmt {
    // "import" followed by '(' is a function call, not an import statement
    if (
      this.peekToken() === Token.Import &&
      this.peekTokenAt(1) === Token.LParen
    ) {
      this.tokens[this.pos] = { ...this.tokens[this.pos], token: Token.Ident };
    }

    const tok = this.peekToken();

    switch (tok) {
      case Token.If:
        return this.parseIf();
      case Token.For:
        return this.parseFor();
      case Token.While:
        return this.parseWhile();
      case Token.Switch:
        return this.parseSwitch();
      case Token.Try:
        return this.parseTryCatch();
      case Token.Import:
        return this.parseImport();
      case Token.ClassDef:
        return this.parseClassDef();
      case Token.Global:
        return this.parseGlobal();
      case Token.Persistent:
        return this.parsePersistent();
      case Token.Break: {
        const token = this.tokens[this.pos];
        this.pos++;
        return {
          type: "Break",
          span: this.spanFrom(token.position, token.end),
        };
      }
      case Token.Continue: {
        const token = this.tokens[this.pos];
        this.pos++;
        return {
          type: "Continue",
          span: this.spanFrom(token.position, token.end),
        };
      }
      case Token.Return: {
        const token = this.tokens[this.pos];
        this.pos++;
        return {
          type: "Return",
          span: this.spanFrom(token.position, token.end),
        };
      }
      case Token.Function:
        return this.parseFunction();
      case Token.LBracket:
        // Multi-assign like [a,b] = f() or [X{N}, W{N}] = f()
        return this.tryParseMultiAssign();
      default:
        if (
          this.peekToken() === Token.Ident &&
          this.peekTokenAt(1) === Token.Assign
        ) {
          const nameToken = this.next()!;
          if (!this.consume(Token.Assign)) {
            throw this.errorWithExpected("expected assignment operator", "'='");
          }
          const expr = this.parseExpr();
          const span = this.spanFrom(nameToken.position, expr.span.end);
          return {
            type: "Assign",
            name: nameToken.lexeme,
            expr,
            suppressed: false,
            span,
          };
        } else if (this.isSimpleAssignmentAhead()) {
          const name = this.expectIdent();
          const start = this.tokens[this.pos - 1].position;
          if (!this.consume(Token.Assign)) {
            throw this.errorWithExpected("expected assignment operator", "'='");
          }
          const expr = this.parseExpr();
          const span = this.spanFrom(start, expr.span.end);
          return { type: "Assign", name, expr, suppressed: false, span };
        } else if (this.peekToken() === Token.Ident) {
          // First, try complex lvalue assignment
          const lv = this.tryParseLValueAssign();
          if (lv) return lv;

          // Command-form at statement start
          if (this.canStartCommandForm()) {
            const nameToken = this.next()!;
            const command = COMMAND_VERBS.find(
              cmd => cmd.name.toLowerCase() === nameToken.lexeme.toLowerCase()
            );

            let args: Expr[];
            if (command) {
              // Known command verb: token-based parsing + normalization
              args = this.parseCommandArgs();
              args = this.normalizeCommandArgs(command, args);
            } else {
              // General command syntax: raw source → char vectors
              args = this.parseCommandArgsGeneral();
            }

            const end = this.lastTokenEnd();
            const span = this.spanFrom(nameToken.position, end);
            return {
              type: "ExprStmt",
              expr: { type: "FuncCall", name: nameToken.lexeme, args, span },
              suppressed: false,
              span,
            };
          } else {
            // Ambiguous adjacency check
            if (
              this.peekTokenAt(1) === Token.Ident &&
              (this.peekTokenAt(2) === Token.LParen ||
                this.peekTokenAt(2) === Token.Dot ||
                this.peekTokenAt(2) === Token.LBracket ||
                this.peekTokenAt(2) === Token.LBrace ||
                this.peekTokenAt(2) === Token.Transpose)
            ) {
              // For known command verbs, parse as verb(expr)
              // e.g., "colormap hsv(600)" → colormap(hsv(600))
              const verbName = this.tokens[this.pos]?.lexeme;
              const isCommandVerb = COMMAND_VERBS.some(
                cmd => cmd.name.toLowerCase() === verbName?.toLowerCase()
              );
              if (isCommandVerb) {
                const nameToken = this.next()!;
                const arg = this.parseExpr();
                const span = this.spanFrom(nameToken.position, arg.span.end);
                return {
                  type: "ExprStmt",
                  expr: {
                    type: "FuncCall",
                    name: nameToken.lexeme,
                    args: [arg],
                    span,
                  },
                  suppressed: false,
                  span,
                };
              }
              // General command syntax: treat as command form with raw args
              const nameToken = this.next()!;
              const args = this.parseCommandArgsGeneral();
              const end = this.lastTokenEnd();
              const span = this.spanFrom(nameToken.position, end);
              return {
                type: "ExprStmt",
                expr: {
                  type: "FuncCall",
                  name: nameToken.lexeme,
                  args,
                  span,
                },
                suppressed: false,
                span,
              };
            }
            const expr = this.parseExpr();
            const span = expr.span;
            return { type: "ExprStmt", expr, suppressed: false, span };
          }
        } else {
          const expr = this.parseExpr();
          const span = expr.span;
          return { type: "ExprStmt", expr, suppressed: false, span };
        }
    }
  }

  // ── LValue Assignment ────────────────────────────────────────────────

  private tryParseLValueAssign(): Stmt | null {
    const save = this.pos;

    if (this.peekToken() !== Token.Ident) {
      return null;
    }

    const baseToken = this.next()!;
    const baseSpan = this.spanFrom(baseToken.position, baseToken.end);
    let base: Expr = { type: "Ident", name: baseToken.lexeme, span: baseSpan };

    while (true) {
      if (this.consume(Token.LParen)) {
        const args: Expr[] = [];
        if (!this.consume(Token.RParen)) {
          args.push(this.parseExpr());
          while (this.consume(Token.Comma)) {
            args.push(this.parseExpr());
          }
          if (!this.consume(Token.RParen)) {
            // Could be Name=Value syntax inside a function call, not
            // a subscripted assignment.  Backtrack to let the expression
            // parser handle it with Name=Value desugaring.
            this.pos = save;
            return null;
          }
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(base.span.start, end);
        base = { type: "Index", base, indices: args, span };
      } else if (this.consume(Token.LBracket)) {
        const idxs: Expr[] = [];
        idxs.push(this.parseExpr());
        while (this.consume(Token.Comma)) {
          idxs.push(this.parseExpr());
        }
        if (!this.consume(Token.RBracket)) {
          throw this.errorWithExpected("expected ']'", "]");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(base.span.start, end);
        base = { type: "Index", base, indices: idxs, span };
      } else if (this.consume(Token.LBrace)) {
        const idxs: Expr[] = [];
        idxs.push(this.parseExpr());
        while (this.consume(Token.Comma)) {
          idxs.push(this.parseExpr());
        }
        if (!this.consume(Token.RBrace)) {
          throw this.errorWithExpected("expected '}'", "}");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(base.span.start, end);
        base = { type: "IndexCell", base, indices: idxs, span };
      } else if (this.peekToken() === Token.Dot) {
        // Check for .' (non-conjugate transpose) - don't consume
        if (this.peekTokenAt(1) === Token.Transpose) break;
        // Check for .+ or .- - don't consume
        if (
          this.peekTokenAt(1) === Token.Plus ||
          this.peekTokenAt(1) === Token.Minus
        )
          break;

        this.pos++; // consume '.'

        if (this.consume(Token.LParen)) {
          const nameExpr = this.parseExpr();
          if (!this.consume(Token.RParen)) {
            throw this.errorWithExpected(
              "expected ')' after dynamic field expression",
              ")"
            );
          }
          const end = this.lastTokenEnd();
          const span = this.spanFrom(base.span.start, end);
          base = { type: "MemberDynamic", base, nameExpr, span };
        } else {
          const name = this.expectMemberName();
          const end = this.lastTokenEnd();
          const span = this.spanFrom(base.span.start, end);
          base = { type: "Member", base, name, span };
        }
      } else {
        break;
      }
    }

    if (!this.consume(Token.Assign)) {
      this.pos = save;
      return null;
    }

    const rhs = this.parseExpr();
    const stmtSpan = this.spanBetween(base.span, rhs.span);

    // Convert to statement
    if (base.type === "Member") {
      return {
        type: "AssignLValue",
        lvalue: { type: "Member", base: base.base, name: base.name },
        expr: rhs,
        suppressed: false,
        span: stmtSpan,
      };
    } else if (base.type === "MemberDynamic") {
      return {
        type: "AssignLValue",
        lvalue: {
          type: "MemberDynamic",
          base: base.base,
          nameExpr: base.nameExpr,
        },
        expr: rhs,
        suppressed: false,
        span: stmtSpan,
      };
    } else if (base.type === "Index") {
      return {
        type: "AssignLValue",
        lvalue: { type: "Index", base: base.base, indices: base.indices },
        expr: rhs,
        suppressed: false,
        span: stmtSpan,
      };
    } else if (base.type === "IndexCell") {
      return {
        type: "AssignLValue",
        lvalue: { type: "IndexCell", base: base.base, indices: base.indices },
        expr: rhs,
        suppressed: false,
        span: stmtSpan,
      };
    } else if (base.type === "Ident") {
      return {
        type: "Assign",
        name: base.name,
        expr: rhs,
        suppressed: false,
        span: stmtSpan,
      };
    }

    this.pos = save;
    return null;
  }

  // ── Multi-assign ─────────────────────────────────────────────────────

  tryParseMultiAssign(): Stmt {
    const save = this.pos;

    if (!this.consume(Token.LBracket)) {
      throw this.error("not a multi-assign");
    }
    const start = this.tokens[this.pos - 1].position;
    const lvalues: LValue[] = [];

    // Parse first lvalue
    const firstLval = this.parseMultiAssignLValue();
    if (!firstLval) {
      // Not a multi-assign, rewind and parse as expression
      this.pos = save;
      const expr = this.parseExpr();
      const span = expr.span;
      return { type: "ExprStmt", expr, suppressed: false, span };
    }
    lvalues.push(firstLval);

    while (
      this.consume(Token.Comma) ||
      this.peekToken() === Token.Ident ||
      this.peekToken() === Token.Tilde
    ) {
      const lval = this.parseMultiAssignLValue();
      if (!lval) {
        // Not a multi-assign, rewind and parse as expression
        this.pos = save;
        const expr = this.parseExpr();
        const span = expr.span;
        return { type: "ExprStmt", expr, suppressed: false, span };
      }
      lvalues.push(lval);
    }

    if (!this.consume(Token.RBracket)) {
      // Not a multi-assign, rewind and parse as expression
      this.pos = save;
      const expr = this.parseExpr();
      const span = expr.span;
      return { type: "ExprStmt", expr, suppressed: false, span };
    }

    if (!this.consume(Token.Assign)) {
      // Not a multi-assign, rewind and parse as expression
      this.pos = save;
      const expr = this.parseExpr();
      const span = expr.span;
      return { type: "ExprStmt", expr, suppressed: false, span };
    }

    const rhs = this.parseExpr();
    const span = this.spanFrom(start, rhs.span.end);
    return { type: "MultiAssign", lvalues, expr: rhs, suppressed: false, span };
  }

  private parseMultiAssignLValue(): LValue | null {
    // Handle tilde (ignore output)
    if (this.consume(Token.Tilde)) {
      return { type: "Ignore" };
    }

    // Parse identifier
    if (this.peekToken() !== Token.Ident) {
      return null; // Not a valid lvalue for multi-assign
    }

    const nameToken = this.next()!;
    let base: Expr = {
      type: "Ident",
      name: nameToken.lexeme,
      span: this.spanFrom(nameToken.position, nameToken.end),
    };

    // Parse postfix operations (indexing, field access)
    while (true) {
      if (this.consume(Token.LParen)) {
        const indices: Expr[] = [];
        if (!this.consume(Token.RParen)) {
          indices.push(this.parseExpr());
          while (this.consume(Token.Comma)) {
            indices.push(this.parseExpr());
          }
          if (!this.consume(Token.RParen)) {
            throw this.errorWithExpected("expected ')' after indices", ")");
          }
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(base.span.start, end);
        base = { type: "Index", base, indices, span };
      } else if (this.consume(Token.LBracket)) {
        const indices: Expr[] = [];
        indices.push(this.parseExpr());
        while (this.consume(Token.Comma)) {
          indices.push(this.parseExpr());
        }
        if (!this.consume(Token.RBracket)) {
          throw this.errorWithExpected("expected ']' after indices", "]");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(base.span.start, end);
        base = { type: "Index", base, indices, span };
      } else if (this.consume(Token.LBrace)) {
        const indices: Expr[] = [];
        indices.push(this.parseExpr());
        while (this.consume(Token.Comma)) {
          indices.push(this.parseExpr());
        }
        if (!this.consume(Token.RBrace)) {
          throw this.errorWithExpected("expected '}' after cell indices", "}");
        }
        const end = this.lastTokenEnd();
        const span = this.spanFrom(base.span.start, end);
        base = { type: "IndexCell", base, indices, span };
      } else if (this.peekToken() === Token.Dot) {
        // Check for .' (non-conjugate transpose) - don't consume
        if (this.peekTokenAt(1) === Token.Transpose) break;
        // Check for .+ or .- - don't consume
        if (
          this.peekTokenAt(1) === Token.Plus ||
          this.peekTokenAt(1) === Token.Minus
        )
          break;

        this.pos++; // consume '.'

        if (this.consume(Token.LParen)) {
          const nameExpr = this.parseExpr();
          if (!this.consume(Token.RParen)) {
            throw this.errorWithExpected(
              "expected ')' after dynamic field expression",
              ")"
            );
          }
          const end = this.lastTokenEnd();
          const span = this.spanFrom(base.span.start, end);
          base = { type: "MemberDynamic", base, nameExpr, span };
        } else {
          const name = this.expectMemberName();
          const end = this.lastTokenEnd();
          const span = this.spanFrom(base.span.start, end);
          base = { type: "Member", base, name, span };
        }
      } else {
        break;
      }
    }

    // Convert Expr to LValue
    if (base.type === "Ident") {
      return { type: "Var", name: base.name };
    } else if (base.type === "Index") {
      return { type: "Index", base: base.base, indices: base.indices };
    } else if (base.type === "IndexCell") {
      return { type: "IndexCell", base: base.base, indices: base.indices };
    } else if (base.type === "Member") {
      return { type: "Member", base: base.base, name: base.name };
    } else if (base.type === "MemberDynamic") {
      return {
        type: "MemberDynamic",
        base: base.base,
        nameExpr: base.nameExpr,
      };
    }

    throw this.error("invalid lvalue in multi-assign");
  }
}

// Import COMMAND_VERBS for inline usage
import { COMMAND_VERBS } from "./commands.js";
