/**
 * CommandParser - Command form parsing methods
 */

import { Token } from "../lexer/index.js";
import { Expr } from "./types.js";
import { COMMAND_VERBS, CommandVerb, extractKeyword } from "./commands.js";
import { ExpressionParser } from "./ExpressionParser.js";

export class CommandParser extends ExpressionParser {
  // ── Command Form ─────────────────────────────────────────────────────

  canStartCommandForm(): boolean {
    const current = this.tokens[this.pos];
    if (!current) return false;

    const verb = current.lexeme;
    const command = this.lookupCommand(verb);
    const zeroArgAllowed =
      command?.argKind.type === "Any" ||
      (command?.argKind.type === "Keyword" && command.argKind.optional);

    let i = 1;
    let sawArg = false;

    // Skip ellipsis (line continuation), but NOT newlines – newlines terminate commands
    while (this.peekTokenAt(i) === Token.Ellipsis) {
      i++;
      // Skip the newline after ellipsis (it's a continuation)
      if (this.peekTokenAt(i) === Token.Newline) i++;
    }

    // If the very next non-ellipsis token is a newline, there are no args on this line
    if (this.peekTokenAt(i) === Token.Newline) {
      if (!zeroArgAllowed) return false;
      return true;
    }

    // At least one simple arg must follow
    const argTokens = [
      Token.Ident,
      Token.Integer,
      Token.Float,
      Token.Char,
      Token.Str,
      Token.End,
      Token.Global,
      Token.Persistent,
    ];
    if (!argTokens.includes(this.peekTokenAt(i)!)) {
      if (!zeroArgAllowed) return false;
    } else {
      sawArg = true;
    }

    // Consume all contiguous simple args (on the same line)
    while (true) {
      const tok = this.peekTokenAt(i);
      if (argTokens.includes(tok!)) {
        sawArg = true;
        i++;
      } else if (tok === Token.Ellipsis) {
        i++;
        // Skip the newline after ellipsis (it's a continuation)
        if (this.peekTokenAt(i) === Token.Newline) i++;
      } else {
        // Newlines and everything else stop the lookahead
        break;
      }
    }

    if (!sawArg && !zeroArgAllowed) return false;

    // If next token begins indexing/member, do not use command-form
    const nextTok = this.peekTokenAt(i);
    if (
      nextTok === Token.LParen ||
      nextTok === Token.Dot ||
      nextTok === Token.LBracket ||
      nextTok === Token.LBrace ||
      nextTok === Token.Transpose ||
      nextTok === Token.Assign
    ) {
      return false;
    }

    return true;
  }

  parseCommandArgs(): Expr[] {
    const args: Expr[] = [];
    while (true) {
      // Newlines terminate command arguments (like semicolons)
      if (this.peekToken() === Token.Newline) break;
      // Ellipsis is line continuation – skip it and the following newline
      if (this.consume(Token.Ellipsis)) {
        this.consume(Token.Newline);
        continue;
      }

      const tok = this.peekToken();
      if (tok === Token.Ident) {
        const token = this.next()!;
        const span = this.spanFrom(token.position, token.end);
        args.push({ type: "Ident", name: token.lexeme, span });
      } else if (tok === Token.End) {
        const token = this.tokens[this.pos];
        this.pos++;
        const span = this.spanFrom(token.position, token.end);
        args.push({ type: "Ident", name: "end", span });
      } else if (tok === Token.Global || tok === Token.Persistent) {
        const token = this.next()!;
        const span = this.spanFrom(token.position, token.end);
        args.push({ type: "Ident", name: token.lexeme, span });
      } else if (tok === Token.Integer || tok === Token.Float) {
        const token = this.next()!;
        const span = this.spanFrom(token.position, token.end);
        args.push({ type: "Number", value: token.lexeme, span });
      } else if (tok === Token.Char) {
        const token = this.next()!;
        const span = this.spanFrom(token.position, token.end);
        args.push({ type: "Char", value: token.lexeme, span });
      } else if (tok === Token.Str) {
        const token = this.next()!;
        const span = this.spanFrom(token.position, token.end);
        args.push({ type: "String", value: token.lexeme, span });
      } else if (
        tok === Token.Slash ||
        tok === Token.Star ||
        tok === Token.Backslash ||
        tok === Token.Plus ||
        tok === Token.Minus ||
        tok === Token.LParen ||
        tok === Token.Dot ||
        tok === Token.LBracket ||
        tok === Token.LBrace ||
        tok === Token.Transpose
      ) {
        break;
      } else {
        break;
      }
    }
    return args;
  }

  private lookupCommand(name: string): CommandVerb | undefined {
    return COMMAND_VERBS.find(
      cmd => cmd.name.toLowerCase() === name.toLowerCase()
    );
  }

  normalizeCommandArgs(command: CommandVerb, args: Expr[]): Expr[] {
    if (command.argKind.type === "Keyword") {
      const { allowed, optional } = command.argKind;
      if (args.length === 0) {
        if (!optional) {
          throw this.error(
            `'${command.name}' command syntax requires an argument`
          );
        }
        return args;
      }
      if (args.length > 1 && !command.argKind.multiKeyword) {
        throw this.error(
          `'${command.name}' command syntax accepts only one argument`
        );
      }
      // Validate and normalize all keyword args
      const keywords: string[] = [];
      for (const arg of args) {
        const keyword = extractKeyword(arg);
        if (!keyword) {
          throw this.error(
            `'${command.name}' command syntax expects a keyword argument`
          );
        }
        if (
          allowed !== undefined &&
          !allowed.some(a => a.toLowerCase() === keyword.toLowerCase())
        ) {
          throw this.error(
            `'${command.name}' command syntax does not support '${keyword}'`
          );
        }
        keywords.push(keyword);
      }
      if (keywords.length === 1) {
        const span = args[0].span;
        return [{ type: "String", value: `"${keywords[0]}"`, span }];
      }
      // Multiple keywords: join with space (e.g. "axis equal tight")
      const span = args[0].span;
      return [{ type: "String", value: `"${keywords.join(" ")}"`, span }];
    }
    return args;
  }
}
