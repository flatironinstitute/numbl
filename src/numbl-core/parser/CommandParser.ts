/**
 * CommandParser - Command form parsing methods
 */

import { Token } from "../lexer/index.js";
import { Expr } from "./types.js";
import { ExpressionParser } from "./ExpressionParser.js";

export class CommandParser extends ExpressionParser {
  // ── Command Form ─────────────────────────────────────────────────────

  /**
   * Detect whether the current position starts a command-form statement.
   *
   * Uses raw source scanning: an identifier followed by horizontal whitespace
   * and then non-operator content is command form.  Zero-arg invocations
   * (e.g. `hold` alone) are handled by the expression parser + lowering
   * which resolves bare identifiers to zero-arg function calls.
   */
  canStartCommandForm(): boolean {
    const current = this.tokens[this.pos];
    if (!current || current.token !== Token.Ident) return false;

    // --- raw-source lookahead ---
    let srcPos = current.end;

    // Must have at least one horizontal whitespace char after the identifier
    if (
      srcPos >= this.input.length ||
      (this.input[srcPos] !== " " && this.input[srcPos] !== "\t")
    ) {
      return false;
    }

    // Skip horizontal whitespace
    while (
      srcPos < this.input.length &&
      (this.input[srcPos] === " " || this.input[srcPos] === "\t")
    ) {
      srcPos++;
    }

    // End-of-statement / end-of-input → not command form (zero-arg handled
    // by expression parser resolving bare Ident to function call)
    if (srcPos >= this.input.length) return false;
    const ch = this.input[srcPos];
    if (ch === "\n" || ch === "\r" || ch === ";" || ch === "%") {
      return false;
    }

    // Assignment → not command form  (but == is comparison, not assignment)
    if (ch === "=" && this.input[srcPos + 1] !== "=") return false;

    // Opening paren → function call syntax, not command form
    if (ch === "(") return false;

    // Check if first arg looks like a command argument vs an operator.
    // Command args start with: letter, digit, quote, or special prefixes.
    // Operators start with: +, *, /, \, ^, &, |, <, >, ~, ?, :, [, {
    if (this.isCommandArgStart(srcPos)) {
      return true;
    }

    // Not recognized as a command argument start → not command form
    return false;
  }

  /**
   * Parse command arguments from the raw source text.
   *
   * Each whitespace-delimited token becomes a Char (character vector) node,
   * matching MATLAB's command-syntax semantics.  Single-quoted strings
   * preserve interior spaces.
   *
   * Call this AFTER consuming the verb token with next().
   */
  parseCommandArgsGeneral(): Expr[] {
    const args: Expr[] = [];

    // Start scanning right after the verb token
    const verbToken = this.tokens[this.pos - 1];
    let scanPos = verbToken.end;

    while (scanPos < this.input.length) {
      const ch = this.input[scanPos];

      // Skip horizontal whitespace
      if (ch === " " || ch === "\t") {
        scanPos++;
        continue;
      }

      // Stop at end-of-statement markers
      if (ch === "\n" || ch === "\r" || ch === ";" || ch === "%") break;

      // Handle , as argument separator (skip it)
      if (ch === ",") {
        scanPos++;
        continue;
      }

      // Handle line continuation (... in raw source)
      if (
        ch === "." &&
        scanPos + 2 < this.input.length &&
        this.input[scanPos + 1] === "." &&
        this.input[scanPos + 2] === "."
      ) {
        scanPos += 3;
        // Skip to next line
        while (scanPos < this.input.length && this.input[scanPos] !== "\n") {
          scanPos++;
        }
        if (scanPos < this.input.length) scanPos++; // skip the newline
        continue;
      }

      // Parse one argument
      const argStart = scanPos;

      if (ch === "'") {
        // Single-quoted string: scan to matching closing quote
        // Handle '' as escaped quote inside
        scanPos++; // skip opening quote
        while (scanPos < this.input.length) {
          if (this.input[scanPos] === "'") {
            if (
              scanPos + 1 < this.input.length &&
              this.input[scanPos + 1] === "'"
            ) {
              scanPos += 2; // escaped quote ''
            } else {
              scanPos++; // closing quote
              break;
            }
          } else {
            scanPos++;
          }
        }
        const text = this.input.substring(argStart, scanPos);
        const span = this.spanFrom(argStart, scanPos);
        args.push({ type: "Char", value: text, span });
      } else {
        // Unquoted argument: read non-whitespace until delimiter
        while (scanPos < this.input.length) {
          const c = this.input[scanPos];
          if (
            c === " " ||
            c === "\t" ||
            c === "\n" ||
            c === "\r" ||
            c === ";" ||
            c === "%" ||
            c === ","
          ) {
            break;
          }
          // Check for ... (line continuation)
          if (
            c === "." &&
            scanPos + 2 < this.input.length &&
            this.input[scanPos + 1] === "." &&
            this.input[scanPos + 2] === "."
          ) {
            break;
          }
          scanPos++;
        }
        const text = this.input.substring(argStart, scanPos);
        const span = this.spanFrom(argStart, scanPos);
        args.push({ type: "Char", value: `'${text}'`, span });
      }
    }

    // Advance token stream past all tokens consumed by raw scanning
    while (this.pos < this.tokens.length) {
      const tok = this.tokens[this.pos];
      if (tok.position >= scanPos) break;
      this.pos++;
    }

    return args;
  }

  /**
   * Check whether the character at `pos` in `this.input` can start a
   * command-syntax argument (as opposed to a binary operator).
   */
  private isCommandArgStart(pos: number): boolean {
    const ch = this.input[pos];
    if (!ch) return false;

    // Letters, underscore, digits → always a command arg
    if (/[A-Za-z_0-9]/.test(ch)) return true;

    // Quoted strings
    if (ch === "'" || ch === '"') return true;

    const next = pos + 1 < this.input.length ? this.input[pos + 1] : "";

    // Dash followed by letter/underscore → flag (e.g. -file)
    if (ch === "-" && /[A-Za-z_]/.test(next)) return true;

    // Dot followed by / or \ → relative path (e.g. ./ .\)
    // Dot followed by . → parent dir or ellipsis (e.g. .. ...)
    // Dot followed by letter → hidden file (e.g. .gitignore)
    if (ch === "." && /[./\\A-Za-z_]/.test(next)) return true;

    // Tilde followed by / or \ → home path (e.g. ~/path)
    if (ch === "~" && (next === "/" || next === "\\")) return true;

    // Everything else (operators: + * / \ ^ & | < > ~ ? : [ { etc.)
    return false;
  }
}
