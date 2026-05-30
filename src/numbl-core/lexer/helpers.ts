/**
 * Helper functions for the lexer.
 */

import { Token } from "./types.js";

export function isAlpha(ch: string): boolean {
  return (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z") || ch === "_";
}

export function isAlnum(ch: string): boolean {
  return isAlpha(ch) || (ch >= "0" && ch <= "9");
}

export function isDigit(ch: string): boolean {
  return ch >= "0" && ch <= "9";
}

export function isWhitespace(ch: string): boolean {
  return ch === " " || ch === "\t" || ch === "\r";
}

/** Check if a token represents a "value" for transpose disambiguation. */
export function isValueToken(tok: Token): boolean {
  return (
    tok === Token.Ident ||
    tok === Token.Integer ||
    tok === Token.Float ||
    tok === Token.True ||
    tok === Token.False ||
    tok === Token.End ||
    tok === Token.RParen ||
    tok === Token.RBracket ||
    tok === Token.RBrace ||
    tok === Token.Str ||
    tok === Token.Char ||
    tok === Token.Transpose
  );
}

/**
 * Find the first line terminator (\n or \r\n or \r) in `s`.
 * Returns [index, length] or undefined.
 */
export function findLineTerminator(
  s: string,
  offset = 0
): [number, number] | undefined {
  for (let i = offset; i < s.length; i++) {
    if (s[i] === "\n") return [i, 1];
    if (s[i] === "\r") {
      return s[i + 1] === "\n" ? [i, 2] : [i, 1];
    }
  }
  return undefined;
}
