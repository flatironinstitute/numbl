import { Token } from "./types.js";

// ── Keyword map ─────────────────────────────────────────────────────────

export const KEYWORDS: ReadonlyMap<string, Token> = new Map([
  ["function", Token.Function],
  ["if", Token.If],
  ["else", Token.Else],
  ["elseif", Token.ElseIf],
  ["for", Token.For],
  ["parfor", Token.ParFor],
  ["while", Token.While],
  ["break", Token.Break],
  ["continue", Token.Continue],
  ["return", Token.Return],
  ["end", Token.End],
  ["classdef", Token.ClassDef],
  ["properties", Token.Properties],
  ["methods", Token.Methods],
  ["events", Token.Events],
  ["enumeration", Token.Enumeration],
  ["arguments", Token.Arguments],
  ["import", Token.Import],
  ["switch", Token.Switch],
  ["case", Token.Case],
  ["otherwise", Token.Otherwise],
  ["try", Token.Try],
  ["catch", Token.Catch],
  ["global", Token.Global],
  ["persistent", Token.Persistent],
  ["true", Token.True],
  ["false", Token.False],
]);

// Tokens that set last_was_value = true when used as a keyword
export const VALUE_KEYWORDS: ReadonlySet<Token> = new Set([
  Token.True,
  Token.False,
]);
