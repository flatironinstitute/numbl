import { describe, it, expect } from "vitest";
import {
  tokenize,
  tokenizeDetailed,
  Token,
  isAlpha,
  isAlnum,
  isDigit,
  isWhitespace,
  isValueToken,
  findLineTerminator,
  KEYWORDS,
  VALUE_KEYWORDS,
} from "../numbl-core/lexer/index.js";
import {
  buildTwoCharMap,
  buildSingleCharMap,
  getSingleCharSpecial,
  TOKEN_CONFIG,
} from "../numbl-core/lexer/token-config.js";

// ── tokenize ──────────────────────────────────────────────────────────

describe("tokenize", () => {
  it("tokenizes a simple assignment", () => {
    const tokens = tokenize("x = 42");
    expect(tokens).toContain(Token.Ident);
    expect(tokens).toContain(Token.Assign);
    expect(tokens).toContain(Token.Integer);
  });

  it("tokenizes arithmetic operators", () => {
    const tokens = tokenize("a + b - c * d / e");
    const ops = tokens.filter(
      t =>
        t === Token.Plus ||
        t === Token.Minus ||
        t === Token.Star ||
        t === Token.Slash
    );
    expect(ops).toHaveLength(4);
  });

  it("tokenizes a single-quoted string (char array)", () => {
    const tokens = tokenize("'hello'");
    expect(tokens).toContain(Token.Char);
  });

  it("tokenizes a double-quoted string", () => {
    const tokens = tokenize('"hello"');
    expect(tokens).toContain(Token.Str);
  });

  it("tokenizes a float literal", () => {
    const tokens = tokenize("3.14");
    expect(tokens).toContain(Token.Float);
  });

  it("tokenizes keywords", () => {
    const tokens = tokenize("if true end");
    expect(tokens).toContain(Token.If);
    expect(tokens).toContain(Token.True);
    expect(tokens).toContain(Token.End);
  });

  it("tokenizes brackets and parens", () => {
    const tokens = tokenize("[1, 2] (3)");
    expect(tokens).toContain(Token.LBracket);
    expect(tokens).toContain(Token.RBracket);
    expect(tokens).toContain(Token.LParen);
    expect(tokens).toContain(Token.RParen);
  });

  it("strips newlines from output", () => {
    const tokens = tokenize("a\nb");
    expect(tokens).not.toContain(Token.Newline);
  });

  it("tokenizes two-character operators", () => {
    const tokens = tokenize("a .* b ./ c .^ d");
    expect(tokens).toContain(Token.DotStar);
    expect(tokens).toContain(Token.DotSlash);
    expect(tokens).toContain(Token.DotCaret);
  });

  it("tokenizes comparison operators", () => {
    const tokens = tokenize("a == b ~= c <= d >= e");
    expect(tokens).toContain(Token.Equal);
    expect(tokens).toContain(Token.NotEqual);
    expect(tokens).toContain(Token.LessEqual);
    expect(tokens).toContain(Token.GreaterEqual);
  });

  it("tokenizes logical operators", () => {
    const tokens = tokenize("a && b || c");
    expect(tokens).toContain(Token.AndAnd);
    expect(tokens).toContain(Token.OrOr);
  });

  it("tokenizes bitwise operators", () => {
    const tokens = tokenize("a & b | c");
    expect(tokens).toContain(Token.And);
    expect(tokens).toContain(Token.Or);
  });

  it("tokenizes braces", () => {
    const tokens = tokenize("{1, 2}");
    expect(tokens).toContain(Token.LBrace);
    expect(tokens).toContain(Token.RBrace);
  });

  it("tokenizes colon", () => {
    const tokens = tokenize("1:10");
    expect(tokens).toContain(Token.Colon);
  });

  it("tokenizes tilde", () => {
    const tokens = tokenize("~x");
    expect(tokens).toContain(Token.Tilde);
  });

  it("tokenizes at sign", () => {
    const tokens = tokenize("@(x) x");
    expect(tokens).toContain(Token.At);
  });

  it("tokenizes dot operator", () => {
    const tokens = tokenize("a.b");
    expect(tokens).toContain(Token.Dot);
  });

  it("tokenizes backslash", () => {
    const tokens = tokenize("A \\ b");
    expect(tokens).toContain(Token.Backslash);
  });

  it("tokenizes dot-backslash", () => {
    const tokens = tokenize("A .\\ b");
    expect(tokens).toContain(Token.DotBackslash);
  });

  it("tokenizes caret (power)", () => {
    const tokens = tokenize("a ^ b");
    expect(tokens).toContain(Token.Caret);
  });

  it("tokenizes all keyword tokens", () => {
    for (const [keyword, token] of KEYWORDS) {
      const tokens = tokenize(keyword);
      expect(tokens).toContain(token);
    }
  });

  it("tokenizes less and greater", () => {
    const tokens = tokenize("a < b > c");
    expect(tokens).toContain(Token.Less);
    expect(tokens).toContain(Token.Greater);
  });
});

// ── tokenizeDetailed ──────────────────────────────────────────────────

describe("tokenizeDetailed", () => {
  it("returns SpannedToken objects with position info", () => {
    const tokens = tokenizeDetailed("x = 1");
    expect(tokens.length).toBeGreaterThan(0);
    for (const t of tokens) {
      expect(t).toHaveProperty("token");
      expect(t).toHaveProperty("lexeme");
      expect(t).toHaveProperty("start");
      expect(t).toHaveProperty("end");
    }
  });

  it("captures correct lexemes", () => {
    const tokens = tokenizeDetailed("myVar = 99");
    const identToken = tokens.find(t => t.token === Token.Ident);
    expect(identToken).toBeDefined();
    expect(identToken!.lexeme).toBe("myVar");
  });

  it("captures span positions correctly", () => {
    const tokens = tokenizeDetailed("abc");
    const tok = tokens.find(t => t.token === Token.Ident);
    expect(tok).toBeDefined();
    expect(tok!.start).toBe(0);
    expect(tok!.end).toBe(3);
  });

  it("includes newline tokens", () => {
    const tokens = tokenizeDetailed("a\nb");
    expect(tokens.some(t => t.token === Token.Newline)).toBe(true);
  });

  it("coalesces multiple newlines", () => {
    const tokens = tokenizeDetailed("a\n\n\nb");
    const newlines = tokens.filter(t => t.token === Token.Newline);
    expect(newlines).toHaveLength(1);
  });

  it("handles line comments", () => {
    const tokens = tokenizeDetailed("x = 1 % this is a comment\ny = 2");
    const idents = tokens.filter(t => t.token === Token.Ident);
    expect(idents.map(t => t.lexeme)).toEqual(["x", "y"]);
  });

  it("handles block comments", () => {
    const tokens = tokenizeDetailed("x = 1\n%{\nblock comment\n%}\ny = 2");
    const idents = tokens.filter(t => t.token === Token.Ident);
    expect(idents.map(t => t.lexeme)).toEqual(["x", "y"]);
  });

  it("handles %{ with content on same line as line comment", () => {
    const tokens = tokenizeDetailed("x = 1\n%{ not a block comment\ny = 2");
    const idents = tokens.filter(t => t.token === Token.Ident);
    expect(idents.map(t => t.lexeme)).toEqual(["x", "y"]);
  });

  it("handles ellipsis (line continuation)", () => {
    const tokens = tokenizeDetailed("x = 1 + ...\n2");
    const ellipsis = tokens.find(t => t.token === Token.Ellipsis);
    expect(ellipsis).toBeDefined();
  });

  it("handles ellipsis at end of file", () => {
    const tokens = tokenizeDetailed("x ...");
    const ellipsis = tokens.find(t => t.token === Token.Ellipsis);
    expect(ellipsis).toBeDefined();
  });

  it("handles section markers %%", () => {
    const tokens = tokenizeDetailed("%% Section Title\nx = 1");
    const section = tokens.find(t => t.token === Token.Section);
    expect(section).toBeDefined();
    expect(section!.lexeme).toContain("Section Title");
  });

  it("section marker not at line start treated as comment", () => {
    const tokens = tokenizeDetailed("x = 1 %% not a section");
    const section = tokens.find(t => t.token === Token.Section);
    expect(section).toBeUndefined();
  });

  it("tokenizes transpose as Transpose after value token", () => {
    const tokens = tokenizeDetailed("x'");
    const transpose = tokens.find(t => t.token === Token.Transpose);
    expect(transpose).toBeDefined();
  });

  it("tokenizes single quote as Char when not adjacent to value", () => {
    const tokens = tokenizeDetailed("x = 'hello'");
    expect(tokens.some(t => t.token === Token.Char)).toBe(true);
  });

  it("handles escaped quotes in strings", () => {
    const tokens = tokenizeDetailed("'it''s'");
    const charTok = tokens.find(t => t.token === Token.Char);
    expect(charTok).toBeDefined();
    expect(charTok!.lexeme).toBe("'it''s'");
  });

  it("handles escaped double quotes in strings", () => {
    const tokens = tokenizeDetailed('"say ""hi"""');
    const strTok = tokens.find(t => t.token === Token.Str);
    expect(strTok).toBeDefined();
  });

  it("produces Error token for unterminated string", () => {
    const tokens = tokenizeDetailed("'unterminated\n");
    expect(tokens.some(t => t.token === Token.Error)).toBe(true);
  });

  it("produces Error token for unknown character", () => {
    const tokens = tokenizeDetailed("$");
    expect(tokens.some(t => t.token === Token.Error)).toBe(true);
  });

  it("tokenizes leading decimal point as float (.5)", () => {
    const tokens = tokenizeDetailed(".5");
    const floatTok = tokens.find(t => t.token === Token.Float);
    expect(floatTok).toBeDefined();
    expect(floatTok!.lexeme).toBe(".5");
  });

  it("tokenizes scientific notation", () => {
    const tokens = tokenizeDetailed("1e10");
    const tok = tokens.find(t => t.token === Token.Float);
    expect(tok).toBeDefined();
    expect(tok!.lexeme).toBe("1e10");
  });

  it("tokenizes scientific notation with sign", () => {
    const tokens = tokenizeDetailed("2.5e-3");
    const tok = tokens.find(t => t.token === Token.Float);
    expect(tok).toBeDefined();
    expect(tok!.lexeme).toBe("2.5e-3");
  });

  it("tokenizes 1..3 as float(1.) dot integer(3)", () => {
    const tokens = tokenizeDetailed("1..3");
    // "1." is a float, then "." is Dot, then "3" is integer
    expect(tokens[0].token).toBe(Token.Float);
    expect(tokens[1].token).toBe(Token.Dot);
    expect(tokens[2].token).toBe(Token.Integer);
  });

  it("handles decimal point before element-wise operator", () => {
    const tokens = tokenizeDetailed("1.*2");
    expect(tokens[0].token).toBe(Token.Integer);
    expect(tokens[1].token).toBe(Token.DotStar);
  });

  it("handles 'end' followed by '(' as identifier", () => {
    const tokens = tokenizeDetailed("end(1)");
    expect(tokens[0].token).toBe(Token.Ident);
    expect(tokens[0].lexeme).toBe("end");
  });

  it("handles semicolon followed by string", () => {
    const tokens = tokenizeDetailed("x; 'hello'");
    expect(tokens.some(t => t.token === Token.Char)).toBe(true);
  });

  it("handles unclosed block comment", () => {
    const tokens = tokenizeDetailed("%{\nunclosed block comment");
    expect(tokens).toBeDefined();
  });
});

// ── helpers ───────────────────────────────────────────────────────────

describe("lexer helpers", () => {
  it("isAlpha recognizes letters and underscore", () => {
    expect(isAlpha("a")).toBe(true);
    expect(isAlpha("Z")).toBe(true);
    expect(isAlpha("_")).toBe(true);
    expect(isAlpha("3")).toBe(false);
    expect(isAlpha(" ")).toBe(false);
  });

  it("isAlnum recognizes letters, underscore, and digits", () => {
    expect(isAlnum("a")).toBe(true);
    expect(isAlnum("5")).toBe(true);
    expect(isAlnum("_")).toBe(true);
    expect(isAlnum(" ")).toBe(false);
  });

  it("isDigit recognizes digits", () => {
    expect(isDigit("0")).toBe(true);
    expect(isDigit("9")).toBe(true);
    expect(isDigit("a")).toBe(false);
  });

  it("isWhitespace recognizes space and tab", () => {
    expect(isWhitespace(" ")).toBe(true);
    expect(isWhitespace("\t")).toBe(true);
    expect(isWhitespace("\r")).toBe(true);
    expect(isWhitespace("a")).toBe(false);
    expect(isWhitespace("\n")).toBe(false);
  });

  it("isValueToken returns true for value tokens", () => {
    expect(isValueToken(Token.Ident)).toBe(true);
    expect(isValueToken(Token.Integer)).toBe(true);
    expect(isValueToken(Token.Float)).toBe(true);
    expect(isValueToken(Token.True)).toBe(true);
    expect(isValueToken(Token.False)).toBe(true);
    expect(isValueToken(Token.RParen)).toBe(true);
    expect(isValueToken(Token.RBracket)).toBe(true);
    expect(isValueToken(Token.RBrace)).toBe(true);
    expect(isValueToken(Token.Str)).toBe(true);
    expect(isValueToken(Token.Char)).toBe(true);
    expect(isValueToken(Token.Transpose)).toBe(true);
  });

  it("isValueToken returns false for non-value tokens", () => {
    expect(isValueToken(Token.Plus)).toBe(false);
    expect(isValueToken(Token.LParen)).toBe(false);
    expect(isValueToken(Token.Assign)).toBe(false);
  });

  it("KEYWORDS map contains expected entries", () => {
    expect(KEYWORDS.get("if")).toBe(Token.If);
    expect(KEYWORDS.get("for")).toBe(Token.For);
    expect(KEYWORDS.get("while")).toBe(Token.While);
    expect(KEYWORDS.get("function")).toBe(Token.Function);
    expect(KEYWORDS.get("end")).toBe(Token.End);
    expect(KEYWORDS.get("else")).toBe(Token.Else);
    expect(KEYWORDS.get("elseif")).toBe(Token.ElseIf);
    expect(KEYWORDS.get("break")).toBe(Token.Break);
    expect(KEYWORDS.get("continue")).toBe(Token.Continue);
    expect(KEYWORDS.get("return")).toBe(Token.Return);
    expect(KEYWORDS.get("switch")).toBe(Token.Switch);
    expect(KEYWORDS.get("case")).toBe(Token.Case);
    expect(KEYWORDS.get("otherwise")).toBe(Token.Otherwise);
    expect(KEYWORDS.get("try")).toBe(Token.Try);
    expect(KEYWORDS.get("catch")).toBe(Token.Catch);
    expect(KEYWORDS.get("classdef")).toBe(Token.ClassDef);
    expect(KEYWORDS.get("global")).toBe(Token.Global);
    expect(KEYWORDS.get("persistent")).toBe(Token.Persistent);
    expect(KEYWORDS.get("import")).toBe(Token.Import);
  });

  it("VALUE_KEYWORDS contains true and false", () => {
    expect(VALUE_KEYWORDS.has(Token.True)).toBe(true);
    expect(VALUE_KEYWORDS.has(Token.False)).toBe(true);
    expect(VALUE_KEYWORDS.has(Token.If)).toBe(false);
  });
});

// ── findLineTerminator ────────────────────────────────────────────────

describe("findLineTerminator", () => {
  it("finds newline", () => {
    expect(findLineTerminator("abc\ndef")).toEqual([3, 1]);
  });

  it("finds carriage return + newline", () => {
    expect(findLineTerminator("abc\r\ndef")).toEqual([3, 2]);
  });

  it("finds carriage return alone", () => {
    expect(findLineTerminator("abc\rdef")).toEqual([3, 1]);
  });

  it("returns undefined when no terminator", () => {
    expect(findLineTerminator("abcdef")).toBeUndefined();
  });

  it("supports offset parameter", () => {
    expect(findLineTerminator("ab\ncd\nef", 4)).toEqual([5, 1]);
  });

  it("returns undefined with offset past all terminators", () => {
    expect(findLineTerminator("ab\ncd", 4)).toBeUndefined();
  });
});

// ── token-config ──────────────────────────────────────────────────────

describe("token-config", () => {
  it("buildTwoCharMap returns map of two-char operators", () => {
    const map = buildTwoCharMap();
    expect(map.get(".*")).toBe(Token.DotStar);
    expect(map.get("./")).toBe(Token.DotSlash);
    expect(map.get(".^")).toBe(Token.DotCaret);
    expect(map.get("==")).toBe(Token.Equal);
    expect(map.get("~=")).toBe(Token.NotEqual);
    expect(map.get("<=")).toBe(Token.LessEqual);
    expect(map.get(">=")).toBe(Token.GreaterEqual);
    expect(map.get("&&")).toBe(Token.AndAnd);
    expect(map.get("||")).toBe(Token.OrOr);
  });

  it("buildSingleCharMap returns map of single-char operators", () => {
    const map = buildSingleCharMap();
    expect(map.get("+")).toBe(Token.Plus);
    expect(map.get("-")).toBe(Token.Minus);
    expect(map.get("*")).toBe(Token.Star);
    expect(map.get("/")).toBe(Token.Slash);
    expect(map.get("=")).toBe(Token.Assign);
    expect(map.get(";")).toBe(Token.Semicolon);
    expect(map.get(",")).toBe(Token.Comma);
    expect(map.get("(")).toBe(Token.LParen);
    expect(map.get(")")).toBe(Token.RParen);
    expect(map.get("[")).toBe(Token.LBracket);
    expect(map.get("]")).toBe(Token.RBracket);
    expect(map.get("{")).toBe(Token.LBrace);
    expect(map.get("}")).toBe(Token.RBrace);
  });

  it("getSingleCharSpecial returns special flag for semicolon", () => {
    const special = getSingleCharSpecial(";");
    expect(special).toBe("semicolonQuoteHeuristic");
  });

  it("getSingleCharSpecial returns undefined for non-special", () => {
    expect(getSingleCharSpecial("+")).toBeUndefined();
  });

  it("TOKEN_CONFIG has expected structure", () => {
    expect(TOKEN_CONFIG.operators).toBeDefined();
    expect(TOKEN_CONFIG.comments).toBeDefined();
    expect(TOKEN_CONFIG.strings).toBeDefined();
    expect(TOKEN_CONFIG.special).toBeDefined();
    expect(TOKEN_CONFIG.numbers).toBeDefined();
    expect(TOKEN_CONFIG.identifiers).toBeDefined();
  });
});
