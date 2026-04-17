// ── Token enum ──────────────────────────────────────────────────────────

export enum Token {
  // Literals
  Integer,
  Float,
  Str,
  Char,
  Ident,

  // Keywords
  Function,
  If,
  Else,
  ElseIf,
  For,
  While,
  Break,
  Continue,
  Return,
  End,
  ClassDef,
  Properties,
  Methods,
  Events,
  Enumeration,
  Arguments,
  Import,
  Switch,
  Case,
  Otherwise,
  Try,
  Catch,
  Global,
  Persistent,
  True,
  False,

  // Single-character operators / punctuation
  Plus,
  Minus,
  Star,
  Slash,
  Backslash,
  Caret,
  And,
  Or,
  Tilde,
  At,
  Question,
  Less,
  Greater,
  Dot,
  Colon,
  Comma,
  Assign,
  Semicolon,
  LParen,
  RParen,
  LBracket,
  RBracket,
  LBrace,
  RBrace,

  // Two-character operators
  DotStar,
  DotSlash,
  DotBackslash,
  DotCaret,
  AndAnd,
  OrOr,
  Equal,
  NotEqual,
  LessEqual,
  GreaterEqual,

  // Special
  Transpose,
  Ellipsis,
  Newline,
  Section,
  Error,

  // Directives (magic comments like %!numbl:assert_jit)
  Directive,
}

// ── Spanned token ───────────────────────────────────────────────────────

export interface SpannedToken {
  token: Token;
  lexeme: string;
  start: number;
  end: number;
}
