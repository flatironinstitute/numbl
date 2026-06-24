import { tokenizeDetailed, Token } from "../lexer/index.js";
import { AbstractSyntaxTree, TokenInfo } from "./types.js";
import { SyntaxError } from "./errors.js";
import { offsetToLine } from "../runtime/index.js";
import { Parser } from "./Parser.js";

// Re-export all types
export * from "./types.js";
export { SyntaxError } from "./errors.js";

/**
 * Parse code with default options.
 */
export function parseMFile(
  input: string,
  fileName: string = "script.m"
): AbstractSyntaxTree {
  return parseWithOptions(input, fileName);
}

/**
 * Parse code with custom options.
 *
 * Uses a two-pass approach for function end-keyword detection:
 * 1. First try parsing assuming all functions use 'end'
 * 2. If that fails with "expected 'end'", try inserting synthetic 'end' tokens
 *    before each top-level 'function' keyword and at EOF, then re-parse.
 *    Functions can omit 'end'.
 */
function parseWithOptions(
  input: string,
  fileName: string = "script.m"
): AbstractSyntaxTree {
  const tokens = tokenizeInput(input);

  // First pass: try with functions requiring 'end'
  try {
    const parser = new Parser(tokens, input, fileName);
    return parser.parseProgram();
  } catch (e) {
    if (!(e instanceof SyntaxError && e.message === "expected 'end'")) {
      throw e;
    }
  }

  // Second pass: insert synthetic 'end' tokens and re-parse
  const fixedTokens = insertFunctionEndTokens(tokens);
  if (!fixedTokens) {
    // Could not determine valid insertion points — re-throw original error
    const parser = new Parser(tokens, input, fileName);
    return parser.parseProgram(); // will throw the original error
  }

  const parser = new Parser(fixedTokens, input, fileName);
  return parser.parseProgram();
}

/**
 * Contextual keywords: words that MATLAB only treats as keywords inside a
 * specific enclosing block, and as ordinary identifiers everywhere else.
 * `properties`/`methods`/`events`/`enumeration` open blocks only directly
 * inside a `classdef`; `arguments` opens a block only at the top of a function
 * body. Outside those contexts they are valid variable/function names
 * (e.g. `events = sortrows(...)`, `methods(obj)`).
 */
const CONTEXTUAL_KEYWORDS = new Set<Token>([
  Token.Properties,
  Token.Methods,
  Token.Events,
  Token.Enumeration,
  Token.Arguments,
]);

/** Plain (non-OOP, non-function) block openers that pair with a single 'end'. */
const SIMPLE_BLOCK_OPENERS = new Set<Token>([
  Token.If,
  Token.For,
  Token.ParFor,
  Token.While,
  Token.Switch,
  Token.Try,
]);

/**
 * Decide whether a contextual keyword at index `i` is opening a block (as
 * opposed to being used as an identifier). A block keyword is followed either
 * by a newline (no attributes) or by a parenthesized attribute/size list that
 * is NOT a `(...) =` indexed assignment.
 */
function contextualKeywordOpensBlock(tokens: TokenInfo[], i: number): boolean {
  const nt = tokens[i + 1];
  if (!nt) return false;
  if (nt.token === Token.Newline) return true;
  if (nt.token === Token.LParen) {
    let depth = 0;
    let j = i + 1;
    for (; j < tokens.length; j++) {
      const tt = tokens[j].token;
      if (tt === Token.LParen) depth++;
      else if (tt === Token.RParen) {
        depth--;
        if (depth === 0) {
          j++;
          break;
        }
      }
    }
    return tokens[j]?.token !== Token.Assign;
  }
  return false;
}

/**
 * The lexer is context-free and always emits the keyword token for contextual
 * keywords. Walk the token stream tracking block nesting and demote any
 * contextual keyword that is really an identifier back to `Token.Ident`.
 */
function reclassifyContextualKeywords(tokens: TokenInfo[]): void {
  const blockStack: ("classdef" | "function" | "block")[] = [];
  let groupDepth = 0;
  let atStmtStart = true;

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const t = tok.token;

    if (t === Token.LParen || t === Token.LBracket || t === Token.LBrace) {
      groupDepth++;
      atStmtStart = false;
      continue;
    }
    if (t === Token.RParen || t === Token.RBracket || t === Token.RBrace) {
      groupDepth--;
      atStmtStart = false;
      continue;
    }
    // Statement separators reset "start of statement" (only at top level).
    if (
      t === Token.Newline ||
      t === Token.Semicolon ||
      (t === Token.Comma && groupDepth === 0)
    ) {
      atStmtStart = true;
      continue;
    }

    if (CONTEXTUAL_KEYWORDS.has(t)) {
      const ctx = blockStack[blockStack.length - 1];
      // `arguments` opens a block at the top of a function body; the other
      // contextual keywords open blocks directly inside a classdef. (numbl
      // also accepts an `arguments` block as a classdef member.)
      const ctxOK =
        t === Token.Arguments
          ? ctx === "function" || ctx === "classdef"
          : ctx === "classdef";
      // A contextual keyword can only open a block at the start of a statement
      // (top level) in the right enclosing block; anywhere else it's an
      // ordinary identifier.
      if (
        groupDepth === 0 &&
        atStmtStart &&
        ctxOK &&
        contextualKeywordOpensBlock(tokens, i)
      ) {
        blockStack.push("block");
      } else {
        tok.token = Token.Ident;
      }
      atStmtStart = false;
      continue;
    }

    if (groupDepth === 0) {
      if (t === Token.ClassDef) blockStack.push("classdef");
      else if (t === Token.Function) blockStack.push("function");
      else if (SIMPLE_BLOCK_OPENERS.has(t)) blockStack.push("block");
      else if (t === Token.End) blockStack.pop();
    }

    atStmtStart = false;
  }
}

/**
 * Tokens that open a block requiring a matching 'end' keyword.
 */
const BLOCK_OPENERS = new Set<Token>([
  Token.If,
  Token.For,
  Token.ParFor,
  Token.While,
  Token.Switch,
  Token.Try,
  // OOP block keywords
  Token.ClassDef,
  Token.Properties,
  Token.Methods,
  Token.Events,
  Token.Enumeration,
]);

/**
 * Insert synthetic 'end' tokens for functions that don't have them.
 *
 * Scans the token array tracking block nesting depth. A top-level 'function'
 * keyword (depth 0) signals the start of a function. When we encounter the
 * next top-level 'function' or reach EOF while inside a function, we insert
 * a synthetic 'end' token.
 *
 * Returns the new token array, or null if the tokens don't match the
 * no-end-functions pattern (e.g., no functions found, or mixed end/no-end).
 */
function insertFunctionEndTokens(tokens: TokenInfo[]): TokenInfo[] | null {
  // Scan tokens tracking block nesting depth.
  // depth=0 means we're at top level (outside any function).
  // depth=1 means we're at the top level of a function body.
  // When we see a 'function' token at depth 0 or 1, it's a sibling function.
  // At depth 0 it starts the first function; at depth 1 it means the previous
  // function didn't have 'end' and a new sibling is starting.

  const result: TokenInfo[] = [];
  let depth = 0;
  let inFunction = false;
  let groupDepth = 0; // tracks nesting inside (), [], {}
  const functionEndStatus: boolean[] = []; // true = had end, false = did not

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];

    // Track grouping constructs — 'end' inside parens/brackets/braces
    // is an indexing sentinel (e.g. a(2:end)), not a block closer.
    if (
      tok.token === Token.LParen ||
      tok.token === Token.LBracket ||
      tok.token === Token.LBrace
    ) {
      groupDepth++;
    } else if (
      tok.token === Token.RParen ||
      tok.token === Token.RBracket ||
      tok.token === Token.RBrace
    ) {
      groupDepth--;
    }

    // A 'function' keyword at depth 0 (not inside any function) or depth 1
    // (at the top level of a function body, i.e. sibling) signals the start
    // of a new top-level function.
    if (
      tok.token === Token.Function &&
      (depth === 0 || (depth === 1 && inFunction))
    ) {
      if (inFunction) {
        // Previous function ended implicitly (no 'end')
        functionEndStatus.push(false);
        depth--; // Close the previous function (depth 1 → 0)
        // Insert synthetic newline + 'end' before this function token.
        // The newline is needed to terminate any command-syntax statement
        // on the last line (e.g. "grid off") so the 'end' token is not
        // consumed as a command argument.
        const prevTok = tokens[i - 1] || tok;
        result.push({
          token: Token.Newline,
          lexeme: "\n",
          position: prevTok.end,
          end: prevTok.end,
        });
        result.push({
          token: Token.End,
          lexeme: "end",
          position: prevTok.end,
          end: prevTok.end,
        });
      }
      inFunction = true;
      depth++; // Enter the new function (depth 0 → 1)
      result.push(tok);
      continue;
    }

    if (BLOCK_OPENERS.has(tok.token)) {
      depth++;
    } else if (
      tok.token === Token.End &&
      groupDepth === 0 &&
      tokens[i - 1]?.token !== Token.Dot
    ) {
      // `end` after `.` is a struct field named 'end' (e.g. `s.end = 1;`),
      // not a block closer — skip it. (Index sentinels like `a(2:end)` are
      // already excluded by the groupDepth check.)
      depth--;
      if (depth === 0 && inFunction) {
        // This 'end' closes the current top-level function
        inFunction = false;
        functionEndStatus.push(true);
      }
    }

    result.push(tok);
  }

  // If we're still inside a function at EOF, insert a closing newline + 'end'.
  // The newline is needed to terminate any command-syntax statement on the
  // last line (e.g. "grid off") so the 'end' token is not consumed as a
  // command argument.
  if (inFunction) {
    functionEndStatus.push(false);
    const lastTok = tokens[tokens.length - 1];
    if (lastTok) {
      result.push({
        token: Token.Newline,
        lexeme: "\n",
        position: lastTok.end,
        end: lastTok.end,
      });
      result.push({
        token: Token.End,
        lexeme: "end",
        position: lastTok.end,
        end: lastTok.end,
      });
    }
  }

  // Validate: either ALL functions had 'end' or NONE did
  const hasEnd = functionEndStatus.filter(x => x).length;
  const noEnd = functionEndStatus.filter(x => !x).length;

  if (noEnd === 0) {
    // All functions already had 'end' — shouldn't reach here, but be safe
    return null;
  }
  if (hasEnd > 0 && noEnd > 0) {
    // Mixed: some have end, some don't — this is an error
    return null;
  }

  // All functions lacked 'end' — return the fixed token array
  return result;
}

/**
 * Tokenize input into a reusable token array.
 */
function tokenizeInput(input: string): TokenInfo[] {
  const toks = tokenizeDetailed(input);
  const tokens: TokenInfo[] = [];

  for (const t of toks) {
    if (t.token === Token.Error) {
      const line = offsetToLine(input, t.start);
      throw new SyntaxError(
        `Invalid token: '${t.lexeme}'`,
        t.start,
        t.lexeme,
        null,
        line
      );
    }
    // Skip layout-only tokens from lexing
    if (t.token === Token.Ellipsis || t.token === Token.Section) {
      continue;
    }
    tokens.push({
      token: t.token,
      lexeme: t.lexeme,
      position: t.start,
      end: t.end,
    });
  }

  reclassifyContextualKeywords(tokens);

  return tokens;
}
