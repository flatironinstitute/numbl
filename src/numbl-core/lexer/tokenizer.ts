import { SpannedToken, Token } from "./types.js";
import { KEYWORDS } from "./keywords.js";
import {
  isAlpha,
  isAlnum,
  isDigit,
  isWhitespace,
  isValueToken,
  findLineTerminator,
} from "./helpers.js";
import {
  TOKEN_CONFIG,
  buildTwoCharMap,
  buildSingleCharMap,
  getSingleCharSpecial,
} from "./token-config.js";

// Build operator lookup maps at module initialization
const TWO_CHAR_OPS = buildTwoCharMap();
const SINGLE_CHAR_OPS = buildSingleCharMap();

/**
 * Tokenize source into a flat token array (no newlines).
 */
export function tokenize(input: string): Token[] {
  return tokenizeDetailed(input)
    .map(t => t.token)
    .filter(tok => tok !== Token.Newline);
}

/**
 * Tokenize source with full span information.
 */
export function tokenizeDetailed(input: string): SpannedToken[] {
  const out: SpannedToken[] = [];
  let pos = 0;
  let lineStart = true;

  while (pos < input.length) {
    // Skip horizontal whitespace (spaces, tabs, carriage returns) but NOT newlines
    if (isWhitespace(input[pos])) {
      pos++;
      continue;
    }

    const ch = input[pos];

    // ── Newlines ──
    if (ch === TOKEN_CONFIG.special.newline.pattern) {
      const start = pos;
      out.push({ token: Token.Newline, lexeme: ch, start, end: pos + 1 });
      pos++;
      // Coalesce multiple newlines if configured
      if (TOKEN_CONFIG.special.newline.coalesceMultiple) {
        while (pos < input.length && input[pos] === "\n") {
          out[out.length - 1].end = pos + 1;
          out[out.length - 1].lexeme += "\n";
          pos++;
        }
      }
      if (TOKEN_CONFIG.special.newline.setsLineStart) {
        lineStart = true;
      }
      continue;
    }

    // ── Ellipsis `...` (line continuation) ──
    const ellipsisCfg = TOKEN_CONFIG.special.ellipsis;
    if (
      input.slice(pos, pos + ellipsisCfg.pattern.length) === ellipsisCfg.pattern
    ) {
      const start = pos;
      pos += ellipsisCfg.pattern.length;
      // Skip remainder of physical line if configured
      if (ellipsisCfg.consumeToEOL) {
        const lt = findLineTerminator(input, pos);
        if (lt) {
          pos = ellipsisCfg.consumeTrailingNewline ? lt[0] + lt[1] : lt[0];
        } else {
          pos = input.length;
        }
      }
      out.push({
        token: Token[ellipsisCfg.token as keyof typeof Token],
        lexeme: ellipsisCfg.pattern,
        start,
        end: start + ellipsisCfg.pattern.length,
      });
      continue;
    }

    // ── Block comment ──
    // %{ only starts a block comment when the rest of the line
    // after %{ is empty or whitespace-only. If there's other content on
    // the same line (e.g. `%{ }` or `%{ not a block comment`), it's a
    // regular line comment.
    const blockCfg = TOKEN_CONFIG.comments.blockComment;
    if (input.slice(pos, pos + blockCfg.start.length) === blockCfg.start) {
      const afterStart = pos + blockCfg.start.length;
      const eolForBlock = input.indexOf("\n", afterStart);
      const restOfLine =
        eolForBlock >= 0
          ? input.slice(afterStart, eolForBlock)
          : input.slice(afterStart);
      if (restOfLine.trim().length > 0) {
        // Not a block comment — fall through to line comment handler
        const eol = input.indexOf("\n", pos);
        pos = eol >= 0 ? eol : input.length;
        continue;
      }
      const rest = input.slice(afterStart);
      const endIdx = rest.indexOf(blockCfg.end);
      if (endIdx >= 0) {
        pos = pos + blockCfg.start.length + endIdx + blockCfg.end.length;
      } else {
        pos = input.length;
      }
      // Consume trailing newline if configured
      if (blockCfg.consumeTrailingNewline) {
        const lt = findLineTerminator(input, pos);
        if (lt && lt[0] === pos) {
          pos += lt[1];
          lineStart = true;
        }
      }
      continue;
    }

    // ── Section marker — must be at line start ──
    const sectionCfg = TOKEN_CONFIG.comments.sectionMarker;
    if (
      input.slice(pos, pos + sectionCfg.pattern.length) === sectionCfg.pattern
    ) {
      if (!sectionCfg.requireLineStart || lineStart) {
        const start = pos;
        let end = pos;
        if (sectionCfg.consumeToEOL) {
          while (end < input.length && input[end] !== "\n") {
            end++;
          }
        }
        const lexeme = input.slice(start, end);
        pos = end;
        // Consume trailing newline if configured
        if (sectionCfg.consumeTrailingNewline) {
          const lt = findLineTerminator(input, pos);
          if (lt && lt[0] === pos) {
            pos += lt[1];
          }
        }
        lineStart = true;
        out.push({
          token: Token[sectionCfg.token as keyof typeof Token],
          lexeme,
          start,
          end,
        });
        continue;
      } else if (sectionCfg.fallbackToLineComment) {
        // Not at line start → treat as line comment
        const eol = input.indexOf("\n", pos);
        pos = eol >= 0 ? eol : input.length;
        continue;
      }
    }

    // ── Line comment ──
    const lineCfg = TOKEN_CONFIG.comments.lineComment;
    if (ch === lineCfg.start) {
      const start = pos;
      const eol = input.indexOf("\n", pos);
      const end = eol >= 0 ? eol : input.length;
      // Check for %!numbl: magic directive
      const DIRECTIVE_PREFIX = "%!numbl:";
      if (
        input.slice(pos, pos + DIRECTIVE_PREFIX.length) === DIRECTIVE_PREFIX
      ) {
        const lexeme = input.slice(start, end).trimEnd();
        out.push({ token: Token.Directive, lexeme, start, end });
      }
      pos = end;
      continue;
    }

    // ── Strings (configured delimiters) ──
    for (const strCfg of TOKEN_CONFIG.strings) {
      if (ch === strCfg.delimiter) {
        const start = pos;

        // Handle transpose disambiguation if configured
        if (strCfg.disambiguation?.type === "transpose") {
          const prev = out.length > 0 ? out[out.length - 1] : undefined;
          const isAdjacent = prev !== undefined && prev.end === start;
          const prevIsValueOrDot =
            prev !== undefined &&
            (prev.token === Token.Dot || isValueToken(prev.token));

          if (
            strCfg.disambiguation.conditions.requireAdjacency &&
            strCfg.disambiguation.conditions.requirePrevValueOrDot &&
            isAdjacent &&
            prevIsValueOrDot
          ) {
            const transposeToken =
              Token[strCfg.disambiguation.transposeToken as keyof typeof Token];
            out.push({
              token: transposeToken,
              lexeme: strCfg.delimiter,
              start,
              end: pos + 1,
            });
            pos++;
            continue;
          }
        }

        // Scan string literal
        let j = pos + 1; // skip opening delimiter
        let ok = false;
        while (j < input.length) {
          const c = input[j];
          if (c === strCfg.delimiter) {
            // Check for escape sequence (doubled delimiter)
            if (j + 1 < input.length && input[j + 1] === strCfg.delimiter) {
              j += 2; // escaped delimiter
            } else {
              ok = true;
              j++; // include closing delimiter
              break;
            }
          } else if (!strCfg.allowMultiline && (c === "\n" || c === "\r")) {
            break; // unterminated
          } else {
            j++;
          }
        }

        if (ok) {
          const lexeme = input.slice(pos, j);
          lineStart = false;
          out.push({
            token: Token[strCfg.token as keyof typeof Token],
            lexeme,
            start: pos,
            end: j,
          });
          pos = j;
        } else {
          // Unterminated → Error
          out.push({
            token: Token.Error,
            lexeme: strCfg.delimiter,
            start: pos,
            end: pos + 1,
          });
          pos++;
        }
        continue;
      }
    }
    if (ch === '"' || ch === "'") {
      // Handled by string config above
      continue;
    }

    // ── Two-character operators (must check before single-char) ──
    if (pos + 1 < input.length) {
      const two = input.slice(pos, pos + 2);
      const twoTok = TWO_CHAR_OPS.get(two);
      if (twoTok !== undefined) {
        out.push({ token: twoTok, lexeme: two, start: pos, end: pos + 2 });
        lineStart = false;
        pos += 2;
        continue;
      }
    }

    // ── Single-character operators / punctuation ──
    {
      const singleTok = SINGLE_CHAR_OPS.get(ch);
      // Don't consume '.' as Dot when it starts a leading-dot float like .9
      // But do consume it if adjacent to a value token (e.g. 1..3 → Float, Dot, Integer)
      const isLeadingDotFloat =
        ch === "." &&
        pos + 1 < input.length &&
        isDigit(input[pos + 1]) &&
        !(
          out.length > 0 &&
          out[out.length - 1].end === pos &&
          isValueToken(out[out.length - 1].token)
        );
      if (singleTok !== undefined && !isLeadingDotFloat) {
        out.push({ token: singleTok, lexeme: ch, start: pos, end: pos + 1 });
        lineStart = false;
        pos++;

        // Handle special behaviors
        const special = getSingleCharSpecial(ch);
        if (special === "semicolonQuoteHeuristic") {
          // After semicolon, eagerly scan for a single-quoted string
          let offset = pos;
          while (
            offset < input.length &&
            (input[offset] === " " ||
              input[offset] === "\t" ||
              input[offset] === "\r")
          ) {
            offset++;
          }
          if (offset < input.length && input[offset] === "'") {
            // Find the string config for single quotes
            const sqCfg = TOKEN_CONFIG.strings.find(s => s.delimiter === "'");
            if (sqCfg) {
              let j = offset + 1;
              let ok = false;
              while (j < input.length) {
                const c = input[j];
                if (c === "'") {
                  if (j + 1 < input.length && input[j + 1] === "'") {
                    j += 2; // escaped quote
                  } else {
                    ok = true;
                    j++;
                    break;
                  }
                } else if (!sqCfg.allowMultiline && c === "\n") {
                  break;
                } else {
                  j++;
                }
              }
              if (ok) {
                const lexeme = input.slice(offset, j);
                const sqToken = Token[sqCfg.token as keyof typeof Token];
                out.push({ token: sqToken, lexeme, start: offset, end: j });
                pos = j;
              }
            }
          }
        }
        continue;
      }
    }

    // ── Numbers ──
    // Also match leading decimal point followed by digit (e.g. .9, .123)
    // but not when adjacent to a value token (e.g. 1..3 where . is the Dot operator)
    const isLeadingDotNum =
      ch === "." &&
      pos + 1 < input.length &&
      isDigit(input[pos + 1]) &&
      !(
        out.length > 0 &&
        out[out.length - 1].end === pos &&
        isValueToken(out[out.length - 1].token)
      );
    if (isDigit(ch) || isLeadingDotNum) {
      const numCfg = TOKEN_CONFIG.numbers;
      const start = pos;

      let isFloat = false;

      if (ch === ".") {
        // Leading decimal point: .9, .123, .5e3, etc.
        isFloat = true;
        pos++; // consume '.'
        while (
          pos < input.length &&
          (isDigit(input[pos]) ||
            (numCfg.allowUnderscores && input[pos] === "_"))
        ) {
          pos++;
        }
      } else {
        // Scan integer part with optional underscores
        pos++;
        while (
          pos < input.length &&
          (isDigit(input[pos]) ||
            (numCfg.allowUnderscores && input[pos] === "_"))
        ) {
          pos++;
        }

        // Check for decimal point (but not `..` or element-wise operators like `.*`)
        if (pos < input.length && input[pos] === numCfg.decimalPoint) {
          const afterDot = pos + 1 < input.length ? input[pos + 1] : "";
          // Check if the dot is part of an element-wise operator
          if (numCfg.dotOperatorPrefixes.includes(afterDot)) {
            // Don't consume the dot — it's part of an element-wise operator
          } else {
            isFloat = true;
            pos++; // consume '.'
            while (
              pos < input.length &&
              (isDigit(input[pos]) ||
                (numCfg.allowUnderscores && input[pos] === "_"))
            ) {
              pos++;
            }
          }
        }
      }

      // Exponent part
      if (pos < input.length && numCfg.exponentChars.includes(input[pos])) {
        isFloat = true;
        pos++;
        if (pos < input.length && (input[pos] === "+" || input[pos] === "-")) {
          pos++;
        }
        while (
          pos < input.length &&
          (isDigit(input[pos]) ||
            (numCfg.allowUnderscores && input[pos] === "_"))
        ) {
          pos++;
        }
      }

      let lexeme = input.slice(start, pos);
      // Strip underscores if configured
      if (numCfg.stripUnderscores) {
        lexeme = lexeme.replace(/_/g, "");
      }
      // Normalize FORTRAN-style exponent markers (d/D → e) so parseFloat works
      if (isFloat) {
        lexeme = lexeme.replace(/[dD]/g, "e");
      }

      const tok = isFloat
        ? Token[numCfg.floatToken as keyof typeof Token]
        : Token[numCfg.integerToken as keyof typeof Token];
      lineStart = false;
      out.push({ token: tok, lexeme, start, end: pos });
      continue;
    }

    // ── Identifiers / keywords ──
    if (isAlpha(ch)) {
      const start = pos;
      pos++;
      while (pos < input.length && isAlnum(input[pos])) {
        pos++;
      }
      const lexeme = input.slice(start, pos);
      let kwTok = KEYWORDS.get(lexeme);
      // 'end' followed by '(' is a function/method name, not a block closer
      if (kwTok === Token.End) {
        let peek = pos;
        while (peek < input.length && isWhitespace(input[peek])) peek++;
        if (peek < input.length && input[peek] === "(") {
          kwTok = undefined;
        }
      }
      if (kwTok !== undefined) {
        lineStart = false;
        out.push({ token: kwTok, lexeme, start, end: pos });
      } else {
        lineStart = false;
        out.push({ token: Token.Ident, lexeme, start, end: pos });
      }
      continue;
    }

    // ── Unknown character → Error ──
    out.push({ token: Token.Error, lexeme: ch, start: pos, end: pos + 1 });
    lineStart = false;
    pos++;
  }

  return out;
}
