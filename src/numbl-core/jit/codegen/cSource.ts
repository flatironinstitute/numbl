/**
 * Text-level passes over emitted C source, built on a small lexer that
 * knows comments and string / character literals.
 *
 * Spec names are mangled with a hash of their argument types
 * (`specializeUserFunction`), so the C a widened spec emits differs from
 * the exact-shape spec it replaces only in those names (and in the type
 * comments) even when the code is the same. `renameCIdentifiers` gives the
 * specs of a module names derived from their position instead, and the
 * host keys its compile cache on `stripCComments` of the result, so such
 * modules share one compiled object.
 */

const IDENT_START = /[A-Za-z_]/;
const IDENT_PART = /[A-Za-z0-9_]/;

/** Calls `onCode` for each run of code (with identifiers split out) and
 *  `onComment` for each comment, in source order. String and character
 *  literals are passed to `onCode` whole. */
function lexC(
  src: string,
  onCode: (text: string, isIdent: boolean) => void,
  onComment: (text: string) => void
): void {
  const n = src.length;
  let i = 0;
  let codeStart = 0;
  const flush = (end: number) => {
    if (end > codeStart) onCode(src.slice(codeStart, end), false);
  };
  while (i < n) {
    const c = src[i];
    if (c === "/" && src[i + 1] === "*") {
      flush(i);
      const close = src.indexOf("*/", i + 2);
      const end = close < 0 ? n : close + 2;
      onComment(src.slice(i, end));
      i = codeStart = end;
    } else if (c === "/" && src[i + 1] === "/") {
      flush(i);
      const nl = src.indexOf("\n", i + 2);
      const end = nl < 0 ? n : nl;
      onComment(src.slice(i, end));
      i = codeStart = end;
    } else if (c === '"' || c === "'") {
      i++;
      while (i < n && src[i] !== c && src[i] !== "\n") {
        i += src[i] === "\\" ? 2 : 1;
      }
      i++;
    } else if (IDENT_START.test(c)) {
      flush(i);
      let j = i + 1;
      while (j < n && IDENT_PART.test(src[j])) j++;
      onCode(src.slice(i, j), true);
      i = codeStart = j;
    } else if (c >= "0" && c <= "9") {
      // A number such as `1e5` or `0x1fUL`: not an identifier.
      i++;
      while (i < n && IDENT_PART.test(src[i])) i++;
    } else {
      i++;
    }
  }
  flush(n);
}

/** `src` with every identifier in `names` replaced by its mapping.
 *  Comments and literals are left as they are. */
export function renameCIdentifiers(
  src: string,
  names: ReadonlyMap<string, string>
): string {
  const out: string[] = [];
  lexC(
    src,
    (text, isIdent) => out.push(isIdent ? (names.get(text) ?? text) : text),
    text => out.push(text)
  );
  return out.join("");
}

/** `src` without its comments. */
export function stripCComments(src: string): string {
  const out: string[] = [];
  lexC(
    src,
    text => out.push(text),
    () => {}
  );
  return out.join("");
}
