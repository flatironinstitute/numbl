const PATH_TOKENS: Record<string, string> = {
  "**/": "(?:.*/)?",
  "**": ".*",
  "*": "[^/]*",
  "?": "[^/]",
};

/**
 * Compile a wildcard pattern to an anchored RegExp.
 *
 * Default (single name): `*` matches any run of characters and `?` any one
 * character, as in `dir('*.m')`, `delete('*.txt')` or `who('a*')`.
 *
 * `paths: true` matches slash-separated paths: `*` and `?` stay within one
 * segment, `**` matches anything, and `**` followed by `/` matches zero or
 * more directory levels (so a double-star prefix before `x.m` also matches a
 * top-level `x.m`).
 */
export function globToRegExp(
  pattern: string,
  opts: { paths?: boolean } = {}
): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  let body: string;
  if (!opts.paths) {
    body = escaped.replace(/\*/g, ".*").replace(/\?/g, ".");
  } else {
    body = escaped.replace(/\*\*\/|\*\*|\*|\?/g, m => PATH_TOKENS[m]);
  }
  return new RegExp("^" + body + "$");
}
