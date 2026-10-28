/**
 * Search-path relative names and private-directory scoping.
 *
 * Function names and private-function scopes are derived from a file's path
 * relative to the search path that contains it, so that `/a/+pkg/f.m` on
 * search path `/a` is `pkg.f` and `/a/private/g.m` is visible from `/a/*.m`.
 */

/** Path of `absolutePath` relative to the most specific (longest) search
 *  path containing it, so overlapping paths (`/a/b` and `/a/b/tools`) work.
 *  Returned unchanged when there are no search paths or none matches. */
export function relativeToSearchPath(
  absolutePath: string,
  searchPaths: readonly string[]
): string {
  if (searchPaths.length === 0) return absolutePath;
  let bestPrefix = "";
  for (const sp of searchPaths) {
    const prefix = sp.endsWith("/") ? sp : sp + "/";
    if (absolutePath.startsWith(prefix) && prefix.length > bestPrefix.length) {
      bestPrefix = prefix;
    }
  }
  return bestPrefix ? absolutePath.slice(bestPrefix.length) : absolutePath;
}

/** Scope key of a directory given as relative path segments: `""` for a
 *  search-path root, otherwise the segments joined with a trailing `/`. */
function scopeKey(dirs: string[]): string {
  return dirs.length > 0 ? dirs.join("/") + "/" : "";
}

/** Scope key under which a file in a `private/` directory is registered (the
 *  directory containing `private/`), or null when the file is not in one.
 *  `parts` is the file's search-path relative path split on `/`. */
export function privateDirScope(parts: string[]): string | null {
  const privateIdx = parts.indexOf("private");
  return privateIdx >= 0 ? scopeKey(parts.slice(0, privateIdx)) : null;
}

/** Scope key of the private functions a file can call: its own directory,
 *  or for a file inside `private/` the directory above (so private functions
 *  can call each other). Matches `privateDirScope` of the callee. */
export function callerPrivateScope(
  fileName: string,
  searchPaths: readonly string[]
): string {
  const parts = relativeToSearchPath(fileName, searchPaths).split("/");
  parts.pop(); // file name
  if (parts.length > 0 && parts[parts.length - 1] === "private") parts.pop();
  return scopeKey(parts);
}
