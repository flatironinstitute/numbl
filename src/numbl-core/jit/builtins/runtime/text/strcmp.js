/* JS sibling of strcmp.h. The input is either a raw JS string (numbl-style
 * scalar handle) or a `{jitTag:"char", value:string}` char wrapper.
 * The helper normalizes either form to its raw bytes before comparing.
 */

function _jit_strcmp_text(v) {
  if (typeof v === "string") return v;
  if (v && v.jitTag === "char" && typeof v.value === "string") return v.value;
  return null;
}

export function jit_strcmp(a, b) {
  const sa = _jit_strcmp_text(a);
  const sb = _jit_strcmp_text(b);
  if (sa === null || sb === null) return 0;
  return sa === sb ? 1 : 0;
}

export function jit_strcmpi(a, b) {
  const sa = _jit_strcmp_text(a);
  const sb = _jit_strcmp_text(b);
  if (sa === null || sb === null) return 0;
  return sa.toLowerCase() === sb.toLowerCase() ? 1 : 0;
}
